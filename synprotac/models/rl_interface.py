import torch 
from typing import Dict, List
from .model import SynthesisGraphTransformer
from pathlib import Path 
from ..data import SynthesisRoute_DataModule
from ..utils.initlib import configure_fs,disable_lib_stdout
from ..data.synthesis_route import SynthesisRoute, ActionType, ActionToken
from ..utils.functional import smiles_to_graph
import os 
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy  
from .rl_module import RL_LightningModule,MolecularScorer   

class Synprotac_RL_Model:
    def __init__(self, 
                 num_atom_classes=100, num_bond_classes=5, num_reaction_classes=91, num_reagent_classes=483, num_action_types=4,
                 hidden_dim=512, 
                 encoder_depth=8, decoder_depth=6, 
                 n_heads=8, dim_head=64, edge_dim=128, 
                 num_out_fps=1, warmup_prob=0.01,
                 max_atoms=70, max_sequence_length=12,
                 num_workers=20, fp_dim=256, prior_checkpoint_path: Path = None, 
                ):

        self.num_atom_classes=num_atom_classes
        self.num_bond_classes=num_bond_classes
        self.num_reaction_classes=num_reaction_classes
        self.num_reagent_classes=num_reagent_classes
        self.num_action_types=num_action_types
        self.hidden_dim=hidden_dim
        self.encoder_depth=encoder_depth
        self.decoder_depth=decoder_depth
        self.n_heads=n_heads
        self.dim_head=dim_head
        self.edge_dim=edge_dim
        self.num_out_fps=num_out_fps
        self.warmup_prob=warmup_prob
        self.max_atoms=max_atoms
        self.max_sequence_length=max_sequence_length
        self.num_workers=num_workers
        self.fp_dim=fp_dim
        self.prior_checkpoint_path=prior_checkpoint_path
        disable_lib_stdout()
        configure_fs()
        self.__build_network_arch()

    def __build_network_arch(self):
        
        self.network= SynthesisGraphTransformer( 
                 num_atom_classes=self.num_atom_classes, 
                 num_bond_classes=self.num_bond_classes, 
                 num_reaction_classes=self.num_reaction_classes,
                 num_reagent_classes=self.num_reagent_classes+1,
                 num_action_types=self.num_action_types,
                 hidden_dim=self.hidden_dim, 
                 encoder_depth=self.encoder_depth, 
                 decoder_depth=self.decoder_depth, 
                 n_heads=self.n_heads, 
                 dim_head=self.dim_head, 
                 edge_dim=self.edge_dim, 
                 num_out_fps=self.num_out_fps, 
                 fp_dim=self.fp_dim, 
                 warmup_prob=self.warmup_prob)

        self.prior_network = deepcopy(self.network)

        if self.prior_checkpoint_path is not None:
            checkpoint = torch.load(self.prior_checkpoint_path, map_location="cpu")
            new_state_dict = {}
            for key in checkpoint['state_dict'].keys():
                new_state_dict[key.replace("model.","")]=checkpoint['state_dict'][key]
            self.prior_network.load_state_dict(new_state_dict)
            print(f"Loaded prior model weights from {self.prior_checkpoint_path}")

        self.network.load_state_dict(self.prior_network.state_dict())

        return

    def create_RL_lightning_module(self,hparams=None,load_cpkt=None):
        default_hparams = {
            "num_out_fps":self.num_out_fps,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "loss_weights": {'action_type': 1.0, 'reaction':1.0, 'fingerprint':1.0},
            "max_sequence_length":7,
            "num_reaction_classes": self.num_reaction_classes,
            "num_reagent_classes": self.num_reagent_classes,
            "num_action_types": self.num_action_types,
        }

        if hparams is not None:
            default_hparams.update(hparams)

        if load_cpkt is not None:
            lightning_module = RL_LightningModule.load_from_checkpoint(
                load_cpkt,
                network=self.network,
                prior_network=self.prior_network,
                **default_hparams
            )

        else:
            lightning_module = RL_LightningModule(
                network=self.network,
                prior_network=self.prior_network,
                **default_hparams
            )

        return lightning_module

    def RL(self, 
        warhead_smiles: str, 
        e3_ligand_smiles: str, 
        warhead_protected_patts: List = [],
        e3_ligand_protected_patts: List = [],
        reaction_templates_file: str = "templates.txt", 
        reagents_file: str = "reagents.txt", 
        scorer: MolecularScorer = None,
        savepath: Path = Path("./models"),
        project_name: str = "Synprotac-RL",
        load_cpkt: Path = None, 
        val_check_epochs: int = 1,
        gradient_clip_val: float = 1.0,
        log_steps: int = 1,
        debug: bool = False,
        ngpus: int = 1,
        batchsize: int = 64,
        epochs: int = 10,
        learning_rate = 1e-4,
        acc_batches = 1,
        rl_samples_path: str = "./rl-samples",
    ):
            
        assert scorer is not None, "Please provide a scorer for RL training."

        empty_synroute=self.create_empty_SynthesisRoutes(warhead_smiles, e3_ligand_smiles)

        self.data_module = SynthesisRoute_DataModule(
            train_routes=[empty_synroute]*batchsize*100*ngpus,
            val_routes=[empty_synroute]*batchsize*2*ngpus,
            max_atoms = self.max_atoms, 
            max_sequence_length = self.max_sequence_length,
            num_workers = self.num_workers,
            batchsize = batchsize
        )
        
        rl_hparams = {
            "warhead": warhead_smiles,
            "e3_ligand": e3_ligand_smiles,
            "warhead_protected_patts": warhead_protected_patts,
            "e3_ligand_protected_patts": e3_ligand_protected_patts,
            "reaction_templates_file": reaction_templates_file,
            "reagents_file": reagents_file,
            "learning_rate": learning_rate,
            "weight_decay": 1e-5,
            "max_sequence_length": self.max_sequence_length,
            "scorer": scorer,
            "reward_sigma": 10,
            "rl_samples_path": rl_samples_path,
            }

        self.lightning_module = self.create_RL_lightning_module(hparams=rl_hparams, load_cpkt=load_cpkt)

        self.lightning_module.init_action_searcher( reaction_templates_file , reagents_file ) 

        os.makedirs("./TensorBoard-RL", exist_ok=True)

        logger = TensorBoardLogger(f"./TensorBoard-RL", name=project_name, version=None)

        lr_monitor = LearningRateMonitor(logging_interval="step")
 
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        checkpointing = ModelCheckpoint(dirpath=savepath, save_top_k=3, every_n_epochs=1, monitor="valid_scores/val", mode="max", save_last=True)

        trainer = pl.Trainer(
                devices=ngpus,
                min_epochs=epochs,
                max_epochs=epochs,
                logger=logger,
                log_every_n_steps=log_steps,
                accumulate_grad_batches=acc_batches,
                gradient_clip_val=gradient_clip_val,
                check_val_every_n_epoch=val_check_epochs,
                callbacks=[lr_monitor, checkpointing],
                precision="32",
                strategy="ddp_find_unused_parameters_true",
        )

        trainer.fit(self.lightning_module, self.data_module)

        return

    def create_empty_SynthesisRoutes(self, warhead_smiles,e3_ligand_smiles):
        # Extract reaction sites from SMILES
        warhead_sites = SynthesisRoute._extract_reaction_sites(warhead_smiles)
        e3_ligand_sites = SynthesisRoute._extract_reaction_sites(e3_ligand_smiles)
        
        # Convert SMILES to graph representations
        warhead_atoms, warhead_bond_indices, warhead_bond_types = smiles_to_graph(warhead_smiles)
        e3_ligand_atoms, e3_ligand_bond_indices, e3_ligand_bond_types = smiles_to_graph(e3_ligand_smiles)
        
        # Create empty synthesis route with only start token
        empty_route = SynthesisRoute(
            warhead_smiles=warhead_smiles,
            e3_ligand_smiles=e3_ligand_smiles,
            warhead_atoms=warhead_atoms,
            e3_ligand_atoms=e3_ligand_atoms,
            warhead_bond_indices=warhead_bond_indices,
            e3_ligand_bond_indices=e3_ligand_bond_indices,
            warhead_bond_types=warhead_bond_types,
            e3_ligand_bond_types=e3_ligand_bond_types,
            warhead_reaction_sites=warhead_sites,
            e3_ligand_reaction_sites=e3_ligand_sites,
            action_tokens=[ActionToken(action_type=ActionType.START, reagent_id=0)],
            final_product_smiles=''
        )
        return empty_route 

    
    
    
