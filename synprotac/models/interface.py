import torch 
from typing import Dict, List
from .model import SynthesisGraphTransformer
from pathlib import Path 
from ..data import SynthesisRoute_DataModule
from .lightning_module import SynthesisRouteLightningModule, Action_Searcher
from ..utils.initlib import configure_fs,disable_lib_stdout
from ..data.synthesis_route import SynthesisRoute, ActionType, ActionToken
from ..utils.functional import smiles_to_graph
import os 
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

class Synprotac_Model:
    def __init__(self,
                 num_atom_classes=100, num_bond_classes=5, num_reaction_classes=91, num_reagent_classes=483, num_action_types=4,
                 hidden_dim=512, 
                 encoder_depth=8, decoder_depth=6, 
                 n_heads=8, dim_head=64, edge_dim=128, 
                 num_out_fps=1, warmup_prob=0.01,
                 max_atoms=70, max_sequence_length=12,
                 num_workers=20,fp_dim=256
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

    def create_lightning_module(self,hparams=None,load_cpkt=None):
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
            lightning_module=SynthesisRouteLightningModule.load_from_checkpoint(
                load_cpkt,
                network=self.network,
                **default_hparams
            )

        else:
            lightning_module = SynthesisRouteLightningModule(
                network=self.network,
                **default_hparams
            )

        return lightning_module

    def Train(self, 
        data_path: Path, epochs: int =1000,
        savepath: Path = Path("./models"),
        project_name: str = "Synprotac",
        load_cpkt: Path = None, 
        val_check_epochs: int = 2,
        gradient_clip_val: float = 1.0,
        log_steps: int = 50,
        debug: bool = False,
        ngpus: int = 1,
        batchsize: int = 64,
        learning_rate = 1e-4,
        acc_batches = 1
    ):
        self.data_module = SynthesisRoute_DataModule(
            train_datafile = data_path / "train.routes",
            val_datafile = data_path/"val.routes",
            test_datafile = data_path/"test.routes",
            max_atoms = self.max_atoms, 
            max_sequence_length = self.max_sequence_length,
            num_workers = self.num_workers,
            batchsize = batchsize
        )

        training_hparams = {"learning_rate": learning_rate,
                            "weight_decay": 1e-5,
                            "warmup_steps": 10000,
                            "max_sequence_length": self.max_sequence_length}
        
        self.lightning_module = self.create_lightning_module(hparams=training_hparams, load_cpkt=load_cpkt)
        
        if not debug:
            os.makedirs("./TensorBoard", exist_ok=True)
            logger = TensorBoardLogger(f"./TensorBoard", name=project_name, version=None)
        else:
            logger = None 
 
        lr_monitor = LearningRateMonitor(logging_interval="step")
 
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        checkpointing = ModelCheckpoint(dirpath=savepath, save_top_k=3, every_n_epochs=1, monitor="val/total_loss", mode="min", save_last=True)

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

    def init_action_searcher(self, reaction_templates_file: str = None, reagents_file: str = None, matrix_file: str = "reaction_to_reagents.pkl"):
        self.action_searcher = Action_Searcher(reaction_templates_file, reagents_file, fp_dim=self.fp_dim)

        if os.path.exists(matrix_file):
            self.action_searcher.load_reaction_to_reagent_matrix(matrix_file)
        else:
            self.action_searcher.build_reaction_to_reagent_matrix()
            self.action_searcher.save_reaction_to_reagent_matrix(matrix_file)
        return 

    def Sample(self, warhead_smiles: str, e3_ligand_smiles: str, reaction_templates_file: str, reagents_file: str, num_samples: int = 10, 
               load_cpkt: Path = None, batchsize: int = None, warhead_protected_patts: List = [], e3_ligand_protected_patts: List = [], matrix_file: str = "reaction_to_reagents.pkl"):
        """
        Sample synthesis routes for given warhead and E3 ligand.
        
        Args:
            warhead_smiles: Warhead SMILES with reaction sites marked by mapnum
            e3_ligand_smiles: E3 ligand SMILES with reaction sites marked by mapnum  
            num_samples: Number of routes to sample
            load_cpkt: Path to checkpoint file to load pretrained weights
            
        Returns:
            List of synthesis route dictionaries
        """ 

        if batchsize is None:
            batchsize = min(32, num_samples)

        empty_synroute=self.create_empty_SynthesisRoutes(warhead_smiles, e3_ligand_smiles)
        
        self.data_module = SynthesisRoute_DataModule(
            test_routes=[empty_synroute]*num_samples, 
            max_atoms = self.max_atoms, 
            max_sequence_length = self.max_sequence_length,
            num_workers = self.num_workers,
            batchsize = batchsize
        )

        self.data_module.setup(stage="test")

        sampling_hparams = {"max_sequence_length": self.max_sequence_length}
        self.lightning_module = self.create_lightning_module(hparams = sampling_hparams, load_cpkt=load_cpkt)

        self.lightning_module.init_action_searcher( reaction_templates_file , reagents_file, matrix_file = matrix_file ) 
        self.lightning_module.init_sample_environment( warhead_smiles , e3_ligand_smiles, warhead_protected_patts, e3_ligand_protected_patts, batchsize)

        test_dataloader=self.data_module.test_dataloader()

        cuda_model= self.lightning_module.to("cuda")
        # Set model to evaluation mode
        cuda_model.eval()

        all_results = []
        for batch in test_dataloader:
            with torch.no_grad():
                batch= {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                # Generate synthesis routes using the model
                routes = cuda_model.generate_routes_batch(
                    batch=batch,
                )
            all_results+=routes

        total_routes=len(all_results)
        valid_routes = [] 
        valid_num=0

        for route in all_results:
            if len(route) > 1:
                if route[-1]["reaction_type"] == "e3_connection":
                    valid_num+=1
                    valid_routes.append(route)

        validity = valid_num/total_routes 
        
        return valid_routes, validity 

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
    
    def Test(self, warhead_smiles, e3_ligand_smiles, reaction_templates_file: str, reagents_file: str, num_samples: int = 10, 
               load_cpkt: Path = None, batchsize: int = None, warhead_protected_patts: List = [], e3_ligand_protected_patts: List = [], matrix_file: str = "reaction_to_reagents.pkl"):
        """
        Sample synthesis routes for given warhead and E3 ligand.
        
        Args:
            warhead_smiles: Warhead SMILES with reaction sites marked by mapnum
            e3_ligand_smiles: E3 ligand SMILES with reaction sites marked by mapnum  
            num_samples: Number of routes to sample
            load_cpkt: Path to checkpoint file to load pretrained weights

        Returns:
            List of synthesis route dictionaries

        """ 

        if batchsize is None:
            batchsize = min(32, num_samples)

        empty_synroute=self.create_empty_SynthesisRoutes(warhead_smiles, e3_ligand_smiles)

        self.data_module = SynthesisRoute_DataModule(
            test_datafile=Path("./data_debug/test.routes"),
            max_atoms = self.max_atoms, 
            max_sequence_length = self.max_sequence_length,
            num_workers = self.num_workers,
            batchsize = batchsize
        )

        self.data_module.setup(test_routes=[empty_synroute]*num_samples, stage="test")
        #self.data_module.setup(stage="test")
        
        sampling_hparams = {"max_sequence_length": self.max_sequence_length}

        self.lightning_module = self.create_lightning_module(hparams = sampling_hparams, load_cpkt=load_cpkt)
        self.lightning_module.init_action_searcher( reaction_templates_file , reagents_file, matrix_file = matrix_file ) 
        self.lightning_module.init_sample_environment( warhead_smiles , e3_ligand_smiles, warhead_protected_patts, e3_ligand_protected_patts, batchsize)

        test_dataloader=self.data_module.test_dataloader()

        cuda_model= self.lightning_module.to("cuda")
        # Set model to evaluation mode
        cuda_model.eval()

        for bid,batch in enumerate(test_dataloader):
            with torch.no_grad():
                batch= {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                # Generate synthesis routes using the model

                cuda_model.validation_step(
                    batch=batch,batch_idx=bid
                )
                routes = cuda_model.generate_routes_batch(
                    batch=batch,
                )

        return