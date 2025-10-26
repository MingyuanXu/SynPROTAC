"""
PyTorch Lightning training module for PROTAC synthesis route generation.
"""
from typing import Dict, Any, List, Optional, Union 
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from rdkit import Chem 
from .model import SynthesisGraphTransformer
from ..data.synthesis_route import ActionType
from ..utils.rdkit import load_rxn_templates, load_building_blocks
from ..chemistry import ReactionTemplate, BuildingBlock
import numpy as np 
from tqdm import tqdm 
from joblib import Parallel, delayed
import pickle 
from ..utils.functional import flatten_list 
import os 
import copy 

class Action_Searcher:
    def __init__(self,rxn_templates_file, reagents_file, matrix_pkl = "./reaction_reagent_matrix.pkl"):
        self.rxn_templates_file = rxn_templates_file
        self.reagents_file=reagents_file
        self._max_reaction_component_num=None 
        self.load_templates()
        self.load_reagents()

    def load_templates(self):
        templates_dicts=load_rxn_templates(self.rxn_templates_file)
        templates=[]
        for dict_data in tqdm(templates_dicts):
            template=ReactionTemplate.from_dict(dict_data)
            templates.append(template)
        self.templates=templates
        return 

    def load_reagents(self):
        reagents_dicts=load_building_blocks(self.reagents_file)
        reagents=[]
        for dict_data in tqdm(reagents_dicts):
            reagent=BuildingBlock.from_dict(dict_data)
            reagents.append(reagent)
        self.reagents = reagents
        return 

    @property
    def max_reaction_component_num(self):
        if self._max_reaction_component_num is None:
            self._max_reaction_component_num=np.max([template.num_reactants for template in self.templates])
        return self._max_reaction_component_num

    @property
    def num_reagents(self):
        return len(self.reagents)
    
    @property
    def num_reaction_classes(self):
        return len(self.templates)

    def _compute_template_row(self, template_idx: int) -> tuple[int, np.ndarray]:
        """
        Compute [n_reagents, max_reaction_component_num] mask for one template.
        Thread-safe: only reads RDKit objects.
        """
        template = self.templates[template_idx]
        patt_list = list(template.reaction.GetReactants())
        n_reagents = len(self.reagents)
        row = np.zeros((n_reagents, self.max_reaction_component_num), dtype=np.float32)

        for j, reagent in enumerate(self.reagents):
            # 对当前模板的每个反应位模式做子结构匹配
            for k, patt in enumerate(patt_list):
                if reagent.mol.HasSubstructMatch(patt):
                    row[j, k] = 1.0
        return template_idx, row

    def build_reaction_to_reagent_matrix(self, n_jobs: int | None = None, backend: str = "threads"):
        """
        Build matrix in parallel over templates:
          shape = [n_templates, n_reagents, max_reaction_component_num]
        Use threads to avoid pickling RDKit objects.
        backend: 'threading'/'threads' => threads, 'loky'/'processes' => processes
        """
        n_templates = len(self.templates)
        n_reagents = len(self.reagents)
        self.matrix = np.zeros((n_templates, n_reagents, self.max_reaction_component_num), dtype=np.float32)

        if n_jobs is None or n_jobs == 0:
            n_jobs = max(1, (os.cpu_count() or 1))
            #print (n_jobs)

        # map backend -> prefer
        if backend in ("threading", "threads"):
            prefer = "threads"
        elif backend in ("loky", "processes", "multiprocessing", "process"):
            prefer = "processes"
        else:
            prefer = None  # let joblib decide

        results = Parallel(n_jobs=n_jobs, prefer=prefer, verbose = 5)(
            delayed(self._compute_template_row)(i) for i in range(n_templates)
        )

        for idx, row in results:
            self.matrix[idx] = row

        return self.matrix

    def load_reaction_to_reagent_matrix(self,matrix_file: str = './reaction_to_reagents.pkl'):
        with open(matrix_file, 'rb') as f:
            self.matrix = pickle.load(f)
        return 
    
    def save_reaction_to_reagent_matrix(self,matrix_file: str = "./reaction_to_reagents.pkl"):
        with open(matrix_file,'wb') as f:
            pickle.dump(self.matrix,f)
        return 

    def generate_reaction_mask(self, substrate, protected_patts, mode="smooth"):
        """Generate a reaction mask for a given reactant and protected patterns."""
        substrate_mol, active_atoms, inactive_atoms = self.mol_from_smiles(substrate, protected_patts)

        reaction_mask = np.zeros((len(self.templates)),dtype= np.float32)
        for i, template in enumerate(self.templates):
            if template.can_apply_to(substrate_mol, active_atoms, inactive_atoms):
                if mode =="strict":
                    reagent_mask = self.matrix[i].T
                    for j,patt in enumerate(template.reaction.GetReactants()):
                        if substrate_mol.HasSubstructMatch(patt):
                            sampled_reactant_id = np.where(reagent_mask[1-j]>0)[0]
                            if len(sampled_reactant_id)<1:
                                continue
                            #print ('sampled_reactant_id',sampled_reactant_id,i,j,)
                            sampled_reactant = self.reagents[sampled_reactant_id[0]].mol
                            products=template.excute(substrate_mol, sampled_reactant)

                            if len(products)>0:
                                reaction_mask[i] = 1
                else:
                    reaction_mask[i] = 1

        #print ("available reaction num", np.sum(reaction_mask))
        if np.sum(reaction_mask)>0:
            return reaction_mask
        else:
            return np.ones((len(self.templates)), dtype=np.float32)

    def generate_connection_mask(self, substrate, connect, substrate_protected_patts, connect_protected_patts):
        substrate_mol, substrate_active_atoms, substrate_inactive_atoms = self.mol_from_smiles(substrate, substrate_protected_patts)
        connect_mol, connect_active_atoms, connect_inactive_atoms = self.mol_from_smiles(connect, connect_protected_patts)

        reaction_mask = np.zeros((len(self.templates)), dtype=np.float32)
        for i, template in enumerate(self.templates):
            if template.can_apply_to(substrate_mol, substrate_active_atoms, substrate_inactive_atoms) and \
                template.can_apply_to(connect_mol, connect_active_atoms, connect_inactive_atoms):
                products=template.excute(substrate_mol,connect_mol)

                if len(products)>0:
                    reaction_mask[i]=1

        #print ("available connection reaction num", np.sum(reaction_mask))

        if np.sum(reaction_mask) > 0:
            return reaction_mask 
        else: 
            return np.ones((len(self.templates)), dtype=np.float32)

    def generate_reagent_mask(self, template_id, substrate: str = None ):    
        #print ('template_id',template_id)    
        reagent_mask = self.matrix[template_id].T
        if substrate is not None:
            reagent_masks_list=[]
            substrate_mol=Chem.MolFromSmiles(substrate)
            for pid,patt in enumerate(self.templates[template_id].reaction.GetReactants()):
                if substrate_mol.HasSubstructMatch(patt):
                    #print('reagent_mask pid', reagent_mask[pid], sum(reagent_mask[pid]))
                    reagent_masks_list.append(reagent_mask[1-pid].astype(int))
            #print ('reagent_mask_list',reagent_masks_list)
            if len(reagent_masks_list)>0:
                mask = np.bitwise_or.reduce(reagent_masks_list)
                #print ('mask',mask)
            else:
                mask = np.ones((len(self.reagents))).astype(int)
        else:
            mask = reagent_mask.any(dim=0)

        return mask 

    def mol_from_smiles(self,smiles,protected_patts :List[str] = []):

        mol=Chem.MolFromSmiles(smiles)
        if mol is None:
            return mol,[],[]

        active_atoms=[atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]
        if len(active_atoms)==0:
            active_atoms = set(range(mol.GetNumAtoms()))

        inactive_atoms=[]
        if len(protected_patts) > 0:
            for patt in protected_patts:
                protected_mol = Chem.MolFromSmiles(patt)
                if protected_mol is None:
                    continue
                protected_matches = mol.GetSubstructMatches(protected_mol)
                for match in protected_matches:
                    for idx in match:
                        inactive_atoms.append(idx)

            inactive_atoms = list(set(inactive_atoms))

        return mol,active_atoms,inactive_atoms
    
    def judge_template_can_keep_protected_patts(self,template, substrate, keeped_patt):
        # can only be used for two-component reactions.
        reagent_mask = self.matrix[template_id].T
        substrate, active_atoms, inactive_atoms = self.mol_from_smiles(substrate, [keeped_patts])
        for i,patt in template.reaction.GetReactants():
            if substrate.HasSubstructMatch(patt):
                try:
                    sampled_reactant_id = torch.where(reagent_mask[1-i]>0)[0]
                except:
                    sampled_reactant_id = 0
                sampled_reactant = self.reagents[sampled_reactant_id].mol

                products=template.excute(substrate, sampled_reactant)
                if len(products)>0:
                    return True 

class SynthesisRouteLightningModule(pl.LightningModule):
    """PyTorch Lightning module for synthesis route generation."""
    
    def __init__(self,
                 network,
                 learning_rate: float = 1e-4,
                 warmup_steps: int = 10000,
                 weight_decay: float = 1e-5,
                 loss_weights: Optional[Dict[str, float]] = None,
                 max_sequence_length : int = 7,
                 num_action_types = 4,
                 num_reaction_classes = 91,
                 num_reagent_classes = 483,
                 **kwargs):
        """
        Initialize the Lightning module.
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            loss_weights: Weights for different loss components
        """
        super().__init__()
        
        # Model parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights or {
            'action_type': 1.0,
            'reaction': 1.0, 
            'reagent': 1.0
        }
        
        # Initialize model
        self.model = network  
        
        # Training step counter for warmup
        self.warmup_steps = 1000
        self.max_sequence_length = max_sequence_length 
        self.num_reaction_classes = num_reaction_classes
        self.num_action_types = num_action_types
        self.num_reagent_classes = num_reagent_classes

    def forward(self, batch: Dict[str, torch.Tensor], **kwargs):
        """Forward pass through the model."""
        return self.model.get_loss_shortcut(batch, **kwargs)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step."""
        # Determine if we're in warmup phase
        warmup = self.global_step < self.warmup_steps
        
        # Get losses
        loss_dict = self(batch, warmup=warmup)
        
        # Compute total weighted loss
        total_loss = 0.0
        for loss_name, loss_value in loss_dict.items():
            weight = self.loss_weights.get(loss_name, 1.0)
            total_loss += weight * loss_value
            
            # Log individual losses
            self.log(f'train/{loss_name}_loss', loss_value, 
                    on_step=True, on_epoch=True, prog_bar=True)
        
        # Log total loss
        self.log('train/total_loss', total_loss, 
                on_step=True, on_epoch=True, prog_bar=True)

        # Log learning rate
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], 
                on_step=True, on_epoch=False)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        # Get losses (no warmup during validation)
        loss_dict = self(batch, warmup=False)
        # Compute total weighted loss
        total_loss = 0.0
        for loss_name, loss_value in loss_dict.items():
            weight = self.loss_weights.get(loss_name, 1.0)
            total_loss += weight * loss_value
            
            # Log individual losses
            self.log(f'val/{loss_name}_loss', loss_value, 
                    on_step=False, on_epoch=True, prog_bar=True)
        
        # Log total loss
        self.log('val/total_loss', total_loss, 
                on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step."""
        # Get losses
        loss_dict= self(batch, warmup=False)
        
        # Compute total weighted loss
        total_loss = 0.0
        for loss_name, loss_value in loss_dict.items():
            weight = self.loss_weights.get(loss_name, 1.0)
            total_loss += weight * loss_value
            
            # Log individual losses
            self.log(f'test/{loss_name}_loss', loss_value, 
                    on_step=False, on_epoch=True)
        
        # Log total loss
        self.log('test/total_loss', total_loss, 
                on_step=False, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Log epoch number
        self.log('epoch', float(self.current_epoch), on_epoch=True)    

    def init_action_searcher(self,reaction_templates_file: str = None, reagents_file: str = None, matrix_file: str = "reaction_to_reagents.pkl"):
        self.action_searcher = Action_Searcher(reaction_templates_file, reagents_file)

        if os.path.exists(matrix_file):
            self.action_searcher.load_reaction_to_reagent_matrix(matrix_file)
        else:
            self.action_searcher.build_reaction_to_reagent_matrix()
            self.action_searcher.save_reaction_to_reagent_matrix(matrix_file)
        return 

    def init_sample_environment(self, warhead, e3_ligand, warhead_protected_patts, e3_ligand_protected_patts, batchsize=8):
        self.current_smiles = [warhead for i in range(batchsize)]
        self.current_reactions = [None for i in range(batchsize)]
        self.current_reagents = [None for i in range(batchsize)]
        self.current_valid_mask = torch.ones(batchsize).long()
        self.current_stop_mask = torch.zeros(batchsize).long()
        self.current_step_id = torch.zeros(batchsize)
        self.routes=[[] for i in range(batchsize)]
        self.warhead = warhead
        self.e3_ligand=e3_ligand 

        self.warhead_protected_patts=warhead_protected_patts
        self.e3_ligand_protected_patts=e3_ligand_protected_patts

        self.current_pred_pos=torch.zeros(batchsize).long()
        self.current_rollback_steps=torch.zeros(batchsize).long()
        return

    def trans_pred_pos_to_padding_mask(self, pred_pos):
        """Transform predicted positions to padding mask."""
        batchsize=len(self.current_pred_pos)
        padding_mask = torch.zeros((batchsize, self.max_sequence_length))

        for i in range(batchsize):
            if pred_pos[i]+1 < self.max_sequence_length:
                padding_mask[i,:pred_pos[i]+1]=1
        
        return padding_mask

    def trans_pred_pos_to_select_mask(self,pred_pos):
        batchsize=len(self.current_pred_pos)
        select_mask = torch.zeros((batchsize,self.max_sequence_length)).long()

        for i in range(batchsize):
            if pred_pos[i]+1 < self.max_sequence_length:
                select_mask[i,pred_pos[i]]=1

        return select_mask.bool()

    def predict_step(self, code, code_padding_mask, action_types, reaction_indices, reagent_indices, predict_postion):
        """Prediction step for inference."""
        # Predict next step (this would need fpindex for full functionality)
        seq_padding_mask = self.trans_pred_pos_to_padding_mask(predict_postion).bool().to(code.device)
        select_mask = self.trans_pred_pos_to_select_mask(predict_postion).to(code.device)
        action_types_logits, reaction_logits, reagent_logits = self.model.predict( 
                                            code,
                                            code_padding_mask,
                                            action_types,
                                            reaction_indices,
                                            reagent_indices,
                                            seq_padding_mask,
                                            select_mask
                                            )
        
        # Sample next action type
        action_types_masks=self.generate_action_types_mask().to(action_types_logits)
        #print ('action_types_masks',action_types_masks)
        #print (action_types_logits)

        next_action_type = self.sample_action_types(action_types_logits,action_types_masks)
        print ('next_action_type',next_action_type)

        reaction_masks = self.generate_reaction_mask(next_action_type).to(reaction_logits)
        #print ('reaction_masks',reaction_masks.shape,reaction_masks.sum(dim=-1))

        next_reaction_indices = self.sample_reaction_indices(reaction_logits,reaction_masks)
        #print ('reaction_indices', next_reaction_indices)

        reagent_masks = self.generate_reagent_mask().to(reagent_logits)
        #print ('reagent_masks',reagent_masks.shape,reagent_masks.sum(dim=-1))

        next_reagent_indices = self.sample_reagent_indices(reagent_logits,reagent_masks)
        #print ('reagent_indices', next_reagent_indices)

        return next_action_type, next_reaction_indices, next_reagent_indices

    def generate_routes_batch(self, batch: Dict[str, torch.Tensor]): 

        with torch.no_grad():
            code, code_padding_mask = self.model.encode_molecules(batch)
            #print ('code',code)
            # Get current sequence information
            action_types = batch['action_types']
            reaction_indices = batch['reaction_indices']
            reagent_indices = batch['reagent_indices']
            batchsize=action_types.shape[0]

            for i in range(self.max_sequence_length):
                #print (f"step {i}, {self.current_pred_pos}")
                if self.current_stop_mask.sum()==action_types.shape[0]:
                    break

                self.current_valid_mask[:]=1
                next_action_type, next_reaction_indices, next_reagent_indices = \
                     self.predict_step(code, code_padding_mask, action_types, reaction_indices, reagent_indices, self.current_pred_pos)

                for j in range(action_types.size(0)):
                    pos = int(self.current_pred_pos[j].item())
                    if pos+1 < action_types.size(1):
                        action_types[j,pos+1]=next_action_type[j]
                        reaction_indices[j,pos+1] = next_reaction_indices[j]
                        reagent_indices[j,pos+1] = next_reagent_indices[j]

                self.update_actions(next_action_type, next_reaction_indices, next_reagent_indices)
                self.update_environment()

                self.current_pred_pos=torch.where(self.current_stop_mask.bool(), self.current_pred_pos, self.current_pred_pos+1)

                self.current_pred_pos=torch.where(
                                                    self.current_valid_mask.bool(), 
                                                    self.current_pred_pos,
                                                    torch.clamp(self.current_pred_pos-self.current_rollback_steps,min=0)
                                                  )

        routes = self.routes
                    
        return routes 

    def generate_action_types_mask(self):
        batchsize = len(self.current_smiles)
        action_types_masks = torch.ones((batchsize, self.num_action_types))
        for i in range(batchsize):
            if self.current_stop_mask[i] == 1:
                action_types_masks[i] = torch.Tensor([1, 0, 0, 0]).to(action_types_masks)
            else:
                if self.current_reactions[i] is None:
                    action_types_masks[i] = torch.Tensor([0, 1, 0, 1]).to(action_types_masks)
                else:
                    action_types_masks[i] = torch.Tensor([0, 0, 1, 0]).to(action_types_masks)

        return action_types_masks            

    def generate_reaction_mask(self, next_action_type):
        """Generate reaction masks for the batch."""
        batchsize = len(self.current_smiles)
        reaction_masks = torch.ones((batchsize, self.num_reaction_classes))
        for i in range(batchsize): 
            if next_action_type[i] ==1:
                if self.current_pred_pos[i]==0:
                    mode="smooth"
                else:
                    mode="strict"
                mask = self.action_searcher.generate_reaction_mask(self.current_smiles[i], self.warhead_protected_patts, mode=mode)
                mask = torch.from_numpy(mask)
                reaction_masks[i]=mask
            elif next_action_type[i]==3:
                mask = self.action_searcher.generate_connection_mask(self.current_smiles[i], self.e3_ligand, self.warhead_protected_patts, self.e3_ligand_protected_patts)
                mask = torch.from_numpy(mask)
                reaction_masks[i]=mask
        return reaction_masks

    def generate_reagent_mask(self):
        batchsize = len(self.current_smiles)
        reagent_masks = torch.ones((batchsize, self.action_searcher.num_reagents+1))
        reagent_masks[:,0]=0
        for i in range(batchsize):
            if self.current_reactions[i] is not None:
                mask = self.action_searcher.generate_reagent_mask(self.current_reactions[i].template_id, self.current_smiles[i]) 
                mask = torch.from_numpy(mask)
                reagent_masks[i,1:] = mask
        return reagent_masks 

    def sample_action_types(self,action_types_logits, action_type_masks=None):
        """Sample action types from logits."""
        softmax=torch.nn.Softmax(dim=-1)
        action_types_logits = softmax(action_types_logits)#*action_type_masks
        action_probability_distribution = torch.distributions.Multinomial(1,probs = action_types_logits)
        action_types_sampled=action_probability_distribution.sample()
        action_types = torch.argmax(action_types_sampled, dim=-1, keepdim=True)
        return action_types        

    def sample_reaction_indices(self, reaction_logits, reaction_masks=None):
        softmax=torch.nn.Softmax(dim=-1)
        reaction_logits = softmax(reaction_logits)*reaction_masks
        reaction_distribution = torch.distributions.Multinomial(1,probs = reaction_logits)
        reaction_sampled = reaction_distribution.sample()
        reaction_indices = torch.argmax(reaction_sampled,dim=-1, keepdim=True)
        return reaction_indices 

    def sample_reagent_indices(self, reagent_logits, reagent_masks=None):
        softmax=torch.nn.Softmax(dim=-1)
        reagent_logits = softmax(reagent_logits)*reagent_masks
        reagent_distribution = torch.distributions.Multinomial(1,probs = reagent_logits)
        reagent_sampled = reagent_distribution.sample()
        reagent_indices = torch.argmax(reagent_sampled,dim=-1, keepdim=True)
        return reagent_indices

    def update_actions(self, next_action_types, next_reaction_indices, next_reagents_indices):
        batchsize=next_action_types.shape[0]
        for i in range(batchsize):
            if next_action_types[i]==1:
                self.current_reactions[i]=self.action_searcher.templates[next_reaction_indices[i]]

            elif next_action_types[i]==2:
                self.current_reagents[i]=self.action_searcher.reagents[next_reagents_indices[i]-1].smiles

            elif next_action_types[i]==3:
                self.current_reactions[i]=self.action_searcher.templates[next_reaction_indices[i]]
                self.current_reagents[i]=self.e3_ligand        
        return             

    def update_environment(self):
        for i in range(len(self.current_smiles)):
            if not self.current_stop_mask[i]:

                fromstate=self.current_smiles[i]
                template= self.current_reactions[i]
                reagent = self.current_reagents[i]

                if reagent == self.e3_ligand:
                    reaction_type="e3_connection"
                else:
                    reaction_type="building_block"
                print (f"Current molecule {i}: {fromstate}, step {int(self.current_step_id[i].clone().detach().item())}, reaction {template}, reagent {reagent}, reaction_type {reaction_type}")
                if template is not None and reagent is not None:
                    fromstate_mol = Chem.MolFromSmiles(fromstate)
                    reagent_mol=Chem.MolFromSmiles(reagent)
                    product=template.excute(fromstate_mol,reagent_mol)

                    print (f"**************")
                    print (i,product,reaction_type)

                    if len(product)>0:
                        self.current_smiles[i] = Chem.MolToSmiles(product[0])
                        self.routes[i].append({'step': int(self.current_step_id[i].clone().detach().item()),
                                                'reaction':template.name,
                                                'from_state':fromstate, 
                                                'reagent':reagent, 
                                                'product':Chem.MolToSmiles(product[0]),
                                                'reaction_type':reaction_type
                                                })
                        
                        self.current_step_id[i]+=1
                        self.current_reactions[i]=None
                        self.current_reagents[i]=None

                        if reaction_type == "e3_connection" or self.current_step_id[i]>4:
                            self.current_stop_mask[i] = 1

                    else:
                        self.current_valid_mask[i]=0
                        self.current_reactions[i]=None
                        self.current_reagents[i]=None
                        if reaction_type == "e3_connection":
                            self.current_rollback_steps[i] = 2
                        else:
                            self.current_rollback_steps[i] = 2
        return

    def show_current_environment(self):
        for i in range(len(self.current_smiles)):
            print (f"*******************Molecule {i}**********************")
            print (self.current_smiles[i])
            print (self.current_reactions[i])
            print (self.current_reagents[i])

