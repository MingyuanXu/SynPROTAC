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
from .lightning_module import Action_Searcher

class MolecularScorer:
    """
    A class for defining the scoring function components.
    """
    def __init__(self,score_functions,score_weights):
        """
        Args:
        ----
            constants (namedtuple) : Contains job parameters as well as global
                                     constants.
        """
        self.score_functions = score_functions # list
        self.score_weights = score_weights
        assert len(self.score_functions) == len(self.score_weights), \
               "`score_functions` and `score_weights` do not match."

    def compute_score( self, mols : list ,subset_id=0, rank=0):
        
        nmols=len(mols)
        valid_mask=torch.zeros(nmols)
        unique_mask=torch.ones(nmols)
        smiles=[]
        
        for mid,mol in enumerate(mols):
            if mol:
                valid_mask[mid]=1
                smi=Chem.MolToSmiles(mol)
                if smi in smiles:
                    unique_mask[mid]=0
                smiles.append(smi)

        contributions=[]
        actual_scores_dict={}

        for func in self.score_functions:
            dealed_scores, actual_scores = func.compute_scores(mols, subset_id=subset_id, rank=rank)
            actual_scores_dict[func.name]=actual_scores
            scores=dealed_scores*unique_mask*valid_mask
            contributions.append(scores)

        if len(contributions) == 1:
            final_score = contributions[0]
        else:
            final_score = contributions[0]*self.score_weights[0]
            for cid,component in enumerate(contributions[1:]):
                final_score *= component*self.score_weights[cid]
        validity=torch.sum(valid_mask)/nmols
        uniqueness=torch.sum(unique_mask)/torch.sum(valid_mask) if torch.sum(valid_mask)>0 else 0
        return final_score,validity,uniqueness, actual_scores_dict

class RL_LightningModule(pl.LightningModule):
    """PyTorch Lightning module for synthesis route generation."""
    
    def __init__(self,
                 network,
                 prior_network,
                 warhead,
                 e3_ligand,
                 warhead_protected_patts = [],
                 e3_ligand_protected_patts = [],
                 scorer = [None],
                 action_searcher = None,
                 reward_sigma = 10,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 loss_weights: Optional[Dict[str, float]] = None,
                 max_sequence_length : int = 7,
                 num_action_types = 4,
                 num_reaction_classes = 91,
                 num_reagent_classes = 483,
                 rl_samples_path = "./rl-samples",
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
        self.prior_model = prior_network
        # Training step counter for warmup
        self.warmup_steps = 1000
        self.max_sequence_length = max_sequence_length 
        self.num_reaction_classes = num_reaction_classes
        self.num_action_types = num_action_types
        self.num_reagent_classes = num_reagent_classes
        self.warhead = warhead
        self.e3_ligand = e3_ligand
        self.warhead_protected_patts = warhead_protected_patts
        self.e3_ligand_protected_patts = e3_ligand_protected_patts
        self.reward_sigma = reward_sigma
        self.scorer = scorer
        self.step_pt=0
        self.rl_samples_path = rl_samples_path
        

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Reinforcement learning training step."""
        
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0 
        savepath=f"{self.rl_samples_path}/train/{self.step_pt}"
        if not os.path.exists(savepath):
            os.makedirs(savepath,exist_ok=True)
        # Get batch size
        batch_size = batch['action_types'].size(0)

        # 初始化环境（假设 batch 包含起始分子）
        self.init_sample_environment(
            warhead=self.warhead,
            e3_ligand=self.e3_ligand,
            warhead_protected_patts=self.warhead_protected_patts,
            e3_ligand_protected_patts=self.e3_ligand_protected_patts,
            batchsize=batch_size
        )
        torch.autograd.set_detect_anomaly(True)
        # Generate routes
        routes, agent_loglikelihoods, prior_loglikelihoods = self.sample_batch(batch)

        scores, route_validity, mol_validity, uniqueness,valid_scores, contribute_scores = self.compute_rewards(routes,subset_id=self.step_pt, rank=rank)
        
        with open(f"{savepath}/routes-{rank}.pkl","wb") as f:
            pickle.dump([routes,scores,contribute_scores], f)

        scores=scores.float().to(agent_loglikelihoods.device)
        uniqueness=uniqueness.float().to(agent_loglikelihoods.device)

        augment_prior_loglikelihoods = prior_loglikelihoods + self.reward_sigma * scores

        total_loss = torch.mean((agent_loglikelihoods - augment_prior_loglikelihoods)**2)

        self.log('route_validity/train', route_validity, on_step=True, prog_bar=True)
        self.log('mol_validity/train', mol_validity, on_step=True, prog_bar=False)
        self.log('uniqueness/train', uniqueness, on_step=True, prog_bar=False)
        self.log('scores/train', scores.mean(), on_step=True, prog_bar=False)
        self.log('valid_scores/train', valid_scores, on_step=True, prog_bar=True)
        self.log('total_loss/train', total_loss, on_step=True, prog_bar=True)
        for key in contribute_scores.keys():
            print (key, contribute_scores[key].mean())
            self.log(f'{key}/train', contribute_scores[key].mean(), on_step=True, prog_bar=False)
        self.step_pt+=1
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Reinforcement learning validation step."""
        # Get batch size
        batch_size = batch['action_types'].size(0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0 
        savepath = f"{self.rl_samples_path}/val/{self.step_pt}"
        if not os.path.exists(savepath):
            os.makedirs(savepath,exist_ok=True)
        
        # 初始化环境（假设 batch 包含起始分子）
        self.init_sample_environment(
            warhead=self.warhead,
            e3_ligand=self.e3_ligand,
            warhead_protected_patts=self.warhead_protected_patts,
            e3_ligand_protected_patts=self.e3_ligand_protected_patts,
            batchsize=batch_size
        )

        # Generate routes
        routes, agent_loglikelihoods, prior_loglikelihoods = self.sample_batch(batch)
        #print (routes)

        scores, route_validity, mol_validity, uniqueness, valid_scores, contribute_scores = self.compute_rewards(routes,subset_id=self.step_pt,rank=rank)
        
        with open(f"{savepath}/routes-{rank}.pkl","wb") as f:
            pickle.dump([routes,scores,contribute_scores], f)

        #print ('agent_loglikelihoods',agent_loglikelihoods)
        self.log('route_validity/val', route_validity,  on_epoch=True, prog_bar=True)
        self.log('mol_validity/val', mol_validity,  on_epoch=True, prog_bar=False)
        self.log('uniqueness/val', uniqueness,  on_epoch=True, prog_bar=False)
        self.log('scores/val', scores.mean(),  on_epoch=True, prog_bar=False)
        self.log('valid_scores/val', valid_scores, on_epoch=True, prog_bar=True)
        for key in contribute_scores.keys():
            print (key,contribute_scores[key].mean())
            try:
                self.log(f'{key}/val', contribute_scores[key].mean(), on_epoch=True, prog_bar=False)
            except:
                pass    
        self.step_pt+=1
        return 

    def compute_rewards(self, routes,subset_id=0, rank=0):

        products = [route[-1]['product'] if len(route) > 0 else None for route in routes]
        
        route_valid_mask = torch.zeros(len(routes))
        for i, route in enumerate(routes):
            try:
                if route[-1]['reaction_type'] =='e3_connection':
                    route_valid_mask[i] =1 
            except:
                pass
        mols=[]
        for product in products:
            if product is not None:
                try:
                    mol = Chem.MolFromSmiles(product)
                    mol_H = Chem.AddHs(mol)
                    Chem.AllChem.EmbedMolecule(mol_H)
                    if mol is not None:
                        mols.append(mol_H)
                    else:
                        mols.append(None)
                except:
                    mols.append(None)
            else:
                mols.append(None)

        scores, mol_validity, uniqueness, contribute_scores = self.scorer.compute_score(mols,subset_id, rank)
        scores=scores*route_valid_mask 
        route_validity = torch.sum(route_valid_mask)/len(route_valid_mask) 
        valid_scores=scores[torch.nonzero(route_valid_mask)].mean() if torch.sum(route_valid_mask)>0 else 0
        #print ('scores:', scores, 'route_validity:', route_validity, 'mol_validity:', mol_validity, 'uniqueness:', uniqueness)

        return scores, route_validity, mol_validity, uniqueness, valid_scores, contribute_scores

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = AdamW(
            self.model.parameters(),
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
        self.current_valid_mask = torch.ones(batchsize).long().to(self.device)
        self.current_stop_mask = torch.zeros(batchsize).long().to(self.device)
        self.current_step_id = torch.zeros(batchsize).to(self.device)
        self.routes=[[] for i in range(batchsize)]
        self.warhead = warhead
        self.e3_ligand=e3_ligand 
        self.warhead_protected_patts=warhead_protected_patts
        self.e3_ligand_protected_patts=e3_ligand_protected_patts
        self.current_pred_pos=torch.zeros(batchsize).long().to(self.device)
        self.current_rollback_steps=torch.zeros(batchsize).long().to(self.device)
        self.route_agent_loglikelihoods=torch.zeros(batchsize,self.max_sequence_length).to(self.device)
        self.route_prior_loglikelihoods=torch.zeros(batchsize,self.max_sequence_length).to(self.device)
        return

    def trans_pred_pos_to_padding_mask(self, pred_pos):
        """Transform predicted positions to padding mask."""
        batchsize=len(self.current_pred_pos)
        padding_mask = torch.zeros((batchsize, self.max_sequence_length))

        for i in range(batchsize):
            if pred_pos[i]+1 < self.max_sequence_length:
                padding_mask[i,:pred_pos[i]+1]=1
        
        return padding_mask

    def trans_pred_pos_to_clean_mask(self,pred_pos):
        """Transform predicted positions to clean mask."""
        batchsize=len(self.current_pred_pos)
        padding_mask = torch.zeros((batchsize, self.max_sequence_length))

        for i in range(batchsize):
            if pred_pos[i]+1 < self.max_sequence_length:
                padding_mask[i,:pred_pos[i]]=1
        
        return padding_mask

    def trans_pred_pos_to_select_mask(self,pred_pos):
        batchsize=len(self.current_pred_pos)
        select_mask = torch.zeros((batchsize,self.max_sequence_length)).long()

        for i in range(batchsize):
            if pred_pos[i]+1 < self.max_sequence_length:
                select_mask[i,pred_pos[i]]=1

        return select_mask.bool()

    def predict_step(self, code, code_prior, code_padding_mask, action_types, reaction_indices, reagent_indices, predict_postion):
        """Prediction step for inference."""
        # Predict next step (this would need fpindex for full functionality)

        seq_padding_mask = self.trans_pred_pos_to_padding_mask(predict_postion).bool().to(code.device)
        select_mask = self.trans_pred_pos_to_select_mask(predict_postion).to(code.device)
        
        action_types_logits_agent, reaction_logits_agent, reagent_logits_agent = self.model.predict( 
                                            code,
                                            code_padding_mask,
                                            action_types,
                                            reaction_indices,
                                            reagent_indices,
                                            seq_padding_mask,
                                            select_mask
                                            )
        
        action_types_logits_prior, reaction_logits_prior, reagent_logits_prior = self.prior_model.predict(
                                            code_prior,
                                            code_padding_mask,
                                            action_types,
                                            reaction_indices,
                                            reagent_indices,
                                            seq_padding_mask,
                                            select_mask
                                            )
        # Sample next action type
        action_types_masks=self.generate_action_types_mask().to(action_types_logits_agent)

        next_action_type,agent_action_loglikelihoods, prior_action_loglikelihoods = self.sample_action_types(action_types_logits_agent,action_types_logits_prior, action_types_masks)

        reaction_masks = self.generate_reaction_mask(next_action_type).to(reaction_logits_agent)

        next_reaction_indices, agent_reaction_loglikelihoods, prior_reaction_loglikelihoods = self.sample_reaction_indices(reaction_logits_agent, reaction_logits_prior, reaction_masks)

        reagent_masks = self.generate_reagent_mask().to(reagent_logits_agent)

        next_reagent_indices, agent_reagent_loglikelihoods, prior_reagent_loglikelihoods = self.sample_reagent_indices(reagent_logits_agent, reagent_logits_prior, reagent_masks)

        return next_action_type, next_reaction_indices, next_reagent_indices,\
               agent_action_loglikelihoods, agent_reaction_loglikelihoods, agent_reagent_loglikelihoods,\
               prior_action_loglikelihoods, prior_reaction_loglikelihoods, prior_reagent_loglikelihoods


    def sample_batch(self, batch: Dict[str, torch.Tensor]): 

        code, code_padding_mask = self.model.encode_molecules(batch)
        code_prior, _ = self.prior_model.encode_molecules(batch)

        # Get current sequence information
        action_types_work = batch['action_types'].clone()
        reaction_indices_work = batch['reaction_indices'].clone()
        reagent_indices_work = batch['reagent_indices'].clone()
        batchsize=action_types_work.shape[0]

        for i in range(self.max_sequence_length):
            #print (f"step{i}, {self.current_pred_pos}")
            if self.current_stop_mask.sum()==action_types_work.shape[0]:
                break

            self.current_valid_mask[:]=1
            next_action_type, next_reaction_indices, next_reagent_indices, \
            agent_action_loglikelihoods, agent_reaction_loglikelihoods, agent_reagent_loglikelihoods, \
            prior_action_loglikelihoods, prior_reaction_loglikelihoods, prior_reagent_loglikelihoods = \
                 self.predict_step(code, code_prior, code_padding_mask, action_types_work, reaction_indices_work, reagent_indices_work, self.current_pred_pos)
            new_action_types=action_types_work.clone()
            new_reaction_indices=reaction_indices_work.clone()
            new_reagent_indices=reagent_indices_work.clone()

            for j in range(batchsize):
                pos = int(self.current_pred_pos[j].item())
                if pos+1 < action_types_work.size(1):
                    new_action_types[j,pos+1]=next_action_type[j]
                    new_reaction_indices[j,pos+1] = next_reaction_indices[j]
                    new_reagent_indices[j,pos+1] = next_reagent_indices[j]
            
            action_types_work=new_action_types
            reaction_indices_work=new_reaction_indices
            reagent_indices_work=new_reagent_indices

            self.update_actions(next_action_type, next_reaction_indices, next_reagent_indices, 
                               agent_action_loglikelihoods, agent_reaction_loglikelihoods, agent_reagent_loglikelihoods, 
                                prior_action_loglikelihoods, prior_reaction_loglikelihoods, prior_reagent_loglikelihoods)

            self.update_environment()

            self.current_pred_pos=torch.where(self.current_stop_mask.bool(), self.current_pred_pos, self.current_pred_pos+1)

            self.current_pred_pos=torch.where(
                                                self.current_valid_mask.bool(), 
                                                self.current_pred_pos,
                                                torch.clamp(self.current_pred_pos-self.current_rollback_steps,min=0)
                                              )

            pos_padding_mask = self.trans_pred_pos_to_clean_mask(self.current_pred_pos).to(self.route_agent_loglikelihoods).long()
            self.route_agent_loglikelihoods=self.route_agent_loglikelihoods*pos_padding_mask
            self.route_prior_loglikelihoods=self.route_prior_loglikelihoods*pos_padding_mask
         
        routes = self.routes
        agent_loglikelihoods=self.route_agent_loglikelihoods.sum(dim=-1)
        prior_loglikelihoods=self.route_prior_loglikelihoods.sum(dim=-1)

        return routes, agent_loglikelihoods, prior_loglikelihoods

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

    def sample_action_types(self,action_types_logits_agent, action_types_logits_prior,  action_type_masks=None):
        
        """Sample action types from logits."""

        softmax=torch.nn.Softmax(dim=-1)
        action_types_logits_agent = softmax(action_types_logits_agent)
        action_types_logits_prior = softmax(action_types_logits_prior)

        #print ('action_types_logits before mask', action_types_logits_agent)
        action_probability_distribution = torch.distributions.Multinomial(1,probs = action_types_logits_agent)
        action_types_sampled=action_probability_distribution.sample()
        #print (action_types_sampled)
        action_types = torch.argmax(action_types_sampled, dim=-1, keepdim=True)

        agent_loglikelihoods = torch.log(action_types_logits_agent.gather(1, action_types)+1e-10).squeeze(-1)
        prior_loglikelihoods = torch.log(action_types_logits_prior.gather(1, action_types)+1e-10).squeeze(-1)
        
        return action_types, agent_loglikelihoods, prior_loglikelihoods        

    def sample_reaction_indices(self, reaction_logits_agent, reaction_logits_prior, reaction_masks=None):

        softmax=torch.nn.Softmax(dim=-1)
        reaction_logits_agent = softmax(reaction_logits_agent)*reaction_masks
        reaction_logits_prior = softmax(reaction_logits_prior)*reaction_masks

        reaction_distribution = torch.distributions.Multinomial(1,probs = reaction_logits_agent)
        reaction_sampled = reaction_distribution.sample()
        reaction_indices = torch.argmax(reaction_sampled,dim=-1, keepdim=True)
        
        agent_loglikelihoods = torch.log(reaction_logits_agent.gather(1, reaction_indices)+1e-10).squeeze(-1)
        prior_loglikelihoods = torch.log(reaction_logits_prior.gather(1, reaction_indices)+1e-10).squeeze(-1)
        
        return reaction_indices, agent_loglikelihoods, prior_loglikelihoods

    def sample_reagent_indices(self, reagent_logits_agent, reagent_logits_prior, reagent_masks=None):
        softmax=torch.nn.Softmax(dim=-1)
        reagent_logits_agent = softmax(reagent_logits_agent)*reagent_masks
        reagent_logits_prior = softmax(reagent_logits_prior)*reagent_masks
        reagent_distribution = torch.distributions.Multinomial(1,probs = reagent_logits_agent)
        reagent_sampled = reagent_distribution.sample()
        reagent_indices = torch.argmax(reagent_sampled,dim=-1, keepdim=True)
        agent_loglikelihoods = torch.log(reagent_logits_agent.gather(1, reagent_indices)+1e-10).squeeze(-1)
        prior_loglikelihoods = torch.log(reagent_logits_prior.gather(1, reagent_indices)+1e-10).squeeze(-1)
        return reagent_indices, agent_loglikelihoods, prior_loglikelihoods

    def update_actions(self, next_action_types, next_reaction_indices, next_reagents_indices, agent_action_likelihoods, agent_reaction_likelihoods, agent_reagent_likelihoods, prior_action_likelihoods, prior_reaction_likelihoods, prior_reagent_likelihoods):
        batchsize=next_action_types.shape[0]
        
        for i in range(batchsize):
            if next_action_types[i]==1:
                self.current_reactions[i]=self.action_searcher.templates[next_reaction_indices[i]]
                #self.route_agent_loglikelihoods[i,self.current_pred_pos[i]]=agent_action_likelihoods[i]+agent_reaction_likelihoods[i]
                #self.route_prior_loglikelihoods[i,self.current_pred_pos[i]]=prior_action_likelihoods[i]+prior_reaction_likelihoods[i]

            elif next_action_types[i]==2:
                self.current_reagents[i]=self.action_searcher.reagents[next_reagents_indices[i]-1].smiles
                #self.route_agent_loglikelihoods[i,self.current_pred_pos[i]]=agent_action_likelihoods[i]+agent_reagent_likelihoods[i]
                #self.route_prior_loglikelihoods[i,self.current_pred_pos[i]]=prior_action_likelihoods[i]+prior_reagent_likelihoods[i]

            elif next_action_types[i]==3:
                self.current_reactions[i]=self.action_searcher.templates[next_reaction_indices[i]]
                self.current_reagents[i]=self.e3_ligand
                #self.route_agent_loglikelihoods[i,self.current_pred_pos[i]]=agent_action_likelihoods[i]+agent_reaction_likelihoods[i]
                #self.route_prior_loglikelihoods[i,self.current_pred_pos[i]]=prior_action_likelihoods[i]+prior_reaction_likelihoods[i]              
        
        action_reaction_mask= ((next_action_types==1) | (next_action_types==3)).long().squeeze(-1)
        action_reagent_mask = (next_action_types==2).long().squeeze(-1)

        
        agent_likelihoods=agent_action_likelihoods+agent_reaction_likelihoods*action_reaction_mask+agent_reagent_likelihoods*action_reagent_mask
        prior_likelihoods=prior_action_likelihoods+prior_reaction_likelihoods*action_reaction_mask+prior_reagent_likelihoods*action_reagent_mask
        pos_select_mask = self.trans_pred_pos_to_select_mask(self.current_pred_pos).to(agent_likelihoods).float()
        
        
        agent_likelihoods=agent_likelihoods.unsqueeze(-1)*pos_select_mask
        prior_likelihoods=prior_likelihoods.unsqueeze(-1)*pos_select_mask
        
        self.route_agent_loglikelihoods=self.route_agent_loglikelihoods*(1-pos_select_mask)+agent_likelihoods
        self.route_prior_loglikelihoods=self.route_prior_loglikelihoods*(1-pos_select_mask)+prior_likelihoods


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
                #print (f"Current molecule {i}: {fromstate}, step {int(self.current_step_id[i].clone().detach().item())}, reaction {template}, reagent {reagent}, reaction_type {reaction_type}")

                if template is not None and reagent is not None:
                    fromstate_mol = Chem.MolFromSmiles(fromstate)
                    reagent_mol=Chem.MolFromSmiles(reagent)
                    product=template.excute(fromstate_mol,reagent_mol)

                    #print (f"**************")
                    #print (i,product,reaction_type)

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
                        self.current_rollback_steps[i] = 2
        return

    def show_current_environment(self):
        for i in range(len(self.current_smiles)):
            print (f"*******************Molecule {i}**********************")
            print (self.current_smiles[i])
            print (self.current_reactions[i])
            print (self.current_reagents[i])

