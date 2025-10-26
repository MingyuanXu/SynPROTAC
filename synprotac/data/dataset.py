"""
Dataset class for PROTAC synthesis routes.
"""
import pickle
import os
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .synthesis_route import SynthesisRoute
from ..chemistry import BuildingBlock
from tqdm import tqdm 

from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class SynthesisRoute_Dataset(Dataset):
    """Dataset for PROTAC synthesis routes."""
    
    def __init__(self, 
                 synroutes: List[SynthesisRoute],
                 max_sequence_length: int = 32,
                 max_atoms: int = 100,
                ):
        """
        Initialize the dataset.
        Args:
            routes_file: Path to routes.pkl file
            max_sequence_length: Maximum sequence length for padding
        """
        self.synroutes = synroutes 
        self.max_sequence_length = max_sequence_length
        self.max_atoms = max_atoms

    def __len__(self) -> int:
        return len(self.synroutes)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training example."""
        route = self.synroutes[idx]
        # Convert to tensor format
        tensor_dict = route.to_tensor_dict(
            max_sequence_length=self.max_sequence_length,
            max_atoms=self.max_atoms,
        )
        return tensor_dict
    
    @staticmethod
    def statistic_routes(synroutes: List[SynthesisRoute]):
        max_warhead_atoms = max(route.warhead_atoms.size(0) for route in synroutes)
        max_e3_ligand_atoms = max(route.e3_ligand_atoms.size(0) for route in synroutes)

        max_sequence_length = max(len(route.action_tokens) for route in synroutes) 

        return {
            "max_warhead_atoms": max_warhead_atoms,
            "max_e3_ligand_atoms": max_e3_ligand_atoms,
            "max_sequence_length": max_sequence_length
        }

class SynthesisRoute_DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_datafile=None,val_datafile=None,test_datafile=None,
                 train_routes=None,val_routes=None,test_routes=None,
                 max_atoms=50,max_sequence_length=12,num_workers=4,
                 batchsize=64):
        
        super().__init__()
        self.train_datafile = train_datafile
        self.val_datafile = val_datafile
        self.test_datafile = test_datafile
        self.train_routes = train_routes
        self.val_routes = val_routes
        self.test_routes = test_routes
        self.max_atoms = max_atoms
        self.max_sequence_length = max_sequence_length
        self.num_workers = num_workers
        self.batchsize=batchsize

    @staticmethod
    def load_synthesis_routes(filepath):
        data_file=filepath.read_bytes()
        datas=pickle.loads(data_file)
        synroutes=[]
        failed_num=0
        for data in tqdm(datas):
            try:
                route=SynthesisRoute.from_bytes(data)
                if len(route.warhead_reaction_sites)>1 or len(route.e3_ligand_reaction_sites)>1:
                    failed_num+=1
                    continue
                synroutes.append(route)
            except:
                failed_num+=1
        print (f"Loaded {len(synroutes)} synthesis routes from {filepath}, failed to load {failed_num} routes.")
        return synroutes

    def setup_dataset(self, synroutes=None, mode=None):
        
        dataset=SynthesisRoute_Dataset( 
                            synroutes = synroutes,
                            max_sequence_length = self.max_sequence_length,
                            max_atoms = self.max_atoms,
                        )
        
        return dataset 

    def setup(self,stage=None,):
        if stage == "fit" or stage is None:

            if self.train_routes is None:
                assert self.train_datafile.exists(), f"Train data file {self.train_datafile} must be provided when train_routes is None."
                train_routes=self.load_synthesis_routes(self.train_datafile)
            else:
                train_routes=self.train_routes

            self.trainset=self.setup_dataset(synroutes=train_routes)
            self.train_routes=None 

            if self.val_routes is None:
                val_routes=self.load_synthesis_routes(self.val_datafile)
            else:
                val_routes=self.val_routes

            self.valset=self.setup_dataset(synroutes=val_routes)
            self.val_routes=None
            
        if stage == "test" or stage is None:
            if self.test_routes  is None:
                test_routes = self.load_synthesis_routes(self.test_datafile)
            else:
                test_routes=self.test_routes

            self.testset=self.setup_dataset(synroutes=test_routes)
            self.test_routes=None
            
    def create_dataloader(self,dataset,shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batchsize,
            num_workers=self.num_workers,
            shuffle=shuffle,  # Lightning 会自动关闭 shuffle 并用 DistributedSampler
            pin_memory=True,
            drop_last=False,
        ) 
    
    def train_dataloader(self):
        return self.create_dataloader(self.trainset, shuffle=True)
    
    def val_dataloader(self):
        return self.create_dataloader(self.valset)
    
    def test_dataloader(self):
        return self.create_dataloader(self.testset)
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # 假设 batch 是一个字典，递归地将所有 tensor 转到 device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, dict):
                batch[k] = self.transfer_batch_to_device(v, device, dataloader_idx)
            # 如果有 list/tuple 也可以递归处理
        return batch
    