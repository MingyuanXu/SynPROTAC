"""
Data structures for PROTAC synthesis routes.
"""

import dataclasses

from typing import List, Dict, Any, Optional, Tuple
from enum import IntEnum
import torch
import pickle 

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from ..utils.functional import smiles_to_graph, adjs_from_edges
from ..comparm import GP
_T = torch.Tensor



class ActionType(IntEnum):
    """Action types for synthesis route generation."""
    START = 0
    REACTION = 1  # Select reaction template
    BUILDING_BLOCK = 2  # Select building block reagent
    E3_CONNECTION = 3  # Connect E3 ligand

def atomic_to_atomids(atomics, atom_type_to_id):
    #print ('atom_type_to_id',atom_type_to_id)
    atomids=[atom_type_to_id[int(atomic)] for atomic in atomics]
    return torch.Tensor(atomids).long()

@dataclasses.dataclass
class ActionToken:
    """Single action token in synthesis route."""
    action_type: ActionType
    reaction_id: Optional[int] = None  # For REACTION and E3_CONNECTION
    reagent_id: Optional[int] = None  # Optional ID for reagent

@dataclasses.dataclass
class SynthesisRoute:
    """Complete synthesis route data structure for training."""
    # Input molecules
    warhead_smiles: str
    e3_ligand_smiles: str
    warhead_atoms: _T
    e3_ligand_atoms: _T
    warhead_bond_indices: _T
    e3_ligand_bond_indices: _T
    warhead_bond_types:  _T
    e3_ligand_bond_types: _T
    # Reaction site information (atom indices with [OH:1] etc.)
    warhead_reaction_sites: Optional[List[int]] = None
    e3_ligand_reaction_sites: Optional[List[int]] = None
    final_product_smiles: str = ''
    # Action sequence
    action_tokens: List[ActionToken] = dataclasses.field(default_factory=list)
    
    def __post_init__(self):
        n_warhead_atoms=len(self.warhead_atoms)
        n_e3_ligand_atoms=len(self.e3_ligand_atoms)
        
        if n_warhead_atoms>GP.max_atoms or n_e3_ligand_atoms>GP.max_atoms:
            raise ValueError(f"Too many atoms in warhead or e3 ligand: {n_warhead_atoms},{n_e3_ligand_atoms}")
        
        for atom in self.warhead_atoms:
            if int(atom) not in GP.atom_types:
                raise ValueError(f"Unknown atom type in warhead: {atom}")

        for atom in self.e3_ligand_atoms:
            if int(atom) not in GP.atom_types:
                raise ValueError(f"Unknown atom type in e3 ligand: {atom}")

    @classmethod
    def from_route_dict(cls, route_dict: Dict[str, Any], 
                       max_sequence_length: int = 32) -> 'SynthesisRoute':
        """Create SynthesisRoute from route dictionary."""
        warhead_smiles = route_dict['warhead']
        e3_ligand_smiles = route_dict['e3_ligand'] 
        
        warhead_atoms, warhead_bond_indices, warhead_bond_types = smiles_to_graph(warhead_smiles)
        e3_ligand_atoms, e3_ligand_bond_indices, e3_ligand_bond_types = smiles_to_graph(e3_ligand_smiles)
        
        # Extract reaction sites from SMILES with atom mapping
        warhead_sites = cls._extract_reaction_sites(warhead_smiles)
        e3_ligand_sites = cls._extract_reaction_sites(e3_ligand_smiles)
        
        action_tokens = [ActionToken(action_type=ActionType.START,reagent_id=0)]
        for reaction in route_dict['reactions']:

            reaction_name = reaction["reaction"]
            reaction_type = reaction["reaction_type"]
            reagent_id = reaction["reagent_id"]

            if reaction_type == 'building_block':
                reaction_id = int(reaction_name.split(' ')[1])
                action_tokens.append(ActionToken(
                    action_type = ActionType.REACTION,
                    reaction_id = reaction_id
                ))
                action_tokens.append(ActionToken(
                    action_type = ActionType.BUILDING_BLOCK,
                    reagent_id = reagent_id+1
                ))
            else:
                reaction_id = int(reaction_name.split(' ')[1].split('_')[0])
                action_tokens.append(ActionToken(
                    action_type = ActionType.E3_CONNECTION,
                    reaction_id = reaction_id
                ))


        return cls(
            warhead_smiles = warhead_smiles,
            e3_ligand_smiles = e3_ligand_smiles,
            warhead_atoms = warhead_atoms,
            warhead_bond_indices = warhead_bond_indices,
            warhead_bond_types = warhead_bond_types,
            warhead_reaction_sites = warhead_sites,
            e3_ligand_atoms = e3_ligand_atoms,
            e3_ligand_bond_indices = e3_ligand_bond_indices,
            e3_ligand_bond_types = e3_ligand_bond_types,
            e3_ligand_reaction_sites = e3_ligand_sites,
            action_tokens = action_tokens,
            final_product_smiles = route_dict['final_product']
        )

    @staticmethod
    def from_bytes(bytes):
        obj=pickle.loads(bytes)

        return SynthesisRoute(
            warhead_smiles = obj['warhead_smiles'],
            e3_ligand_smiles = obj['e3_ligand_smiles'],
            warhead_atoms = obj['warhead_atoms'],
            warhead_bond_indices = obj['warhead_bond_indices'],
            warhead_bond_types = obj['warhead_bond_types'],
            warhead_reaction_sites = obj['warhead_reaction_sites'],
            e3_ligand_atoms = obj['e3_ligand_atoms'],
            e3_ligand_bond_indices = obj['e3_ligand_bond_indices'],
            e3_ligand_bond_types = obj['e3_ligand_bond_types'],
            e3_ligand_reaction_sites = obj['e3_ligand_reaction_sites'],
            action_tokens = obj["action_tokens"],
            final_product_smiles = obj['final_product_smiles'],
        )

    def to_bytes(self):
        data_dict={
            'warhead_smiles': self.warhead_smiles,
            'e3_ligand_smiles': self.e3_ligand_smiles,
            'warhead_atoms': self.warhead_atoms,
            'e3_ligand_atoms': self.e3_ligand_atoms,
            'warhead_bond_indices': self.warhead_bond_indices,
            'e3_ligand_bond_indices': self.e3_ligand_bond_indices,
            'warhead_bond_types': self.warhead_bond_types,
            'e3_ligand_bond_types': self.e3_ligand_bond_types,
            'warhead_reaction_sites': self.warhead_reaction_sites,
            'e3_ligand_reaction_sites': self.e3_ligand_reaction_sites,
            'final_product_smiles': self.final_product_smiles,
            'action_tokens': self.action_tokens,
        }
        bytes=pickle.dumps(data_dict)
        return bytes

    @staticmethod
    def _extract_reaction_sites(smiles: str) -> List[int]:
        """Extract reaction site atom indices from SMILES with atom mapping."""

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []
        
        reaction_sites = []
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() > 0:  # Atoms with mapping like [OH:1]
                reaction_sites.append(atom.GetIdx())

        if len(reaction_sites) == 0:
            raise ValueError(f"No reaction sites found in SMILES: {smiles}")

        return reaction_sites
    
    @property
    def warhead_atom_ids(self):
        warhead_atom_ids=atomic_to_atomids(self.warhead_atoms, GP.atom_type_to_id)
        return warhead_atom_ids 

    @property
    def e3_ligand_atom_ids(self):
        e3_ligand_atom_ids=atomic_to_atomids(self.e3_ligand_atoms, GP.atom_type_to_id)
        return e3_ligand_atom_ids

    @property
    def warhead_adjs(self):
        natoms=len(self.warhead_atoms)
        warhead_adjs = adjs_from_edges(natoms, self.warhead_bond_indices, self.warhead_bond_types)
        return warhead_adjs

    @property
    def e3_ligand_adjs(self):
        natoms=len(self.e3_ligand_atoms)
        e3_ligand_adjs = adjs_from_edges(natoms, self.e3_ligand_bond_indices, self.e3_ligand_bond_types)
        return e3_ligand_adjs

    def to_tensor_dict(self, max_sequence_length: int = 32, max_atoms =256) -> Dict[str, torch.Tensor]:
        """Convert to tensor dictionary for model training."""  
        warhead_natoms= len(self.warhead_atoms)
        e3_ligand_natoms= len(self.e3_ligand_atoms)   
        warhead_atom_ids = torch.zeros(max_atoms, dtype=torch.long)
        warhead_atom_ids[:warhead_natoms]=self.warhead_atom_ids

        warhead_atom_mask = torch.zeros(max_atoms, dtype=torch.long)
        warhead_atom_mask[:warhead_natoms] = 1

        e3_ligand_atom_ids=torch.zeros(max_atoms, dtype=torch.long)
        e3_ligand_atom_ids[:e3_ligand_natoms]=self.e3_ligand_atom_ids

        e3_ligand_atom_mask = torch.zeros(max_atoms,dtype=torch.long)
        e3_ligand_atom_mask[:e3_ligand_natoms] = 1

        warhead_adjs = torch.zeros((max_atoms,max_atoms), dtype=torch.long)
        warhead_adjs[:warhead_natoms,:warhead_natoms] = self.warhead_adjs

        e3_ligand_adjs = torch.zeros((max_atoms,max_atoms),dtype=torch.long)
        e3_ligand_adjs[:e3_ligand_natoms,:e3_ligand_natoms] = self.e3_ligand_adjs

        warhead_reaction_site_mask=torch.zeros((max_atoms), dtype=torch.long)
        e3_ligand_reaction_site_mask = torch.zeros((max_atoms), dtype=torch.long)

        warhead_reaction_site_mask[self.warhead_reaction_sites] = 1
        e3_ligand_reaction_site_mask[self.e3_ligand_reaction_sites] = 1
        
        seq_len = min(len(self.action_tokens), max_sequence_length)
        
        # Action type sequence
        action_types = torch.zeros(max_sequence_length, dtype=torch.long)
        
        # Reaction indices sequence  
        reaction_indices = torch.zeros(max_sequence_length, dtype=torch.long)
        
        # Reagent fingerprints sequence
        reagent_indices = torch.zeros(max_sequence_length, dtype=torch.long)
        
        # Padding mask (True for valid positions)
        seq_mask = torch.zeros(max_sequence_length, dtype=torch.bool)
        
        for i in range(seq_len):
            token = self.action_tokens[i]
            action_types[i] = token.action_type
            seq_mask[i] = True
            
            if token.reaction_id is not None:
                reaction_indices[i] = token.reaction_id
            
            if token.reagent_id is not None:
                # Ensure fingerprint has correct dimension
                reagent_indices[i] = token.reagent_id
        
        return {
            'warhead_natoms': torch.Tensor([warhead_natoms]).long(),
            'warhead_atom_ids': warhead_atom_ids,
            'warhead_adjs': warhead_adjs,
            'warhead_atom_mask': warhead_atom_mask,
            'e3_ligand_natoms': torch.Tensor([e3_ligand_natoms]).long(),
            'e3_ligand_atom_ids': e3_ligand_atom_ids,
            'e3_ligand_adjs': e3_ligand_adjs,
            'e3_ligand_atom_mask': e3_ligand_atom_mask,
            'action_types': action_types,
            'reaction_indices': reaction_indices,
            'reagent_indices': reagent_indices,
            'seq_mask': seq_mask,
            'warhead_reaction_site_mask': warhead_reaction_site_mask,
            'e3_ligand_reaction_site_mask': e3_ligand_reaction_site_mask
        }
