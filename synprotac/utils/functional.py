from typing import Union
from rdkit import Chem 
from ..comparm import GP 

import torch
from scipy.spatial.transform import Rotation

_T = torch.Tensor
TupleRot = tuple[float, float, float]

def smiles_to_graph(smiles: str):

    mol=Chem.MolFromSmiles(smiles)
    atoms = []
    bond_indices=[]
    bond_types=[]

    if mol:
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        atoms = torch.Tensor(atoms).long()
        natoms = len(atoms)
        # Keep the original edge list format for compatibility
        bond_indices = []
        bond_types = []

        for bond in mol.GetBonds():
            bond_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bond_types.append(GP.bond_type_to_id[bond.GetBondType()])

        bond_indices = torch.Tensor(bond_indices).long()
        bond_types = torch.Tensor(bond_types).long()
        
    return atoms, bond_indices, bond_types 

def adjs_from_edges(natoms: int, bond_indices: _T, bond_types: _T):
    adjs=torch.zeros((natoms,natoms)).long()
    from_indices = bond_indices[:,0]
    end_indices = bond_indices[:,1]
    adjs[from_indices,end_indices] = bond_types
    adjs[end_indices,from_indices] = bond_types 
    return adjs 

def flatten_list(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result 

import torch

def tanimoto_similarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    并行计算A和B的Tanimoto相似度
    A: [N, D] 二值指纹
    B: [M, D] 二值指纹
    返回: [N, M] 相似度矩阵
    """
    # 交集
    intersect = torch.matmul(A.float(), B.float().t())  # [N, M]
    # 并集
    sum_A = A.sum(dim=1, keepdim=True)  # [N, 1]
    sum_B = B.sum(dim=1, keepdim=True)  # [M, 1]
    union = sum_A + sum_B.t() - intersect  # [N, M]
    # 防止除零
    eps = 1e-8
    similarity = intersect / (union + eps)
    return similarity