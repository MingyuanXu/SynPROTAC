"""
Molecular Protection Module

Implements molecular site protection for PROTAC synthesis to prevent
unwanted reactions at warhead and E3 ligand reactive sites.
"""

import logging
from typing import List, Tuple, Dict, Optional, Set
from rdkit import Chem
from rdkit.Chem import rdFMCS

logger = logging.getLogger(__name__)

def label_breaking_atoms(protac_smiles, warhead_smiles):
    protac = Chem.MolFromSmiles(protac_smiles)
    warhead = Chem.MolFromSmiles(warhead_smiles)
    match = protac.GetSubstructMatch(warhead)
    if not match:
        raise ValueError("Warhead not found in protac")
    match_set = set(match)
    breaking_atoms = set()
    for bond in protac.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # 找到连接warhead和非warhead的键
        if (a1 in match_set) != (a2 in match_set):
            # 只保留warhead端的原子编号
            breaking_atoms.add(match.index(a1) if a1 in match_set else match.index(a2))
    # 标注映射号
    warhead_edit = Chem.RWMol(warhead)
    for i, atom in enumerate(warhead_edit.GetAtoms()):
        if i in breaking_atoms:
            atom.SetAtomMapNum(1)  # 你可以根据需要设置不同的映射号
    return Chem.MolToSmiles(warhead_edit)

def draw_molecule_with_map_atoms(mol_smiles):
    mol = Chem.MolFromSmiles(mol_smiles)
    if not mol:
        raise ValueError("Invalid SMILES string")
    # 获取需要高亮的原子
    highlight_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]
    img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, size=(400, 300))
    return img

def remove_dummy_and_label(mol):
    """将dummy原子的邻居加映射号，并移除dummy原子"""
    rw_mol = Chem.RWMol(mol)
    map_num = 1  # 映射号从1开始
    dummy_idxs = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() == 0]
    for idx in sorted(dummy_idxs, reverse=True):
        atom = rw_mol.GetAtomWithIdx(idx)
        neighbors = atom.GetNeighbors()
        for n in neighbors:
            n.SetAtomMapNum(map_num)
            map_num+=1
        rw_mol.RemoveAtom(idx)
    return rw_mol.GetMol()
