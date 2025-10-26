import os
import sys
from pathlib import Path

import logging
from typing import List, Tuple, Dict, Optional, Set
from rdkit import Chem

from rdkit.Chem import AllChem, Draw,rdFMCS
import numpy as np 

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

def draw_molecule_with_dummpy_atoms(mol_smiles):
    mol = Chem.MolFromSmiles(mol_smiles)
    if not mol:
        raise ValueError("Invalid SMILES string")
    # 获取需要高亮的原子
    highlight_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
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


# 修复的反应模板 - 使用更简单有效的SMARTS
# 定义多个反应模板
def load_rxn_templates(rxn_file):
    with open(rxn_file, 'r') as f:
        rxns=[line.strip() for line in f if line.strip()]
    templates=[]
    for sid,rxn in enumerate(rxns):
        vars=rxn.split("\t")
        if len(vars) > 1:
            smarts=rxn.split('\t')[0]
            prior=rxn.split('\t')[1]
        else:
            smarts=rxn
            prior=1
        templates.append({
            'smarts': smarts,
            'id': sid,
            'name': f'Reaction {sid}',
            'priority': prior
        })
    return templates 

def load_building_blocks(bb_file):
    with open(bb_file, 'r') as f:
        bb_data = [line.strip() for line in f if line.strip()]
    building_blocks = []
    for bid,smiles in enumerate(bb_data):
        building_blocks.append({
            'smiles': smiles,
            'id': str(bid),
            'type': str(bid)
        })
    return building_blocks

from rdkit.Chem import rdmolops

def get_far_fragment(smiles: str, mapnum: int = 1, radius: int = 2, keep_largest: bool = True) -> str:
    """
    提取距离带 mapnum 的中心原子超过 radius 条键的原子组成的碎片 SMILES。
    - smiles: 含有原子映射号的输入 SMILES
    - mapnum: 中心原子的映射号
    - radius: 键距离阈值（默认2）
    - keep_largest: 多片段时是否只返回最大片段
    返回: 碎片 SMILES（可能包含多个片段用'.'分隔；若 keep_largest=True 则返回最大片段）
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 找到中心原子索引（支持有多个相同 mapnum 的中心时取最近距离）
    centers = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == mapnum]
    if not centers:
        raise ValueError(f"No atom with mapnum={mapnum} in: {smiles}")

    dmat = rdmolops.GetDistanceMatrix(mol)
    all_idx = list(range(mol.GetNumAtoms()))

    # 收集距离 <= radius 的近邻原子
    near = set()
    for i in all_idx:
        min_d = min(dmat[i, c] for c in centers)
        if min_d <= radius:
            near.add(i)

    # 距离 > radius 的远端原子
    far = [i for i in all_idx if i not in near]
    if not far:
        return ""

    # 清理映射号，输出更干净
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)

    frag_smiles = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=far,
        isomericSmiles=True,
        canonical=True,
        allHsExplicit=False,
        allBondsExplicit=False,
    )

    if not keep_largest or "." not in frag_smiles:
        return frag_smiles

    # 多片段时选最大片段
    frags = frag_smiles.split(".")
    sizes = []
    for s in frags:
        m = Chem.MolFromSmiles(s)
        sizes.append(m.GetNumAtoms() if m is not None else 0)
    return frags[int(np.argmax(sizes))]