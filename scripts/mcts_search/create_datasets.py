#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
from synprotac.chemistry import Parallel_Synthesizable_PROTAC_Search, Mini_Synthesizable_PROTAC_Search 
from synprotac.utils.rdkit import draw_molecule_with_map_atoms, load_rxn_templates, load_building_blocks, get_far_fragment
from rdkit import Chem 
from rdkit.Chem import AllChem, Draw
import random 
from tqdm import tqdm
import numpy as np
from rdkit import RDLogger

# 禁用所有rdApp.*相关的警告
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def collects_synprotac_trees_with_MCTS(warhead, e3_ligand, protected_patts, templates, building_blocks, savepath, njobs=1, nprocs=1, num_paths_per_proc=1):

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    warhead_img = draw_molecule_with_map_atoms(warhead)
    warhead_img.save(savepath / "warhead.png")
    e3_ligand_img = draw_molecule_with_map_atoms(e3_ligand)
    e3_ligand_img.save(savepath / "e3_ligand.png")
    # 这里是使用MCTS算法收集PROTAC合成树的代码
    protac_target=None
    Parallel_Synthesizable_PROTAC_Search(
        warhead=warhead,
        e3_ligand=e3_ligand,
        templates=templates,
        building_blocks=building_blocks,
        protected_patts=protected_patts,
        max_depth=3,
        max_iterations=1000,
        protac_target=protac_target,
        mini_template_pool_size=50,
        mini_building_block_size=100,
        savepath=savepath,
        njobs=56,  # 并行任务数
        nprocs=28,  # 并行进程数
        num_paths_per_proc=100
    )
    return 

# 定义构建块（包含更多种类的官能团）
templates=load_rxn_templates("templates.txt")
building_blocks=load_building_blocks("reagents.txt")

import pandas as pd 
warhead_df=pd.read_csv('warhead_dealed.txt')
e3ligand_df=pd.read_csv("e3_ligand_dealed.txt")


with open("testset.csv",'r') as f:
    for lid,line in enumerate(f.readlines()[1:2]):
        #try:
            protac,warhead,linker_id,e3_ligand=line.strip().split(',')
            savepath=Path(f"Protac_Synthesis_Tree/{lid}")
            warhead=warhead.replace("R1",'*')
            e3_ligand=e3_ligand.replace("R2",'*')
            logger.info(f"--Warhead:{warhead}")
            logger.info(f"--E3Ligand:{e3_ligand}")
            dealed_warhead=warhead_df[warhead_df["original_smiles"]==warhead]["dealed_smiles"].values[0]
            logger.info(f"--Warhead dealed as: {dealed_warhead}")
            dealed_e3ligand=e3ligand_df[e3ligand_df["original_smiles"]==e3_ligand]["dealed_smiles"].values[0]
            logger.info(f"--E3Ligand dealed as: {dealed_e3ligand}")
            logger.info(dealed_e3ligand)
     
            protected_warhead_patt= get_far_fragment(dealed_warhead,mapnum=1,radius=2)
            logger.info(f"--Protected Warhead Pattern: {protected_warhead_patt}")
     
            protected_e3ligand_patt= get_far_fragment(dealed_e3ligand,mapnum=1,radius=2)
            logger.info(f"--Protected E3Ligand Pattern: {protected_e3ligand_patt}")
            protected_patts=[protected_warhead_patt, protected_e3ligand_patt]
     
            collects_synprotac_trees_with_MCTS(
                warhead=dealed_warhead,
                e3_ligand=dealed_e3ligand,
                protected_patts=protected_patts,
                templates=templates,
                building_blocks=building_blocks,
                savepath=savepath,
                njobs=40,
                nprocs=20,
                num_paths_per_proc=100
            )
        #except Exception as e:
        #    logger.info(f"Error occurred while processing {lid}: {e}")

