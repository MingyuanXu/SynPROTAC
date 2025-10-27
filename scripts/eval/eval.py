from synprotac.models import Synprotac_Model
from synprotac.chemistry.reaction_search import ChemReactionSearch, ReactionTemplate, BuildingBlock
from synprotac.utils.rdkit import draw_molecule_with_map_atoms, load_rxn_templates, load_building_blocks
import random 
import torch 
from synprotac.comparm import GP ,Update_PARAMS
import pickle,os
from tqdm import tqdm 
from pathlib import Path 
from synprotac.chemistry import visualize_path
from synprotac.utils.rdkit import draw_molecule_with_map_atoms, load_rxn_templates, load_building_blocks, get_far_fragment
from rdkit import Chem 
from rdkit.Chem import AllChem, Draw
import random 
from tqdm import tqdm
import numpy as np
from rdkit import RDLogger
import logging
# 禁用所有rdApp.*相关的警告
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import argparse as arg 
parser = arg.ArgumentParser(description="Synprotac model")
parser.add_argument('-i','--input')
args = parser.parse_args()
jsonfile = args.input

GP=Update_PARAMS(GP,jsonfile)

os.environ["CUDA_VISIBLE_DEVICES"]=GP.CUDA_VISIBLE_DEVICES
os.environ["CUDA_LAUNCH_BLOCKING"]="0"

model=Synprotac_Model(
    num_atom_classes = len(GP.atom_types)+1,
    num_bond_classes = len(GP.bond_types)+1,
    num_reaction_classes = 91,
    num_reagent_classes = 483,
    num_action_types = 4,
    max_sequence_length = 15
)

import pandas as pd 
warhead_df=pd.read_csv('warhead_dealed.txt')
e3ligand_df=pd.read_csv("e3_ligand_dealed.txt")

with open("testset.csv",'r') as f:
    lines=f.readlines()
    for lid,line in enumerate(lines[1:]):
        #try:
            savepath=Path(f"samples/{lid}")
            if os.path.exists(savepath/f"routes-{lid}.pkl"):
                continue 
            print (f"{lid},job")

            protac,warhead,linker_id,e3_ligand=line.strip().split(',')
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

            valid_routes, validity = model.Sample(
                warhead_smiles = dealed_warhead,
                e3_ligand_smiles = dealed_e3ligand,
                reaction_templates_file = "templates.txt",
                reagents_file = "reagents.txt",
                batchsize=100,
                num_samples=100,
                load_cpkt='../../pretrained_models/synprotac.ckpt',  # 加载预训练权重
                warhead_protected_patts = [protected_warhead_patt],
                e3_ligand_protected_patts = [protected_e3ligand_patt],
            )

            logger.info(f"Validity of SynPROTAC for {lid}th pair of warhead and E3-Ligand : {validity}")

            if not os.path.exists(savepath):
                os.makedirs(savepath)

            with open(savepath/f"routes-{lid}.pkl",'wb') as f:
                pickle.dump(valid_routes, f)
        #except:
        #    pass
