from synprotac.models import Synprotac_RL_Model
import torch 
from synprotac.comparm import GP ,Update_PARAMS
import pickle,os
from tqdm import tqdm 
from pathlib import Path 

from synprotac.models.scores.scores import SimilarityScore
from synprotac.models import MolecularScorer


import argparse as arg 
parser = arg.ArgumentParser(description="Train a Synprotac model")
parser.add_argument('-i','--input')
args = parser.parse_args()
jsonfile = args.input

GP=Update_PARAMS(GP,jsonfile)

os.environ["CUDA_VISIBLE_DEVICES"]=GP.CUDA_VISIBLE_DEVICES
os.environ["CUDA_LAUNCH_BLOCKING"]="0"

sim_score=SimilarityScore(
    target_smiles=["CC1=C(C)C2=C(S1)N1C(C)=NN=C1[C@H](CC(=O)NCCCCCCCCNC(=O)COC1=CC=CC3=C1C(=O)N(C1CCC(=O)NC1=O)C3=O)N=C2C1=CC=C(Cl)C=C1"],
    tanimoto_k=0.1,
    cutoff=0.75
)

scorer=MolecularScorer(
    score_functions=[sim_score],
    score_weights=[1.0],
)

model=Synprotac_RL_Model(
    num_atom_classes = len(GP.atom_types)+1,
    num_bond_classes = len(GP.bond_types)+1,
    num_reaction_classes = 91,
    num_reagent_classes = 483,
    num_action_types = 4,
    max_sequence_length = 10,
    prior_checkpoint_path = "../../pretrained_models/synprotac.ckpt",
)

model.RL(
    warhead_smiles='[OH:1]C1=CC=CC2=C1C(N(C3CCC(NC3=O)=O)C2=O)=O',
    e3_ligand_smiles='Cc1c(C)c(C(c2ccc(Cl)cc2)=N[C@H](c3n4c(C)nn3)CC([OH:1])=O)c4s1',
    warhead_protected_patts = ['O=C(N(C1CCC(NC1=O)=O)C2=O)C3=C2C=CC=C3'],
    e3_ligand_protected_patts = ['Cc1sc2c(C(c3ccc(Cl)cc3)=N[C@@H2]c4nnc(C)n42)c1C'],
    reaction_templates_file = "templates.txt",
    reagents_file = "reagents.txt",
    scorer = scorer,
    savepath=Path("./models"),
    project_name="Synprotac-RL",
    load_cpkt=None,
    epochs=100000,
    batchsize=GP.batchsize,
    learning_rate=GP.learning_rate,
    ngpus=1
)
