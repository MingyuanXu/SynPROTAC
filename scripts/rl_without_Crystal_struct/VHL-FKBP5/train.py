from synprotac.models import Synprotac_RL_Model
import torch 
from synprotac.comparm import GP ,Update_PARAMS
import pickle,os
from tqdm import tqdm 
from pathlib import Path 

from synprotac.models.scores.scores import SimilarityScore
from synprotac.models.scores.constrained_dockingscores import Constrained_DockingScore
from synprotac.models import MolecularScorer


import argparse as arg 
parser = arg.ArgumentParser(description="Train a Synprotac model")
parser.add_argument('-i','--input')
args = parser.parse_args()
jsonfile = args.input

GP=Update_PARAMS(GP,jsonfile)

os.environ["CUDA_VISIBLE_DEVICES"]=GP.CUDA_VISIBLE_DEVICES
os.environ["CUDA_LAUNCH_BLOCKING"]="0"

"""
sim_score=SimilarityScore(
    target_smiles=["CC1=C(C)C2=C(S1)N1C(C)=NN=C1[C@H](CC(=O)NCCCCCCCCNC(=O)COC1=CC=CC3=C1C(=O)N(C1CCC(=O)NC1=O)C3=O)N=C2C1=CC=C(Cl)C=C1"],
    tanimoto_k=0.1,
    cutoff=0.75
)
"""

dockingscore = Constrained_DockingScore(
    target_pdb='8pc2_model.pdb',
    reflig_sdf='ref_ligand.sdf',
    warhead_smiles = "c1ccc([C@@H](CCc2cc(OC)c(OC)cc2)OC([C@H]3N(C([C@@H](C4CCCCC4)c5cc(OC)c(OC)c(OC)c5)=O)CCCC3)=O)cc1",
    e3_smiles = "FC1(C(N[C@@H](C(C)(C)C)C(N2C[C@@H](C[C@H]2C(NCc3ccc(c4scnc4C)cc3)=O)O)=O)=O)CC1",
    low_threshold=-12,
    high_threshold=-1,
    jobpath=Path('./Constrained_Docking'),
    boxsize=[32,38,32],
    strained_energy_cutoff=-10.0,
    max_workers=20,
    refine_only=True,
    target_match_topk=4,
    score_transform="leaky_window",
    outside_penalty=0.1,
    score_clip_min=-1.0,
)

scorer=MolecularScorer(
    score_functions=[dockingscore],
    score_weights=[1.0],
)

model=Synprotac_RL_Model(
    num_atom_classes = len(GP.atom_types)+1,
    num_bond_classes = len(GP.bond_types)+1,
    num_reaction_classes = 91,
    num_reagent_classes = 483,
    num_action_types = 4,
    max_sequence_length = 10,
    prior_checkpoint_path = Path("./models/synprotac_prior.ckpt"),
)

model.RL(
    warhead_smiles='[OH:2]c1ccc([C@@H](CCc2cc(OC)c(OC)cc2)OC([C@H]3N(C([C@@H](C4CCCCC4)c5cc(OC)c(OC)c(OC)c5)=O)CCCC3)=O)cc1',
    e3_ligand_smiles='FC1(C(N[C@@H](C(C)(C)C)C(N2C[C@@H](C[C@H]2C(NCc3ccc(c4scnc4C)cc3[NH2:1])=O)O)=O)=O)CC1',
    warhead_protected_patts = ['c1ccc([C@@H](CCc2cc(OC)c(OC)cc2)OC([C@H]3N(C([C@@H](C4CCCCC4)c5cc(OC)c(OC)c(OC)c5)=O)CCCC3)=O)cc1'],
    e3_ligand_protected_patts = ['FC1(C(N[C@@H](C(C)(C)C)C(N2C[C@@H](C[C@H]2C(NCc3ccc(c4scnc4C)cc3)=O)O)=O)=O)CC1'],
    reaction_templates_file = "templates.txt",
    reagents_file = "reagents.txt",
    scorer = scorer,
    savepath=Path("./models"),
    project_name="Synprotac-RL",
    load_cpkt=None,
    epochs=100000,
    batchsize=GP.batchsize,
    learning_rate=GP.learning_rate,
    ngpus=1,
    rl_samples_path="./rl-samples",
)
