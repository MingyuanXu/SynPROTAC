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


dockingscore = Constrained_DockingScore(
    target_pdb='7277_model_D1416_localfix.pdb',
    reflig_sdf='ref_ligand.sdf',
    warhead_smiles = "O=c1c(cccc2)c2n(c3c4cccc3)c(N4C5CCCC5)n1",
    e3_smiles = "O=C(NCc1ccc(c2scnc2C)cc1)[C@H]3N(C([C@H](C(C)(C)C)NC(C4(CC4)F)=O)=O)C[C@@H](C3)O",
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
    warhead_smiles='O=c1c(cccc2)c2n(c3c4cc(C5CC[NH:2]CC5)cc3)c(N4C6CCCC6)n1',
    e3_ligand_smiles='O=C(N[C@@H]([Br:1])c1ccc(c2scnc2C)cc1)[C@H]3N(C([C@H](C(C)(C)C)NC(C4(CC4)F)=O)=O)C[C@@H](C3)O',
    warhead_protected_patts = ['O=c1c(cccc2)c2n(c3c4cccc3)c(N4C5CCCC5)n1'],
    e3_ligand_protected_patts = ['O=C(N)[C@H]1N(C([C@H](C(C)(C)C)NC(C2(CC2)F)=O)=O)C[C@@H](C1)O','Cc1ncsc1c2ccccc2'],
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
