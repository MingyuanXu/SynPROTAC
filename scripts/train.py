from synprotac.models import Synprotac_Model
import torch 
from synprotac.comparm import GP ,Update_PARAMS
import pickle,os
from tqdm import tqdm 
from pathlib import Path 

import argparse as arg 
parser = arg.ArgumentParser(description="Train a Synprotac model")
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
    max_sequence_length = 10
)

model.Train(
    data_path=Path("./data_mini"),
    epochs=1000,
    batchsize=GP.batchsize,
    learning_rate=GP.learning_rate,
    project_name="Synprotac_debug",
    savepath=Path("./models"),
    ngpus=2
)
