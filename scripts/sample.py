from synprotac.models import Synprotac_Model
from synprotac.chemistry.reaction_search import ChemReactionSearch, ReactionTemplate, BuildingBlock
from synprotac.utils.rdkit import draw_molecule_with_map_atoms, load_rxn_templates, load_building_blocks

import torch 
from synprotac.comparm import GP ,Update_PARAMS
import pickle,os
from tqdm import tqdm 
from pathlib import Path 
from synprotac.chemistry import visualize_path
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
    max_sequence_length = 15
)

e3_ligand='Cc1c(C)c(C(c2ccc(Cl)cc2)=N[C@H](c3n4c(C)nn3)CC([OH:1])=O)c4s1'
warhead='[OH:1]C1=CC=CC2=C1C(N(C3CCC(NC3=O)=O)C2=O)=O'

protected_patts = ['O=C(N(C1CCC(NC1=O)=O)C2=O)C3=C2C=CC=C3','Cc1sc2c(C(c3ccc(Cl)cc3)=N[C@@H2]c4nnc(C)n42)c1C']
templates=load_rxn_templates("templates.txt")
building_blocks=load_building_blocks("reagents_new.txt")
searcher = ChemReactionSearch()
searcher.load_reaction_templates(templates)
searcher.load_building_blocks(building_blocks)
templates_for_e3_ligand=searcher.get_suitable_rxn_templates_for_specific_molecule(e3_ligand,protected_patts=protected_patts)
templates_for_warhead=searcher.get_suitable_rxn_templates_for_specific_molecule(warhead,protected_patts=protected_patts)
print ('Suitable templates for e3 ligand:',[tmp.template_id for tmp in templates_for_e3_ligand])
print ('Suitable templates for warhead:',[tmp.template_id for tmp in templates_for_warhead])
print(f"{len(templates_for_e3_ligand)},{len(templates_for_warhead)}") 

for i in range(10):
    try:
        valid_routes, validity = model.Sample(
            warhead_smiles = warhead,
            e3_ligand_smiles = e3_ligand,
            reaction_templates_file = "templates.txt",
            reagents_file = "reagents.txt",
            batchsize=100,
            num_samples=100,
            load_cpkt='./models/synprotac_prior.ckpt',  # 加载预训练权重
            warhead_protected_patts = ['O=C(N(C1CCC(NC1=O)=O)C2=O)C3=C2C=CC=C3'],
            e3_ligand_protected_patts = ['Cc1sc2c(C(c3ccc(Cl)cc3)=N[C@@H2]c4nnc(C)n42)c1C']
        )
 
        print (validity)
 
        savepath= Path(f"./samples/{i}")
        if not os.path.exists(savepath):
            os.makedirs(savepath)
 
        with open(savepath/f"routes-{i}.pkl",'wb') as f:
            pickle.dump(valid_routes, f)
    except:
        pass
