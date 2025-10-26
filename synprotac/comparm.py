from rdkit import Chem 
import json
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

class GPARAMS():
    def __init__(self):
        self.n_atom_classes = 100
        self.n_bond_classes = 5
        self.n_reaction_classes = 483

        self.hidden_dim = 512
        self.n_heads = 8
        self.dim_head = 64
        self.edge_dim = 128
        self.output_norm = False
        self.max_atoms = 70
        
        self.gencoder_depth = 8
        self.gencoder_rel_pos_emb = False
        self.decoder_depth = 6
        self.decoder_pe_max_len = 32
        self.decoder_fingerprint_dim = 256
        self.decoder_dim_fp_embed_hidden = 256

        self.n_token_types=5
        self.bond_types = [ Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC, Chem.BondType.ZERO]
        self.atom_types = [1,5,6,7,8,9,15,16,17,35,53]

        self.learning_rate = 1e-4
        self.batchsize = 256
        self.CUDA_VISIBLE_DEVICES = "0,1"
        self.fp_dim=256 # for Morgan Finger Print
        self.avoid_substructures=["C(=O)O","C(=O)OC(=O)"]
        self.DockStream_root_path='/mnt_191/myxu/synprotac/envs/DockStream'

    def update(self):
        self.bond_id_to_type = {i+1: bond for i, bond in enumerate(self.bond_types)}
        self.atom_id_to_type = {i+1: atom for i, atom in enumerate(self.atom_types)}
        self.bond_type_to_id = {bond: i+1 for i, bond in enumerate(self.bond_types)}
        self.atom_type_to_id = {atom: i+1 for i, atom in enumerate(self.atom_types)}
        self.fp_generator = GetMorganGenerator(radius=2, fpSize=self.fp_dim)

def Loaddict2obj(dict,obj):
    objdict=obj.__dict__
    for i in dict.keys():
        if i not in objdict.keys():
            print ("%s not is not a standard setting option!"%i)
        objdict[i]=dict[i]
    obj.__dict__==objdict

def Update_PARAMS(obj,jsonfile):
    with open(jsonfile,'r') as f:
        jsondict=json.load(f)
        Loaddict2obj(jsondict,obj)

    obj.update()
    return obj

GP=GPARAMS()
GP.update()
