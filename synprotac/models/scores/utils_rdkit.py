from rdkit.Chem.Draw import rdMolDraw2D
#from IPython.display import Image
import copy
import numpy as np
from rdkit import Chem 
from rdkit.Chem import AllChem,rdmolfiles,rdFMCS,Draw
import copy
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from tqdm import tqdm 



def tanimoto_similarities(mol1,mol2):
    if GP.syssetting.similarity_type=='Morgan':
        fp1 = AllChem.GetMorganFingerprint(mol1, GP.similarity_radius, useCounts=True, useFeatures=True)
        fp2 = AllChem.GetMorganFingerprint(mol2, GP.similarity_radius, useCounts=True, useFeatures=True)
    similarity= DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity

def match_substructures(mols,smarts=[],submols=[]):
    submols+=[Chem.MolFromSmarts(subst) for subst in smarts if Chem.MolFromSmarts(subst)]
    print (submols)
    if len(submols)>0:
        matches=[]
        for mol in mols:
            if mol:
                match = any([mol.HasSubstructMatch(submol) for submol in submols])
                if match:
                    matches.append(1)
                else:
                    matches.append(0)
            else:
                matches.append(0)
        return np.array(matches)
    else:
        return np.zeros(len(mols))

def mcs_similarity(mols,smarts=[],submols=[]):
    similarity_scores=[]
    submols+=[Chem.MolFromSmarts(submol) for subst in smarts if Chem.MolFromSmarts(subst)]
    for mol in mols:
        max_max_similarity=0
        try:
            for submol in submols:
                mcs=rdFMCS.FindMCS([mol,submol])
                #print (mcs.smartsString)
                patt=Chem.MolFromSmarts(mcs.smartsString)
                matched_substructures=mol.GetSubstructMatches(patt)
                #print (matched_substructures)
                max_similarity=0
                msubsets=Get_fragmols(mol,matched_substructures)
                for msub in msubsets:
                #    print (matched_substruct)
                #    msubst=Get_fragmols(mol,matched_substruct)
                    similarity=tanimoto_similarities(msub,submol)
                    if similarity>max_similarity:
                        max_similarity=similarity
                if max_similarity>max_max_similarity:
                    max_max_similarity=max_similarity
        except:
            max_max_similarity=0
        similarity_scores.append(max_max_similarity)
    return np.array(similarity_scores)

def Change_mol_xyz(rdkitmol,coords):
    molobj=copy.deepcopy(rdkitmol)
    conformer=molobj.GetConformer()
    id=conformer.GetId()
    for cid,xyz in enumerate(coords):
        conformer.SetAtomPosition(cid,Point3D(float(xyz[0]),float(xyz[1]),float(xyz[2])))
    conf_id=molobj.AddConformer(conformer)
    molobj.RemoveConformer(id)
    return molobj

def Gen_ETKDG_structures(rdkitmol,nums=1,basenum=50,mode='opt+lowest',withh=False,ifwrite=False,path='./mol'):
    mol=copy.deepcopy(rdkitmol)
    mol_h=Chem.AddHs(mol)
    confids=AllChem.EmbedMultipleConfs(mol_h,basenum)
    confs=[]
    energies=[]
    mollist=[]
    for cid,c in enumerate(confids):
        conformer=mol_h.GetConformer(c)
        tmpmol=Chem.Mol(mol_h)
        ff=AllChem.UFFGetMoleculeForceField(tmpmol)
        if 'opt' in mode:
            ff.Minimize()
        uffenergy=ff.CalcEnergy()
        energies.append(uffenergy)
        if not withh:
            tmpmol=Chem.RemoveHs(tmpmol) 
        optconf=tmpmol.GetConformer(0).GetPositions()
        confs.append(optconf)
        mollist.append(tmpmol)
    lowest_ids=np.argsort(energies)
    lowest_confs=[confs[i] for i in lowest_ids[:nums]]
    if ifwrite:
        for i in lowest_ids[:nums]:
            rdmolfiles.MolToMolFile(mollist[i],f'{path}.mol2')
    return [mollist[i] for i in lowest_ids[:nums]]
        
def Drawmols(rdkitmol,filename='Mol.png',permindex=[],cliques=[]):
    reindex=np.zeros(len(permindex))
    for pid,p in enumerate(permindex):
        reindex[p]=pid 
    mol=copy.deepcopy(rdkitmol)
    Chem.rdDepictor.Compute2DCoords(mol)
    hatomlist=[]
    hbondlist=[]
    colors=[(1,1,0),(1,0,1),(1,0,0),(0,1,1),(0,1,0),(0,0,1)]
    if len(cliques)>0:
        for clique in cliques:
            if len(clique)>1:
                clique_bonds=[]
                hatomlist.append([int(a) for a in clique])
                for bond in mol.GetBonds():
                    a1=bond.GetBeginAtom().GetIdx()
                    a2=bond.GetEndAtom().GetIdx()
                    if a1 in clique and a2 in clique:
                        clique_bonds.append(bond.GetIdx())
                hbondlist.append(clique_bonds)
        atom_colors={}
        bond_colors={}
        atomlist=[]
        bondlist=[]
        for i,(hl_atom,hl_bond) in enumerate(zip(hatomlist,hbondlist)):
            #print (hl_atom,hl_bond)
            hl_atom=list(hl_atom)
            for at in hl_atom:
                atom_colors[at]=colors[i%6]
                atomlist.append(at)
            for bt in hl_bond:
                bond_colors[bt]=colors[i%6]
                bondlist.append(bt)

    options=rdMolDraw2D.MolDrawOptions()
    options.addAtomIndices=True
    draw=rdMolDraw2D.MolDraw2DCairo(500,500)
    for i in range(len(reindex)):
        mol.GetAtomWithIdx(i).SetProp("atomNote",':'+str(int(reindex[i])))
    draw.SetDrawOptions(options)
    #print (type(atomlist[0]),type(atom_colors),type(bondlist[0]),type(bond_colors))
    rdMolDraw2D.PrepareAndDrawMolecule(draw,mol,highlightAtoms=atomlist,
                                                highlightAtomColors=atom_colors,
                                                highlightBonds=bondlist,
                                                highlightBondColors=bond_colors)
    draw.FinishDrawing()
    draw.WriteDrawingText(filename)

def SmilesToSVG(smiles,legends=None,fname='mol.svg'):
    mols=[]
    vlegends=[]
    for sid,smi in enumerate(smiles):
        mol=Chem.MolFromSmiles(smi)
        if mol:
            Chem.AllChem.Compute2DCoords(mol)
            mols.append(mol)
            if legends:
                vlegends.append(legends[sid])
    img=Draw.MolsToGridImage(mols,legends=vlegends,molsPerRow=5,subImgSize=(250,250),useSVG=True)
    with open (fname,'w') as f:
        f.write(img)
    return 

def Neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol

def Get_fragmols(mol,cliques):
    natoms=mol.GetNumAtoms()
    frags=[]
    for clique in cliques:
        ringfrag=Chem.RWMol(mol)
        ringfrag.BeginBatchEdit()
        for i in range(natoms):
            if i not in clique:
                ringfrag.RemoveAtom(i)
        ringfrag.CommitBatchEdit()
        frag=ringfrag.GetMol()
        #Chem.Kekulize(frag)
        frag=Neutralize_atoms(frag)
        frags.append(frag)
    return frags

def Save_smiles_list(smileslist,filename):
    with open(filename,'w')  as f:
        for smiles in smileslist:
            if smiles:
                f.write(smiles+'\n')
            else:
                f.write('None\n')

def Load_smiles_list(filename):
    with open(filename,'r')  as f:
        smileslist=[line.strip() for line in f.readlines() if '.' not in line]
    return smileslist

def Save_smarts_list(smartslist,filename):
    with open(filename,'w')  as f:
        for smiles in smartslist:
            f.write(smiles+'\n')

def Load_smarts_list(filename):
    with open(filename,'r')  as f:
        smartslist=[line.strip() for line in f.readlines()]
    return smartslist

def analysis_molecules_properties(smis):
    molwts=[]
    qeds=[]
    tpsas=[]
    logps=[]
    hbas=[]
    hbds=[]
    for smi in tqdm(smis):
        if smi:
            mol=Chem.MolFromSmiles(smi)
            qed=QED.qed(mol)
            logp  = Descriptors.MolLogP(mol)
            tpsa  = Descriptors.TPSA(mol)
            molwt = Descriptors.ExactMolWt(mol)
            hba   = rdMolDescriptors.CalcNumHBA(mol)
            hbd   = rdMolDescriptors.CalcNumHBD(mol)
            molwts.append(molwt)
            qeds.append(qed)
            logps.append(logp)
            tpsas.append(tpsa)
            hbas.append(hba)
            hbds.append(hbd)
    return molwts,qeds,tpsas,logps,hbas,hbds

from rdkit import Chem 
import numpy as np

def rdkit_center_of_mass(mol: Chem.Mol, conformer_id: int = 0) -> np.ndarray:
    """
    计算 RDKit 分子的质量中心（按原子质量加权）
    """
    conf = mol.GetConformer(conformer_id)
    if conf is None:
        raise ValueError("Molecule has no conformer with id %r" % conformer_id)
    coords = []
    masses = []
    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        coords.append([p.x, p.y, p.z])
        masses.append(atom.GetMass())  # 原子质量（可能是同位素校正后的）
    coords = np.array(coords, dtype=float)
    masses = np.array(masses, dtype=float)
    if masses.sum() == 0:
        raise ValueError("Total mass is zero")
    com = (coords * masses[:, None]).sum(axis=0) / masses.sum()
    return com

def move_molecule_to_target(mol: Chem.Mol, target: list, mass_weighted: bool = True, ) -> Chem.Mol:
    """
    将分子移动到目标位置
    Args:
        mol: RDKit 分子对象
        target: 目标位置 [x, y, z]
        mass_weighted: 是否按原子质量加权计算中心
        copy: 是否返回分子的副本
    Returns:
        移动后的分子对象
    """
    if mass_weighted:
        center = rdkit_center_of_mass(mol)
    else:
        conf = mol.GetConformer()
        if conf is None:
            raise ValueError("Molecule has no conformer")
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        center = coords.mean(axis=0)
    translation = np.array(target) - center
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        new_p = p + translation
        conf.SetAtomPosition(i, new_p)
    return mol

def strain_energy_per_atom(mol:Chem.Mol, force_field:str='MMFF94', max_iterations:int=500) -> tuple:
    """
    计算分子的应变能
    Args:
        mol: RDKit 分子对象
        force_field: 力场类型，支持 'MMFF94' 和 'UFF'
        max_iterations: 最大优化迭代次数
    Returns:
        (收敛状态码, 应变能)
    """
    if force_field == 'MMFF94':
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    elif force_field == 'UFF':
        ff = AllChem.UFFGetMoleculeForceField(mol)
    else:
        raise ValueError("Unsupported force field: %s" % force_field)
    ff.Initialize()
    energy_before=ff.CalcEnergy()
    status = ff.Minimize(maxIts=max_iterations)
    energy_after = ff.CalcEnergy()
    energy_diff=energy_after-energy_before
    natoms=mol.GetNumAtoms()
    strain_energy_per_atom=energy_diff/natoms if natoms>0 else float('inf')
    return strain_energy_per_atom

def pdbqt_to_sdf(pdbqt_path:str, sdf_path:str) -> None:
    """
    将 PDBQT 文件转换为 SDF 文件
    Args:
        pdbqt_path: 输入的 PDBQT 文件路径
        sdf_path: 输出的 SDF 文件路径
    """
    from rdkit import Chem
    from openbabel import pybel
    mols=list(pybel.readfile("pdbqt", pdbqt_path))
    writer = Chem.SDWriter(sdf_path)
    for pybelmol in mols:
        mol_block=pybelmol.write("mol")
        mol=Chem.MolFromMolBlock(mol_block)
        writer.write(mol)
    writer.close()
    return

def pdbqt_to_rdkitmols(pdbqt_path:str) -> list:
    """
    将 PDBQT 文件转换为 SDF 文件
    Args:
        pdbqt_path: 输入的 PDBQT 文件路径
        sdf_path: 输出的 SDF 文件路径
    """
    from rdkit import Chem
    from openbabel import pybel
    mols=list(pybel.readfile("pdbqt", pdbqt_path))
    rdkitmols=[]
    for pybelmol in mols:
        mol_block=pybelmol.write("mol")
        mol=Chem.MolFromMolBlock(mol_block)
        rdkitmols.append(mol)
    return rdkitmols 