from rdkit import Chem
from .utils_rdkit import rdkit_center_of_mass ,move_molecule_to_target
import numpy as np 
from typing import List 
import os 
from pathlib import Path
from .scores import BaseScore
from .vina import AutoDock_Vina_Docking
import math 
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import torch 
from .scores import reverse_sigmoid_transformation
from .suppress_output import suppress_output

class DockingScore(BaseScore):
    def __init__(self,
                 target_pdb: str,
                 reflig_sdf: str = None,
                 low_threshold: float = -10.0,
                 high_threshold: float = -6.0,
                 center: List[float] = None,
                 boxsize: List[float] = [30,30,30],
                 jobpath: Path = Path("./Docking"),
                 max_workers: int = 8,
                 strained_energy_cutoff: float = -4.0,
                 name="DockingScore",
                 exhaustiveness: int = 8,
                 ):

        super().__init__()
        self.target_pdb=target_pdb
        self.reflig_sdf=reflig_sdf
        self.low_threshold=low_threshold
        self.high_threshold=high_threshold
        self.max_workers=max_workers
        self.strained_energy_cutoff=strained_energy_cutoff
        self.name=name 
        self.exhaustiveness=exhaustiveness
        assert center is not None or reflig_sdf is not None, "Either center or reflig_sdf must be provided"

        if center is not None:
            self.center=center
        else:
            self.center=self.cal_pocket_center()

        if not os.path.exists(jobpath):
            os.makedirs(jobpath)

        self.jobpath=jobpath
        self.boxsize=boxsize

        self.prepare_target()
    
    def prepare_target(self):
        os.system(f"cp {self.target_pdb} {self.jobpath}/receptor.pdb")
        prepare_target_cmd=f"mk_prepare_receptor.py -i {self.jobpath}/receptor.pdb -o {self.jobpath}/receptor -p -v --box_size {self.boxsize[0]} {self.boxsize[1]} {self.boxsize[2]} --box_center {self.center[0]} {self.center[1]} {self.center[2]}"
        try:
            os.system(prepare_target_cmd)
        except:
            raise RuntimeError("Target preparation failed,please check the input target pdb file")
        pass

    def docking_ligand(self,ligand, output_path,idx=0):
        ligand=move_molecule_to_target(ligand, target=self.center, mass_weighted=True)
        writer=Chem.SDWriter(f"{output_path}/ligand_{idx}.sdf")
        writer.write(ligand)
        writer.close()
        prepare_ligand_cmd=f"mk_prepare_ligand.py -i {output_path}/ligand_{idx}.sdf -o {output_path}/ligand_{idx}.pdbqt"
        os.system(prepare_ligand_cmd)
        with suppress_output():
            best_score, strained_energy=AutoDock_Vina_Docking(
                            output_path=output_path, 
                            receptor_pdbqt=f"{self.jobpath}/receptor.pdbqt", 
                            ligand_pdbqt=f"{output_path}/ligand_{idx}.pdbqt",
                            mass_center=self.center,
                            box_size=self.boxsize,
                            output_name=f"pose_{idx}.pdbqt",
                          )
        return best_score, strained_energy 

    def cal_pocket_center(self):
        ref_ligand=Chem.SDMolSupplier(self.reflig_sdf,removeHs=False)[0]
        center=rdkit_center_of_mass(ref_ligand)
        print ("Receptor pocket center (x,y,z):",center)
        return center 
    
    def _dock_wrapper(self, ligand, prepare_path, ligand_id):
        try:
            score, strained_energy = self.docking_ligand(ligand, output_path=prepare_path, idx=ligand_id)
        except Exception as e:
            print(f"Error occurred while docking ligand {ligand_id}: {e}")
            score = 0
            strained_energy = 0
        return ligand_id, score, strained_energy

    def compute_scores(self, mols: List[Chem.Mol], subset_id: int = 0, rank: int = 0):
        ligands = mols
        prepare_path = self.jobpath / f"subset_{rank}_{subset_id}"
        if os.path.exists(prepare_path) is False:
            os.makedirs(prepare_path)
        
        affinity_scores = np.full(len(ligands), float("nan"), dtype=np.float32)
        strained_energies = np.full(len(ligands), float("nan"), dtype=np.float32)

        if self.max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) * 2)
        else:
            max_workers = self.max_workers

        tasks = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for ligand_id, ligand in enumerate(ligands):
                if ligand is None:
                    continue
                tasks.append(ex.submit(self._dock_wrapper,
                                       ligand, prepare_path, ligand_id))

            for fut in as_completed(tasks):
                try:
                    ligand_id, score, strained_energy = fut.result()
                    affinity_scores[ligand_id] = score
                    strained_energies[ligand_id] = strained_energy
                except Exception as e:
                    print(f"Unhandled exception in docking task: {e}")

        with open(prepare_path / "docking_scores.txt", "w") as f:
        
            for ligand_id, score, strained_energy in zip(range(len(affinity_scores)),affinity_scores, strained_energies):
        
                f.write(f"ligand {ligand_id}: affinity: {score} kcal/mol, strained energy: {strained_energy} kcal/mol per atom\n")
        
        affinity_scores = np.where(strained_energies<self.strained_energy_cutoff, 0, affinity_scores)
        
        final_scores = reverse_sigmoid_transformation(affinity_scores, _low=self.low_threshold, _high=self.high_threshold, _k=0.25)

        return torch.tensor(final_scores), torch.tensor(affinity_scores)
