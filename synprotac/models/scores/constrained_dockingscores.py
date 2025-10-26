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
from .dockingscore import DockingScore 
from .constrained_opt import FragmentConstraint, generate_constrained_conformers_all_matches, load_molecule
from .suppress_output import suppress_output

class Constrained_DockingScore(DockingScore):
    def __init__(self,
                 target_pdb: str,
                 reflig_sdf: str,
                 warhead_smiles: str,
                 e3_smiles: str,
                 low_threshold: float = -10,
                 high_threshold: float = -6,
                 boxsize: List[float] = [30,30,30],
                 jobpath: Path = Path("./Docking"),
                 max_workers: int = 8,
                 strained_energy_cutoff: float = -4.0,
                 name: str = "Constrained_DockingScore",
                 refine_only: bool = True,
                 exhaustiveness: int = 2,
                 ):

        super().__init__(target_pdb=target_pdb,
                         reflig_sdf=reflig_sdf,
                         low_threshold=low_threshold,
                         high_threshold=high_threshold,
                         boxsize=boxsize,
                         jobpath=jobpath,
                         max_workers=max_workers,
                         strained_energy_cutoff=strained_energy_cutoff,
                         exhaustiveness=exhaustiveness,
                         )
        self.warhead_smiles=warhead_smiles
        self.e3_smiles=e3_smiles
        self.ref_protac=Chem.SDMolSupplier(reflig_sdf,removeHs=False)[0]
        self.name=name
        self.refine_only=refine_only
        return

    def prepare_constrained_ligands(self,ligand:Chem.Mol,output_path:Path, idx=0):

        ligand=move_molecule_to_target(ligand, target = self.center, mass_weighted = True)
        
        constraints = [
            FragmentConstraint(smiles=self.warhead_smiles, label="warhead", match_index=0),
            FragmentConstraint(smiles=self.e3_smiles, label="E3 ligand", match_index=0),
        ]

        optimized = generate_constrained_conformers_all_matches(
            reference=self.ref_protac,
            target=ligand,
            constraints=constraints,
            force_field="MMFF94",
            max_iterations=500,
            keep_hydrogens=True,
        )

        optimized_mols = [opt.molecule for opt in optimized if opt.molecule is not None]
        lowest_energy_conf_id=0
        lowest_energy_molobj=None 
        lowest_energy= float('inf')
        for i, opt in enumerate(optimized):
            if opt.energy is not None:
                if opt.energy < lowest_energy:
                    lowest_energy=opt.energy
                    lowest_energy_conf_id = i
                    lowest_energy_molobj = opt.molecule

        if lowest_energy_molobj is not None:
            writer = Chem.SDWriter(output_path / f"constrained_ligand_{idx}.sdf")
            writer.write(lowest_energy_molobj)
            writer.close()
            return 1
        else:
            print(f"Warning: No constrained conformer generated for ligand {idx}")
            return 0

    def docking_ligand(self,ligand, output_path,idx=0):
        constrained_flag=self.prepare_constrained_ligands(ligand, output_path, idx=idx)
        if constrained_flag:
            prepare_ligand_cmd=f"mk_prepare_ligand.py -i {output_path}/constrained_ligand_{idx}.sdf -o {output_path}/constrained_ligand_{idx}.pdbqt"
            os.system(prepare_ligand_cmd)
            
            best_score,strain_energy=AutoDock_Vina_Docking(
                            output_path=output_path, 
                            receptor_pdbqt=f"{self.jobpath}/receptor.pdbqt", 
                            ligand_pdbqt=f"{output_path}/constrained_ligand_{idx}.pdbqt",
                            mass_center=self.center,
                            box_size=self.boxsize,
                            output_name=f"pose_{idx}",
                            refine_only=self.refine_only,
                            exhaustiveness=self.exhaustiveness
                          )
        else:
            best_score=0
            strain_energy=0

        return  best_score, strain_energy
    
    def _dock_wrapper(self, ligand, prepare_path, ligand_id):
        try:
            with suppress_output():
                score, strained_energy = self.docking_ligand(ligand, output_path=prepare_path, idx=ligand_id)
        except Exception as e:
            print(f"Error occurred while docking ligand {ligand_id}: {e}")
            score = 0
            strained_energy = 0
        return ligand_id, score, strained_energy
