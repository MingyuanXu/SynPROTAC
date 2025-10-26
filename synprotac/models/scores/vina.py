
from rdkit import Chem 
import numpy as np
from pathlib import Path
import os 
from vina import Vina
from .utils_rdkit import pdbqt_to_rdkitmols,strain_energy_per_atom



def estimate_strained_energy(pdbqt_path):
    mols=pdbqt_to_rdkitmols(pdbqt_path=pdbqt_path)
    strain_energies_per_atom=[]
    for mol in mols:
        try:
            se_per_atom = strain_energy_per_atom(mol)
        except Exception as e:
            se_per_atom = float('nan')
        strain_energies_per_atom.append(se_per_atom)
    return strain_energies_per_atom

def AutoDock_Vina_Docking(
                            output_path, 
                            receptor_pdbqt, 
                            ligand_pdbqt,
                            mass_center,
                            box_size=[30,30,30],
                            output_name="vina.pdbqt",
                            exhaustiveness=8,
                            n_poses=20,
                            refine_only=False
                          ):
    
    v = Vina(sf_name='vina')
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(center=mass_center, box_size=box_size)

    # Score the current pose
    energy = v.score()
    #print('Score before minimization: %.3f (kcal/mol)' % energy[0])

    # Minimized locally the current pose
    energy_minimized = v.optimize()
    #print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
    v.write_pose(f'{output_path}/{output_name}.pdbqt', overwrite=True)
    best_score = energy_minimized[0]
    strain_energies_per_atom=estimate_strained_energy(pdbqt_path=f'{output_path}/{output_name}.pdbqt')
    strain_energy=strain_energies_per_atom[0] if len(strain_energies_per_atom)>0 else 0

    # Dock the ligand
    if not refine_only:
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        v.write_poses(f'{output_path}/{output_name}.pdbqt', n_poses=n_poses, overwrite=True)
        scores = v.energies(n_poses=n_poses)
        mols=pdbqt_to_rdkitmols(pdbqt_path=f'{output_path}/{output_name}.pdbqt')
        strain_energies_per_atom=estimate_strained_energy(pdbqt_path=f'{output_path}/{output_name}.pdbqt')

        assert len(strain_energies_per_atom) == len(scores), "Length mismatch between strain energies and scores"

        #print ('strain_energies_per_atom',strain_energies_per_atom)
        #print ('scores',scores) 
        if len(scores) > 0:
            best_score = scores[0][0]
            strain_energy = strain_energies_per_atom[0]
        else:
            best_score = 0
            strain_energy = 0

    return best_score,strain_energy


