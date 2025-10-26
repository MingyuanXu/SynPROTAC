"""Command-line entry point for constrained PROTAC linker optimization."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from rdkit import Chem

from workflow import FragmentConstraint, generate_constrained_conformer, load_molecule,generate_constrained_conformers_all_matches

logger = logging.getLogger("constrained_opt")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")

def generate_constrained_protac_conformers_with_reference(reference_sdf, target_sdf, warhead_smiles, e3_smiles, output_path) -> int:
    
    output_path = Path("optimized_target.sdf")
    logger.info("Loading reference molecule: %s", reference_sdf)
    reference = load_molecule(reference_sdf)
    logger.info("Loading target molecule: %s", target_sdf)
    target = load_molecule(target_sdf)

    constraints = [
        FragmentConstraint(smiles=warhead_smiles, label="warhead", match_index=0),
        FragmentConstraint(smiles=e3_smiles, label="E3 ligand", match_index=0),
    ]

    optimized = generate_constrained_conformers_all_matches(
        reference=reference,
        target=target,
        constraints=constraints,
        force_field="MMFF94",
        max_iterations=500,
        keep_hydrogens=True,
    )
    
    low_energy_conf_id=0
    low_energy_molobj=None 
    for i, opt in enumerate(optimized):
        if opt.energy is not None:
            if opt.energy < optimized[low_energy_conf_id].energy:
                low_energy_conf_id = i
                low_energy_molobj = opt.molecule

    logger.info("Lowest energy conformer ID: %d with energy %.4f kcal/mol", low_energy_conf_id, optimized[low_energy_conf_id].energy)
    writer = Chem.SDWriter(str(output_path))
    writer.write(low_energy_molobj)
    writer.close()
    if low_energy_molobj is None:
        logger.error("No optimized conformer was generated.")
    logger.info("Wrote optimized conformer to %s", output_path)

    return 

#if __name__ == "__main__":
if True:
    import sys
    _configure_logging(verbose=True)
    reference_sdf = "ref_ligand.sdf"
    target_sdf = "gen_ligand.sdf"
    warhead_smiles = "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1"
    e3_smiles = "Cc1sc2c(C(c3ccc(Cl)cc3)=N[C@@H2]c4nnc(C)n42)c1C"
    output_path = Path("optimized_target.sdf")
    generate_constrained_protac_conformers_with_reference(reference_sdf, target_sdf, warhead_smiles, e3_smiles, output_path)

