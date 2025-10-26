"""Core routines for constrained PROTAC linker optimization.

This module exposes helper functions to:
- load reference and target molecules from SMILES/SDF/MOL inputs
- match user-provided SMARTS patterns (warhead and E3 ligand fragments)
- transfer the reference fragment coordinates to the target molecule
- embed and optimize the target while freezing the matched fragments
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem.rdchem import Conformer, Mol
from rdkit.Geometry import Point3D

logger = logging.getLogger(__name__)


@dataclass
class FragmentConstraint:
    """Definition of a fragment to be constrained during optimization."""

    smiles: str
    label: str
    match_index: int = 0  # use the first match by default


@dataclass(frozen=True)
class FragmentMatchAssignment:
    """Mapping between a fragment match on the reference and target molecules."""

    label: str
    smiles: str
    reference_match: Tuple[int, ...]
    target_match: Tuple[int, ...]
    reference_match_index: int
    target_match_index: int


@dataclass
class ConstrainedConformerResult:
    """Container for a generated constrained conformer and its match metadata."""

    molecule: Mol
    matches: List[FragmentMatchAssignment]
    convergence_code: int
    energy: Optional[float] = None


def load_molecule(path_or_smiles: str, add_hydrogens: bool = True) -> Mol:
    """Load an RDKit molecule from a file path or SMILES string."""

    if os.path.isfile(path_or_smiles):
        ext = os.path.splitext(path_or_smiles)[1].lower()

        if ext == ".sdf":
            supplier = Chem.SDMolSupplier(path_or_smiles, removeHs=False)
            mol = supplier[0] if supplier and supplier[0] is not None else None
            if mol is not None:
                Chem.SanitizeMol(mol)
        elif ext in {".mol", ".mol2"}:
            mol = Chem.MolFromMolFile(path_or_smiles, removeHs=False)
            if mol is not None:
                Chem.SanitizeMol(mol)
        else:
            raise ValueError(f"Unsupported extension '{ext}' for {path_or_smiles}")
        
    else:
        mol = Chem.MolFromSmiles(path_or_smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol) if add_hydrogens else mol
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.UFFOptimizeMolecule(mol)

    if mol is None:
        raise ValueError(f"Failed to parse molecule from '{path_or_smiles}'")

    mol = Chem.AddHs(mol, addCoords=True) if add_hydrogens else mol
    return mol


def _ensure_conformer(mol: Mol, random_seed: int = 0xF00D) -> Mol:
    """Guarantee that *mol* has at least one 3D conformer."""

    if mol.GetNumConformers():
        return mol

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, params)

    if status == -1:
        raise RuntimeError("Embedding failed; consider adjusting SMARTS or providing a 3D template")

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as exc:  # noqa: BLE001 - RDKit raises generic Exception
        logger.warning("UFF pre-optimization failed: %s", exc)
    return mol


def _match_fragment(mol: Mol, constraint: FragmentConstraint) -> Tuple[int, ...]:
    query = Chem.MolFromSmiles(constraint.smiles)
    if query is None:
        raise ValueError(f"Invalid SMILES for {constraint.label}: {constraint.smiles}")

    matches = mol.GetSubstructMatches(query, uniquify=False)
    if not matches:
        raise ValueError(f"No matches found for {constraint.label} using SMILES {constraint.smiles}")

    if constraint.match_index >= len(matches):
        raise IndexError(
            f"match_index {constraint.match_index} out of range for {constraint.label}; "
            f"only {len(matches)} match(es) available"
        )
    match = matches[constraint.match_index]
    logger.debug("Selected match for %s: %s", constraint.label, match)
    return match


def _build_coord_map(
    reference: Mol,
    target: Mol,
    constraints: Sequence[FragmentConstraint],
    conformer_id: int = 0,
) -> Dict[int, Point3D]:
    """Construct a coordinate map that anchors selected atoms in the target."""

    ref_conf = reference.GetConformer(conformer_id)
    coord_map: Dict[int, Point3D] = {}

    for constraint in constraints:
        ref_match = _match_fragment(reference, constraint)
        tgt_match = _match_fragment(target, constraint)
        if len(ref_match) != len(tgt_match):
            raise ValueError(
                f"Fragment size mismatch for {constraint.label}: "
                f"reference {len(ref_match)} atoms vs target {len(tgt_match)} atoms"
            )
        for ref_idx, tgt_idx in zip(ref_match, tgt_match):
            pt = ref_conf.GetAtomPosition(ref_idx)
            coord_map[tgt_idx] = Point3D(pt.x, pt.y, pt.z)
        logger.info(
            "Mapped %d atoms for %s (ref match %s -> target match %s)",
            len(ref_match),
            constraint.label,
            ref_match,
            tgt_match,
        )
    return coord_map


def _build_coord_map_from_assignments(
    ref_conf: Conformer,
    assignments: Sequence[FragmentMatchAssignment],
) -> Dict[int, Point3D]:
    coord_map: Dict[int, Point3D] = {}
    for assignment in assignments:
        ref_match = assignment.reference_match
        tgt_match = assignment.target_match
        if len(ref_match) != len(tgt_match):
            raise ValueError(
                f"Fragment size mismatch for {assignment.label}: "
                f"reference {len(ref_match)} atoms vs target {len(tgt_match)} atoms"
            )
        for ref_idx, tgt_idx in zip(ref_match, tgt_match):
            pt = ref_conf.GetAtomPosition(ref_idx)
            coord_map[tgt_idx] = Point3D(pt.x, pt.y, pt.z)
    return coord_map


def _iter_fragment_match_assignments(
    reference: Mol,
    target: Mol,
    constraints: Sequence[FragmentConstraint],
) -> Iterator[List[FragmentMatchAssignment]]:
    """Yield all combinations of fragment match assignments between reference and target."""

    if not constraints:
        return

    options: List[List[FragmentMatchAssignment]] = []

    for constraint in constraints:
        query = Chem.MolFromSmiles(constraint.smiles)
        if query is None:
            raise ValueError(f"Invalid SMILES for {constraint.label}: {constraint.smiles}")

        ref_matches = reference.GetSubstructMatches(query, uniquify=False)
        tgt_matches = target.GetSubstructMatches(query, uniquify=False)

        if not ref_matches:
            raise ValueError(f"No matches found in reference for {constraint.label}")
        if not tgt_matches:
            raise ValueError(f"No matches found in target for {constraint.label}")

        assignments: List[FragmentMatchAssignment] = []
        for ref_idx, ref_match in enumerate(ref_matches):
            for tgt_idx, tgt_match in enumerate(tgt_matches):
                if len(ref_match) != len(tgt_match):
                    continue
                assignments.append(
                    FragmentMatchAssignment(
                        label=constraint.label,
                        smiles=constraint.smiles,
                        reference_match=ref_match,
                        target_match=tgt_match,
                        reference_match_index=ref_idx,
                        target_match_index=tgt_idx,
                    )
                )

        if not assignments:
            raise ValueError(
                f"No size-compatible match pairs found for {constraint.label}; check fragment definition"
            )

        options.append(assignments)

    for combo in product(*options):
        yield list(combo)


def _place_fragments_and_optimize(
    target: Mol,
    coord_map: Dict[int, Point3D],
    force_field: str,
    max_iterations: int,
    random_seed: int,
    keep_hydrogens: bool,
) -> Tuple[Mol, int]:
    """Set constrained atom coordinates and run the constrained optimization."""

    _ensure_conformer(target, random_seed=random_seed)
    conf = target.GetConformer()
    for atom_idx, point in coord_map.items():
        conf.SetAtomPosition(atom_idx, point)

    convergence, energy = optimize_with_fixed_fragments(
        target,
        fixed_atoms=coord_map.keys(),
        force_field=force_field,
        max_iterations=max_iterations,
    )

    result = target if keep_hydrogens else RemoveHs(target)
    return result, convergence, energy 


def optimize_with_fixed_fragments(
    mol: Mol,
    fixed_atoms: Iterable[int],
    force_field: str = "UFF",
    max_iterations: int = 500,
    conf_id: int = 0,
) -> int:
    """Optimize a molecule while keeping *fixed_atoms* frozen.

    Returns the RDKit force-field convergence code (0 indicates success).
    """

    force_field = force_field.upper()
    fixed_atoms = list(dict.fromkeys(int(idx) for idx in fixed_atoms))

    if force_field == "UFF":
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    elif force_field in {"MMFF", "MMFF94", "MMFF94S"}:
        variant = "MMFF94" if force_field == "MMFF" else force_field
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
        if props is None:
            raise ValueError("MMFF parameters unavailable for given molecule")
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    else:
        raise ValueError(f"Unsupported force field '{force_field}'")

    for idx in fixed_atoms:
        ff.AddFixedPoint(idx)
    ff.Initialize()
    status = ff.Minimize(maxIts=max_iterations)
    energy = ff.CalcEnergy()
    logger.info("Force-field termination code: %s", status)
    return status, energy  


def generate_constrained_conformer(
    reference: Mol,
    target: Mol,
    constraints: Sequence[FragmentConstraint],
    force_field: str = "UFF",
    max_iterations: int = 500,
    random_seed: int = 0xF00D,
    keep_hydrogens: bool = False,
) -> Mol:
    """Generate an optimized conformer for *target* with constrained fragments.

    The returned molecule is a copy of *target* containing the optimized
    coordinates (hydrogens removed by default).
    """

    if not constraints:
        raise ValueError("At least one fragment constraint must be provided")

    ref = Chem.Mol(reference)
    tgt = Chem.Mol(target)

    ref = Chem.AddHs(ref, addCoords=True)
    tgt = Chem.AddHs(tgt)

    _ensure_conformer(ref, random_seed=random_seed)

    coord_map = _build_coord_map(ref, tgt, constraints)

    tgt, convergence, energy = _place_fragments_and_optimize(
        tgt,
        coord_map,
        force_field=force_field,
        max_iterations=max_iterations,
        random_seed=random_seed,
        keep_hydrogens=keep_hydrogens,
    )
    
    if convergence != 0:
        logger.warning("Optimization did not fully converge (code %s)", convergence)

    return tgt, energy


def generate_constrained_conformers_all_matches(
    reference: Mol,
    target: Mol,
    constraints: Sequence[FragmentConstraint],
    force_field: str = "UFF",
    max_iterations: int = 500,
    random_seed: int = 0xF00D,
    keep_hydrogens: bool = False,
    max_results: Optional[int] = None,
) -> List[ConstrainedConformerResult]:
    """Enumerate all match combinations and generate constrained conformers for each.

    Returns a list of :class:`ConstrainedConformerResult` objects containing the
    optimized molecule, convergence code, and the reference/target match pairs
    that were applied.
    """

    if not constraints:
        raise ValueError("At least one fragment constraint must be provided")

    ref = Chem.Mol(reference)
    tgt_template = Chem.Mol(target)

    ref = Chem.AddHs(ref, addCoords=True)
    tgt_template = Chem.AddHs(tgt_template)

    _ensure_conformer(ref, random_seed=random_seed)
    ref_conf = ref.GetConformer()

    results: List[ConstrainedConformerResult] = []

    for combo_idx, assignments in enumerate(
        _iter_fragment_match_assignments(ref, tgt_template, constraints),
        start=1,
    ):
        if max_results is not None and combo_idx > max_results:
            logger.debug("Reached max_results=%d; stopping enumeration", max_results)
            break

        coord_map = _build_coord_map_from_assignments(ref_conf, assignments)
        tgt_candidate = Chem.Mol(tgt_template)

        combo_seed = random_seed + combo_idx
        optimized, convergence, energy = _place_fragments_and_optimize(
            tgt_candidate,
            coord_map,
            force_field=force_field,
            max_iterations=max_iterations,
            random_seed=combo_seed,
            keep_hydrogens=keep_hydrogens,
        )

        if convergence != 0:
            logger.warning(
                "Optimization did not fully converge for combination %d (code %s)",
                combo_idx,
                convergence,
            )

        summary = "; ".join(
            f"{assignment.label}[ref#{assignment.reference_match_index} -> tgt#{assignment.target_match_index}]"
            for assignment in assignments
        )
        logger.info("Generated constrained conformer #%d using matches: %s", combo_idx, summary)
        energy_per_atom=energy/optimized.GetNumAtoms() if optimized is not None else float('inf')
        results.append(
            ConstrainedConformerResult(
                molecule=optimized,
                matches=list(assignments),
                convergence_code=convergence,
                energy=energy_per_atom,
            )
        )

    if not results:
        logger.warning(
            "No constrained conformers were generated; verify the fragment definitions and matches"
        )

    return results
