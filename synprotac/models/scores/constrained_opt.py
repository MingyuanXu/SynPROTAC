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

import numpy as np

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
    max_target_matches_per_constraint: Optional[int] = None,
) -> Iterator[List[FragmentMatchAssignment]]:
    """Yield combinations of fragment assignments with fixed reference match and full target matches.

    Design choice:
    - Reference side uses a single deterministic match (constraint.match_index),
      avoiding combinatorial blow-up from symmetric matches in the reference ligand.
        - Target side can keep all matches, or a diverse top-k subset to balance
            coverage and speed.
    """

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

        if constraint.match_index >= len(ref_matches):
            raise IndexError(
                f"match_index {constraint.match_index} out of range for {constraint.label}; "
                f"only {len(ref_matches)} reference match(es) available"
            )

        selected_ref_idx = constraint.match_index
        selected_ref_match = ref_matches[selected_ref_idx]

        selected_target_matches = _select_diverse_target_matches(
            query=query,
            target=target,
            target_matches=tgt_matches,
            max_keep=max_target_matches_per_constraint,
        )

        assignments: List[FragmentMatchAssignment] = []
        for tgt_idx, tgt_match in selected_target_matches:
            if len(selected_ref_match) != len(tgt_match):
                continue
            assignments.append(
                FragmentMatchAssignment(
                    label=constraint.label,
                    smiles=constraint.smiles,
                    reference_match=selected_ref_match,
                    target_match=tgt_match,
                    reference_match_index=selected_ref_idx,
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


def _select_diverse_target_matches(
    query: Mol,
    target: Mol,
    target_matches: Sequence[Tuple[int, ...]],
    max_keep: Optional[int],
) -> List[Tuple[int, Tuple[int, ...]]]:
    """Keep up to ``max_keep`` target matches that are maximally diverse.

    Diversity is computed from a geometric descriptor focused on connection
    behavior:
    - anchor atom positions (priority: mapped atoms in query)
    - vector from anchor centroid to match centroid (captures linker direction)
    """

    indexed = [(idx, match) for idx, match in enumerate(target_matches)]
    if max_keep is None or max_keep <= 0 or len(indexed) <= max_keep:
        return indexed

    # Deduplicate exact same target atom assignment first.
    unique_indexed: List[Tuple[int, Tuple[int, ...]]] = []
    seen = set()
    for idx, match in indexed:
        if match in seen:
            continue
        seen.add(match)
        unique_indexed.append((idx, match))

    if len(unique_indexed) <= max_keep:
        return unique_indexed

    _ensure_conformer(target)
    conf = target.GetConformer()

    query_anchor_positions = [atom.GetIdx() for atom in query.GetAtoms() if atom.GetAtomMapNum() > 0]
    if not query_anchor_positions:
        query_anchor_positions = [0]

    feature_vectors = []
    for _, match in unique_indexed:
        coords = []
        for atom_idx in match:
            p = conf.GetAtomPosition(atom_idx)
            coords.append(np.array([p.x, p.y, p.z], dtype=float))

        match_centroid = np.mean(coords, axis=0)

        anchor_coords = []
        for query_idx in query_anchor_positions:
            if query_idx >= len(match):
                continue
            tgt_idx = match[query_idx]
            p = conf.GetAtomPosition(tgt_idx)
            anchor_coords.append(np.array([p.x, p.y, p.z], dtype=float))

        if not anchor_coords:
            anchor_coords = [match_centroid]
        anchor_centroid = np.mean(anchor_coords, axis=0)

        direction = match_centroid - anchor_centroid
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-8:
            direction = direction / direction_norm
        else:
            direction = np.zeros(3, dtype=float)

        # Flatten feature: anchor centroid + linker direction + match centroid.
        feature = np.concatenate([anchor_centroid, direction, match_centroid])
        feature_vectors.append(feature)

    feature_vectors = np.asarray(feature_vectors)

    # Farthest-point sampling for diversity.
    selected = [0]
    min_dists = np.linalg.norm(feature_vectors - feature_vectors[0], axis=1)
    while len(selected) < max_keep:
        next_idx = int(np.argmax(min_dists))
        if next_idx in selected:
            break
        selected.append(next_idx)
        new_dists = np.linalg.norm(feature_vectors - feature_vectors[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return [unique_indexed[i] for i in selected]


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
    max_target_matches_per_constraint: Optional[int] = None,
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
    _ensure_conformer(tgt_template, random_seed=random_seed + 17)
    ref_conf = ref.GetConformer()

    results: List[ConstrainedConformerResult] = []

    for combo_idx, assignments in enumerate(
        _iter_fragment_match_assignments(
            ref,
            tgt_template,
            constraints,
            max_target_matches_per_constraint=max_target_matches_per_constraint,
        ),
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
