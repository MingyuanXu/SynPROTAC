"""
Simplified FingerprintIndex for PROTAC synthesis.
"""
import dataclasses
from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
import torch
from sklearn.neighbors import BallTree

from .mol import Molecule, FingerprintOption


@dataclasses.dataclass
class QueryResult:
    index: int
    molecule: Molecule
    fingerprint: np.ndarray
    distance: float


class FingerprintIndex:
    """Simplified fingerprint index for reagent retrieval."""
    
    def __init__(self, molecules: Iterable[Molecule], fp_option: FingerprintOption) -> None:
        super().__init__()
        self._molecules = tuple(molecules)
        self._fp_option = fp_option
        self._fp = self._init_fingerprint()
        self._tree = self._init_tree()

    @property
    def molecules(self) -> tuple[Molecule, ...]:
        return self._molecules

    @property
    def fp_option(self) -> FingerprintOption:
        return self._fp_option

    def _init_fingerprint(self) -> np.ndarray:
        """Initialize fingerprints for all molecules."""
        fps = []
        for mol in self._molecules:
            fp = mol.get_fingerprint(self._fp_option)
            fps.append(fp)
        return np.array(fps, dtype=np.float32)

    def _init_tree(self) -> BallTree:
        """Initialize BallTree for fast similarity search."""
        tree = BallTree(self._fp, metric="manhattan")
        return tree

    def __getitem__(self, index: int) -> tuple[Molecule, np.ndarray]:
        return self._molecules[index], self._fp[index]

    def query(self, q: np.ndarray, k: int) -> List[List[QueryResult]]:
        """
        Query for similar molecules.
        
        Args:
            q: Query fingerprints of shape (bsz, fp_dim)
            k: Number of top results to return
            
        Returns:
            List of lists of QueryResult objects
        """
        if q.ndim == 1:
            q = q.reshape(1, -1)
            
        bsz = q.shape[0]
        dist, idx = self._tree.query(q, k=k)
        
        results: List[List[QueryResult]] = []
        for i in range(bsz):
            res: List[QueryResult] = []
            for j in range(k):
                index = int(idx[i, j])
                res.append(
                    QueryResult(
                        index=index,
                        molecule=self._molecules[index],
                        fingerprint=self._fp[index],
                        distance=dist[i, j],
                    )
                )
            results.append(res)
        return results

    def query_cuda(self, q: torch.Tensor, k: int) -> List[List[QueryResult]]:
        """
        CUDA-accelerated query for similar molecules.
        
        Args:
            q: Query fingerprints of shape (bsz, fp_dim)
            k: Number of top results to return
            
        Returns:
            List of lists of QueryResult objects
        """
        if q.dim() == 1:
            q = q.unsqueeze(0)
            
        bsz = q.size(0)
        
        # Convert to numpy for sklearn BallTree
        q_np = q.detach().cpu().numpy()
        
        # Use regular query method
        return self.query(q_np, k)


def create_reagent_index(reagent_smiles_list: List[str], 
                        fp_option: FingerprintOption = None) -> FingerprintIndex:
    """
    Create a fingerprint index from a list of reagent SMILES.
    
    Args:
        reagent_smiles_list: List of SMILES strings
        fp_option: Fingerprint options
        
    Returns:
        FingerprintIndex object
    """
    if fp_option is None:
        fp_option = FingerprintOption.morgan_for_building_blocks()
    
    molecules = []
    for smiles in reagent_smiles_list:
        try:
            mol = Molecule(smiles)
            if mol.is_valid:
                molecules.append(mol)
        except Exception:
            # Skip invalid molecules
            continue
    
    return FingerprintIndex(molecules, fp_option)
