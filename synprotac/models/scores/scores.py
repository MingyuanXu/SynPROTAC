"""
This class is used for defining the scoring function(s) which can be used during
fine-tuning.
"""
# load general packages and functions
from collections import namedtuple
import torch,pickle 
from rdkit import DataStructs,Chem
from rdkit.Chem import QED, AllChem
import numpy as np
import sklearn,math
from sklearn import svm,metrics
from tqdm import tqdm
import random
from typing import Dict, Any, List, Optional, Union 

def sigmoid_transformation(scores,_low=0,_high=1.0,_k=0.25):
    def _exp(pred_val, low, high, k):
        try:
            return math.pow(10, (10 * k * (pred_val - (low + high) * 0.5) / (low - high)))
        except:
            return 0
    transformed = [1 / (1 + _exp(pred_val, _low, _high, _k)) for pred_val in scores]
    return np.array(transformed, dtype=np.float32)

def reverse_sigmoid_transformation(scores,_low, _high, _k):
    def _reverse_sigmoid_formula(value, low, high, k):
        try:
            return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
        except:
            return 0
    transformed = [_reverse_sigmoid_formula(pred_val, _low, _high, _k) for pred_val in scores]
    return np.array(transformed, dtype=np.float32)

def leaky_window_reverse_sigmoid_transformation(
    scores,
    _low,
    _high,
    _k=0.25,
    _outside_penalty=0.25,
    _clip_min=-1.0,
):
    """
    Single-sided leaky-linear reward.

    Piecewise definition:
    - score <= _low: reward = 1.0
    - _low < score <= _high: linearly mapped from 1.0 to 0.0
    - score > _high: linear penalty with slope controlled by _outside_penalty

    Notes:
    - _k is kept for backward compatibility and is unused in this linear variant.
    - _outside_penalty is the penalty slope (normalized by interval width).
    """
    scores_arr = np.asarray(scores, dtype=np.float32)
    width = max(float(_high - _low), 1e-6)

    transformed = np.empty_like(scores_arr, dtype=np.float32)
    low_mask = scores_arr <= _low
    window_mask = (scores_arr > _low) & (scores_arr <= _high)
    high_mask = scores_arr > _high

    transformed[low_mask] = 1.0
    transformed[window_mask] = (_high - scores_arr[window_mask]) / width
    transformed[high_mask] = -float(_outside_penalty) * (scores_arr[high_mask] - _high) / width

    if _clip_min is not None:
        transformed = np.maximum(transformed, float(_clip_min))

    return transformed.astype(np.float32)

class BaseScore:
    """Base class for molecular scoring components."""
    def __init__(self,):
        pass 
    def compute_scores(self, mols: List[Chem.Mol], subset_id: int = 0):
        """Compute scores for a list of molecules. Subclasses must implement this."""
        raise NotImplementedError

class QEDScore(BaseScore):
    """Compute QED (Quantitative Estimate of Drug-likeness) scores."""
    def compute_scores(self, mols: List[Chem.Mol], subset_id: int = 0):
        qed_scores = []
        for mol in mols:
            if mol:
                try:
                    score = QED.qed(mol)
                except:
                    score = 0.0
            else:
                score = 0.0
            qed_scores.append(score)
        return torch.tensor(qed_scores), torch.tensor(qed_scores)

class SimilarityScore(BaseScore):
    """Compute 2D similarity scores (Tanimoto similarity to target molecules)."""
    def __init__(self, target_smiles: List[str], tanimoto_k: float = 0.2, cutoff=0.75):
        self.target_mols = [Chem.MolFromSmiles(smi) for smi in target_smiles]
        self.target_fps = [AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True) for mol in self.target_mols if mol]
        self.tanimoto_k = tanimoto_k
        self.cutoff = cutoff
    
    def compute_scores(self, mols: List[Chem.Mol],subset_id: int = 0): 
        scores=[]
        similarities = []
        for mol in mols:
            if mol:
                fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
                sim_scores = [DataStructs.TanimotoSimilarity(fp, target_fp) for target_fp in self.target_fps]
                max_sim = max(sim_scores) if sim_scores else 0.0
                score = max(max_sim-self.cutoff,0) / self.tanimoto_k
            else:
                max_sim=0.0
                score = 0.0
            scores.append(score)
            similarities.append(max_sim)
        return torch.tensor(scores), torch.tensor(similarities)