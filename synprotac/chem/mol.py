"""
Simplified Molecule class for PROTAC synthesis.
"""
import dataclasses
from functools import cache, cached_property
from typing import Literal, overload, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw

from .base import Drawable


@dataclasses.dataclass(frozen=True, eq=True, unsafe_hash=True)
class FingerprintOption:
    type: str = "morgan"
    # Morgan
    morgan_radius: int = 2
    morgan_n_bits: int = 256

    def __post_init__(self):
        supported_types = ("morgan",)
        if self.type not in supported_types:
            raise ValueError(f"Unsupported fingerprint type: {self.type}")

    @classmethod
    def morgan_for_building_blocks(cls):
        return FingerprintOption(
            type="morgan",
            morgan_radius=2,
            morgan_n_bits=256,
        )

    @property
    def dim(self) -> int:
        if self.type == "morgan":
            return self.morgan_n_bits
        raise ValueError(f"Unsupported fingerprint type: {self.type}")


class Molecule(Drawable):
    def __init__(self, smiles: str) -> None:
        super().__init__()
        self._smiles = smiles.strip()

    @classmethod
    def from_rdmol(cls, rdmol: Chem.Mol) -> "Molecule":
        return cls(Chem.MolToSmiles(rdmol))

    def __getstate__(self):
        return self._smiles

    def __setstate__(self, state):
        self._smiles = state

    @property
    def smiles(self) -> str:
        return self._smiles

    @cached_property
    def _rdmol(self):
        return Chem.MolFromSmiles(self._smiles)

    @cached_property
    def is_valid(self) -> bool:
        return self._rdmol is not None

    @cached_property
    def csmiles(self) -> str:
        if self._rdmol is None:
            return self._smiles
        return Chem.MolToSmiles(self._rdmol, canonical=True, isomericSmiles=False)

    @cached_property
    def num_atoms(self) -> int:
        if self._rdmol is None:
            return 0
        return self._rdmol.GetNumAtoms()

    def draw(self, size: int = 100, svg: bool = False):
        if self._rdmol is None:
            # Return a placeholder image for invalid molecules
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (size, size), 'white')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Invalid", fill='red')
            return img
            
        if svg:
            return Draw._moltoSVG(self._rdmol, sz=(size, size), highlights=[], legend=[], kekulize=True)
        else:
            return Draw.MolToImage(self._rdmol, size=(size, size), kekulize=True)

    def __hash__(self) -> int:
        return hash(self._smiles)

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Molecule) and self.csmiles == __value.csmiles

    @overload
    def get_fingerprint(self, option: FingerprintOption) -> np.ndarray: ...

    @overload
    def get_fingerprint(self, option: FingerprintOption, as_bitvec: Literal[True]) -> Sequence[Literal[0, 1]]: ...

    @overload
    def get_fingerprint(self, option: FingerprintOption, as_bitvec: Literal[False]) -> np.ndarray: ...

    def get_fingerprint(self, option: FingerprintOption, as_bitvec: bool = False):
        return self._get_fingerprint(option, as_bitvec)

    @cache
    def _get_fingerprint(self, option: FingerprintOption, as_bitvec: bool):
        if self._rdmol is None:
            # Return zero fingerprint for invalid molecules
            if as_bitvec:
                return [0] * option.dim
            return np.zeros(option.dim, dtype=np.float32)
            
        if option.type == "morgan":
            bit_vec = AllChem.GetMorganFingerprintAsBitVect(self._rdmol, option.morgan_radius, option.morgan_n_bits)
        else:
            raise ValueError(f"Unsupported fingerprint type: {option.type}")

        if as_bitvec:
            return bit_vec
        
        feat = np.zeros(option.dim, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bit_vec, feat)
        return feat
