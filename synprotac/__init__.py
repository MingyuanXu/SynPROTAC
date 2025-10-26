"""
PROTAC Synthesis Planning Package

A Python package for PROTAC (PROteolysis TArgeting Chimera) linker synthesis 
planning using Monte Carlo Tree Search (MCTS) algorithm.
"""

__version__ = "1.0.0"
__author__ = "PROTAC Research Team"

# Import the main chemistry module
from . import chemistry
from . import utils
from . import retrosynthesis


__all__ = ['chemistry','utils','retrosynthesis']
