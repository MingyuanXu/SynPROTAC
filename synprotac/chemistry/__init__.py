"""
PROTAC Linker Synthesis Package

A Monte Carlo Tree Search (MCTS) based package for PROTAC linker synthesis path planning.

Core components:
- MCTSPlanner: Main MCTS algorithm for synthesis path search
- ChemReactionSearch: Chemical reaction matching and processing
- PathVisualizer: Synthesis path visualization
- SynthesisInterface: Main API interface
"""

from .mcts_planner import MCTSPlanner, run_protac_mcts_search
from .reaction_search import ChemReactionSearch, ReactionTemplate, BuildingBlock, create_reaction_searcher  
from .path_visualizer import ReactionPathVisualizer, visualize_path
from .synthesis_interface import SynthesisInterface, Synthesizable_PROTAC_Search 
from .parallel import Mini_Synthesizable_PROTAC_Search, Parallel_Synthesizable_PROTAC_Search

__all__ = [
    # Core MCTS
    'MCTSPlanner',
    'run_protac_mcts_search',
    
    # Reaction Search
    'ChemReactionSearch', 
    'ReactionTemplate',
    'BuildingBlock',
    'create_reaction_searcher',
    
    # Visualization
    'ReactionPathVisualizer',
    'visualize_path', 
    'create_molecule_image',

    # Main Interface
    'SynthesisInterface',
    "Synthesizable_PROTAC_Search",  # PROTAC synthesis search interface
    "Mini_Synthesizable_PROTAC_Search",  # Mini PROTAC synthesis search for parallel processing
    "Parallel_Synthesizable_PROTAC_Search",  # Parallel PROTAC synthesis search for large datasets
]
