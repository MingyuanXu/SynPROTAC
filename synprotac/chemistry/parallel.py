import os
import sys
import logging
from pathlib import Path

from .synthesis_interface import Synthesizable_PROTAC_Search
from .reaction_search import ChemReactionSearch, ReactionTemplate, BuildingBlock
from ..utils.rdkit import draw_molecule_with_map_atoms, load_rxn_templates, load_building_blocks

from rdkit import Chem 
from rdkit.Chem import AllChem, Draw
import random 
from tqdm import tqdm
import numpy as np
import math 
def minimize_reaction_space(warhead,e3_ligand,templates,building_blocks, protected_patts=[],select_templates_num=200):
    """
    最小化反应空间
    """
    warhead_mol = Chem.MolFromSmiles(warhead)
    e3_ligand_mol = Chem.MolFromSmiles(e3_ligand)
    searcher = ChemReactionSearch()
    searcher.load_reaction_templates(templates)
    searcher.load_building_blocks(building_blocks)
    templates_for_e3_ligand=searcher.get_suitable_rxn_templates_for_specific_molecule(e3_ligand,protected_patts=protected_patts)
    templates_for_warhead=searcher.get_suitable_rxn_templates_for_specific_molecule(warhead,protected_patts=protected_patts)
    print(f"{len(templates_for_e3_ligand)},{len(templates_for_warhead)}") 
    random.shuffle(templates_for_e3_ligand)
    random.shuffle(templates_for_warhead)
    templates_for_e3_ligand = templates_for_e3_ligand[:10]
    templates_for_warhead = templates_for_warhead[:10]
    sampled_num=select_templates_num - len(templates_for_e3_ligand) - len(templates_for_warhead)
    random_templates = random.sample(searcher.reaction_templates, sampled_num) 
    filtered_templates = templates_for_e3_ligand + templates_for_warhead+random_templates
    filtered_templates = [tp.to_dict() for tp in filtered_templates]

    return filtered_templates

def Mini_Synthesizable_PROTAC_Search(warhead, e3_ligand, templates, building_blocks, 
                                    protected_patts=[], 
                                    max_depth=2,
                                    max_iterations=1000,
                                    protac_target=None,
                                    mini_template_pool_size=50,
                                    mini_building_block_size=2000,
                                    savepath="results.pkl",
                                    num_paths=10,
                                    tqdm_position=0):
    log_file = f"{str(savepath)}.log"
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
        
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(process)d %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"子进程日志启动: {log_file}")
    filtered_templates=minimize_reaction_space(warhead, e3_ligand, templates, building_blocks, 
                                           protected_patts=protected_patts,
                                           select_templates_num=mini_template_pool_size)

    filtered_building_blocks = random.sample(building_blocks, mini_building_block_size)  # 随机选择500个构建块
    random.shuffle(filtered_building_blocks)
    
    Synthesizable_PROTAC_Search(warhead=warhead,
                        e3_ligand=e3_ligand, 
                        protected_patts=protected_patts,
                        rxn_templates=filtered_templates,
                        building_blocks=filtered_building_blocks,
                        max_depth=max_depth,
                        max_iterations=max_iterations,
                        protac_target=protac_target, 
                        savepath=savepath,
                        num_paths=num_paths,
                        tqdm_position=tqdm_position,
                        )
    
    return 


def Parallel_Synthesizable_PROTAC_Search(warhead, e3_ligand, templates, building_blocks, 
                                    protected_patts=[], 
                                    max_depth=2,
                                    max_iterations=1000,
                                    protac_target=None,
                                    mini_template_pool_size=50,
                                    mini_building_block_size=2000,
                                    savepath="Synthesizable_PROTACs",
                                    num_paths_per_proc=100,
                                    njobs=14,
                                    nprocs=14):

    from multiprocessing import Pool,Queue,Manager,Process
    manager=Manager()
    DQueue=manager.Queue()

    if not os.path.exists(savepath):
        os.system(f'mkdir -p {savepath}')

    p=Pool(nprocs)
    resultlist=[]
    for i in range(njobs):
        result=p.apply_async(Mini_Synthesizable_PROTAC_Search,
                                (
                                    warhead,
                                    e3_ligand,
                                    templates,
                                    building_blocks,
                                    protected_patts,
                                    max_depth,
                                    max_iterations,
                                    protac_target,
                                    mini_template_pool_size,
                                    mini_building_block_size,
                                    savepath/f'{i}.pkl',
                                    num_paths_per_proc,
                                    i
                                )
                            )

        resultlist.append(result)

    for i in range(len(resultlist)):
        tmp=resultlist[i].get()

    p.terminate()
    p.join()
    return
