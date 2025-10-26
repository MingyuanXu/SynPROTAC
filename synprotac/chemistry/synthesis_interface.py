"""
Synthesis Interface Module

Main API interface for PROTAC synthesis planning using MCTS.
This module ties together all the other components to provide a simple interface.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import time
import os 
import pickle 


from .mcts_planner import MCTSPlanner
from .reaction_search import ChemReactionSearch, create_reaction_searcher


logger = logging.getLogger(__name__)

class SynthesisInterface:
    """
    PROTAC合成规划接口
    
    这是主要的API接口，整合了MCTS规划、反应搜索和可视化功能。
    """
    
    def __init__(self, 
                 max_depth: int = 5,
                 max_iterations: int = 1000,
                 exploration_weight: float = 1.4,
                 max_reactions_per_node: int = 50,
                 simulation_steps: int = 1):
        """
        初始化合成接口
        
        Args:
            max_depth: MCTS最大搜索深度
            max_iterations: MCTS最大迭代次数
            exploration_weight: MCTS探索权重
        """
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.max_reactions_per_node = max_reactions_per_node
        self.simulation_steps = simulation_steps

        # 初始化组件
        self.mcts_planner = MCTSPlanner(
            max_depth=max_depth,
            max_iterations=max_iterations,
            exploration_weight=exploration_weight,
            simulation_steps=self.simulation_steps,
        )
        
        self.reaction_searcher = ChemReactionSearch()

        
        # 保护机制相关
        self.protector = None
        
        # 起始分子集合（用于目标检查）
        self.target_molecules: set = set()
        
        # 搜索统计
        self.search_history = []
    
    def load_data(self, 
                  reaction_templates: List[Dict],
                  building_blocks: List[Dict],
                  target_molecules: List[str],
                  rxn_bb_map_file="rxn_bb_map.pkl") -> Dict[str, int]:
        """
        加载搜索所需的数据
        
        Args:
            reaction_templates: 反应模板列表
            building_blocks: 构建块列表  
            start_molecules: 起始分子SMILES列表
            connect_molecules: 连接分子SMILES列表
            target_molecules: 目标分子SMILES列表
            
        Returns:
            加载统计信息
        """
        logger.info("开始加载合成规划数据...")
        
        # 加载反应模板和构建块
        num_templates = self.reaction_searcher.load_reaction_templates(reaction_templates)
        num_blocks = self.reaction_searcher.load_building_blocks(building_blocks)

        #if os.path.exists(rxn_bb_map_file):
        #    self.reaction_searcher.load_rxn_bb_map(rxn_bb_map_file)
        #else:
        #   self.reaction_searcher.build_rxn_to_building_block_map()
            #self.reaction_searcher.save_rxn_bb_map(rxn_bb_map_file)
        self.reaction_searcher.build_rxn_to_building_block_map()
        
        stats = {
            'reaction_templates': num_templates,
            'building_blocks': num_blocks,
        }
        
        logger.info(f"数据加载完成: {stats}")
        return stats
    
    def find_synthesis_paths(self, 
                           warhead: str,
                           e3_ligand: str,
                           protac_target: str = None,
                           num_paths: int = 3,
                           protected_patts: List[str] = [],
                           savepath = "synthesis_tree.pkl",
                           tqdm_position=0) -> List[Dict]:
        
        """
        简化的PROTAC合成路径搜索
        
        Args:
            warhead: warhead分子SMILES (起始分子)
            e3_ligand: E3配体分子SMILES (需要连接的分子)
            protac_target: 目标PROTAC分子SMILES (可选验证)
            num_paths: 要找的路径数量
            
        Returns:
            找到的合成路径列表
        """
        logger.info(f"开始PROTAC合成路径搜索")
        logger.info(f"  Warhead: {warhead}")
        logger.info(f"  E3 Ligand: {e3_ligand}")
        if protac_target:
            logger.info(f"  目标PROTAC: {protac_target}")
        
        start_time = time.time()
        
        # 根据是否包含占位符选择合适的动作生成器
        action_generator = self._create_action_generator(protected_patts)
        e3_connector = self._create_e3_connector(e3_ligand, protected_patts)
        
        # 执行MCTS搜索 (使用新的接口)
        paths = self.mcts_planner.search(
            warhead=warhead,
            e3_ligand=e3_ligand,
            action_generator=action_generator,
            e3_connector=e3_connector,
            state_evaluator=None,
            num_paths=num_paths,
            tqdm_position=tqdm_position
        )
        path_datas={"warhead": warhead, "e3_ligand": e3_ligand, "protac_target": protac_target, "protected_patts": protected_patts} 
        path_datas["paths"] = paths
        with open(savepath,"wb") as f:
            pickle.dump(path_datas, f)
        logger.info(f"搜索完成: 找到{len(paths)}条PROTAC合成路径，用时{time.time() - start_time:.2f}秒")
        return 
    
    def _create_action_generator(self, protected_patts: List[str] = []) -> Any:
        """创建目标导向的动作生成器 - 符合新的MCTS接口"""
        def action_generator(current_state: str) -> List[Tuple]:
            """
            目标导向的动作生成器
            
            Args:
                current_state: 当前分子状态
                target_state: 目标E3 ligand状态
                
            Returns:
                List[Tuple[action_info, new_state]]
            """
            actions = []
            # 只获取常规构建块反应，不包括E3连接
            reactions = self.reaction_searcher.get_applicable_reactions(current_state, protected_patts=protected_patts)
            
            for reaction_info, product_smiles in reactions:  # 限制数量
                # 构建动作信息
                action_info = {
                    'reaction': reaction_info.get('reaction_name', 'unknown'),
                    'reagent': reaction_info.get('reagent', ''),
                    'template': reaction_info.get('reaction', ''),
                    'reaction_type': 'building_block'
                }
                
                if product_smiles and len(product_smiles) > 5:
                    actions.append((action_info, product_smiles))
            
            return actions
        
        return action_generator

    
    def _create_e3_connector(self, e3_ligand_smiles: str, protected_patts: List[str] = []) -> Any:
        """创建E3连接器函数"""
        specific_templates=self.reaction_searcher.get_suitable_rxn_templates_for_specific_molecule(e3_ligand_smiles, protected_patts)
        if len(specific_templates)==0:
            raise ValueError("没有找到适用于E3 ligand的反应模板, 请增加反应模板、修改保护模式或更宽松的反应位置定义")
        
        def e3_connector(current_smiles: str) -> List[Tuple]:
            """
            获取当前分子与E3 ligand的连接反应
            
            Args:
                current_smiles: 当前分子SMILES
                e3_ligand_smiles: E3配体分子SMILES
                
            Returns:
                List[Tuple[reaction_info, product_smiles]]
            """
            # 使用reaction_searcher的连接反应方法
            connection_reactions = self.reaction_searcher.get_connection_reactions(
                current_smiles, e3_ligand_smiles, protected_patts=protected_patts, specific_templates=specific_templates
            )
            
            # 为E3连接反应添加特殊标识
            e3_connection_reactions = []
            for reaction_info, product_smiles in connection_reactions:
                # 更新反应信息，标记为E3连接
                e3_reaction_info = reaction_info.copy()
                e3_reaction_info['reaction_name'] = f"{reaction_info['reaction_name']}_E3_Connection"
                e3_reaction_info['reaction_type'] = 'e3_connection'
                e3_reaction_info['reagent_id'] = 'e3_ligand'
            
                e3_connection_reactions.append((e3_reaction_info, product_smiles))
            
            return e3_connection_reactions
        
        return e3_connector
    
    def _create_state_evaluator(self):
        """创建PROTAC专用状态评估器函数"""
        def state_evaluator(molecule_smiles: str, target_smiles: str, depth: int) -> float:
            """
            评估分子状态的价值，考虑与E3 ligand的连接潜力
            
            Args:
                molecule_smiles: 当前分子SMILES字符串
                target_smiles: 目标分子SMILES（E3 ligand）
                depth: 当前深度
                
            Returns:
                状态评估分数
            """
            # 基础合成可达性
            accessibility = self.reaction_searcher.evaluate_synthetic_accessibility(molecule_smiles)
            
            # 深度惩罚（鼓励更短的路径）
            depth_penalty = 0.2 * depth
            
            # 连接潜力奖励（具有双官能团的分子得分更高）
            connection_potential = self._evaluate_connection_potential(molecule_smiles)
            
            # PROTAC分子量合理性
            molecular_weight_bonus = self._evaluate_protac_molecular_weight(molecule_smiles)
            
            # 综合分数
            final_score = (accessibility * 0.3 + 
                         connection_potential * 0.5 + 
                         molecular_weight_bonus * 0.2 - 
                         depth_penalty)
            
            return max(0.0, min(1.0, final_score))
        
        return state_evaluator
    
    
    def _evaluate_connection_potential(self, molecule_smiles: str) -> float:
        """评估分子的连接潜力"""
        # 检查分子中的可反应官能团数量
        bb=BuildingBlock(molecule_smiles)
        reactive_groups = bb.reactive_group_num
        
        # 双官能团分子得分更高（适合做linker）
        if reactive_groups >= 2:
            return 0.8
        elif reactive_groups == 1:
            return 0.5
        else:
            return 0.1
    
    def _evaluate_protac_molecular_weight(self, molecule_smiles: str) -> float:
        """评估PROTAC分子量的合理性"""
        mw = self.reaction_searcher.calculate_molecular_weight(molecule_smiles)
        
        # PROTAC理想分子量范围 500-1500 Da
        if 500 <= mw <= 1500:
            return 1.0
        elif 400 <= mw <= 2000:
            return 0.7
        else:
            return 0.3
    
    def get_search_statistics(self) -> Dict:
        """获取搜索统计信息"""
        mcts_stats = self.mcts_planner.get_search_statistics()
        reaction_stats = self.reaction_searcher.get_statistics()
        
        interface_stats = {
            'total_searches': len(self.search_history),
            'average_search_time': (
                sum(s['search_time'] for s in self.search_history) / 
                len(self.search_history) if self.search_history else 0
            ),
            'total_paths_found': sum(s['num_paths_found'] for s in self.search_history)
        }
        
        return {
            'interface': interface_stats,
            'mcts': mcts_stats,
            'reactions': reaction_stats
        }


# 便捷函数
def create_synthesis_planner(reaction_templates: List[Dict],
                           building_blocks: List[Dict],
                           target_molecules: List[str],
                           max_depth: int = 5,
                           max_iterations: int = 1000) -> SynthesisInterface:
    """
    创建配置好的合成规划器
    
    Args:
        reaction_templates: 反应模板数据
        building_blocks: 构建块数据
        target_molecules: 目标分子列表
        max_depth: 最大搜索深度
        max_iterations: 最大迭代次数
        
    Returns:
        配置好的SynthesisInterface实例
    """
    planner = SynthesisInterface(
        max_depth=max_depth,
        max_iterations=max_iterations
    )
    
    planner.load_data(
        reaction_templates=reaction_templates,
        building_blocks=building_blocks,
        target_molecules=target_molecules
    )
    
    return planner


def Synthesizable_PROTAC_Search(warhead: str,
                            e3_ligand: str,
                            rxn_templates: List[Dict],
                            building_blocks: List[Dict],
                            protected_patts: List[str] = [],
                            max_depth: int = 5,
                            max_iterations: int = 1000,
                            protac_target: Optional[str] = None,   
                            savepath="Protac_Synthesis_Tree",
                            num_paths: int = 3,
                            tqdm_position=0) -> List[Dict]:

    """
    快速合成路径搜索
    
    Args:
        start_molecule: 起始分子SMILES
        target_molecules: 目标分子列表
        reaction_templates: 反应模板数据
        building_blocks: 构建块数据
        num_paths: 路径数量
        
    Returns:
        找到的路径列表
    """

    planner = create_synthesis_planner(
        reaction_templates=rxn_templates,
        building_blocks=building_blocks,
        target_molecules=[protac_target],
        max_depth=max_depth,
        max_iterations=max_iterations
    )

    planner.find_synthesis_paths(
        warhead=warhead,
        e3_ligand=e3_ligand,
        protac_target=protac_target,
        num_paths=num_paths,
        protected_patts=protected_patts,
        savepath=savepath,
        tqdm_position=tqdm_position
    )
    return 
