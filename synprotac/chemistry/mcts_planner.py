"""
MCTS Planner Module

Implementation of Monte Carlo Tree Search algorithm for PROTAC linker synthesis.
This module focuses on finding connection paths from warhead molecules to E3 ligand molecules.

Key Features:
- Target-directed search: from warhead (A) to E3 ligand (B)
- Connection-aware rewards: prioritizes paths that can connect to target
- Enhanced UCB scoring: bonus for nodes closer to target connection
- PROTAC-specific path extraction: includes final molecule formation

Algorithm Flow:
1. Start from warhead molecule (state A)
2. Use MCTS to explore intermediate states through chemical reactions
3. Reward paths that can connect to E3 ligand (state B)
4. Return complete PROTAC formation paths: A -> linkers -> B
"""

import math
import random
import logging
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from tqdm import tqdm 

logger = logging.getLogger(__name__)

@dataclass
class MCTSNode:
    """MCTS树节点"""
    
    def __init__(self, state: str, parent: Optional['MCTSNode'] = None, action: Dict = None):
        self.state = state  # 分子SMILES
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.depth = parent.depth + 1 if parent else 0
        self.action = action  # 产生此节点的动作
        
        # 扩展状态跟踪
        self.is_fully_expanded = False  # 是否已完全扩展
        self.expansion_attempts = 0     # 扩展尝试次数
        self.available_actions = None   # 缓存的可用动作
        
        # 反应信息 (向后兼容)
        self.reaction_used = action.get('reaction') if action else None
        self.reagent_used = action.get('reagent') if action else None
    
    def add_child(self, child_node: 'MCTSNode'):
        """添加子节点"""
        self.children.append(child_node)
    
    @property
    def average_reward(self) -> float:
        """平均奖励"""
        return self.total_reward / self.visits if self.visits > 0 else 0.0
    
    @property
    def is_leaf(self) -> bool:
        """是否为叶节点"""
        return len(self.children) == 0
    
    @property
    def is_expandable(self) -> bool:
        """是否可以扩展（未完全扩展且未达到最大深度）"""
        return not self.is_fully_expanded and self.depth < 100  # 使用较大值，实际由max_depth控制
    
    def mark_fully_expanded(self):
        """标记节点为完全扩展"""
        self.is_fully_expanded = True
    
    def ucb_score(self, exploration_weight: float = 1.4) -> float:
        """UCB (Upper Confidence Bound) 分数"""
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0:
            return self.average_reward
        
        exploitation = self.average_reward
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        
        return exploitation + exploration
        
    @property
    def path_length(self) -> int:
        """计算从根到当前节点的路径长度"""
        length = 0
        node = self
        while node.parent is not None:
            length += 1
            node = node.parent
        return length


class MCTSPlanner:
    """
    Monte Carlo Tree Search规划器 - 专用于PROTAC合成路径搜索
    
    实现完整的MCTS算法：Selection → Expansion → Simulation → Backpropagation
    目标：从warhead分子开始，找到连接到E3 ligand的合成路径
    """
    
    def __init__(self, 
                 max_depth: int = 5,
                 max_iterations: int = 1000,
                 exploration_weight: float = 1.4,
                 simulation_steps=1):
        """
        初始化MCTS规划器
        
        Args:
            max_depth: 最大搜索深度
            max_iterations: 最大迭代次数  
            exploration_weight: UCB探索权重
        """
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.simulation_steps = simulation_steps
        
        # 统计信息
        self.stats = {
            'total_iterations': 0,
            'successful_expansions': 0,
            'failed_expansions': 0,
            'target_connections_found': 0,
            'backtrack_count': 0,        # 回溯次数
            'fully_expanded_nodes': 0,   # 完全扩展的节点数
            'max_tree_depth': 0          # 树的最大深度
        }
    
    def search(self, 
               warhead: str,
               e3_ligand: str,
               action_generator: Callable,
               e3_connector: Callable = None,
               state_evaluator: Callable = None,
               num_paths: int = 3,
               target_molecule:str = None,
               tqdm_position=0) -> List[Dict]:
        """
        MCTS搜索：从warhead开始找到连接E3 ligand的路径
        
        Args:
            warhead: 起始warhead分子SMILES
            e3_ligand: 最终E3 ligand分子SMILES  
            action_generator: 动作生成器 f(current_state) -> List[(action_info, new_state)]
            connection_checker: 连接检查器 f(current_state) -> bool
            e3_connector: E3连接器 f(current_state, e3_ligand) -> List[(reaction_info, product_smiles)]
            state_evaluator: 状态评估器 f(state, target, depth) -> float
            num_paths: 要找的路径数量
            target_molecule: 目标分子SMILES（用于评估）
            
        Returns:
            找到的PROTAC合成路径列表
        """
        logger.info(f"开始MCTS搜索PROTAC合成路径")
        logger.info(f"  Warhead: {warhead}")
        logger.info(f"  E3 ligand: {e3_ligand}")
        
        # 使用默认评估器如果未提供
        if state_evaluator is None:
            state_evaluator = self._default_state_evaluator
            
        # 重置统计
        self._reset_stats()
        
        # 创建根节点
        root = MCTSNode(state=warhead)
        found_paths = []
        found_product = []
        iteration_bar=tqdm(range(self.max_iterations), position=tqdm_position, desc=f"Iterations")

        # MCTS主循环
        for iteration in iteration_bar:
            self.stats['total_iterations'] = iteration + 1
            
            # 1. Selection: 选择最有希望的节点
            node = self._select(root, e3_connector)
            # 2. Expansion: 扩展选中的节点
            if node.depth < self.max_depth:
                new_node = self._expand(node,  action_generator)
                if new_node:
                    self.stats['successful_expansions'] += 1

                    # 3. Simulation: 评估新节点的价值
                    reward = self._simulate(new_node, target_molecule, action_generator, 
                                          state_evaluator, e3_connector)
                    
                    # 4. Backpropagation: 反向传播奖励
                    self._backpropagate(new_node, reward)
                    
                    # 检查是否找到目标连接
                    connection_reactions = e3_connector(new_node.state)

                    if len(connection_reactions) > 0:
                        path = self._extract_path(new_node,connection_reactions)
                        product=path.get('final_product', None)
                        if self._is_unique_path(path, found_paths) and (product not in found_product):
                            found_paths.append(path)
                            found_product.append(product)
                            self.stats['target_connections_found'] += 1
                            logger.info(f"找到PROTAC路径 #{len(found_paths)}, 深度: {new_node.depth}")
                            # 找到足够路径就停止
                            if len(found_paths) >= num_paths:
                                break
                else:
                    self.stats['failed_expansions'] += 1

            iteration_bar.update(1)
            iteration_bar.set_description(f"Iterations: {iteration}, Found Paths: {len(found_paths)}")
            iteration_bar.refresh()
        
        logger.info(f"MCTS搜索完成: 找到{len(found_paths)}条路径，迭代{self.stats['total_iterations']}次,{self.stats['failed_expansions']}次扩展失败")
        return found_paths[:num_paths]

    def _select(self, root: MCTSNode, e3_connector: Callable) -> MCTSNode:
        """
        Selection阶段: 智能选择最有希望的可扩展节点
        支持回溯以探索不同分支，提高搜索多样性
        """
        current = root
        
        # 首先尝试常规的向下选择
        path = [current]
        
        while not current.is_leaf and current.depth < self.max_depth:
            if not current.children:
                break
            
            # 检查是否有可扩展的子节点
            expandable_children = [child for child in current.children if child.is_expandable]
            
            if not expandable_children:
                # 所有子节点都已完全扩展，标记当前节点
                current.mark_fully_expanded()
                self.stats['fully_expanded_nodes'] += 1
                break
            
            # 计算增强的UCB分数
            def enhanced_ucb(node):
                base_ucb = node.ucb_score(self.exploration_weight)
                
                # 为可扩展节点增加奖励
                expandability_bonus = 5.0 if node.is_expandable else 0.0
                
                # 为能连接到目标的节点增加奖励
                connection_bonus = 0.0
                connection_reactions = e3_connector(node.state)
                if len(connection_reactions) > 0:
                    connection_bonus = 10.0  # 连接奖励
                
                # 深度多样性奖励：适度奖励较浅的节点以增加多样性
                depth_diversity_bonus = max(0, (self.max_depth - node.depth) * 0.5)
                
                return base_ucb + expandability_bonus + connection_bonus + depth_diversity_bonus
            
            # 选择最佳子节点
            current = max(current.children, key=enhanced_ucb)
            path.append(current)
        
        # 如果当前节点不可扩展或达到最大深度，尝试回溯找到可扩展的节点
        if not current.is_expandable or current.depth >= self.max_depth:
            # 从路径中回溯找到第一个可扩展的节点
            for node in reversed(path[:-1]):  # 排除当前节点
                if node.is_expandable and node.depth < self.max_depth:
                    self.stats['backtrack_count'] += 1
                    return node
            
            # 如果路径上没有可扩展节点，从根开始广度优先搜索
            return self._find_best_expandable_node(root, e3_connector)
        
        # 更新树的最大深度统计
        self.stats['max_tree_depth'] = max(self.stats['max_tree_depth'], current.depth)
        
        return current
    
    def _find_best_expandable_node(self, root: MCTSNode, e3_connector: Callable) -> MCTSNode:
        """
        从整个树中找到最佳的可扩展节点（当常规选择失败时使用）
        使用广度优先搜索确保探索多样性
        """
        from collections import deque
        
        queue = deque([root])
        expandable_nodes = []
        
        # 广度优先搜索收集所有可扩展节点
        while queue:
            node = queue.popleft()
            
            if node.is_expandable and node.depth < self.max_depth:
                expandable_nodes.append(node)
            
            # 添加子节点到队列
            queue.extend(node.children)
        
        if not expandable_nodes:
            # 没有可扩展节点，返回根节点
            self.stats['backtrack_count'] += 1
            return root
        
        # 从可扩展节点中选择最佳的一个
        def node_priority(node):
            # 基础UCB分数
            base_score = node.ucb_score(self.exploration_weight)
            
            # 连接潜力奖励
            connection_bonus = 0.0
            connect_reactions=e3_connector(node.state)
            if len(connect_reactions)>0:
                connection_bonus = 15.0
            
            # 访问次数多样性：奖励访问较少的节点
            visit_diversity = max(0, 20 - node.visits) * 0.5
            
            # 深度多样性：适度奖励不同深度的节点
            depth_bonus = (self.max_depth - node.depth) * 0.3
            
            return base_score + connection_bonus + visit_diversity + depth_bonus
        
        best_node = max(expandable_nodes, key=node_priority)
        self.stats['backtrack_count'] += 1
        return best_node
    
    def _expand(self, node: MCTSNode, action_generator: Callable) -> Optional[MCTSNode]:
        """
        Expansion阶段: 智能扩展节点，支持部分扩展和多样性
        支持传递深度参数给动作生成器以实现映射号控制
        """
        if node.depth >= self.max_depth:
            node.mark_fully_expanded()
            return None
        

        # 获取或缓存可能的动作，传递深度参数支持映射号控制
        if node.available_actions is None:
            # 尝试传递深度参数（新的映射号控制动作生成器）
            node.available_actions = action_generator(node.state)
            
            if not node.available_actions:
                node.mark_fully_expanded()
                return None
        
        # 找到尚未扩展的动作
        existing_states = {child.state for child in node.children}
        unexplored_actions = []
        
        for action_info, new_state in node.available_actions:
            if new_state not in existing_states:
                unexplored_actions.append((action_info, new_state))
        
        if not unexplored_actions:
            # 所有动作都已扩展
            node.mark_fully_expanded()
            return None
        
        # 选择一个未探索的动作（可以是随机或智能选择）
        action_info, new_state = random.choice(unexplored_actions)
        
        # 检查是否已经存在相同状态的子节点（额外安全检查）
        for child in node.children:
            if child.state == new_state:
                # 如果只剩这一个动作，标记为完全扩展
                if len(unexplored_actions) == 1:
                    node.mark_fully_expanded()
                return None
        
        # 创建新子节点
        new_node = MCTSNode(
            state=new_state,
            parent=node,
            action={
                'reaction': action_info.get('reaction', 'unknown'),
                'reagent': action_info.get('reagent', ''),
                'product': new_state,
                'depth': node.depth
            }
        )
        
        node.add_child(new_node)
        node.expansion_attempts += 1
        
        # 如果这是最后一个未探索的动作，标记节点为完全扩展
        if len(unexplored_actions) == 1:
            node.mark_fully_expanded()
        
        return new_node
    
    def _simulate(self, node: MCTSNode, target_state: str, action_generator: Callable, 
                  state_evaluator: Callable, e3_connector: Callable) -> float:
        """
        Simulation阶段: 从当前节点模拟到终止状态
        评估达到目标连接的可能性
        """
        current_state = node.state
        current_depth = node.depth
        
        # 如果已经能连接到目标，给高奖励
        connection_reactions=e3_connector(current_state)
        if len(connection_reactions) > 0:
            base_reward = state_evaluator(current_state, target_state,current_depth)
            connection_bonus = 50.0  # 成功连接奖励
            depth_penalty = current_depth * 2.0  # 鼓励短路径
            return base_reward + connection_bonus - depth_penalty
        
        # 进行有限步数的随机模拟
        simulation_steps = 0
        max_sim_steps = min(self.simulation_steps, self.max_depth - current_depth)
        best_reward = state_evaluator(current_state, target_state, current_depth)
        
        while simulation_steps < max_sim_steps and current_depth < self.max_depth:
            try:
                # 获取下一步可能的动作
                actions = action_generator(current_state)
                if not actions:
                    break
                
                # 随机选择动作
                _, new_state = random.choice(actions)
                current_state = new_state
                current_depth += 1
                simulation_steps += 1
                
                # 评估新状态
                reward = state_evaluator(current_state, target_state, current_depth)

                # 检查是否达到目标连接
                connection_reactions = e3_connector(current_state)
                if len(connection_reactions) > 0:
                    connection_bonus = 50.0
                    depth_penalty = current_depth * 2.0
                    return reward + connection_bonus - depth_penalty
                
                
                best_reward = max(best_reward, reward)
                
            except Exception as e:
                logger.debug(f"模拟步骤失败: {str(e)}")
                break
        
        # 返回最佳奖励减去深度惩罚
        return best_reward - current_depth * 1.0
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation阶段: 沿路径反向传播奖励
        """

        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def _extract_path(self, node: MCTSNode, connection_reactions: List[Dict]) -> Dict:
        """
        提取从根节点到当前节点的完整路径，并添加E3连接步骤
        """
        path_nodes = []
        current = node
        
        # 收集路径上的所有节点
        while current is not None:
            path_nodes.append(current)
            current = current.parent
            
        path_nodes.reverse()  # 转换为正向路径
        
        # 构建反应序列
        reactions = []
        for i in range(1, len(path_nodes)):
            node = path_nodes[i]
            if node.action:
                reactions.append({
                    'step': i,
                    'reaction': node.action.get('reaction', 'unknown'),
                    'reagent': node.action.get('reagent', ''),
                    'product': node.state,
                    'from_state': path_nodes[i-1].state,
                    'reaction_type': 'building_block'
                })
        
        # 获取最终中间产物
        final_intermediate = path_nodes[-1].state
        
        # 添加E3连接步骤
        final_product = final_intermediate  # 默认值
        
        reaction_info, product_smiles = connection_reactions[0]

        # 添加E3连接步骤
        e3_step = {
            'step': len(reactions) + 1,
            'reaction': reaction_info.get('reaction_name'),
            'reagent': reaction_info.get('reagent'),
            'product': product_smiles,
            'from_state': final_intermediate,
            'reaction_type': 'e3_connection'
        }
        reactions.append(e3_step)
        final_product = product_smiles
        
        return {
            'warhead': path_nodes[0].state,
            'intermediate_product': final_intermediate,
            'final_product': final_product,
            'e3_ligand': reaction_info.get('reagent', 'E3_ligand'),
            'reactions': reactions,
            'path_length': len(reactions),
            'score': node.total_reward / max(node.visits, 1),
            'final_depth': node.depth,
            'connection_ready': True  # 表示可以连接到E3 ligand
        }
    
    def _reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_iterations': 0,
            'successful_expansions': 0,
            'failed_expansions': 0,
            'target_connections_found': 0,
            'backtrack_count': 0,
            'fully_expanded_nodes': 0,
            'max_tree_depth': 0
        }
    
    def _is_unique_path(self, new_path: Dict, existing_paths: List[Dict]) -> bool:
        """检查路径是否唯一（基于反应序列）"""
        if not existing_paths:
            return True
            
        new_signature = self._get_path_signature(new_path['reactions'])
        
        for existing_path in existing_paths:
            existing_signature = self._get_path_signature(existing_path['reactions'])
            if new_signature == existing_signature:
                return False
        
        return True
    
    def _get_path_signature(self, reactions: List[Dict]) -> tuple:
        """生成路径的唯一签名"""
        # 基于反应类型和试剂的组合
        signature_parts = []
        for reaction in reactions:
            part = f"{reaction.get('reaction', '')}_{reaction.get('reagent', '')}"
            signature_parts.append(part)
        return tuple(signature_parts)
    
    def _default_state_evaluator(self, state: str, target: str, depth: int) -> float:
        """默认的状态评估函数"""
        # 基础评分：分子复杂度
        base_score = len(state) * 0.01
        
        # 深度惩罚：鼓励较短路径
        depth_penalty = depth * 0.1
        
        # 目标相似性：简单的字符串相似度
        similarity = 0.0
        if target is not None:
            if len(target) > 5:
                target_fragment = target[:5]
                if target_fragment in state:
                    similarity = 0.5
        
        return base_score + similarity - depth_penalty
    
    def get_search_statistics(self) -> Dict:
        """获取搜索统计信息"""
        return self.stats.copy()
 

# 便捷函数
def run_protac_mcts_search(warhead: str,
                          e3_ligand: str,
                          action_generator: Callable,
                          connection_checker: Callable,
                          state_evaluator: Callable = None,
                          max_depth: int = 5,
                          max_iterations: int = 1000,
                          num_paths: int = 3) -> List[Dict]:
    """
    便捷的PROTAC MCTS搜索函数
    
    Args:
        warhead: warhead分子SMILES
        e3_ligand: E3 ligand分子SMILES
        action_generator: 动作生成器 f(current_state) -> List[(action_info, new_state)]
        connection_checker: 连接检查器 f(current_state) -> bool
        state_evaluator: 状态评估器 f(state, target, depth) -> float (可选)
        max_depth: 最大搜索深度
        max_iterations: 最大迭代次数
        num_paths: 要找的路径数量
        
    Returns:
        PROTAC合成路径列表
    """
    planner = MCTSPlanner(
        max_depth=max_depth,
        max_iterations=max_iterations
    )
    
    return planner.search(
        warhead=warhead,
        e3_ligand=e3_ligand,
        action_generator=action_generator,
        connection_checker=connection_checker,
        state_evaluator=state_evaluator,
        num_paths=num_paths
    )
