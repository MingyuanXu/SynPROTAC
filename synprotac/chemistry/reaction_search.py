"""
Chemical Reaction Search Module

Handles chemical reaction matching, building block search, and molecule processing
using RDKit for PROTAC synthesis planning.
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
import numpy as np
from tqdm import tqdm 
import pickle 
from ..comparm import GP 
from rdkit.DataStructs import ConvertToNumpyArray
import random 
logger = logging.getLogger(__name__)

class ReactionTemplate:
    """反应模板类"""
    
    def __init__(self, template_smarts: str, template_id: int = None, 
                 name: str = None, priority: int = 1):
        """
        初始化反应模板
        
        Args:
            template_smarts: SMARTS模板字符串
            template_id: 模板ID
            name: 模板名称
            priority: 优先级
        """
        self.template_smarts = template_smarts
        self.template_id = template_id 
        self.name = name or "Unknown Reaction"
        self.priority = priority
        
        try:
            self.reaction = AllChem.ReactionFromSmarts(template_smarts)
            if self.reaction is None:
                raise ValueError(f"无法解析反应模板: {template_smarts}")
                
            # 验证反应
            self.num_reactants = self.reaction.GetNumReactantTemplates()
            self.num_products = self.reaction.GetNumProductTemplates()
            self.is_valid = True
            
        except Exception as e:
            logger.warning(f"反应模板解析失败 {template_id}: {str(e)}")
            self.reaction = None
            self.is_valid = False

    def can_apply_to(self, mol: Chem.Mol, active_atoms: List[int]=[], inactive_atoms: List[int]=[]) -> bool:
        #检查反应是否可应用于分子
        if not self.is_valid or mol is None:
            return False

        if len(active_atoms) ==0:
            active_atoms = list(range(mol.GetNumAtoms()))
        
        flag=False
        for i,patt in enumerate(self.reaction.GetReactants()):
            # 遍历所有匹配
            for match in mol.GetSubstructMatches(patt):
                for atom_idx in active_atoms:
                    if atom_idx in match:
                        flag = True

        for i,patt in enumerate(self.reaction.GetReactants()):
            for match in mol.GetSubstructMatches(patt):
                for atom_idx in inactive_atoms:
                    if atom_idx in match:
                        flag = False

        return flag

    def to_dict(self) -> Dict:
        """将模板转换为字典格式"""
        return {
            'smarts': self.template_smarts,
            'id': self.template_id,
            'name': self.name,
            'priority': self.priority
        }

    @staticmethod
    def from_dict(template_dict): 
        return ReactionTemplate(
            template_smarts=template_dict.get('smarts', ''),
            template_id=template_dict.get('id'),
            name=template_dict.get('name'),
            priority=template_dict.get('priority', 1)
        )

    def excute(self,mol1,mol2):

        avoid_parts=[Chem.MolFromSmiles(part) for part in GP.avoid_substructures]

        try:
            # 尝试不同的反应物顺序
            reactants_combinations = [
                (mol1, mol2),
                (mol2, mol1)
            ]
            total_atomnums= mol1.GetNumAtoms() + mol2.GetNumAtoms()
            
            products = []
            seen_smiles = set()  # 去重

            for reactants in reactants_combinations:
                    reaction_products = self.reaction.RunReactants(reactants)
                    #print (reaction_products)
                    for product_set in reaction_products:
                        for product in product_set:
                            if product is not None:
                                # 清理产物结构
                                avoid_flag=False
                                for part in avoid_parts:
                                    if product.HasSubstructMatch(part):
                                        avoid_flag=True
                                if not avoid_flag:
                                    #try:
                                        Chem.SanitizeMol(product)
                                        # 去重检查
                                        product_smiles = Chem.MolToSmiles(product)
                                        #print (product_smiles)
                                        product_atomnums= product.GetNumAtoms()
                                        flag=False
                                        if product_smiles not in seen_smiles and product_atomnums > total_atomnums*0.75:
                                            seen_smiles.add(product_smiles)
                                            products.append(product)
                                            flag=True
                                        #if not flag:
                                        #    print ('product is something wrong', [Chem.MolToSmiles(reactant) for reactant in reactants], Chem.MolToSmiles(product), total_atomnums, product_atomnums, self.template_id)
                                    #except:
                                    #    continue
            return products
            
        except Exception as e:
            logger.debug(f"反应应用失败: {str(e)}")
            return []


class BuildingBlock:
    """构建块类"""
    
    def __init__(self, smiles: str, mol_id: str = None, properties: Dict = None):
        """
        初始化构建块
        
        Args:
            smiles: SMILES字符串
            mol_id: 分子ID
            properties: 分子属性字典
        """
        self.smiles = smiles
        self.mol_id = mol_id or f"bb_{hash(smiles)}"
        self.properties = properties or {}
        
        try:
        #if True:
            self.mol = Chem.MolFromSmiles(smiles)
            if self.mol is None:
                raise ValueError(f"无法解析SMILES: {smiles}")
            
            # 计算基本属性
            self.mw = Descriptors.MolWt(self.mol)
            self.logp = Descriptors.MolLogP(self.mol)
            self.hbd = Descriptors.NumHDonors(self.mol)
            self.hba = Descriptors.NumHAcceptors(self.mol)
            self.rotatable_bonds = Descriptors.NumRotatableBonds(self.mol)
            self.is_valid = True
            self._reactive_group_num = None
            self.fp=self.calc_fp(self.mol)
        except Exception as e:
            logger.warning(f"构建块解析失败 {mol_id}: {str(e)}")
            self.mol = None
            self.is_valid = False
            self.fp = np.zeros(GP.fp_dim, dtype=np.float32)
    
    def passes_drug_like_filter(self) -> bool:
        """检查是否通过类药性筛选（Lipinski规则）"""
        if not self.is_valid:
            return False
        
        return (self.mw <= 500 and 
                self.logp <= 5 and 
                self.hbd <= 5 and 
                self.hba <= 10)

    @property
    def reactive_group_num(self) -> int:
        """
        计算分子中的反应性基团数量
        
        Args:
            mol_smiles: 分子SMILES
            
        Returns:
            反应性基团数量
        """
        if self.mol is None:
            return 0
        if self._reactive_group_num is not None:
            return self._reactive_group_num
        else: 
            # 常见的反应性基团SMARTS模式
            reactive_patterns = [
                '[NH2]',        # 伯胺
                '[NH]',         # 仲胺  
                '[OH]',         # 羟基
                'C(=O)O',       # 羧酸
                'C(=O)Cl',      # 酰氯
                'C#C',          # 炔基
                "Cl",
                "Br",
                "I",
                "C#N"
            ]
            
            reactive_count = 0
            for pattern in reactive_patterns:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol is not None:
                    matches = self.mol.GetSubstructMatches(patt_mol)
                    reactive_count += len(matches)
            self._reactive_group_num = reactive_count 
            return self._reactive_group_num
    
    def to_dict(self) -> Dict:
        """将构建块转换为字典格式"""
        return {
            'smiles': self.smiles,
            'id': self.mol_id,
            'type': self.mol_id
        }

    @staticmethod
    def from_dict(bb_data):
        return BuildingBlock(bb_data.get('smiles', ''), bb_data.get('id', ''), bb_data.get('properties', {}))

    @staticmethod
    def calc_fp(mol):
        arr = np.zeros(GP.fp_dim, dtype=np.float32)
        fp = GP.fp_generator.GetFingerprint(mol)
        ConvertToNumpyArray(fp, arr)
        return arr 

class ChemReactionSearch:
    """
    化学反应搜索引擎
    
    负责反应模板匹配、构建块搜索和分子处理。
    """
    
    def __init__(self):
        """初始化反应搜索引擎"""
        self.reaction_templates: List[ReactionTemplate] = []
        self.building_blocks: List[BuildingBlock] = []
        
        # 性能优化的缓存
        self._reaction_cache = {}
        self._similarity_cache = {}
        
        # 统计信息
        self.stats = {
            'total_reactions_attempted': 0,
            'successful_reactions': 0,
            'failed_reactions': 0,
            'cache_hits': 0
        }

    def load_reaction_templates(self, templates_data: List[Dict]) -> int:
        """
        加载反应模板
        
        Args:
            templates_data: 模板数据列表，每个元素包含 'smarts', 'id', 'name' 等字段
            
        Returns:
            成功加载的模板数量
        """
        logger.info(f"开始加载{len(templates_data)}个反应模板...")
        
        valid_templates = 0
        for template_data in templates_data:
            template = ReactionTemplate(
                template_smarts=template_data.get('smarts', ''),
                template_id=template_data.get('id'),
                name=template_data.get('name'),
                priority=template_data.get('priority', 1)
            )
            
            if template.is_valid:
                self.reaction_templates.append(template)
                valid_templates += 1
        logger.info(f"成功加载{valid_templates}个有效反应模板")
        return valid_templates
    
    def load_building_blocks(self, building_blocks_data: List[Dict]) -> int:
        """
        加载构建块
        Args:
            building_blocks_data: 构建块数据列表，包含 'smiles', 'id' 等字段
        Returns:
            成功加载的构建块数量
        """
        logger.info(f"开始加载{len(building_blocks_data)}个构建块...")
        
        valid_blocks = 0
        for bb_data in building_blocks_data:
            building_block = BuildingBlock(
                smiles=bb_data.get('smiles', ''),
                mol_id=bb_data.get('id'),
                properties=bb_data.get('properties', {})
            )
            
            if building_block.is_valid and building_block.passes_drug_like_filter():
                self.building_blocks.append(building_block)
                valid_blocks += 1
        
        logger.info(f"成功加载{valid_blocks}个有效构建块")
        return valid_blocks

    def build_rxn_to_building_block_map(self) -> Dict[str, List[BuildingBlock]]:
        """
        构建反应到构建块的映射
        
        Returns:
            反应ID到构建块列表的映射
        """
        self.rxn_to_bb_map = {}
        
        for template in tqdm(self.reaction_templates):
            if not template.is_valid:
                continue
            
            for building_block in self.building_blocks:
                if not building_block.is_valid:
                    continue
                
                # 检查构建块是否可以与反应模板匹配
                if template.can_apply_to(building_block.mol):
                    if template.template_id not in self.rxn_to_bb_map:
                        self.rxn_to_bb_map[template.template_id] = []
                    self.rxn_to_bb_map[template.template_id].append(building_block)
        
        return

    def save_rxn_bb_map(self, file_path: str):
        """
        保存反应到构建块的映射到文件
        
        Args:
            file_path: 保存路径
        """
        import json
        with open(file_path, 'wb') as f:
            pickle.dump(self.rxn_to_bb_map, f)
        logger.info(f"反应到构建块的映射已保存到 {file_path}")

    def load_rxn_bb_map(self, file_path: str):
        """
        从文件加载反应到构建块的映射
        
        Args:
            file_path: 文件路径
        """
        import json
        try:
            with open(file_path, 'rb') as f:
                self.rxn_to_bb_map = pickle.load(f)
            logger.info(f"成功加载反应到构建块的映射从 {file_path}")
        except Exception as e:
            logger.error(f"加载映射失败: {str(e)}")
            self.rxn_to_bb_map = {}
    
    def get_suitable_rxn_templates_for_specific_molecule(self, mol_smiles: str, protected_patts: List[str] = []) -> List[Tuple[Dict, str]]:
        """
        获取适用于特定分子的反应模板
        
        Args:
            mol_smiles: 分子SMILES字符串
            protected_patts: 保护基团的SMARTS模式列表
            
        Returns:
            适用的反应模板列表，每个元素为 (反应信息, 产物SMILES)
        """
        # 检查缓存
        
        try:
            mol,active_atoms,inactive_atoms = self.mol_from_smiles(mol_smiles, protected_patts)
            
            if mol is None:
                return []
            
            applicable_rxn_templates = []

            for template in self.reaction_templates:
                if not template.can_apply_to(mol, active_atoms, inactive_atoms):
                    continue

                applicable_rxn_templates.append(template)

            return applicable_rxn_templates
            
        except Exception as e:
            logger.error(f"获取特定分子反应失败: {str(e)}")
            return []
    
    def get_applicable_reactions(self, mol_smiles: str, protected_patts: List=[]) -> List[Tuple[Dict, str]]:
        """
        获取可应用于分子的反应
        
        Args:
            mol_smiles: 分子SMILES字符串
            
        Returns:
            可应用的反应列表，每个元素为 (反应信息, 产物SMILES)
        """
        # 检查缓存

        cache_key = f"reactions_{mol_smiles}"
        if cache_key in self._reaction_cache:
            self.stats['cache_hits'] += 1
            return self._reaction_cache[cache_key]
        
        try:
            mol,active_atoms,inactive_atoms = self.mol_from_smiles(mol_smiles, protected_patts)
            if mol is None:
                return []
            
            applicable_reactions = []
            
            for template in self.reaction_templates:
                if not template.can_apply_to(mol, active_atoms, inactive_atoms):
                    continue
                
                self.stats['total_reactions_attempted'] += 1
                
                # 尝试与构建块反应
                for building_block in self.rxn_to_bb_map.get(template.template_id, []):
                    try:
                        products = self._apply_reaction(
                            template, mol, building_block.mol
                        )
                        
                        for product_mol in products:
                            if product_mol and self._is_valid_product(product_mol):
                                product_smiles = Chem.MolToSmiles(product_mol)
                                
                                reaction_info = {
                                    'reaction': template.template_id,
                                    'reaction_name': template.name,
                                    'reagent': building_block.smiles,
                                    'reagent_id': building_block.mol_id,
                                    'template_smarts': template.template_smarts
                                }
                                
                                applicable_reactions.append((reaction_info, product_smiles))
                                self.stats['successful_reactions'] += 1
                                
                    except Exception as e:
                        logger.debug(f"反应失败: {str(e)}")
                        self.stats['failed_reactions'] += 1
                        continue

            random.shuffle(applicable_reactions)
            # 限制结果数量以提高性能
            if len(applicable_reactions) > 50:
                applicable_reactions = applicable_reactions[:50]
            
            # 缓存结果
            self._reaction_cache[cache_key] = applicable_reactions
            return applicable_reactions
            
        except Exception as e:
            logger.error(f"获取可应用反应失败: {str(e)}")
            return []
    
    def _apply_reaction(self, template: ReactionTemplate, mol1: Chem.Mol, 
                       mol2: Chem.Mol) -> List[Chem.Mol]:
        """应用反应模板"""
        return template.excute(mol1,mol2)
    
    def _is_valid_product(self, mol: Chem.Mol) -> bool:
        """检查产物是否有效"""
        if mol is None:
            return False
        
        # 基本有效性检查
        if mol.GetNumAtoms() < 3 or mol.GetNumAtoms() > 250:
            return False
        
        # 检查是否有合理的分子量
        mw = Descriptors.MolWt(mol)
        if mw < 50 or mw > 1800:
            return False
        
        # 检查是否过于复杂
        if Descriptors.NumRotatableBonds(mol) > 100:
            return False
        
        return True
            
    
    def find_similar_molecules(self, query_smiles: str, 
                              similarity_threshold: float = 0.6) -> List[BuildingBlock]:
        """
        寻找相似分子
        
        Args:
            query_smiles: 查询分子SMILES
            similarity_threshold: 相似度阈值
            
        Returns:
            相似构建块列表
        """

        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            return []
        
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
        similar_blocks = []
        
        for building_block in self.building_blocks:
            if not building_block.is_valid:
                continue
            
            try:
                bb_fp = AllChem.GetMorganFingerprintAsBitVect(building_block.mol, 2, nBits=2048)
                similarity = AllChem.DataStructs.TanimotoSimilarity(query_fp, bb_fp)
                
                if similarity >= similarity_threshold:
                    similar_blocks.append(building_block)
                    
            except Exception:
                continue
        
        # 按相似度排序
        similar_blocks.sort(key=lambda bb: AllChem.DataStructs.TanimotoSimilarity(
            query_fp, AllChem.GetMorganFingerprintAsBitVect(bb.mol, 2, nBits=2048)
        ), reverse=True)
        
        return similar_blocks[:20]  # 限制返回数量
            
    
    def evaluate_synthetic_accessibility(self, mol_smiles: str) -> float:
        """
        评估合成可达性
        
        Args:
            mol_smiles: 分子SMILES
            
        Returns:
            合成可达性分数 (0-1, 1表示容易合成)
        """
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
            return 0.0
        
        # 简化的合成可达性评估
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        
        # 使用替代的复杂度计算方法
        try:
            complexity = rdMolDescriptors.BertzCT(mol)
        except AttributeError:
            # 如果BertzCT不可用，使用简单的原子数作为复杂度指标
            complexity = mol.GetNumAtoms() * 10
        
        # 计算分数（启发式方法）
        mw_score = 1.0 - min(1.0, max(0.0, (mw - 200) / 400))
        logp_score = 1.0 - min(1.0, max(0.0, abs(logp - 2) / 5))
        flexibility_score = min(1.0, rotatable_bonds / 15)
        complexity_score = 1.0 - min(1.0, complexity / 1000)
        
        # 综合评分
        final_score = (mw_score * 0.3 + logp_score * 0.2 + 
                      flexibility_score * 0.2 + complexity_score * 0.3)
        
        return max(0.0, min(1.0, final_score))
            
    
    def get_statistics(self) -> Dict:
        """获取搜索统计信息"""
        total_attempts = self.stats['total_reactions_attempted']
        success_rate = (self.stats['successful_reactions'] / total_attempts 
                       if total_attempts > 0 else 0.0)
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'num_templates': len(self.reaction_templates),
            'num_building_blocks': len(self.building_blocks),
            'cache_size': len(self._reaction_cache)
        }
    
    def clear_cache(self):
        """清理缓存"""
        self._reaction_cache.clear()
        self._similarity_cache.clear()

    @staticmethod 
    def mol_from_smiles(smiles,protected_patts :List[str] = []):

        mol=Chem.MolFromSmiles(smiles)
        if mol is None:
            return mol,[],[]

        active_atoms=[atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]
        if len(active_atoms)==0:
            active_atoms = set(range(mol.GetNumAtoms()))

        inactive_atoms=[]
        if len(protected_patts) > 0:
            for patt in protected_patts:
                protected_mol = Chem.MolFromSmiles(patt)
                
                if protected_mol is None:
                    continue
                protected_matches = mol.GetSubstructMatches(protected_mol)
                
                for match in protected_matches:
                    for idx in match:
                        inactive_atoms.append(idx)

            inactive_atoms = list(set(inactive_atoms))

        return mol,active_atoms,inactive_atoms

    def get_connection_reactions(self, mol1_smiles: str, mol2_smiles: str, protected_patts: List[str] =[], specific_templates: List[ReactionTemplate] = []) -> bool:
        """
        检查两个分子是否可以通过反应模板库中的反应连接

        Args:
            mol1_smiles: 第一个分子的SMILES
            mol2_smiles: 第二个分子的SMILES
            
        Returns:
            是否可以连接
        """
        applicable_templates = specific_templates if len(specific_templates)>0 else self.reaction_templates

        mol1, active_atoms_in_mol1, inactive_atoms_in_mol1 = self.mol_from_smiles(mol1_smiles,protected_patts)

        mol2, active_atoms_in_mol2, inactive_atoms_in_mol2 = self.mol_from_smiles(mol2_smiles,protected_patts)
        
        connection_reactions = [] 
        if mol1 is None or mol2 is None:
            return connection_reactions

        # 遍历所有反应模板，检查是否有能连接这两个分子的反应
        for template in applicable_templates:
            if not template.is_valid:
                continue
            
            # 检查反应是否需要两个反应物
            if template.num_reactants != 2:
                continue
            
            if template.can_apply_to(mol1, active_atoms_in_mol1, inactive_atoms_in_mol1) and \
                template.can_apply_to(mol2, active_atoms_in_mol2, inactive_atoms_in_mol2):
                reactants_combinations = [
                    (mol1, mol2),
                    (mol2, mol1)
                ]
                for reactants in reactants_combinations: 
                    products = self._apply_reaction(template, reactants[0], reactants[1])
                    if products and any(self._is_valid_product(p) for p in products):
                        for product_mol in products:
                            product_smiles = Chem.MolToSmiles(product_mol)        
                            reaction_info = {
                                        'reaction': template.template_id,
                                        'reaction_name': f"{template.name}_Connection",
                                        'reagent': mol2_smiles,  # 第二个反应物作为试剂
                                        'reagent_id': f"reagent_{hash(mol2_smiles)}",
                                        'template_smarts': template.template_smarts,
                                        'reaction_type': 'connection'
                                    }
                            connection_reactions.append((reaction_info, product_smiles))

        return connection_reactions        
    
    def calculate_molecular_weight(self, mol_smiles: str) -> float:
        """
        计算分子量
        
        Args:
            mol_smiles: 分子SMILES
            
        Returns:
            分子量 (Da)
        """
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
            return 0.0
            
        return Descriptors.MolWt(mol)
    
    def calculate_molecular_similarity(self, mol1_smiles: str, mol2_smiles: str) -> float:
        """
        计算两个分子的相似性
        
        Args:
            mol1_smiles: 第一个分子SMILES
            mol2_smiles: 第二个分子SMILES
            
        Returns:
            Tanimoto相似性分数 (0-1)
        """
        mol1 = Chem.MolFromSmiles(mol1_smiles)
        mol2 = Chem.MolFromSmiles(mol2_smiles)
        
        if mol1 is None or mol2 is None:
            return 0.0
            
        # 计算Morgan指纹
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        # 计算Tanimoto相似性
        return DataStructs.TanimotoSimilarity(fp1, fp2)


# 便捷函数
def create_reaction_searcher(templates_data: List[Dict], 
                           building_blocks_data: List[Dict]) -> ChemReactionSearch:
    """
    创建配置好的反应搜索器
    
    Args:
        templates_data: 反应模板数据
        building_blocks_data: 构建块数据
        
    Returns:
        配置好的ChemReactionSearch实例
    """
    searcher = ChemReactionSearch()
    searcher.load_reaction_templates(templates_data)
    searcher.load_building_blocks(building_blocks_data)
    return searcher
