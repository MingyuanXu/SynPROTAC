"""
PROTAC反应路径可视化工具
改进版本，支持真实的反应步骤可视化
"""
import logging
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw
import os

# 设置日志
logger = logging.getLogger(__name__)

class ReactionPathVisualizer:
    """PROTAC合成反应路径可视化器"""
    
    def __init__(self, img_size: Tuple[int, int] = (250, 250), 
                 font_size: int = 12, 
                 padding: int = 30):
        """
        初始化可视化器
        
        Args:
            img_size: 分子图像尺寸
            font_size: 字体大小
            padding: 边距
        """
        self.img_size = img_size
        self.font_size = font_size
        self.padding = padding
        self.arrow_size = (180, 100)
        self.horizontal_spacing = 40  # 水平间距
        self.vertical_spacing = 50    # 垂直间距
        
        # 颜色配置
        self.colors = {
            'background': 'white',
            'warhead': '#E8F5E8',      # 浅绿色
            'reagent': '#FFF3E0',      # 浅橙色
            'intermediate': '#E3F2FD',  # 浅蓝色
            'final_product': '#FCE4EC', # 浅粉色
            'arrow': '#666666',         # 灰色箭头
            'text': '#333333'           # 深灰色文字
        }
    
    def draw_molecule_with_label(self, smiles: str, label: str, 
                               bg_color: str = 'white') -> Image.Image:
        """
        绘制带标签的分子结构
        
        Args:
            smiles: SMILES字符串
            label: 分子标签
            bg_color: 背景颜色
            
        Returns:
            PIL Image对象
        """
        try:
            # 解析SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._create_error_image(f"Invalid SMILES: {smiles}")
            
            # 生成分子图像
            mol_img = Draw.MolToImage(mol, size=self.img_size)
            
            # 创建带标签的图像，增加标签区域高度
            label_height = 50  # 增加标签高度
            total_height = self.img_size[1] + label_height
            
            combined_img = Image.new('RGB', (self.img_size[0], total_height), bg_color)
            
            # 粘贴分子图像
            combined_img.paste(mol_img, (0, 0))
            
            # 添加标签
            draw = ImageDraw.Draw(combined_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)  # 稍微增大字体
            except:
                font = ImageFont.load_default()
            
            # 计算文字位置（居中），增加上边距
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (self.img_size[0] - text_width) // 2
            text_y = self.img_size[1] + 10  # 增加间距
            if label !='reagent':
                draw.text((text_x, text_y), label, fill=self.colors['text'], font=font)
            
            return combined_img
            
        except Exception as e:
            logger.warning(f"绘制分子失败 {smiles}: {str(e)}")
            return self._create_error_image(f"Error: {str(e)}")
    
    def _create_error_image(self, error_msg: str) -> Image.Image:
        """创建错误提示图像"""
        img = Image.new('RGB', self.img_size, 'lightgray')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
        
        # 多行显示错误信息
        lines = error_msg.split(' ')
        line_height = 15
        start_y = self.img_size[1] // 2 - len(lines) * line_height // 2
        
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (self.img_size[0] - text_width) // 2
            text_y = start_y + i * line_height
            draw.text((text_x, text_y), line, fill='red', font=font)
        
        return img
    
    def _extract_reaction_steps(self, path_data: Dict) -> List[Dict]:
        """按顺序提取每一步反应的详细信息"""
        reactions = path_data.get('reactions', [])
        if not reactions:
            return []
        
        # 按step字段排序
        sorted_reactions = sorted(reactions, key=lambda x: x.get('step', 0))
        
        reaction_steps = []
        
        for i, reaction in enumerate(sorted_reactions):
            step_info = {
                'step_number': reaction.get('step', i + 1),
                'reaction_name': reaction.get('reaction', f'Reaction {i + 1}'),
                'reaction_type': reaction.get('reaction_type', 'unknown'),
                'reactant_smiles': reaction.get('from_state', ''),
                'reagent_smiles': reaction.get('reagent', ''),
                'product_smiles': reaction.get('product', ''),
                'is_final_step': i == len(sorted_reactions) - 1
            }
            
            # 如果没有from_state，使用warhead作为第一步的反应物
            if not step_info['reactant_smiles'] and i == 0:
                step_info['reactant_smiles'] = path_data.get('warhead', '')
            
            reaction_steps.append(step_info)
        
        return reaction_steps
    
    def _create_reaction_step_image(self, step_info: Dict) -> Image.Image:
        """创建单个反应步骤的图像"""
        step_components = []
        
        # 1. 反应物
        if step_info['reactant_smiles']:
            reactant_label = f"Reactant\n(Step {step_info['step_number']})"
            reactant_img = self.draw_molecule_with_label(
                step_info['reactant_smiles'],
                reactant_label,
                self.colors['warhead'] if step_info['step_number'] == 1 else self.colors['intermediate']
            )
            step_components.append(reactant_img)
        
        # 2. 反应箭头和试剂
        arrow_text = f"Step {step_info['step_number']}\n{step_info['reaction_name']}"
        
        if step_info['reagent_smiles']:
            # 创建试剂图像
            reagent_img = self.draw_molecule_with_label(
                step_info['reagent_smiles'],
                "Reagent",
                self.colors['reagent']
            )
            arrow_img = self._create_arrow_with_reagent(arrow_text, reagent_img)
        else:
            arrow_img = self._create_arrow_image(arrow_text)
        
        step_components.append(arrow_img)
        
        # 3. 产物
        if step_info['product_smiles']:
            if step_info['is_final_step']:
                product_label = "Final PROTAC"
                product_color = self.colors['final_product']
            else:
                product_label = f"Intermediate\n(Step {step_info['step_number']})"
                product_color = self.colors['intermediate']
            
            product_img = self.draw_molecule_with_label(
                step_info['product_smiles'],
                product_label,
                product_color
            )
            step_components.append(product_img)
        
        # 水平组合这一步的所有组件
        return self._combine_step_components(step_components)
    
    def _create_arrow_image(self, text: str) -> Image.Image:
        """创建简单的反应箭头（无试剂）"""
        arrow_width = 220
        arrow_height = 120  # 增加高度以容纳文本
        
        arrow_img = Image.new('RGB', (arrow_width, arrow_height), 'white')
        draw = ImageDraw.Draw(arrow_img)
        
        # 绘制箭头
        arrow_start_x = 40
        arrow_end_x = arrow_width - 40
        arrow_center_y = arrow_height // 2
        
        # 箭头线
        draw.line([(arrow_start_x, arrow_center_y), (arrow_end_x, arrow_center_y)], 
                 fill='black', width=3)
        
        # 箭头头部
        head_points = [
            (arrow_end_x, arrow_center_y),
            (arrow_end_x - 15, arrow_center_y - 8),
            (arrow_end_x - 15, arrow_center_y + 8)
        ]
        draw.polygon(head_points, fill='black')
        
        # 反应名称（在箭头上方，增加间距）
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except:
            font = ImageFont.load_default()
        
        # 多行文本处理，增加文本区域
        text_lines = text.split('\n')
        text_start_y = arrow_center_y - len(text_lines) * 8 - 20  # 增加间距
        for i, line in enumerate(text_lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (arrow_width - text_width) // 2
            draw.text((text_x, text_start_y + i * 15), line, fill='black', font=font)
        
        return arrow_img
    
    def _create_arrow_with_reagent(self, arrow_text: str, reagent_img: Image.Image) -> Image.Image:
        """创建带有试剂的反应箭头"""
        arrow_width = 250
        
        # 缩放试剂图像以避免过大
        max_reagent_size = (150, 150)
        if reagent_img.width > max_reagent_size[0] or reagent_img.height > max_reagent_size[1]:
            reagent_img = reagent_img.resize(max_reagent_size, Image.Resampling.LANCZOS)
        
        arrow_height = reagent_img.height + 100  # 增加空间以容纳箭头和文本
        
        # 创建箭头图像
        arrow_img = Image.new('RGB', (arrow_width, arrow_height), 'white')
        draw = ImageDraw.Draw(arrow_img)
        
        # 试剂位置（在上方，居中）
        reagent_x = (arrow_width - reagent_img.width) // 2
        reagent_y = 15
        arrow_img.paste(reagent_img, (reagent_x, reagent_y))
        
        # 绘制箭头（在试剂下方）
        arrow_y = reagent_y + reagent_img.height + 20
        arrow_start_x = 40
        arrow_end_x = arrow_width - 40
        arrow_center_y = arrow_y + 15
        
        # 箭头线
        draw.line([(arrow_start_x, arrow_center_y), (arrow_end_x, arrow_center_y)], 
                 fill='black', width=3)
        
        # 箭头头部
        head_points = [
            (arrow_end_x, arrow_center_y),
            (arrow_end_x - 15, arrow_center_y - 8),
            (arrow_end_x - 15, arrow_center_y + 8)
        ]
        draw.polygon(head_points, fill='black')
        
        # 反应名称（在箭头下方，增加间距）
        text_y = arrow_center_y + 25
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
        
        # 多行文本处理，增加行间距
        text_lines = arrow_text.split('\n')
        for i, line in enumerate(text_lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (arrow_width - text_width) // 2
            draw.text((text_x, text_y + i * 15), line, fill='black', font=font)
        
        return arrow_img
    
    def _combine_step_components(self, components: List[Image.Image]) -> Image.Image:
        """水平组合单个反应步骤的组件"""
        if not components:
            return Image.new('RGB', (100, 100), 'white')
        
        # 计算总宽度和最大高度，增大间距
        total_width = sum(img.width for img in components) + (len(components) - 1) * self.horizontal_spacing
        max_height = max(img.height for img in components)
        
        # 创建组合图像，增加padding
        combined_img = Image.new('RGB', (total_width + self.padding * 2, max_height + self.padding * 2), 'white')
        
        x_offset = self.padding
        for img in components:
            y_offset = (max_height - img.height) // 2 + self.padding
            combined_img.paste(img, (x_offset, y_offset))
            x_offset += img.width + self.horizontal_spacing
        
        return combined_img
    
    def _combine_images_vertically(self, images: List[Image.Image]) -> Image.Image:
        """垂直组合图像"""
        if not images:
            return Image.new('RGB', (100, 100), 'white')
        
        # 计算总尺寸，增大垂直间距
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images) + (len(images) - 1) * self.vertical_spacing
        
        # 创建组合图像，增加padding
        combined_img = Image.new('RGB', (max_width + self.padding * 2, total_height + self.padding * 2), 'white')
        
        y_offset = self.padding
        for img in images:
            x_offset = (max_width - img.width) // 2 + self.padding
            combined_img.paste(img, (x_offset, y_offset))
            y_offset += img.height + self.vertical_spacing
        
        return combined_img
    
    def _add_title(self, img: Image.Image, title: str) -> Image.Image:
        """为图像添加标题"""
        title_height = 50
        new_img = Image.new('RGB', (img.width, img.height + title_height), 'white')
        
        # 粘贴原图像
        new_img.paste(img, (0, title_height))
        
        # 添加标题
        draw = ImageDraw.Draw(new_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (img.width - text_width) // 2
        text_y = 15
        
        draw.text((text_x, text_y), title, fill='black', font=font)
        
        return new_img
    
    def visualize_reaction_path(self, path_data: Dict, save_path: str = None) -> Image.Image:
        """可视化反应路径 - 改进版本，显示每一步反应"""
        try:
            # 提取反应步骤
            reaction_steps = self._extract_reaction_steps(path_data)
            
            if not reaction_steps:
                # 如果没有详细的反应步骤，回退到原始方法
                return self._visualize_simple_path(path_data, save_path)
            
            # 为每个反应步骤创建图像
            step_images = []
            for step_info in reaction_steps:
                step_img = self._create_reaction_step_image(step_info)
                step_images.append(step_img)
            
            # 垂直组合所有步骤
            if step_images:
                combined_img = self._combine_images_vertically(step_images)
            else:
                combined_img = Image.new('RGB', (800, 200), 'white')
                draw = ImageDraw.Draw(combined_img)
                draw.text((50, 50), "No reaction steps found", fill='black')
            
            # 添加标题
            title_text = f"PROTAC Synthesis Pathway ({len(reaction_steps)} steps)"
            final_img = self._add_title(combined_img, title_text)
            
            if save_path:
                final_img.save(save_path)
                print(f"Reaction pathway visualization saved to: {save_path}")
            
            return final_img
            
        except Exception as e:
            print(f"Error in reaction path visualization: {str(e)}")
            # 回退到简单可视化
            return self._visualize_simple_path(path_data, save_path)
    
    def _visualize_simple_path(self, path_data: Dict, save_path: str = None) -> Image.Image:
        """简单的反应路径可视化（回退方法）"""
        molecules = self._extract_molecules(path_data)
        
        # 为每个分子创建图像
        mol_images = []
        for mol_data in molecules:
            mol_img = self.draw_molecule_with_label(
                mol_data['smiles'], 
                mol_data['label'], 
                mol_data['color']
            )
            mol_images.append(mol_img)
        
        # 创建箭头图像
        arrow_imgs = []
        for i in range(len(mol_images) - 1):
            arrow_text = f"Step {i+1}"
            arrow_img = self._create_arrow_image(arrow_text)
            arrow_imgs.append(arrow_img)
        
        # 组合分子和箭头
        components = []
        for i, mol_img in enumerate(mol_images):
            components.append(mol_img)
            if i < len(arrow_imgs):
                components.append(arrow_imgs[i])
        
        # 水平组合
        combined_img = self._combine_step_components(components)
        
        # 添加标题
        final_img = self._add_title(combined_img, "PROTAC Synthesis Pathway")
        
        if save_path:
            final_img.save(save_path)
        
        return final_img
    
    def _extract_molecules(self, path_data: Dict) -> List[Dict]:
        """提取反应路径中的所有分子"""
        molecules = []
        
        # Warhead
        molecules.append({
            'smiles': path_data['warhead'],
            'label': 'Warhead',
            'color': self.colors['warhead']
        })
        
        # 中间产物（如果有）
        if 'intermediate_product' in path_data and path_data['intermediate_product']:
            molecules.append({
                'smiles': path_data['intermediate_product'],
                'label': 'Intermediate',
                'color': self.colors['intermediate']
            })
        
        # 最终产物
        molecules.append({
            'smiles': path_data['final_product'],
            'label': 'Final PROTAC',
            'color': self.colors['final_product']
        })
        
        return molecules

# 便捷函数
def visualize_path(path_data: Dict, save_path: str = None) -> Image.Image:
    """
    快速可视化单个反应路径
    
    Args:
        path_data: 反应路径数据
        save_path: 保存路径
        
    Returns:
        PIL Image对象
    """
    visualizer = ReactionPathVisualizer()
    return visualizer.visualize_reaction_path(path_data, save_path)

# 使用示例
if __name__ == "__main__":
    # 示例数据
    example_path = {
        'warhead': 'Cc1c(C)c(C(c2ccc(Cl)cc2)=N[C@H](c3n4c(C)nn3)CC([OH:1])=O)c4s1',
        'intermediate_product': 'CC(=O)Nc1cc(Cl)ccc1OC(=O)C[C@@H]1N=C(c2ccc(Cl)cc2)c2c(sc(C)c2C)-n2c(C)nnc21',
        'final_product': 'Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@@H](CC(=O)Oc1cccc(N3CCC(=O)NC3=O)c1C)c1nnc(C)n1-2',
        'e3_ligand': 'CC1=C([OH:1])C=CC=C1N2CCC(NC2=O)=O',
        'reactions': [
            {
                'step': 1,
                'reaction': 'Reaction 12',
                'reagent': 'O=C1COc2ccc(Cl)cc2N1',
                'product': 'CC(=O)Nc1cc(Cl)ccc1OC(=O)C[C@@H]1N=C(c2ccc(Cl)cc2)c2c(sc(C)c2C)-n2c(C)nnc21',
                'from_state': 'Cc1c(C)c(C(c2ccc(Cl)cc2)=N[C@H](c3n4c(C)nn3)CC([OH:1])=O)c4s1',
                'reaction_type': 'building_block'
            },
            {
                'step': 2,
                'reaction': 'E3_Connection',
                'reagent': 'CC1=C([OH:1])C=CC=C1N2CCC(NC2=O)=O',
                'product': 'Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@@H](CC(=O)Oc1cccc(N3CCC(=O)NC3=O)c1C)c1nnc(C)n1-2',
                'from_state': 'CC(=O)Nc1cc(Cl)ccc1OC(=O)C[C@@H]1N=C(c2ccc(Cl)cc2)c2c(sc(C)c2C)-n2c(C)nnc21',
                'reaction_type': 'e3_connection'
            }
        ],
        'path_length': 2,
        'score': 48.65,
        'final_depth': 1,
        'connection_ready': True
    }
    
    # 生成可视化图像
    img = visualize_path(example_path, "reaction_path_fixed.png")
    print("反应路径可视化完成！")
