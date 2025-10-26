from rdkit import Chem 
from typing import List, Dict, Optional, Tuple, Any, Callable
from tqdm import tqdm 

def reverse_reaction_smarts(smarts: str) -> str:
    """将正向反应SMARTS转换为逆向（用于逆合成分析）"""
    if '>>' not in smarts:
        print (smarts)
        raise ValueError("不是合法的反应SMARTS")
    left, right = smarts.split('>>')
    return f"{right}>>{left}"

def reverse_templates(templates:List[Dict] =[]) -> List[Dict]:
    """批量将模板列表逆向"""
    reversed_templates = []
    for tpl in tqdm(templates):
        try:
            new_tpl = tpl.copy()
            new_tpl['smarts'] = reverse_reaction_smarts(tpl['smarts'])
            reversed_templates.append(new_tpl)
        except:
            pass
    return reversed_templates
