import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Union, Optional, Any, Dict

@dataclass
class Parameter:
    """
    参数定义类。
    负责定义单个参数的属性（类型、范围、是否对数标度），
    并提供从归一化空间 [0, 1] 到物理空间的映射功能。
    """
    name: str
    type: str  # 支持 'float', 'int', 'categorical'
    
    # === 连续/整数参数配置 ===
    low: Optional[float] = None
    high: Optional[float] = None
    log_scale: bool = False
    
    # === 分类参数配置 ===
    # choices 可以是字符串列表，也可以是数字列表
    choices: Optional[List[Any]] = field(default_factory=list)

    def __post_init__(self):
        """简单的参数校验"""
        if self.type in ['float', 'int']:
            if self.low is None or self.high is None:
                raise ValueError(f"Parameter '{self.name}' of type '{self.type}' must have 'low' and 'high' defined.")
            if self.high <= self.low:
                raise ValueError(f"Parameter '{self.name}': high must be greater than low.")
        
        if self.type == 'categorical':
            if not self.choices:
                raise ValueError(f"Parameter '{self.name}' of type 'categorical' must have 'choices' defined.")

    def transform(self, val_norm: float) -> Union[float, int, Any]:
        """
        核心映射函数：将 [0, 1] 的归一化值转换为物理值。
        
        Args:
            val_norm: 范围在 [0, 1] 之间的浮点数 (通常来自 BO 算法)
        
        Returns:
            对应的物理参数值
        """
        # 1. 裁剪以防止数值误差导致的越界
        val_norm = np.clip(val_norm, 0.0, 1.0)
        
        # 2. 分类变量处理
        if self.type == 'categorical':
            # 将 [0, 1] 均匀切分为 len(choices) 份
            n_choices = len(self.choices)
            # 例如 3 个选项：[0, 0.33) -> 0, [0.33, 0.66) -> 1, [0.66, 1.0] -> 2
            idx = int(val_norm * n_choices)
            # 边界处理：当 val_norm=1.0 时，idx 会等于 n_choices，需要减 1
            idx = min(idx, n_choices - 1)
            return self.choices[idx]
        
        # 3. 连续/整数变量处理
        elif self.type in ['float', 'int']:
            if self.log_scale:
                # Log10 空间插值
                # 物理值 = 10^(log_low + norm * (log_high - log_low))
                # 注意：前提是 low 和 high 必须 > 0
                if self.low <= 0:
                    raise ValueError(f"Log scale parameter '{self.name}' must have positive bounds.")
                
                log_low = np.log10(self.low)
                log_high = np.log10(self.high)
                phys_val = 10**(log_low + val_norm * (log_high - log_low))
            else:
                # 线性空间插值
                phys_val = self.low + val_norm * (self.high - self.low)
            
            # 类型转换
            if self.type == 'int':
                return int(round(phys_val))
            return phys_val
            
        raise ValueError(f"Unknown parameter type: {self.type}")

class ParameterSpace:
    """
    参数空间管理类。
    管理一组 Parameter 对象，提供批量解码功能。
    """
    def __init__(self, parameters: List[Parameter]):
        self.params = parameters
        self.dim = len(parameters)
        self.param_names = [p.name for p in parameters]
        
    def decode(self, x_norm: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        将一行归一化向量解析为参数字典。
        
        Args:
            x_norm: shape 为 (dim,) 的 Tensor 或 numpy array
        
        Returns:
            Dict: {param_name: phys_value}
        """
        # 统一转为 numpy
        if isinstance(x_norm, torch.Tensor):
            x_vals = x_norm.detach().cpu().numpy()
        else:
            x_vals = x_norm
            
        if len(x_vals) != self.dim:
            raise ValueError(f"Input dimension {len(x_vals)} does not match parameter space dimension {self.dim}")
            
        decoded = {}
        for i, param in enumerate(self.params):
            decoded[param.name] = param.transform(x_vals[i])
            
        return decoded
