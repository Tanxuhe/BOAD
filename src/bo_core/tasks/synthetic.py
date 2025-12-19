import torch
from .base import BaseTask
# 引用你原有的函数工厂
from bo_core.test_functions import get_function

class SyntheticTask(BaseTask):
    """
    包装传统的数学合成测试函数 (如 Stybtang, Rosenbrock)。
    完全兼容旧的配置文件结构。
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.func_name = cfg.problem.name
        self.func = get_function(self.func_name)
        self._dim = cfg.problem.dim
        
        # 兼容旧配置：从 bounds_low/high 读取边界
        # 如果配置是单值列表 [-5]，则扩展到所有维度
        low_val = cfg.problem.bounds_low[0]
        high_val = cfg.problem.bounds_high[0]
        
        self._bounds = torch.tensor(
            [[low_val] * self._dim, [high_val] * self._dim],
            device=self.device, 
            dtype=self.dtype
        )
        
        # 缓存范围用于归一化映射
        self.range_val = high_val - low_val
        self.min_val = low_val

    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def bounds(self) -> torch.Tensor:
        # 注意：这里我们返回物理边界给 BO 核心吗？
        # 依照 BaseTask 约定，BO 核心最好在 [0, 1] 运行。
        # 但为了最小化对旧代码逻辑的冲击，这里我们可以做一个内部映射：
        # BaseTask 默认返回 [0, 1]，我们在 evaluate 里手动把 [0, 1] 映射回 [-5, 5]
        return super().bounds 

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, dim) 范围 [0, 1]
        """
        # 1. 映射回物理空间 (e.g., [-5, 5])
        x_phys = x * self.range_val + self.min_val
        
        # 2. 调用原有函数
        # 原函数期望输入 (batch, dim)
        y = self.func(x_phys)
        
        # 3. 确保输出维度 (batch, 1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
            
        return y
