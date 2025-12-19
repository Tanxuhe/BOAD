from abc import ABC, abstractmethod
import torch
from typing import Optional, Dict, Any

class BaseTask(ABC):
    """
    实际优化任务的抽象基类。
    所有具体任务（Rover, MIP, NAS 等）都必须继承此类。
    """
    def __init__(self, cfg: Any):
        """
        初始化任务。
        
        Args:
            cfg: 全局配置对象 (FullConfig)
        """
        self.cfg = cfg
        self.device = torch.device(cfg.experiment.device)
        # 统一使用 double 精度以保证数值稳定性
        self.dtype = torch.double 
        
        # 尝试从配置中读取最优值（如果存在），用于计算 Regret
        # 实际任务中通常为 None
        self.optimal_value: Optional[float] = None
        if hasattr(cfg.problem, 'optimal_value'):
            self.optimal_value = cfg.problem.optimal_value

    @property
    @abstractmethod
    def dim(self) -> int:
        """返回问题的总维度"""
        pass
    
    @property
    def bounds(self) -> torch.Tensor:
        """
        返回 BO 算法使用的搜索空间边界。
        对于高维 BO，通常建议在归一化空间 [0, 1]^D 中进行搜索，
        具体的物理参数映射由各 Task 内部的 ParameterSpace 处理。
        
        Returns:
            torch.Tensor: shape (2, dim), 第一行为下界，第二行为上界。
        """
        # 默认返回 [0, 1] 边界
        return torch.tensor(
            [[0.0] * self.dim, [1.0] * self.dim], 
            device=self.device, 
            dtype=self.dtype
        )

    @abstractmethod
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        核心评估函数。
        
        Args:
            x: shape (batch_size, dim), 范围在 [0, 1] 之间的 Tensor
            
        Returns:
            y: shape (batch_size, 1), 目标函数值。
               注意：BO 通常假设是 最大化 问题。
               如果原问题是最小化（如 Loss, Runtime），请在此函数内部取负。
        """
        pass
    
    def get_optimal_value(self) -> Optional[float]:
        """获取理论最优值（用于 Log 记录）"""
        return self.optimal_value
