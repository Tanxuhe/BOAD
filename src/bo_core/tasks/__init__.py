from typing import Any
from .base import BaseTask
from .synthetic import SyntheticTask
from .rover import RoverTrajectoryTask
from .lassobench import LassoIndependentTask
from .nas import NASTask
from .mip import MIPTask

class TaskFactory:
    """
    任务工厂类：根据配置文件的 problem.type 创建对应的任务实例。
    """
    @staticmethod
    def create_task(cfg: Any) -> BaseTask:
        """
        Args:
            cfg: FullConfig 对象
            
        Returns:
            具体任务的实例 (继承自 BaseTask)
        """
        # 获取任务类型，默认为 'synthetic' 以兼容旧配置
        task_type = getattr(cfg.problem, 'type', 'synthetic').lower()
        
        print(f"[TaskFactory] Initializing task: {task_type} ...")
        
        if task_type == 'synthetic':
            return SyntheticTask(cfg)
            
        elif task_type == 'rover':
            return RoverTrajectoryTask(cfg)
            
        elif task_type in ['svm', 'dna', 'lasso', 'lassobench']:
            # 统一由 LassoIndependentTask 处理
            return LassoIndependentTask(cfg)
            
        elif task_type == 'nas':
            return NASTask(cfg)
            
        elif task_type == 'mip':
            return MIPTask(cfg)
            
        else:
            raise ValueError(f"Unknown task type: {task_type}. Supported: synthetic, rover, lasso, nas, mip")
