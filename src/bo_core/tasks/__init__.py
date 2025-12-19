from typing import Any
from .base import BaseTask

# 导入具体任务 (后续批次提供的文件)
# 为了避免循环依赖或文件未创建报错，我们使用局部导入或在 factory 中导入
# 这里仅预留结构

class TaskFactory:
    """
    任务工厂类，根据配置创建对应的 Task 实例。
    """
    @staticmethod
    def create_task(cfg: Any) -> BaseTask:
        """
        Args:
            cfg: FullConfig 对象
            
        Returns:
            BaseTask 的具体子类实例
        """
        # 从配置中获取任务类型
        # 假设 config.problem.type 存在，默认为 'synthetic' 以兼容旧代码
        task_type = getattr(cfg.problem, 'type', 'synthetic').lower()
        
        if task_type == 'synthetic':
            from .synthetic import SyntheticTask
            return SyntheticTask(cfg)
            
        elif task_type == 'rover':
            from .rover import RoverTrajectoryTask
            return RoverTrajectoryTask(cfg)
            
        elif task_type in ['svm', 'dna', 'lasso', 'lassobench']:
            from .lassobench import LassoIndependentTask
            return LassoIndependentTask(cfg)
            
        elif task_type == 'nas':
            from .nas import NASTask
            return NASTask(cfg)
            
        elif task_type == 'mip':
            from .mip import MIPTask
            return MIPTask(cfg)
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
