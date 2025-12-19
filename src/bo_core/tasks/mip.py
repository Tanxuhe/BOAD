import torch
import numpy as np
import time
from .base import BaseTask
from .parameter import Parameter, ParameterSpace

try:
    from pyscipopt import Model
    HAS_SCIP = True
except ImportError:
    HAS_SCIP = False

class MIPTask(BaseTask):
    """
    74维 MIP 参数调优任务。
    如果 PySCIPOpt 不可用，自动降级为 Mock 模式（模拟复杂地形）。
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        task_conf = getattr(cfg.problem, 'task_config', {})
        self.instance_name = task_conf.get('instance', 'qiu')
        self.time_limit = task_conf.get('time_limit', 10.0)
        self.mps_path = f"data/mip_instances/{self.instance_name}.mps"
        
        # === 定义 74 个参数 ===
        # 为了演示，我们混合定义几种类型。
        # 在真实实验中，你需要根据 MIPLIB 的文档列出所有重要参数。
        self.params = []
        for i in range(74):
            # 模拟：前 10 个是分类 (branching strategy 等)
            if i < 10:
                self.params.append(Parameter(f"p{i}", "categorical", choices=[-1, 0, 1, 2, 3]))
            # 后面的全是浮点/整数 (heuristics freq, limits 等)
            else:
                self.params.append(Parameter(f"p{i}", "float", low=0.0, high=1.0))
                
        self.param_space = ParameterSpace(self.params)
        
        if not HAS_SCIP:
            print(f"[MIPTask] PySCIPOpt not found. Running in MOCK mode.")

    @property
    def dim(self) -> int:
        return 74

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        results = []
        x_np = x.detach().cpu().numpy()
        
        for i in range(x.shape[0]):
            config = self.param_space.decode(x[i])
            
            if HAS_SCIP:
                score = self._run_scip(config)
            else:
                score = self._run_mock(config)
                
            results.append(score)
            
        return torch.tensor(results, device=self.device, dtype=self.dtype).unsqueeze(-1)

    def _run_mock(self, config):
        """模拟一个非凸、多模态的函数"""
        # 简单的 Ackley-like 模拟
        vals = np.array(list(config.values()))
        # 标准化到 [-5, 5]
        v_scaled = (vals - 0.5) * 10 
        
        # Sum of squares
        term1 = -0.2 * np.sqrt(np.mean(v_scaled**2))
        term2 = np.mean(np.cos(2 * np.pi * v_scaled))
        y = -20 * np.exp(term1) - np.exp(term2) + 20 + np.e
        
        # MIP 目标是最小化时间，我们返回 -time
        # 这里 y 是一个正数（类似 error），我们取负
        return -y

    def _run_scip(self, config):
        try:
            model = Model()
            # 安静模式
            model.hideOutput()
            model.readProblem(self.mps_path)
            
            # 设置参数 (伪代码映射，实际需要查 SCIP 文档)
            # model.setParam("separating/maxrounds", config['p0']) ...
            # 这里简单演示：我们假设 config['p0'] 控制 heuristics
            # model.setRealParam("heuristics/alp/freq", config['p10'])
            
            model.setRealParam("limits/time", self.time_limit)
            
            model.optimize()
            
            # 获取结果
            solve_time = model.getSolvingTime()
            gap = model.getGap() # 0.0 表示最优
            
            # 评分函数：优先看时间，如果超时看 Gap
            score = -solve_time
            if gap > 1e-4:
                # 惩罚项
                score -= gap * 1000.0
                
            return score
            
        except Exception as e:
            print(f"[SCIP Error] {e}")
            return -1e5 # 失败惩罚
