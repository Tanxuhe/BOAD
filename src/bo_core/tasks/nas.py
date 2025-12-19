import torch
from .base import BaseTask
from .parameter import Parameter, ParameterSpace

try:
    from nas_201_api import NASBench201API as API
    HAS_NAS = True
except ImportError:
    HAS_NAS = False

class NASTask(BaseTask):
    """
    NAS-Bench-201 架构搜索任务。
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # [修复] Fail Fast: 必须检查依赖
        if not HAS_NAS:
            raise RuntimeError("NAS-Bench-201 library not found. Please install: pip install nas-bench-201")
            
        self.dim_edges = 6
        self.ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        
        self.params = [
            Parameter(f"edge_{i}", "categorical", choices=self.ops) 
            for i in range(self.dim_edges)
        ]
        self.param_space = ParameterSpace(self.params)
        
        task_conf = getattr(cfg.problem, 'task_config', {})
        self.dataset = task_conf.get('dataset', 'cifar10-valid')
        self.db_path = task_conf.get('data_path', 'data/NAS-Bench-201-v1_1-096897.pth')
        
        print(f"[NASTask] Loading NAS-Bench-201 API from {self.db_path} ...")
        try:
            self.api = API(self.db_path, verbose=False)
        except Exception as e:
            # [修复] 数据文件不存在也直接抛出异常
            raise FileNotFoundError(f"Failed to load NAS Database at {self.db_path}. Error: {e}")

    @property
    def dim(self) -> int:
        return 6

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        results = []
        for i in range(x.shape[0]):
            # 1. 解码参数
            config = self.param_space.decode(x[i])
            
            # 2. 构建架构字符串
            ops = [config[f"edge_{j}"] for j in range(6)]
            arch_str = f"|{ops[0]}~0|+|{ops[1]}~0|{ops[2]}~1|+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
            
            try:
                idx = self.api.query_index_by_arch(arch_str)
                info = self.api.get_more_info(idx, self.dataset, is_random=False)
                # 使用验证集精度 (0~100)
                acc = info['valid-accuracy']
                results.append(acc)
            except Exception as e:
                print(f"[NAS] Query failed: {e}")
                results.append(0.0)
                
        return torch.tensor(results, device=self.device, dtype=self.dtype).unsqueeze(-1)
