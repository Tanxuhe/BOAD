import torch
from .base import BaseTask
from .parameter import Parameter, ParameterSpace

# 尝试导入，如果环境没装，则在运行时报错
try:
    from nas_201_api import NASBench201API as API
    HAS_NAS = True
except ImportError:
    HAS_NAS = False

class NASTask(BaseTask):
    """
    NAS-Bench-201 架构搜索任务。
    搜索空间：6 条边，每条边 5 种操作。
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dim_edges = 6
        self.ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        
        # 定义参数空间：6 个 Categorical 变量
        self.params = [
            Parameter(f"edge_{i}", "categorical", choices=self.ops) 
            for i in range(self.dim_edges)
        ]
        self.param_space = ParameterSpace(self.params)
        
        task_conf = getattr(cfg.problem, 'task_config', {})
        self.dataset = task_conf.get('dataset', 'cifar10-valid')
        # 数据库路径 (必须手动下载上传)
        self.db_path = task_conf.get('data_path', 'data/NAS-Bench-201-v1_1-096897.pth')
        
        self.api = None
        if HAS_NAS:
            print(f"[NASTask] Loading NAS-Bench-201 API from {self.db_path} ...")
            # 这一步比较耗时(加载2GB数据)，建议只做一次
            try:
                self.api = API(self.db_path, verbose=False)
            except Exception as e:
                print(f"[Error] Failed to load NAS API: {e}")
                HAS_NAS = False

    @property
    def dim(self) -> int:
        return 6

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        results = []
        if not HAS_NAS or self.api is None:
            # Fallback for testing code logic without data
            return torch.zeros(x.shape[0], 1, device=self.device) - 1.0

        for i in range(x.shape[0]):
            # 1. 解码参数
            config = self.param_space.decode(x[i])
            
            # 2. 构建架构结构字符串
            # NAS-Bench-201 的节点结构: (1<-0), (2<-0, 2<-1), (3<-0, 3<-1, 3<-2)
            # 对应的边索引顺序: 0, 1, 2, 3, 4, 5
            # 格式: |op0~0|+|op1~0|op2~1|+|op3~0|op4~1|op5~2|
            ops = [config[f"edge_{j}"] for j in range(6)]
            arch_str = f"|{ops[0]}~0|+|{ops[1]}~0|{ops[2]}~1|+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
            
            # 3. 查询精度
            try:
                idx = self.api.query_index_by_arch(arch_str)
                info = self.api.get_more_info(idx, self.dataset, is_random=False)
                # 获取验证集精度 (0~100)
                acc = info['valid-accuracy']
                results.append(acc)
            except Exception as e:
                print(f"[NAS] Query failed: {e}")
                results.append(0.0)
                
        return torch.tensor(results, device=self.device, dtype=self.dtype).unsqueeze(-1)
