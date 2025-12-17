import torch
import numpy as np
import random
import os
import time
import logging
import json
from functools import wraps
from torch.quasirandom import SobolEngine
from pathlib import Path

# ==========================================
# 1. 基础环境设置与日志
# ==========================================

def setup_seed(seed):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_experiment_dir(exp_name, base_dir=None):
    """
    创建实验目录。
    如果不指定 base_dir，默认在项目根目录下的 logs/ 中创建
    """
    if base_dir is None:
        # 获取项目根目录 (假设 utils.py 在 src/bo_core/ 下)
        root_dir = Path(__file__).parent.parent.parent
        base_dir = root_dir / "logs"
    
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def setup_logger(exp_dir, name="BO_Exp"):
    """配置 Python logging"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 清理旧 Handlers
    if logger.hasHandlers():
        logger.handlers.clear()
            
    log_file = os.path.join(exp_dir, "experiment.log")
    fh = logging.FileHandler(log_file, mode='w')
    ch = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

class JSONLLogger:
    """结构化日志记录器"""
    def __init__(self, exp_dir):
        self.log_path = os.path.join(exp_dir, "history.jsonl")
        # 清空旧日志
        with open(self.log_path, 'w') as f:
            pass
        
    def log(self, data_dict):
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(data_dict) + "\n")

# ==========================================
# 2. 张量与采样工具
# ==========================================

def expand_tensor(tensor, batch_shape):
    if len(batch_shape) == 0:
        return tensor
    return tensor.expand(batch_shape + tensor.shape)

def generate_lhs_samples(n_samples, dim, bounds, seed=None):
    """生成初始采样点 (Sobol)"""
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_cand = sobol.draw(n_samples)
    
    lower = bounds[0].cpu()
    upper = bounds[1].cpu()
    
    # 确保类型匹配
    X_cand = X_cand.to(dtype=lower.dtype, device=lower.device)
    
    X_scaled = lower + X_cand * (upper - lower)
    return X_scaled

def generate_oracle_decomposition(dim, structure_type, param):
    """生成 Oracle 分解结构"""
    decomposition = []
    
    if structure_type == 'block':
        block_size = int(param)
        for i in range(0, dim, block_size):
            group = list(range(i, min(i + block_size, dim)))
            decomposition.append(group)
            
    elif structure_type == 'sparse':
        effective_dim = int(param)
        for i in range(effective_dim):
            decomposition.append([i])
        if effective_dim < dim:
            decomposition.append(list(range(effective_dim, dim)))

    elif structure_type == 'dense':
        decomposition.append(list(range(dim)))
        
    return decomposition

# ==========================================
# 3. 实验辅助逻辑
# ==========================================

def simple_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        print(f"[{func.__name__}] cost: {t1-t0:.4f}s")
        return result
    return wrapper

class AdaptiveFrequencyScheduler:
    def __init__(self, initial_freq=20, loose_freq=50, switch_point=100):
        self.initial_freq = initial_freq
        self.loose_freq = loose_freq
        self.switch_point = switch_point
        
    def check_trigger(self, iter_idx):
        if iter_idx < self.switch_point:
            return iter_idx % self.initial_freq == 0
        else:
            return iter_idx % self.loose_freq == 0
