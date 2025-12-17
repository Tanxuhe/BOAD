import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Any
import os

@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42
    device: str = "cuda"

@dataclass
class ProblemConfig:
    name: str
    dim: int
    bounds_low: List[float]
    bounds_high: List[float]
    optimal_value: Optional[float] = None
    wrapper: Optional[str] = None
    params: dict = field(default_factory=dict)

@dataclass
class OracleConfig:
    type: Optional[str] = None
    param: Optional[int] = None

@dataclass
class OptimizationConfig:
    n_initial: int = 10
    n_total: int = 100
    switch_threshold: int = 20
    # [新增] 冷启动间隔
    cold_start_interval: int = 10 

@dataclass
class AlgorithmConfig:
    decomposition_method: str = "shap" 
    decomp_freq: int = 20
    
    # [新增] 频率调度参数
    # 如果为 None，代码中会自动设为 decomp_freq * 2
    decomp_loose_freq: Optional[int] = None 
    # 切换到宽松频率的迭代轮数
    decomp_switch_point: int = 150          
    
    # ... (其他参数保持不变)
    background_source: str = "data"
    data_sample_ratio: float = 0.5
    grid_sample_count: int = 100
    
    interaction_threshold: float = 0.25
    importance_threshold: float = 0.02
    
    beta_scaling: float = 0.75
    ls_prior: float = 0.5

@dataclass
class FullConfig:
    experiment: ExperimentConfig
    problem: ProblemConfig
    optimization: OptimizationConfig
    algorithm: AlgorithmConfig
    oracle: Optional[OracleConfig] = None

    @classmethod
    def load(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
            
        return cls(
            experiment=ExperimentConfig(**raw.get('experiment', {})),
            problem=ProblemConfig(**raw.get('problem', {})),
            optimization=OptimizationConfig(**raw.get('optimization', {})),
            algorithm=AlgorithmConfig(**raw.get('algorithm', {})),
            oracle=OracleConfig(**raw.get('oracle', {})) if 'oracle' in raw else None
        )
