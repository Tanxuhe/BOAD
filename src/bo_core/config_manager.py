import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import os

@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42
    device: str = "cuda"

@dataclass
class ProblemConfig:
    # === 通用配置 ===
    name: str                    # 任务名称 (用于日志或合成函数名)
    dim: int                     # 维度 (部分任务会自动覆盖此值)
    type: str = "synthetic"      # [新增] 任务类型: synthetic, rover, lasso, nas, mip
    
    # === 合成函数专用 ===
    bounds_low: List[float] = field(default_factory=lambda: [-5.0])
    bounds_high: List[float] = field(default_factory=lambda: [5.0])
    optimal_value: Optional[float] = None
    
    # === [新增] 实际任务专用配置 ===
    # 用于存放 dataset, data_path, instance_name, time_limit 等杂项
    task_config: Dict[str, Any] = field(default_factory=dict)
    
    wrapper: Optional[str] = None
    params: dict = field(default_factory=dict)

@dataclass
class OracleConfig:
    type: Optional[str] = None
    param: Optional[int] = None

@dataclass
class OptimizationConfig:
    n_initial: int = 20
    n_total: int = 300
    switch_threshold: int = 50
    cold_start_interval: int = 20 

@dataclass
class AlgorithmConfig:
    decomposition_method: str = "friedman" 
    decomp_freq: int = 25
    
    decomp_loose_freq: Optional[int] = None 
    decomp_switch_point: int = 150          
    
    background_source: str = "data"
    data_sample_ratio: float = 0.5
    grid_sample_count: int = 100
    
    interaction_threshold: float = 0.25
    importance_threshold: float = 0.02
    
    beta_scaling: float = 2.0
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
