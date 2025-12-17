import argparse
import torch
import sys
import os
from pathlib import Path

# 确保能找到 bo_core 包
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bo_core.config_manager import FullConfig
from bo_core.optimizer import AdaptiveBO
# [修正] 引入 setup_experiment_dir
from bo_core.utils import setup_seed, generate_lhs_samples, generate_oracle_decomposition, setup_experiment_dir
from bo_core.test_functions import get_function

def main():
    parser = argparse.ArgumentParser(description="Run High-Dim BO Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading config from {args.config}...")
    cfg = FullConfig.load(args.config)
    
    # 2. Setup Environment
    setup_seed(cfg.experiment.seed)
    
    # 3. Prepare Problem
    func = get_function(cfg.problem.name)
    bounds = torch.tensor([cfg.problem.bounds_low * cfg.problem.dim, 
                           cfg.problem.bounds_high * cfg.problem.dim])
    
    # 4. Initial Sampling (LHS)
    print(f"Generating {cfg.optimization.n_initial} initial samples...")
    X_init = generate_lhs_samples(cfg.optimization.n_initial, cfg.problem.dim, bounds)
    Y_init = func(X_init).unsqueeze(-1)
    
    if cfg.experiment.device == "cuda" and torch.cuda.is_available():
        X_init = X_init.cuda()
        Y_init = Y_init.cuda()
        
    # 5. Initial Decomposition (Default or Oracle)
    if cfg.oracle and cfg.oracle.type:
        print(f"Using Oracle Decomposition: {cfg.oracle.type}")
        decomp = generate_oracle_decomposition(cfg.problem.dim, cfg.oracle.type, cfg.oracle.param)
    else:
        decomp = [list(range(cfg.problem.dim))]

    # [修正] 6. Setup Output Directory (确保目录存在)
    # 这会在 logs/ 下创建对应的实验文件夹
    exp_dir = setup_experiment_dir(cfg.experiment.name)
    print(f"Experiment output directory: {exp_dir}")

    # 7. Initialize Optimizer
    optimizer = AdaptiveBO(
        config=cfg,
        X_init=X_init,
        Y_init=Y_init,
        bounds=bounds,
        func=func,
        decomposition=decomp,
        output_dir=exp_dir # 传入创建好的绝对路径/相对路径
    )
    
    # 8. Run
    optimizer.optimize()

if __name__ == "__main__":
    main()
