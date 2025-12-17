import argparse
import torch
import sys
import os
from pathlib import Path

# 确保能找到 bo_core 包
# 如果您安装了 pip install -e .，这一步其实不需要，但为了保险起见保留
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bo_core.config_manager import FullConfig
from bo_core.optimizer import AdaptiveBO
from bo_core.utils import setup_seed, generate_lhs_samples, generate_oracle_decomposition
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
    
    if cfg.problem.device == "cuda" and torch.cuda.is_available():
        X_init = X_init.cuda()
        Y_init = Y_init.cuda()
        
    # 5. Initial Decomposition (Default or Oracle)
    if cfg.oracle and cfg.oracle.type:
        print(f"Using Oracle Decomposition: {cfg.oracle.type}")
        decomp = generate_oracle_decomposition(cfg.problem.dim, cfg.oracle.type, cfg.oracle.param)
    else:
        # Default: start with all-in-one group or one-by-one? 
        # Usually start with full dim group until first structure learning
        decomp = [list(range(cfg.problem.dim))]

    # 6. Initialize Optimizer
    optimizer = AdaptiveBO(
        config=cfg,
        X_init=X_init,
        Y_init=Y_init,
        bounds=bounds,
        func=func,
        decomposition=decomp,
        output_dir=f"logs/{cfg.experiment.name}"
    )
    
    # 7. Run
    optimizer.optimize()

if __name__ == "__main__":
    main()
