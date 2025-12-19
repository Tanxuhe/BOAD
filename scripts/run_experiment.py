import argparse
import torch
import sys
import os
from pathlib import Path

# 添加 src 路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bo_core.config_manager import FullConfig
from bo_core.optimizer import AdaptiveBO
from bo_core.utils import setup_seed, generate_lhs_samples, generate_oracle_decomposition, setup_experiment_dir
# [修改] 引入 TaskFactory
from bo_core.tasks import TaskFactory

def main():
    parser = argparse.ArgumentParser(description="Run High-Dim BO Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading config from {args.config}...")
    cfg = FullConfig.load(args.config)
    
    # 2. Setup Environment
    setup_seed(cfg.experiment.seed)
    
    # 3. [修改] Create Task Instance
    # 这里会根据 config 自动创建 Rover, NAS, 或 Synthetic 任务
    task = TaskFactory.create_task(cfg)
    print(f"Task created: {task.__class__.__name__} (Dim: {task.dim})")
    
    # 4. Initial Sampling (LHS)
    print(f"Generating {cfg.optimization.n_initial} initial samples...")
    # 使用 task.bounds 来生成样本
    # 注意：generate_lhs_samples 需要 bounds 是 tensor
    X_init = generate_lhs_samples(cfg.optimization.n_initial, task.dim, task.bounds)
    
    # [修改] 使用 task.evaluate
    print("Evaluating initial samples...")
    Y_init = task.evaluate(X_init)
    
    if cfg.experiment.device == "cuda" and torch.cuda.is_available():
        X_init = X_init.cuda()
        Y_init = Y_init.cuda()
        
    # 5. Initial Decomposition
    if cfg.oracle and cfg.oracle.type:
        print(f"Using Oracle Decomposition: {cfg.oracle.type}")
        decomp = generate_oracle_decomposition(task.dim, cfg.oracle.type, cfg.oracle.param)
    else:
        decomp = [list(range(task.dim))]

    # 6. Setup Output Directory
    exp_dir = setup_experiment_dir(cfg.experiment.name)
    print(f"Experiment output directory: {exp_dir}")

    # 7. Initialize Optimizer
    optimizer = AdaptiveBO(
        config=cfg,
        task=task,  # [修改] 传入 task 对象
        X_init=X_init,
        Y_init=Y_init,
        decomposition=decomp,
        output_dir=exp_dir
    )
    
    # 8. Run
    optimizer.optimize()

if __name__ == "__main__":
    main()
