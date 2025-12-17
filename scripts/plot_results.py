import argparse
import sys
import os
import glob
from pathlib import Path

# 添加 src 到路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bo_core.components.visualizer import ExperimentVisualizer

def main():
    parser = argparse.ArgumentParser(description="Plot Regret Curves from Logs")
    
    # 允许传入多组实验
    # 用法: --labels "Algo A" "Algo B" --patterns "logs/AlgoA_*" "logs/AlgoB_*"
    parser.add_argument("--labels", nargs='+', required=True, help="List of algorithm names")
    parser.add_argument("--patterns", nargs='+', required=True, help="List of glob patterns for log directories")
    parser.add_argument("--title", type=str, default="Optimization Performance")
    parser.add_argument("--output", type=str, default="comparison_plot.png")
    parser.add_argument("--linear", action="store_true", help="Use linear scale instead of log scale")
    
    args = parser.parse_args()
    
    if len(args.labels) != len(args.patterns):
        print("Error: Number of labels must match number of directory patterns.")
        return
        
    viz = ExperimentVisualizer()
    experiments = {}
    
    for label, pattern in zip(args.labels, args.patterns):
        # 扩展通配符
        dirs = glob.glob(pattern)
        # 过滤掉非目录
        dirs = [d for d in dirs if os.path.isdir(d)]
        
        if not dirs:
            print(f"[Warn] No directories found for pattern: {pattern}")
        else:
            print(f"Found {len(dirs)} runs for '{label}'")
            experiments[label] = dirs
            
    if not experiments:
        print("No valid data found to plot.")
        return
        
    viz.plot_regret(
        experiments, 
        title=args.title, 
        save_path=args.output, 
        log_scale=not args.linear
    )

if __name__ == "__main__":
    main()
