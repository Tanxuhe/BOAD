import argparse
import sys
import os
import glob
from pathlib import Path

# 添加 src 到路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bo_core.components.visualizer import ExperimentVisualizer

def main():
    parser = argparse.ArgumentParser(description="Plot Regret or Best Value Curves")
    
    parser.add_argument("--labels", nargs='+', required=True, help="List of algorithm names")
    parser.add_argument("--patterns", nargs='+', required=True, help="List of glob patterns")
    parser.add_argument("--title", type=str, default="Optimization Performance")
    parser.add_argument("--output", type=str, default="result_plot.png", help="Output filename")
    parser.add_argument("--linear", action="store_true", help="Force linear scale (useful for y_best)")
    
    args = parser.parse_args()
    
    if len(args.labels) != len(args.patterns):
        print("Error: Number of labels must match number of directory patterns.")
        return
        
    viz = ExperimentVisualizer()
    experiments = {}
    first_valid_dir = None
    
    for label, pattern in zip(args.labels, args.patterns):
        dirs = glob.glob(pattern)
        dirs = [d for d in dirs if os.path.isdir(d)]
        
        if not dirs:
            print(f"[Warn] No directories found for pattern: {pattern}")
        else:
            print(f"Found {len(dirs)} runs for '{label}'")
            experiments[label] = dirs
            if first_valid_dir is None:
                first_valid_dir = dirs[0]
            
    if not experiments:
        print("No valid data found to plot.")
        return

    # 智能路径处理
    save_path = args.output
    if os.path.dirname(save_path) == "" and first_valid_dir:
        save_path = os.path.join(first_valid_dir, args.output)
        print(f"[Auto-Save] Output path adjusted to: {save_path}")
        
    viz.plot_regret(
        experiments, 
        title=args.title, 
        save_path=save_path, 
        log_scale=not args.linear
    )

if __name__ == "__main__":
    main()
