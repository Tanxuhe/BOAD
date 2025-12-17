import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional
import glob

class ExperimentVisualizer:
    def __init__(self):
        pass

    def load_history(self, log_dir: str) -> pd.DataFrame:
        """读取单个实验目录下的 history.jsonl"""
        path = os.path.join(log_dir, "history.jsonl")
        if not os.path.exists(path):
            print(f"[Warn] No history.jsonl found in {log_dir}")
            return pd.DataFrame()
        
        data = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    continue
        return pd.DataFrame(data)

    def aggregate_results(self, exp_dirs: List[str]) -> pd.DataFrame:
        """
        聚合多个实验（不同种子）的结果。
        返回一个 DataFrame，包含 iter, regret_mean, regret_std 等
        """
        all_regrets = []
        max_len = 0
        
        for d in exp_dirs:
            df = self.load_history(d)
            if df.empty or 'regret' not in df.columns:
                continue
            
            # 确保按 iter 排序
            if 'iter' in df.columns:
                df = df.sort_values('iter')
                
            regrets = df['regret'].values
            # 过滤掉 None (如果 optimal_value 未设置)
            if np.any(pd.isnull(regrets)):
                continue
                
            all_regrets.append(regrets)
            max_len = max(max_len, len(regrets))
            
        if not all_regrets:
            return pd.DataFrame()
            
        # Pad with last value (Or NaN) - standard behavior is to fill forward
        # 转为矩阵 (N_runs, Max_Iter)
        matrix = np.full((len(all_regrets), max_len), np.nan)
        for i, r in enumerate(all_regrets):
            matrix[i, :len(r)] = r
            # Forward fill if some runs stopped early
            if len(r) < max_len:
                matrix[i, len(r):] = r[-1]
                
        # Calculate stats
        mean_regret = np.nanmean(matrix, axis=0)
        std_regret = np.nanstd(matrix, axis=0)
        # Standard Error = Std / Sqrt(N)
        stderr_regret = std_regret / np.sqrt(len(all_regrets))
        
        iters = np.arange(max_len)
        return pd.DataFrame({
            'iter': iters,
            'mean': mean_regret,
            'std': std_regret,
            'stderr': stderr_regret
        })

    def plot_regret(self, 
                    experiments: Dict[str, List[str]], 
                    title: str = "Regret vs Iterations",
                    save_path: str = "regret_plot.png",
                    log_scale: bool = True):
        """
        experiments: 字典，Key是算法名称 (Label), Value是该算法对应的多个实验目录列表
        例如: {
            "Adaptive BO": ["logs/Exp1_Seed1", "logs/Exp1_Seed2"],
            "Standard BO": ["logs/Exp2_Seed1", "logs/Exp2_Seed2"]
        }
        """
        plt.figure(figsize=(10, 6))
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for idx, (label, dirs) in enumerate(experiments.items()):
            df_agg = self.aggregate_results(dirs)
            if df_agg.empty:
                print(f"[Warn] No valid data for {label}")
                continue
            
            x = df_agg['iter']
            y = df_agg['mean']
            err = df_agg['stderr'] # 使用标准误作为阴影，或者用 'std'
            
            color = colors[idx % len(colors)]
            plt.plot(x, y, label=label, color=color, linewidth=2)
            plt.fill_between(x, y - err, y + err, color=color, alpha=0.2)
            
        plt.xlabel("Iterations")
        plt.ylabel("Simple Regret (Log Scale)" if log_scale else "Simple Regret")
        if log_scale:
            plt.yscale('log')
            
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
