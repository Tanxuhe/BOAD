import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional

class ExperimentVisualizer:
    def __init__(self):
        pass

    def load_history(self, log_dir: str) -> pd.DataFrame:
        """读取单个实验目录下的 history.jsonl"""
        path = os.path.join(log_dir, "history.jsonl")
        if not os.path.exists(path):
            print(f"   [Debug] File not found: {path}")
            return pd.DataFrame()
        
        data = []
        with open(path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"   [Debug] JSON decode error at line {line_num+1} in {path}")
                    continue
        return pd.DataFrame(data)

    def aggregate_results(self, exp_dirs: List[str]) -> pd.DataFrame:
        """
        聚合多个实验结果。
        """
        all_series = []
        max_len = 0
        
        print(f"[Viz] Aggregating {len(exp_dirs)} directories...")
        
        for d in exp_dirs:
            df = self.load_history(d)
            
            # 1. 基础检查
            if df.empty:
                print(f"   [Skip] {d}: DataFrame is empty.")
                continue
            
            if 'regret' not in df.columns:
                print(f"   [Skip] {d}: No 'regret' column found. Columns: {df.columns.tolist()}")
                continue
            
            # 2. 排序与清洗
            if 'iter' in df.columns:
                df = df.sort_values('iter')
            
            # [核心修复] 只丢弃 Regret 为空的行，而不是丢弃整个文件
            original_len = len(df)
            df_clean = df.dropna(subset=['regret'])
            cleaned_len = len(df_clean)
            
            if cleaned_len == 0:
                print(f"   [Skip] {d}: All regret values are None/NaN. (Check optimal_value config)")
                continue
                
            if cleaned_len < original_len:
                print(f"   [Info] {d}: Dropped {original_len - cleaned_len} rows with NaN regret.")

            # 3. 提取数据
            regrets = df_clean['regret'].values
            all_series.append(regrets)
            max_len = max(max_len, len(regrets))
            print(f"   [Load] {d}: Loaded {len(regrets)} iterations. Final regret: {regrets[-1]:.4e}")
            
        if not all_series:
            return pd.DataFrame()
            
        # 4. 对齐数据 (Padding with NaN for shorter runs)
        # 我们创建一个矩阵，行是实验，列是迭代
        matrix = np.full((len(all_series), max_len), np.nan)
        for i, r in enumerate(all_series):
            matrix[i, :len(r)] = r
            # 可选：向前填充最后的值 (Forward Fill)，假设实验停止后 Regret 不变
            if len(r) < max_len:
                 matrix[i, len(r):] = r[-1]
                
        # 5. 计算统计量
        with np.errstate(invalid='ignore'): # 忽略全 NaN 列的警告
            mean_regret = np.nanmean(matrix, axis=0)
            std_regret = np.nanstd(matrix, axis=0)
            # 计算有效样本数
            valid_counts = np.sum(~np.isnan(matrix), axis=0)
            stderr_regret = np.divide(std_regret, np.sqrt(valid_counts), out=np.zeros_like(std_regret), where=valid_counts>0)
        
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
        
        plt.figure(figsize=(10, 6))
        # 预定义一组好看的颜色
        colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
        
        has_data = False
        
        for idx, (label, dirs) in enumerate(experiments.items()):
            df_agg = self.aggregate_results(dirs)
            if df_agg.empty:
                print(f"[Warn] No valid data for label: '{label}'")
                continue
            
            has_data = True
            x = df_agg['iter']
            y = df_agg['mean']
            err = df_agg['stderr']
            
            color = colors[idx % len(colors)]
            plt.plot(x, y, label=label, color=color, linewidth=2)
            plt.fill_between(x, y - err, y + err, color=color, alpha=0.2)
            
        if not has_data:
            print("Error: No data to plot.")
            return

        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel("Simple Regret", fontsize=12)
        if log_scale:
            plt.yscale('log')
            plt.ylabel("Simple Regret (Log Scale)", fontsize=12)
            
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which="both", ls="--", alpha=0.4)
        
        plt.tight_layout()
        
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plt.savefig(save_path, dpi=300)
        print(f"\n[Success] Plot saved to: {save_path}")
        plt.close()
