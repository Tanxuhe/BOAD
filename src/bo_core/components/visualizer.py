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
        自动检测是否包含 Regret，如果没有（实际任务），则聚合 y_best。
        """
        all_series = []
        max_len = 0
        use_regret = True
        
        print(f"[Viz] Aggregating {len(exp_dirs)} directories...")
        
        # 1. 第一遍扫描：确定是否所有实验都有 regret
        # 如果有任何一个实验没有 optimal_value，我们就切换到 y_best 模式
        for d in exp_dirs:
            df = self.load_history(d)
            if df.empty: continue
            
            # 检查 regret 列是否全为 None
            if 'regret' not in df.columns or df['regret'].isnull().all():
                use_regret = False
                break
        
        metric_col = 'regret' if use_regret else 'y_best'
        print(f"   [Info] Aggregation Metric: {metric_col}")

        for d in exp_dirs:
            df = self.load_history(d)
            
            if df.empty:
                print(f"   [Skip] {d}: DataFrame is empty.")
                continue
            
            if 'iter' in df.columns:
                df = df.sort_values('iter')
            
            # [修复] 仅当 metric_col 存在时才提取
            if metric_col not in df.columns:
                # 尝试向后兼容：如果 y_best 没有，用 y_new 的累计最大值填充
                if metric_col == 'y_best' and 'y_new' in df.columns:
                    df['y_best'] = df['y_new'].cummax()
                else:
                    print(f"   [Skip] {d}: Column '{metric_col}' not found.")
                    continue

            # 提取数据
            # 注意：对于 Regret，我们通常希望越小越好；对于 y_best，越大越好
            # 这里只负责提取，不负责转换方向
            vals = df[metric_col].values
            
            # 清洗 NaN (仅针对 Regret，y_best 通常不会是 NaN)
            if use_regret:
                vals = vals[~np.isnan(vals)]
            
            if len(vals) == 0:
                continue

            all_series.append(vals)
            max_len = max(max_len, len(vals))
            print(f"   [Load] {d}: Loaded {len(vals)} iters. Final {metric_col}: {vals[-1]:.4e}")
            
        if not all_series:
            return pd.DataFrame()
            
        # 2. 对齐数据 (Padding)
        matrix = np.full((len(all_series), max_len), np.nan)
        for i, r in enumerate(all_series):
            matrix[i, :len(r)] = r
            # Forward Fill: 假设实验停止后，最优值/Regret 保持不变
            if len(r) < max_len:
                 matrix[i, len(r):] = r[-1]
                
        # 3. 计算统计量
        with np.errstate(invalid='ignore'):
            mean_val = np.nanmean(matrix, axis=0)
            std_val = np.nanstd(matrix, axis=0)
            valid_counts = np.sum(~np.isnan(matrix), axis=0)
            stderr_val = np.divide(std_val, np.sqrt(valid_counts), out=np.zeros_like(std_val), where=valid_counts>0)
        
        iters = np.arange(max_len)
        return pd.DataFrame({
            'iter': iters,
            'mean': mean_val,
            'std': std_val,
            'stderr': stderr_val,
            'metric': metric_col # 标记当前使用的是哪个指标
        })

    def plot_regret(self, 
                    experiments: Dict[str, List[str]], 
                    title: str = "Optimization Performance",
                    save_path: str = "result_plot.png",
                    log_scale: bool = True):
        
        plt.figure(figsize=(10, 6))
        colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
        
        has_data = False
        is_regret_plot = True
        
        for idx, (label, dirs) in enumerate(experiments.items()):
            df_agg = self.aggregate_results(dirs)
            if df_agg.empty:
                print(f"[Warn] No valid data for label: '{label}'")
                continue
            
            has_data = True
            
            # 检测实际绘制的是 Regret 还是 y_best
            if 'metric' in df_agg.columns and df_agg['metric'].iloc[0] == 'y_best':
                is_regret_plot = False
            
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
        
        # 自动调整 Y 轴标签和 Log Scale
        if is_regret_plot:
            plt.ylabel("Simple Regret (Log Scale)" if log_scale else "Simple Regret", fontsize=12)
            if log_scale:
                plt.yscale('log')
        else:
            plt.ylabel("Best Found Value", fontsize=12)
            # y_best 通常不需要 log scale，除非差异极大，这里默认关闭 log
            # 或者仅当 log_scale=True 且数据全是正数时开启
            # 为安全起见，y_best 模式下建议默认线性，除非用户强制要求
            # 这里我们保持用户选项，但如果数据有负数，log scale 会报错，需注意
            if log_scale:
                try:
                    plt.yscale('log')
                except:
                    pass
            
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which="both", ls="--", alpha=0.4)
        
        plt.tight_layout()
        
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plt.savefig(save_path, dpi=300)
        print(f"\n[Success] Plot saved to: {save_path}")
        plt.close()
