import torch
import numpy as np
import pandas as pd
import os
from .base import BaseTask
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.preprocessing import StandardScaler

class LassoIndependentTask(BaseTask):
    """
    高维稀疏特征选择任务 (LassoBench 复刻版)。
    支持 'svm' (388D) 和 'dna' (180D)。
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # 从 task_config 中读取任务名
        task_conf = getattr(cfg.problem, 'task_config', {})
        self.dataset_name = task_conf.get('dataset', 'svm') # 'svm' or 'dna'
        
        # === 数据加载逻辑 ===
        if self.dataset_name == 'svm':
            # SVM 任务基于 Breast Cancer，并扩展维度到 388
            self._dim = 388
            print(f"[LassoTask] Loading Breast Cancer dataset (Simulating SVM-388D)...")
            X_raw, y_raw = load_breast_cancer(return_X_y=True)
            # 扩展特征：添加噪声特征以模拟高维稀疏性
            # 原始 30维 -> 扩展到 388维
            n_samples = X_raw.shape[0]
            n_noise = self._dim - X_raw.shape[1]
            rng = np.random.RandomState(42)
            X_noise = rng.randn(n_samples, n_noise)
            self.X = np.hstack([X_raw, X_noise])
            self.y = y_raw
            
        elif self.dataset_name == 'dna':
            self._dim = 180
            data_path = task_conf.get('data_path', 'data/dna.csv')
            print(f"[LassoTask] Loading DNA dataset from {data_path}...")
            
            if os.path.exists(data_path):
                # 假设 csv 最后一列是 label
                df = pd.read_csv(data_path)
                self.X = df.iloc[:, :-1].values
                self.y = df.iloc[:, -1].values
            else:
                print(f"[Warning] DNA data not found. Generating synthetic sparse data (180D).")
                self.X, self.y = make_regression(
                    n_samples=200, n_features=180, n_informative=15, 
                    noise=0.5, random_state=42
                )
        
        # 标准化数据 (这对线性模型很重要)
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    @property
    def dim(self) -> int:
        return self._dim

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [0, 1]^D. 
        我们将其解释为特征权重 (Feature Weights)。
        如果 w_i < threshold，则丢弃该特征。
        然后在选中特征子集上训练 Ridge 回归。
        """
        results = []
        x_np = x.detach().cpu().numpy()
        threshold = 0.3  # 特征选择阈值
        
        for i in range(x.shape[0]):
            weights = x_np[i]
            
            # 1. 特征选择
            # 只有权重 > 阈值的特征才被选中
            selected_mask = weights > threshold
            n_selected = np.sum(selected_mask)
            
            if n_selected == 0:
                # 惩罚：未选中任何特征
                results.append(-100.0)
                continue
            
            # 2. 子集训练
            X_sub = self.X[:, selected_mask]
            
            # 使用 Ridge 而不是 Lasso，因为我们已经手动做了选择
            model = Ridge(alpha=1.0) 
            
            # 3折交叉验证 (使用 Neg MSE)
            # score 是负均方误差 (越大越好)
            scores = cross_val_score(model, X_sub, self.y, cv=3, scoring='neg_mean_squared_error')
            mean_score = scores.mean()
            
            # 3. 添加稀疏性奖励/惩罚
            # 我们希望特征越少越好 (Occam's razor)
            # Penalty = lambda * number_of_features
            sparsity_penalty = 0.005 * n_selected
            
            final_score = mean_score - sparsity_penalty
            results.append(final_score)
            
        return torch.tensor(results, device=self.device, dtype=self.dtype).unsqueeze(-1)
