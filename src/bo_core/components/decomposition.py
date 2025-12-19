import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Tuple, Any
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

try:
    import shap
except ImportError:
    pass 

class BaseDecompositionFinder:
    # ... (Base class unchanged) ...
    def __init__(self, X: Any, Y: Any, 
                 importance_threshold_ratio: float = 0.02,
                 interaction_threshold_ratio: float = 0.25,
                 seed: int = 42): 
        
        if hasattr(X, 'cpu'):
            X_np = X.cpu().numpy()
        else:
            X_np = np.array(X)
            
        self.dimension = X_np.shape[1]
        self.feature_names = [f'X{i}' for i in range(self.dimension)]

        if hasattr(Y, 'cpu'): 
            Y_np = Y.cpu().numpy().ravel()
        else: 
            Y_np = np.array(Y).ravel()

        mask = np.isfinite(Y_np)
        if not np.all(mask):
            X_np = X_np[mask]
            Y_np = Y_np[mask]
            
        self.X_full = pd.DataFrame(X_np, columns=self.feature_names)
        self.Y_full = Y_np

        self.imp_ratio = importance_threshold_ratio
        self.int_ratio = interaction_threshold_ratio
        
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=seed, 
            n_jobs=-1
        )

    def find(self) -> Tuple[List[List[int]], int]:
        raise NotImplementedError

class SHAPDecompositionFinder(BaseDecompositionFinder):
    # ... (SHAP logic unchanged) ...
    def find(self) -> Tuple[List[List[int]], int]:
        model = clone(self.rf_model)
        model.fit(self.X_full, self.Y_full)
        
        try:
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            print(f"[SHAP] TreeExplainer failed: {e}. Returning simple decomposition.")
            return [[i] for i in range(self.dimension)], -1

        shap_interaction_values = explainer.shap_interaction_values(self.X_full)
        global_interactions = np.abs(shap_interaction_values).mean(0)
        main_effects = np.diag(global_interactions)
        
        max_imp = main_effects.max() + 1e-9
        effective_dims = []
        for i in range(self.dimension):
            if main_effects[i] / max_imp > self.imp_ratio:
                effective_dims.append(i)
                
        if not effective_dims:
            effective_dims = [np.argmax(main_effects)]
            
        G = nx.Graph()
        G.add_nodes_from(effective_dims)
        
        for i in effective_dims:
            for j in effective_dims:
                if i >= j: continue
                interaction_val = global_interactions[i, j] * 2
                denom = interaction_val + main_effects[i] + main_effects[j] + 1e-9
                score = interaction_val / denom
                
                if score > self.int_ratio:
                    G.add_edge(i, j)
        
        decomposition = []
        visited = set()
        for comp in nx.connected_components(G):
            group = sorted(list(comp))
            decomposition.append(group)
            visited.update(group)
            
        unimportant_group = [i for i in range(self.dimension) if i not in visited]
        if unimportant_group:
            decomposition.append(unimportant_group)
            unimp_idx = len(decomposition) - 1
        else:
            unimp_idx = -1
            
        return decomposition, unimp_idx

class FriedmanHDecompositionFinder(BaseDecompositionFinder):
    """
    基于 Empirical Friedman's H-statistic 的结构发现器。
    """
    def __init__(self, X: Any, Y: Any, 
                 importance_threshold_ratio: float = 0.02,
                 interaction_threshold_ratio: float = 0.25,
                 seed: int = 42,
                 background_source: str = 'data',  
                 data_sample_ratio: float = 0.5,   
                 grid_sample_count: int = 100):
        
        super().__init__(X, Y, importance_threshold_ratio, interaction_threshold_ratio, seed)
        self.bg_source = background_source
        self.data_ratio = np.clip(data_sample_ratio, 0.01, 1.0)
        self.grid_count = grid_sample_count

    def _compute_pdp_vectorized(self, model, X_t: np.ndarray, X_bg: np.ndarray, 
                              fix_cols: List[int]) -> np.ndarray:
        T, D = X_t.shape
        B = X_bg.shape[0]
        
        # [优化] 使用 tile 构建大矩阵
        # 注意: 如果 T*B 很大 (如 > 500k)，可能会导致内存压力
        # 但相比 Python 循环，向量化预测速度快得多
        X_batch = np.tile(X_bg, (T, 1)) 
        
        for col in fix_cols:
            col_vals = np.repeat(X_t[:, col], B) 
            X_batch[:, col] = col_vals
            
        max_batch = 500000 
        total_samples = X_batch.shape[0]
        
        if total_samples <= max_batch:
            preds = model.predict(X_batch)
        else:
            preds = np.zeros(total_samples)
            for start in range(0, total_samples, max_batch):
                end = min(start + max_batch, total_samples)
                preds[start:end] = model.predict(X_batch[start:end])
        
        pdp_values = preds.reshape(T, B).mean(axis=1)
        return pdp_values

    def _get_background_data(self, rng):
        N = self.X_full.shape[0]
        X_vals = self.X_full.values
        
        if self.bg_source == 'uniform':
            mins = X_vals.min(axis=0)
            maxs = X_vals.max(axis=0)
            X_bg = rng.uniform(low=mins, high=maxs, size=(self.grid_count, self.dimension))
            return X_bg
        else: 
            n_samples = int(N * self.data_ratio)
            n_samples = max(10, min(n_samples, N))
            idx_bg = rng.choice(N, n_samples, replace=False)
            return X_vals[idx_bg]

    def find(self) -> Tuple[List[List[int]], int]:
        X_vals = self.X_full.values
        N, D = X_vals.shape
        rng = np.random.RandomState(self.rf_model.random_state)
        
        if N > 2000:
            idx_t = rng.choice(N, 2000, replace=False)
            X_t = X_vals[idx_t]
        else:
            X_t = X_vals
        
        X_bg = self._get_background_data(rng)
        
        model = clone(self.rf_model)
        # 使用 values 避免 sklearn 警告
        model.fit(self.X_full.values, self.Y_full)
        
        f_hat_raw = model.predict(X_t)
        global_mean = np.mean(f_hat_raw)
        f_hat_centered = f_hat_raw - global_mean
        denom_imp = np.sum(f_hat_centered ** 2) + 1e-9

        # Stage 1: Importance
        importances = {}
        cached_pd_1d = {} 
        
        for j in tqdm(range(D), desc="[H-Stat] Screening", leave=False):
            fix_cols = [c for c in range(D) if c != j]
            pd_minus_j = self._compute_pdp_vectorized(model, X_t, X_bg, fix_cols)
            pd_minus_j_centered = pd_minus_j - global_mean
            
            numerator = np.sum((f_hat_centered - pd_minus_j_centered) ** 2)
            importances[j] = numerator / denom_imp

            pd_j = self._compute_pdp_vectorized(model, X_t, X_bg, [j])
            cached_pd_1d[j] = pd_j - global_mean

        max_imp = max(importances.values()) if importances else 1.0
        effective_dims = [j for j, imp in importances.items() if (imp / max_imp) > self.imp_ratio]
        
        if not effective_dims:
            effective_dims = [max(importances, key=importances.get)]

        # Stage 2: Interactions
        G = nx.Graph()
        G.add_nodes_from(effective_dims)
        
        pairs = []
        for i_idx in range(len(effective_dims)):
            for j_idx in range(i_idx + 1, len(effective_dims)):
                pairs.append((effective_dims[i_idx], effective_dims[j_idx]))
        
        # [优化提示] 如果 D 很大，这里的循环会非常慢。
        # 建议通过提高 importance_threshold 来减少 effective_dims 的数量
        for i, j in tqdm(pairs, desc="[H-Stat] Interaction", leave=False):
            pd_ij = self._compute_pdp_vectorized(model, X_t, X_bg, [i, j])
            pd_ij_centered = pd_ij - global_mean
            
            pd_i_centered = cached_pd_1d[i]
            pd_j_centered = cached_pd_1d[j]
            
            interaction_term = pd_ij_centered - pd_i_centered - pd_j_centered
            numerator = np.sum(interaction_term ** 2)
            denominator = np.sum(pd_ij_centered ** 2) + 1e-9
            
            h_squared = numerator / denominator
            if h_squared > self.int_ratio:
                G.add_edge(i, j)

        decomposition = []
        visited = set()
        for comp in nx.connected_components(G):
            group = sorted(list(comp))
            decomposition.append(group)
            visited.update(group)
            
        unimportant_group = [i for i in range(D) if i not in visited]
        if unimportant_group:
            decomposition.append(unimportant_group)
            unimp_idx = len(decomposition) - 1
        else:
            unimp_idx = -1
            
        return decomposition, unimp_idx
