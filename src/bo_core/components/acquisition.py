import torch
import numpy as np
from scipy.stats import qmc
from typing import Dict, List, Optional

def dynamic_beta(t: int, scaling_factor: float = 2.0) -> float:
    return scaling_factor * np.log(2 * (t + 2))

class ParallelSubspaceOptimizer:
    """
    [Parallel Execution Layer] 并行子空间采集函数优化器。
    """
    def __init__(self, 
                 batch_kernel,           
                 train_x_batch: torch.Tensor, 
                 precomputed_matrices: Dict, 
                 bounds: torch.Tensor,        
                 iteration: int, 
                 device: torch.device, 
                 dtype: torch.dtype,
                 beta_scaling: float = 2.0):
        
        self.kernel = batch_kernel
        self.train_x = train_x_batch
        self.L = precomputed_matrices['L']         
        self.alpha = precomputed_matrices['alpha'] 
        self.bounds = bounds
        self.device = device
        self.dtype = dtype
        
        self.num_batches = train_x_batch.size(0)
        self.n_train = train_x_batch.size(1)
        self.sub_dim = train_x_batch.size(2)
        
        self.beta = dynamic_beta(iteration, scaling_factor=beta_scaling)
        
        self.num_samples = min(10000, max(2000, 2000 * self.sub_dim))
        self.opt_steps = min(200, max(50, 50 + 10 * self.sub_dim))
        self.num_top_k = min(100, max(20, 20 * self.sub_dim))
        self.lr = 0.05

    def _calc_ucb_batch(self, X_cand):
        # K_cand: (B, N_c, N_t)
        K_cand = self.kernel(X_cand, self.train_x).evaluate()
        
        # mu: (B, N_c)
        mu = torch.matmul(K_cand, self.alpha)
        
        # v: (B, N_t, N_c)
        v = torch.linalg.solve_triangular(self.L, K_cand.transpose(-1, -2), upper=False)
        var_red = torch.sum(v.square(), dim=-2)
        
        # own_var: (B, N_c)
        own_var = self.kernel(X_cand, X_cand).diag()
        var = (own_var - var_red).clamp(min=1e-9)
        
        return mu.squeeze(-1) + self.beta * torch.sqrt(var)

    def optimize_step(self):
        lb, ub = self.bounds[0], self.bounds[1]
        
        # Phase 1: Batch LHS (CPU loop, GPU safe)
        sampler = qmc.LatinHypercube(d=self.sub_dim, seed=None)
        cands = torch.empty((self.num_batches, self.num_samples, self.sub_dim), 
                            device=self.device, dtype=self.dtype)
        
        # LHS 采样不需要梯度
        with torch.no_grad():
            for i in range(self.num_batches):
                sample_np = sampler.random(self.num_samples)
                cands[i] = lb + torch.tensor(sample_np, device=self.device, dtype=self.dtype) * (ub - lb)
        
            # Phase 2: Top-K
            ucb_vals = self._calc_ucb_batch(cands)
        
        _, top_indices = ucb_vals.topk(self.num_top_k, dim=-1)
        
        batch_idx = torch.arange(self.num_batches, device=self.device).unsqueeze(1).expand(-1, self.num_top_k)
        cands_opt = cands[batch_idx, top_indices].clone()
        
        # Phase 3: Batch Adam
        cands_opt.requires_grad_(True)
        optimizer = torch.optim.Adam([cands_opt], lr=self.lr)
        
        for _ in range(self.opt_steps):
            optimizer.zero_grad()
            loss = -self._calc_ucb_batch(cands_opt).sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                cands_opt.clamp_(min=lb, max=ub)
        
        # Phase 4: Final Select
        with torch.no_grad():
            final_vals = self._calc_ucb_batch(cands_opt)
            best_indices = final_vals.argmax(dim=-1)
            result = cands_opt[torch.arange(self.num_batches), best_indices]
            
        return result.detach()
