import torch
import gpytorch
import math
from typing import List
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, AdditiveKernel
from gpytorch.priors import GammaPrior, LogNormalPrior
from gpytorch.means import ConstantMean

class AdditiveStructureGP(gpytorch.models.ExactGP):
    """
    [Core Model] 加性高斯过程模型。
    已集成高维自适应先验 (LogNormal Prior)。
    """
    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        decomposition: List[List[int]],
        unimportant_group_idx: int = -1,
        use_matern: bool = True,
        lengthscale_prior_mean: float = 0.5  # [兼容性保留] 实际上不再使用此参数
    ):
        if train_Y.dim() > 1:
            train_Y = train_Y.squeeze()
        super().__init__(train_X, train_Y, likelihood)
        
        self._decomposition = decomposition
        self._validate_decomposition(decomposition, train_X.size(-1))

        # Mean Module
        self.mean_module = ConstantMean()
        
        # Kernel Factory
        base_kernel_class = MaternKernel if use_matern else RBFKernel
        base_kernel_kwargs = {"nu": 2.5} if use_matern else {}
        
        sub_kernels = []
        device = train_X.device

        for i, group_dims in enumerate(decomposition):
            active_dims = torch.tensor(group_dims, dtype=torch.long, device=device)
            group_dim_size = len(group_dims)
            
            # === [修改] 自适应高维先验逻辑 ===
            # 参考论文: "Standard GP is All You Need for High-Dimensional Bayesian Optimization"
            # Formula: mu = sqrt(2) + 0.5 * ln(D), sigma = sqrt(3)
            # 这里的 D 是当前子组的维度
            d_val = max(1, group_dim_size)
            prior_mu = math.sqrt(2) + 0.5 * math.log(d_val)
            prior_sigma = math.sqrt(3)
            
            ls_prior = LogNormalPrior(prior_mu, prior_sigma)
            # ===============================
            
            # Base Kernel
            base_kernel = base_kernel_class(
                ard_num_dims=group_dim_size, 
                active_dims=active_dims,
                lengthscale_prior=ls_prior, # 应用自适应先验
                **base_kernel_kwargs
            )
            
            # Scale Kernel
            if i == unimportant_group_idx:
                # Soft Constraint for dummy group
                soft_mask_prior = GammaPrior(0.5, 50.0)
                scale_kernel = ScaleKernel(base_kernel, outputscale_prior=soft_mask_prior)
                scale_kernel.outputscale = 1e-4 
            else:
                active_prior = GammaPrior(2.0, 2.0)
                scale_kernel = ScaleKernel(base_kernel, outputscale_prior=active_prior)
            
            sub_kernels.append(scale_kernel)
        
        self.covar_module = AdditiveKernel(*sub_kernels)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _validate_decomposition(self, decomposition, input_dim):
        all_dims = [d for group in decomposition for d in group]
        if len(all_dims) > 0 and max(all_dims) >= input_dim:
             raise ValueError(f"Decomposition index {max(all_dims)} out of bounds {input_dim}")
