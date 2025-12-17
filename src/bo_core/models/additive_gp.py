import torch
import gpytorch
from typing import List
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, AdditiveKernel
from gpytorch.priors import GammaPrior
from gpytorch.means import ConstantMean

class AdditiveStructureGP(gpytorch.models.ExactGP):
    """
    [Core Model] 加性高斯过程模型。
    """
    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        decomposition: List[List[int]],
        unimportant_group_idx: int = -1,
        use_matern: bool = True,
        lengthscale_prior_mean: float = 0.5 
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
        
        ls_beta = 6.0
        ls_alpha = lengthscale_prior_mean * ls_beta
        
        sub_kernels = []
        device = train_X.device

        for i, group_dims in enumerate(decomposition):
            active_dims = torch.tensor(group_dims, dtype=torch.long, device=device)
            
            # Base Kernel
            base_kernel = base_kernel_class(
                ard_num_dims=len(group_dims), 
                active_dims=active_dims,
                lengthscale_prior=GammaPrior(ls_alpha, ls_beta), 
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
