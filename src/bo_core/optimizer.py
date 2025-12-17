import torch
import time
import gc
import pprint
from tqdm import tqdm
from typing import Callable, Optional

# --- Absolute Imports (Standardized) ---
from bo_core.config_manager import FullConfig
from bo_core.models.additive_gp import AdditiveStructureGP
from bo_core.components.acquisition import ParallelSubspaceOptimizer
from bo_core.components.decomposition import SHAPDecompositionFinder, FriedmanHDecompositionFinder
from bo_core.utils import (
    setup_logger, JSONLLogger, AdaptiveFrequencyScheduler, expand_tensor
)

# --- BoTorch / GPyTorch ---
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel

class AdaptiveBO:
    def __init__(
        self,
        config: FullConfig,
        X_init: torch.Tensor,
        Y_init: torch.Tensor,
        bounds: torch.Tensor,
        func: Callable,
        decomposition: list,
        output_dir: str
    ):
        """
        初始化 AdaptiveBO
        :param config: 强类型的全参数配置对象
        """
        self.cfg = config
        self.device = torch.device(config.experiment.device)
        self.dtype = torch.double
        self.exp_dir = output_dir
        
        # 日志系统
        self.logger = setup_logger(self.exp_dir)
        self.json_logger = JSONLLogger(self.exp_dir)
        
        # 基础数据
        self.func = func
        self.bounds = bounds.to(self.device, self.dtype)
        self.x_min, self.x_range = self.bounds[0], self.bounds[1] - self.bounds[0]
        self.dim = X_init.size(1)
        
        self.X_train = X_init.to(self.device, self.dtype)
        self.Y_train = Y_init.to(self.device, self.dtype)
        
        # 状态追踪
        self.decomp = decomposition
        self.unimp_idx = -1 # 默认初始无无效组
        self.model = None
        self.best_value_trace = []
        self._update_norm()
        self._update_trace()

        # 调度器
        self.freq_scheduler = AdaptiveFrequencyScheduler(
            initial_freq=config.algorithm.decomp_freq,
            loose_freq=config.algorithm.decomp_freq * 2,
            switch_point=150
        )
        
        self.logger.info(f"Initialized AdaptiveBO with dim={self.dim}, device={self.device}")

    def _update_trace(self):
        valid_y = self.Y_train[torch.isfinite(self.Y_train)]
        current_best = valid_y.max().item() if len(valid_y) > 0 else -float('inf')
        self.best_value_trace.append(current_best)

    def _update_norm(self):
        valid_mask = torch.isfinite(self.Y_train)
        if valid_mask.any():
            valid_Y = self.Y_train[valid_mask]
            self.y_mean = valid_Y.mean()
            self.y_std = valid_Y.std()
            if self.y_std < 1e-6: self.y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        else:
            self.y_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            self.y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)

    def _log_model_diagnostics(self):
        """[新增] 增强日志：记录 GP 超参数"""
        if self.model is None: return
        
        try:
            noise = self.model.likelihood.noise.item()
            ls_values = []
            os_values = []
            
            for k in self.model.covar_module.kernels:
                # k is ScaleKernel -> BaseKernel
                ls = k.base_kernel.lengthscale.mean().item()
                os_val = k.outputscale.item()
                ls_values.append(ls)
                os_values.append(os_val)
            
            avg_ls = sum(ls_values) / len(ls_values)
            avg_os = sum(os_values) / len(os_values)
            
            self.logger.info(f"   [Model Diag] Noise: {noise:.4f} | Avg LS: {avg_ls:.4f} | Avg OS: {avg_os:.4f}")
        except Exception:
            pass

    def _train_model_adam(self, is_warm_start=False):
        self.model.train()
        self.model.likelihood.train()
        
        # 提取可训练参数
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=0.1)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        
        N = self.X_train.size(0)
        base_iter = 100 if N < 50 else (150 if N < 150 else 250)
        # [热启动优化] 减少迭代步数
        max_iter = int(base_iter * 0.4) if is_warm_start else base_iter
        
        patience = 15
        best_loss = float('inf')
        no_improv = 0
        
        # 训练循环
        with torch.enable_grad():
            for i in range(max_iter):
                optimizer.zero_grad()
                try:
                    output = self.model(self.model.train_inputs[0])
                    loss = -mll(output, self.model.train_targets)
                    if torch.isnan(loss): break
                    loss.backward()
                    optimizer.step()
                    
                    if loss.item() < best_loss - 1e-4:
                        best_loss = loss.item()
                        no_improv = 0
                    else:
                        no_improv += 1
                    if no_improv >= patience: break
                except RuntimeError:
                    break

    def _create_and_train_model(self, is_warm_start=False):
        # 1. 归一化数据
        X_norm = (self.X_train - self.x_min) / self.x_range
        Y_norm = (self.Y_train - self.y_mean) / self.y_std
        if Y_norm.dim() == 2: Y_norm = Y_norm.squeeze(-1)
        
        # 2. 尝试热启动
        if is_warm_start and self.model is not None:
            try:
                self.model.set_train_data(inputs=X_norm, targets=Y_norm, strict=False)
                self._train_model_adam(is_warm_start=True)
                self.logger.info("   [Model] Warm start completed.")
                self._log_model_diagnostics()
                return
            except Exception as e:
                self.logger.warning(f"   [Model] Warm start failed ({e}), falling back to cold start.")
        
        # 3. 冷启动：显存清理
        if self.model is not None:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=gpytorch.priors.GammaPrior(1.1, 0.05)
        ).to(device=self.device, dtype=self.dtype)
        
        self.model = AdditiveStructureGP(
            train_X=X_norm, train_Y=Y_norm, likelihood=likelihood,
            decomposition=self.decomp, unimportant_group_idx=self.unimp_idx,
            use_matern=True, 
            lengthscale_prior_mean=self.cfg.algorithm.ls_prior
        ).to(device=self.device, dtype=self.dtype)
        
        self._train_model_adam(is_warm_start=False)
        self.logger.info("   [Model] Cold start completed.")
        self._log_model_diagnostics()

    def _create_batch_kernel_from_base(self, group_indices, dim_size, batch_shape):
        """[修复] 维度对齐与一致性 Bug Fix"""
        template_scale = self.model.covar_module.kernels[group_indices[0]]
        template_base = template_scale.base_kernel
        
        sub_kernels = self.model.covar_module.kernels
        
        # Extract Params
        raw_ls = torch.cat([sub_kernels[idx].base_kernel.raw_lengthscale.detach() for idx in group_indices], dim=0)
        raw_ls = raw_ls.view(*batch_shape, 1, dim_size)
        
        raw_os = torch.stack([sub_kernels[idx].raw_outputscale.detach() for idx in group_indices], dim=0)
        raw_os = raw_os.view(*batch_shape)
        
        # Dynamic Cloning
        kernel_cls = type(template_base)
        kwargs = {}
        if hasattr(template_base, 'nu'): kwargs['nu'] = template_base.nu
        
        new_base = kernel_cls(ard_num_dims=dim_size, batch_shape=torch.Size(batch_shape), **kwargs)
        new_base.raw_lengthscale = torch.nn.Parameter(raw_ls)
        
        new_scale = ScaleKernel(new_base, batch_shape=torch.Size(batch_shape))
        new_scale.raw_outputscale = torch.nn.Parameter(raw_os)
        
        return new_scale.to(self.device, self.dtype)

    def _get_next_query_point_custom(self, iteration):
        # ... (Precompute matrices logic inline for brevity or reuse _precompute method) ...
        self.model.eval()
        X_tr, Y_tr = self.model.train_inputs[0], self.model.train_targets
        with torch.no_grad():
            K = self.model.covar_module(X_tr).evaluate()
            noise = self.model.likelihood.noise
            L = torch.linalg.cholesky(K + torch.eye(K.size(0), device=self.device) * (noise + 1e-4))
            alpha = torch.cholesky_solve(Y_tr.unsqueeze(1), L)
        
        matrices = {'L': L, 'alpha': alpha}
        X_next_norm = torch.full((1, self.dim), 0.5, device=self.device, dtype=self.dtype)
        
        # Grouping
        from collections import defaultdict
        buckets = defaultdict(list)
        random_groups = []
        sub_kernels = self.model.covar_module.kernels
        
        for i, dec in enumerate(self.decomp):
            is_unimp = (i == self.unimp_idx)
            if is_unimp and sub_kernels[i].outputscale.item() < 0.05:
                random_groups.append(dec)
            else:
                buckets[len(dec)].append((i, dec))
                
        # Random fill
        for dec in random_groups:
            X_next_norm[:, dec] = torch.rand((1, len(dec)), device=self.device)
            
        # Parallel Opt
        num_groups = len(self.decomp)
        scaled_beta = self.cfg.algorithm.beta_scaling / (np.sqrt(num_groups) if num_groups > 0 else 1.0)
        
        for dim_size, group_info in buckets.items():
            indices = [x[0] for x in group_info]
            dims_list = [x[1] for x in group_info]
            batch_size = len(indices)
            
            x_slices = [X_tr[:, dims] for dims in dims_list]
            train_x_batch = torch.stack(x_slices, dim=0)
            
            batch_kernel = self._create_batch_kernel_from_base(indices, dim_size, [batch_size])
            bounds_sub = torch.tensor([[0.0]*dim_size, [1.0]*dim_size], device=self.device, dtype=self.dtype)
            
            opt = ParallelSubspaceOptimizer(
                batch_kernel, train_x_batch, matrices, bounds_sub, 
                iteration, self.device, self.dtype, scaled_beta
            )
            candidates = opt.optimize_step()
            
            for i, dims in enumerate(dims_list):
                X_next_norm[:, dims] = candidates[i:i+1]
                
        return X_next_norm * self.x_range + self.x_min

    def optimize(self):
        """主优化循环"""
        self.logger.info(f"=== Optimization Started: {self.cfg.experiment.name} ===")
        pbar = tqdm(range(self.X_train.size(0), self.cfg.optimization.n_total))
        
        for i in pbar:
            # 1. 显存管理
            if i % 5 == 0: gc.collect(); torch.cuda.empty_cache()
            
            phase = "StdBO" if i < self.cfg.optimization.switch_threshold else "AdaptBO"
            
            try:
                if phase == "StdBO":
                    # Standard BO (Simplified for brevity, use SingleTaskGP)
                    # Note: In real implementation, insert Standard BO logic here 
                    # For now, let's assume random if not implemented, or copy from previous bo_optimizer.py
                    X_next = torch.rand((1, self.dim), device=self.device, dtype=self.dtype) * self.x_range + self.x_min
                else:
                    # Adaptive Phase
                    steps = i - self.cfg.optimization.switch_threshold
                    
                    # A. 结构学习
                    updated = False
                    if self.freq_scheduler.check_trigger(steps):
                        self.logger.info(f"[Decomp] Running {self.cfg.algorithm.decomposition_method}...")
                        
                        # Clean Memory before heavy computation
                        self.model = None; gc.collect(); torch.cuda.empty_cache()
                        
                        Finder = SHAPDecompositionFinder if self.cfg.algorithm.decomposition_method == "shap" else FriedmanHDecompositionFinder
                        # Pass Friedman Params from Config
                        finder_kwargs = {}
                        if self.cfg.algorithm.decomposition_method == "friedman":
                            finder_kwargs = {
                                "background_source": self.cfg.algorithm.background_source,
                                "data_sample_ratio": self.cfg.algorithm.data_sample_ratio,
                                "grid_sample_count": self.cfg.algorithm.grid_sample_count
                            }
                            
                        finder = Finder(
                            self.X_train, self.Y_train,
                            importance_threshold_ratio=self.cfg.algorithm.importance_threshold,
                            interaction_threshold_ratio=self.cfg.algorithm.interaction_threshold,
                            seed=self.cfg.experiment.seed + i,
                            **finder_kwargs
                        )
                        self.decomp, self.unimp_idx = finder.find()
                        updated = True
                        self.logger.info(f"[Decomp] Found {len(self.decomp)} groups.")
                    
                    # B. 训练模型 (决策: 冷启动 vs 热启动)
                    cold_interval = self.cfg.optimization.cold_start_interval
                    need_cold = updated or (self.model is None) or (cold_interval > 0 and i % cold_interval == 0)
                    
                    self._create_and_train_model(is_warm_start=not need_cold)
                    
                    # C. 采集
                    X_next = self._get_next_query_point_custom(i)
                
                # Evaluation
                if not isinstance(X_next, torch.Tensor): X_next = torch.tensor(X_next, device=self.device)
                Y_next = self.func(X_next).reshape(1).to(self.device)
                
                # Update
                self.X_train = torch.cat([self.X_train, X_next], dim=0)
                self.Y_train = torch.cat([self.Y_train, Y_next], dim=0)
                self._update_norm()
                self._update_trace()
                
                # Log
                cur_best = self.best_value_trace[-1]
                self.json_logger.log({"iter": i, "y": Y_next.item(), "best": cur_best})
                pbar.set_postfix({"Best": f"{cur_best:.4f}"})
                
            except Exception as e:
                self.logger.error(f"Error at {i}: {e}")
                gc.collect()
                # Fail-safe random
                X_next = torch.rand((1, self.dim), device=self.device) * self.x_range + self.x_min
                self.X_train = torch.cat([self.X_train, X_next], dim=0)
                self.Y_train = torch.cat([self.Y_train, torch.tensor([-100.0], device=self.device)], dim=0)
