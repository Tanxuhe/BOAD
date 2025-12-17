import torch
import time
import gc
import pprint
import numpy as np
from tqdm import tqdm
from typing import Callable, Optional

# [关键] 导入 gpytorch 防止 NameError
import gpytorch 

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
        
        # 读取理论最优值用于计算 Regret
        self.optimal_value = config.problem.optimal_value
        if self.optimal_value is None:
            self.logger.warning("No optimal_value provided in config. Regret cannot be calculated.")
        
        # 状态追踪
        self.decomp = decomposition
        self.unimp_idx = -1 # 默认初始无无效组
        
        # [Oracle Logic] 判断是否固定结构
        self.fixed_decomp = (self.cfg.oracle is not None and self.cfg.oracle.type is not None)
        if self.fixed_decomp:
            self.logger.info(f"[Oracle] Decomposition is FIXED. Type: {self.cfg.oracle.type}")
            if self.cfg.oracle.type == 'sparse':
                self.unimp_idx = len(self.decomp) - 1
        
        self.model = None
        self.best_value_trace = []
        self._update_norm()
        self._update_trace()

        # 调度器
        initial_freq = config.algorithm.decomp_freq
        loose_freq = config.algorithm.decomp_loose_freq
        if loose_freq is None:
            loose_freq = int(initial_freq * 2)
        switch_point = config.algorithm.decomp_switch_point

        self.freq_scheduler = AdaptiveFrequencyScheduler(
            initial_freq=initial_freq,
            loose_freq=loose_freq,
            switch_point=switch_point
        )
        
        self.logger.info(f"Initialized AdaptiveBO with dim={self.dim}, device={self.device}")
        self.logger.info(f"[Scheduler] Freq: {initial_freq} -> {loose_freq} after iter {switch_point}")

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
        """增强日志：记录 GP 超参数"""
        if self.model is None: return
        try:
            noise = self.model.likelihood.noise.item()
            ls_values = []
            os_values = []
            for k in self.model.covar_module.kernels:
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
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=0.1)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        
        N = self.X_train.size(0)
        base_iter = 100 if N < 50 else (150 if N < 150 else 250)
        max_iter = int(base_iter * 0.4) if is_warm_start else base_iter
        
        patience = 15
        best_loss = float('inf')
        no_improv = 0
        
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
        X_norm = (self.X_train - self.x_min) / self.x_range
        Y_norm = (self.Y_train - self.y_mean) / self.y_std
        if Y_norm.dim() == 2: Y_norm = Y_norm.squeeze(-1)
        
        # 尝试热启动
        if is_warm_start and self.model is not None:
            try:
                self.model.set_train_data(inputs=X_norm, targets=Y_norm, strict=False)
                self._train_model_adam(is_warm_start=True)
                self.logger.info("   [Model] Warm start completed.")
                self._log_model_diagnostics()
                return
            except Exception as e:
                self.logger.warning(f"   [Model] Warm start failed ({e}), falling back to cold start.")
        
        # 冷启动
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

    def _run_standard_bo_step(self):
        # 确保 mask 是 1D 的 (N,)，以匹配 X_train 的第一维
        valid_mask = torch.isfinite(self.Y_train).view(-1)
        
        train_x = self.X_train[valid_mask]
        train_y = self.Y_train[valid_mask]
        
        train_x_norm = normalize(train_x, self.bounds)
        
        y_std_val = train_y.std()
        if y_std_val < 1e-9: y_std_val = 1.0
        train_y_std = (train_y - train_y.mean()) / y_std_val
        if train_y_std.dim() == 1: train_y_std = train_y_std.unsqueeze(-1)
            
        model = SingleTaskGP(train_x_norm, train_y_std)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        beta = 2.0 * np.log(2 * (train_x.size(0) + 2))
        acq = UpperConfidenceBound(model, beta=beta)
        
        candidates, _ = optimize_acqf(
            acq, 
            bounds=torch.tensor([[0.0]*self.dim, [1.0]*self.dim], device=self.device, dtype=self.dtype),
            q=1, num_restarts=10, raw_samples=512
        )
        del model; del acq
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return unnormalize(candidates, self.bounds)

    def _create_batch_kernel_from_base(self, group_indices, dim_size, batch_shape):
        template_scale = self.model.covar_module.kernels[group_indices[0]]
        template_base = template_scale.base_kernel
        sub_kernels = self.model.covar_module.kernels
        
        raw_ls = torch.cat([sub_kernels[idx].base_kernel.raw_lengthscale.detach() for idx in group_indices], dim=0)
        raw_ls = raw_ls.view(*batch_shape, 1, dim_size)
        
        raw_os = torch.stack([sub_kernels[idx].raw_outputscale.detach() for idx in group_indices], dim=0)
        raw_os = raw_os.view(*batch_shape)
        
        kernel_cls = type(template_base)
        kwargs = {}
        if hasattr(template_base, 'nu'): kwargs['nu'] = template_base.nu
        
        new_base = kernel_cls(ard_num_dims=dim_size, batch_shape=torch.Size(batch_shape), **kwargs)
        new_base.raw_lengthscale = torch.nn.Parameter(raw_ls)
        
        new_scale = ScaleKernel(new_base, batch_shape=torch.Size(batch_shape))
        new_scale.raw_outputscale = torch.nn.Parameter(raw_os)
        return new_scale.to(self.device, self.dtype)

    def _get_next_query_point_custom(self, iteration):
        self.model.eval()
        X_tr, Y_tr = self.model.train_inputs[0], self.model.train_targets
        with torch.no_grad():
            K = self.model.covar_module(X_tr).evaluate()
            noise = self.model.likelihood.noise
            eye = torch.eye(K.size(0), device=self.device)
            L = None
            for jit in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
                try: 
                    L = torch.linalg.cholesky(K + eye * (noise + jit))
                    break
                except: continue
            if L is None:
                L = torch.eye(K.size(0), device=self.device) * torch.sqrt(noise + K.diag().mean())
            alpha = torch.cholesky_solve(Y_tr.unsqueeze(1), L)
        
        matrices = {'L': L, 'alpha': alpha}
        X_next_norm = torch.full((1, self.dim), 0.5, device=self.device, dtype=self.dtype)
        
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
        
        for dec in random_groups:
            # [修正点 1] 添加 dtype=self.dtype 防止类型不匹配
            X_next_norm[:, dec] = torch.rand((1, len(dec)), device=self.device, dtype=self.dtype)
            
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
            if i % 5 == 0: gc.collect(); torch.cuda.empty_cache()
            phase = "StdBO" if i < self.cfg.optimization.switch_threshold else "AdaptBO"
            
            try:
                if phase == "StdBO":
                    X_next = self._run_standard_bo_step()
                else:
                    steps = i - self.cfg.optimization.switch_threshold
                    updated = False
                    
                    if not self.fixed_decomp and self.freq_scheduler.check_trigger(steps):
                        self.logger.info(f"[Decomp] Running {self.cfg.algorithm.decomposition_method}...")
                        self.model = None; gc.collect(); torch.cuda.empty_cache()
                        
                        Finder = SHAPDecompositionFinder if self.cfg.algorithm.decomposition_method == "shap" else FriedmanHDecompositionFinder
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
                        
                        effective_groups = [sorted(g) for idx, g in enumerate(self.decomp) if idx != self.unimp_idx]
                        effective_groups.sort(key=lambda x: x[0] if x else 0)
                        
                        log_msg = {
                            "iter": i, "event": "decomposition", 
                            "method": self.cfg.algorithm.decomposition_method,
                            "groups": effective_groups
                        }
                        self.json_logger.log(log_msg)
                        self.logger.info(f"[Decomp] Found {len(self.decomp)} groups. Updated.")

                    cold_interval = self.cfg.optimization.cold_start_interval
                    need_cold = updated or (self.model is None) or (cold_interval > 0 and i % cold_interval == 0)
                    self._create_and_train_model(is_warm_start=not need_cold)
                    X_next = self._get_next_query_point_custom(i)
                
                # Evaluation
                if not isinstance(X_next, torch.Tensor): X_next = torch.tensor(X_next, device=self.device)
                
                # 确保 Y_next 是 2D (1, 1)，以匹配 Y_train
                Y_next = self.func(X_next).reshape(1, 1).to(self.device)
                
                # Update
                self.X_train = torch.cat([self.X_train, X_next], dim=0)
                self.Y_train = torch.cat([self.Y_train, Y_next], dim=0)
                self._update_norm()
                self._update_trace()
                
                cur_best = self.best_value_trace[-1]
                
                # Regret 计算与记录
                regret_val = None
                if self.optimal_value is not None:
                    # Regret = Optimal - Current Best (Assuming maximization)
                    regret_val = self.optimal_value - cur_best
                
                self.json_logger.log({
                    "iter": i, 
                    "phase": phase, 
                    "y_new": Y_next.item(), 
                    "y_best": cur_best,
                    "regret": regret_val
                })
                
                log_str = f"Best: {cur_best:.4f}"
                if regret_val is not None:
                    log_str += f" | Regret: {regret_val:.4e}"
                
                pbar.set_postfix_str(log_str)
                
            except Exception as e:
                self.logger.error(f"Error at {i}: {e}")
                gc.collect()
                # [修正点 2] Fail-safe 随机生成也需要指定 dtype
                X_next = torch.rand((1, self.dim), device=self.device, dtype=self.dtype) * self.x_range + self.x_min
                self.X_train = torch.cat([self.X_train, X_next], dim=0)
                # 兜底数据也必须是 (1, 1)
                self.Y_train = torch.cat([self.Y_train, torch.tensor([[-100.0]], device=self.device, dtype=self.dtype)], dim=0)
