import torch
import math
from typing import Callable

# ==============================================================================
# Helper Utilities
# ==============================================================================

def _ensure_tensor(x: torch.Tensor) -> torch.Tensor:
    """Ensures input is at least 2D (Batch, Dim)."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x

# ==============================================================================
# High-Dimensional Wrappers (For Sparse / Dummy dims)
# ==============================================================================

class HighDimWrapper:
    """
    Wraps a core function into a high-dimensional space with effective and dummy dimensions.
    """
    def __init__(self, core_func: Callable, effective_dim: int, total_dim: int):
        self.core_func = core_func
        self.effective_dim = effective_dim
        self.total_dim = total_dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Only use the first 'effective_dim' columns
        return self.core_func(x[..., :self.effective_dim])

# ==============================================================================
# Standard Test Functions (Maximization Mode)
# ==============================================================================

def stybtang_nd(x: torch.Tensor) -> torch.Tensor:
    """
    Styblinski-Tang function (Maximization).
    Global Max is at x ~= -2.903534.
    Range: usually [-5, 5].
    """
    x = _ensure_tensor(x)
    # Original Min: 0.5 * sum(x^4 - 16x^2 + 5x)
    # We return negative for Maximization
    return -0.5 * torch.sum(x**4 - 16 * x**2 + 5 * x, dim=-1)

def michalewicz_nd(x: torch.Tensor, m: int = 10) -> torch.Tensor:
    """
    Michalewicz function (Maximization).
    Range: [0, pi].
    """
    x = _ensure_tensor(x)
    d = x.size(-1)
    i = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    term = torch.sin(x) * torch.pow(torch.sin(i * x**2 / torch.pi), 2 * m)
    return torch.sum(term, dim=-1) 

def rosenbrock_nd(x: torch.Tensor) -> torch.Tensor:
    """
    Rosenbrock function (Maximization).
    Sum of N-1 2D Rosenbrocks.
    Global Max (orig min 0): 0.0 at (1, 1, ..., 1).
    Range: usually [-2.048, 2.048] or [-5, 10].
    """
    x = _ensure_tensor(x)
    
    # X_curr: x_i (first d-1 elements)
    # X_next: x_{i+1} (last d-1 elements)
    X_curr = x[:, :-1]
    X_next = x[:, 1:]
    
    term1 = 100 * (X_next - X_curr**2)**2
    term2 = (1 - X_curr)**2
    
    # Sum over dimensions
    y = torch.sum(term1 + term2, dim=1) # Shape: (Batch,)
    
    # Return negative for maximization
    return -y

def hartmann6d(x: torch.Tensor) -> torch.Tensor:
    """
    Hartmann 6D function (Maximization).
    Standard domain: [0, 1]^6.
    """
    x = _ensure_tensor(x)
    device, dtype = x.device, x.dtype
    
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], device=device, dtype=dtype)
    A = torch.tensor([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ], device=device, dtype=dtype)
    P = 1e-4 * torch.tensor([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ], device=device, dtype=dtype)
    
    # (Batch, 1, 6) - (4, 6) -> (Batch, 4, 6)
    inner_sum = torch.sum(A.unsqueeze(0) * (x.unsqueeze(1) - P.unsqueeze(0))**2, dim=-1)
    return torch.sum(alpha.unsqueeze(0) * torch.exp(-inner_sum), dim=-1)

def ackley_5d_block(x: torch.Tensor) -> torch.Tensor:
    """
    Standard Ackley Function (5D block).
    Global Max (orig min 0): 0.0 at x=[0,0,0,0,0]
    Domain: [-32.768, 32.768]
    """
    x = _ensure_tensor(x)
    d = x.size(-1) 
    
    a = 20; b = 0.2; c = 2 * math.pi
    sum_sq = torch.sum(x**2, dim=-1)
    sum_cos = torch.sum(torch.cos(c * x), dim=-1)
    
    term1 = -a * torch.exp(-b * torch.sqrt(sum_sq / d))
    term2 = -torch.exp(sum_cos / d)
    val = term1 + term2 + a + math.e
    return -val # Maximization

# ==============================================================================
# Block-Additive Functions (Sum of Sub-problems)
# ==============================================================================

def sum_rosenbrock_nd(x: torch.Tensor, block_size: int = 5) -> torch.Tensor:
    """
    Block-Additive Rosenbrock.
    Divides dimension D into blocks of size 'block_size'.
    """
    x = _ensure_tensor(x)
    d = x.size(-1)
    num_blocks = d // block_size
    
    total_val = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    
    for i in range(num_blocks):
        sub_x = x[:, i*block_size : (i+1)*block_size]
        # We reuse the core rosenbrock logic
        # Note: rosenbrock_nd returns -y, so here we sum negative values
        total_val += rosenbrock_nd(sub_x)
        
    return total_val 

def sum_ackley_nd(x: torch.Tensor, block_size: int = 5) -> torch.Tensor:
    """
    Block-Additive Ackley.
    """
    x = _ensure_tensor(x)
    d = x.size(-1)
    num_blocks = d // block_size
    
    total_val = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    
    for i in range(num_blocks):
        sub_x = x[:, i*block_size : (i+1)*block_size]
        total_val += ackley_5d_block(sub_x)
        
    return total_val

def sum_hartmann6d_60d(x: torch.Tensor) -> torch.Tensor:
    """
    Sum of 10 Hartmann6D functions (Total dim = 60).
    """
    x = _ensure_tensor(x)
    total_val = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    # Assuming 60 dims / 6 = 10 blocks
    num_blocks = x.size(-1) // 6
    
    for i in range(num_blocks):
        sub_x = x[:, i*6 : (i+1)*6]
        total_val += hartmann6d(sub_x)
    return total_val

# ==============================================================================
# Function Dispatcher (Factory)
# ==============================================================================

def get_function(name: str) -> Callable:
    """
    根据配置名称获取对应的测试函数。
    支持自动处理 sparse wrapper 和 block_size 参数（通过闭包或 wrapper）。
    """
    # 这里我们只返回基础函数，具体的 wrapper 和 param 绑定逻辑
    # 实际上在 run_experiment.py 和 optimizer.py 中已经有部分处理，
    # 或者我们可以在这里做更高级的封装。
    
    # 简单查表
    lookup = {
        "stybtang_nd": stybtang_nd,
        "michalewicz_nd": michalewicz_nd,
        "rosenbrock_nd": rosenbrock_nd,
        "hartmann6d": hartmann6d,
        "ackley_5d_block": ackley_5d_block,
        "sum_rosenbrock_nd": sum_rosenbrock_nd,
        "sum_ackley_nd": sum_ackley_nd,
        "sum_hartmann6d_60d": sum_hartmann6d_60d,
    }
    
    if name in lookup:
        return lookup[name]
    
    # 别名支持 (Aliases)
    if "stybtang" in name: return stybtang_nd
    if "ackley" in name: return sum_ackley_nd # 默认 Ackley 多为 sum 形式
    if "rosenbrock" in name: return sum_rosenbrock_nd
    
    raise ValueError(f"Unknown function: {name}")

# ==============================================================================
# Wrapper Aliases (Optional, for backward compatibility)
# ==============================================================================
def stybtang_40_dummy_40(x): return HighDimWrapper(stybtang_nd, 40, 80)(x)
def sparse_stybtang_20_in_100(x): return HighDimWrapper(stybtang_nd, 20, 100)(x)
