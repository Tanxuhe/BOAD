import torch
import numpy as np

# 这里是一个示例，您可以直接将您原有的 test_functions.py 内容粘贴覆盖此文件
# 确保函数接收 tensor 输入并返回 tensor 输出

def stybtang_nd(x):
    # x: (batch, dim)
    # y: 0.5 * sum(x^4 - 16x^2 + 5x)
    return 0.5 * (x.pow(4) - 16 * x.pow(2) + 5 * x).sum(dim=-1, keepdim=True)

def ackley(x):
    # Standard Ackley function
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = x.shape[-1]
    
    sum1 = (x ** 2).sum(dim=-1)
    sum2 = (torch.cos(c * x)).sum(dim=-1)
    
    term1 = -a * torch.exp(-b * torch.sqrt(sum1 / d))
    term2 = -torch.exp(sum2 / d)
    
    return term1 + term2 + a + np.exp(1)

def get_function(name):
    name = name.lower()
    if "stybtang" in name:
        return stybtang_nd
    elif "ackley" in name:
        return ackley
    else:
        raise ValueError(f"Unknown function: {name}")
