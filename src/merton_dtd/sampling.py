from __future__ import annotations

import math

import torch


def sample_log_uniform(batch_size: int, low: float, high: float, device: str = "cpu") -> torch.Tensor:
    if low <= 0.0 or high <= 0.0 or low >= high:
        raise ValueError("Need 0 < low < high for log-uniform sampling")
    u = torch.rand(batch_size, device=device)
    log_low, log_high = math.log(low), math.log(high)
    return torch.exp(log_low + (log_high - log_low) * u)
