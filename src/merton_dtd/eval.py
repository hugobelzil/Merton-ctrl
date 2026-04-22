from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from .config import MertonParams, PolicyParams
from .merton import exact_value


def wealth_grid(low: float, high: float, num: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(low), np.log(high), num))


def evaluate_critic_on_grid(
    critic,
    params: MertonParams,
    policy: PolicyParams,
    low: float,
    high: float,
    num: int,
    device: str = "cpu",
) -> dict[str, np.ndarray | float | dict]:
    grid = wealth_grid(low, high, num)
    with torch.no_grad():
        w = torch.tensor(grid, dtype=torch.float32, device=device)
        pred = critic.value(w).detach().cpu().numpy()
    truth = exact_value(grid, params, policy)
    abs_err = np.abs(pred - truth)
    rel_err = abs_err / np.maximum(np.abs(truth), 1e-12)
    return {
        "wealth": grid,
        "pred": pred,
        "truth": truth,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mape": float(rel_err.mean()),
        "params": asdict(params),
        "policy": asdict(policy),
    }
