from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from .config import MertonParams, PolicyParams
from .merton import exact_value, exact_value_coefficient, utility


def wealth_grid(low: float, high: float, num: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(low), np.log(high), num))


def evaluate_critic_on_grid(
    critic,
    params: MertonParams,
    policy: PolicyParams,
    low: float,
    high: float,
    num: int,
    dt: float | None = None,
    device: str = "cpu",
) -> dict[str, np.ndarray | float | dict]:
    """
    Evaluate critic against the closed-form value, and additionally report
    diagnostics that test the noise-bias hypothesis for pure dTD:

      - v_w_mae       : MAE of V_w(w) vs the closed-form V_w(w) = A * w^{-gamma}
      - v_w_norm      : RMS of V_w(w) under the learned critic, on the eval grid
      - v_w_norm_true : RMS of V_w(w) under the closed-form solution, same grid
      - hjb_rmse      : RMS of the (deterministic) HJB residual
                         L^pi V - rho V + U(kappa w),  L^pi V = a w V_w + 0.5 b^2 w^2 V_ww
      - dtd_noise_floor : analytic per-sample noise variance V_w^2 * pi^2 sigma^2 w^2 * dt,
                          averaged over the grid (only computed if dt is given)
    """
    grid = wealth_grid(low, high, num)
    w = torch.tensor(grid, dtype=torch.float32, device=device, requires_grad=True)

    V, Vw, Vww = critic.value_and_derivatives(w)
    pred = V.detach().cpu().numpy()
    Vw_np = Vw.detach().cpu().numpy()
    Vww_np = Vww.detach().cpu().numpy()

    truth = exact_value(grid, params, policy)
    abs_err = np.abs(pred - truth)
    rel_err = abs_err / np.maximum(np.abs(truth), 1e-12)

    # Closed-form derivatives:  V(w) = A * w^{1-g} / (1-g)
    # =>  V_w  = A * w^{-g},   V_ww = -g * A * w^{-g-1}
    g = params.gamma
    A = exact_value_coefficient(params, policy)
    Vw_truth = A * np.power(grid, -g)
    Vww_truth = -g * A * np.power(grid, -g - 1.0)
    v_w_abs_err = np.abs(Vw_np - Vw_truth)

    # Deterministic HJB residual at the learned critic:
    #   L^pi V (w) = a w V_w(w) + 0.5 b^2 w^2 V_ww(w)
    #   HJB(w)    = L^pi V (w) - rho V(w) + U(kappa w)
    a = params.r + policy.pi * (params.mu - params.r) - policy.kappa
    b = policy.pi * params.sigma
    LV = a * grid * Vw_np + 0.5 * (b ** 2) * (grid ** 2) * Vww_np
    U = utility(policy.kappa * grid, g)
    hjb = LV - params.rho * pred + U

    out: dict[str, np.ndarray | float | dict] = {
        "wealth": grid,
        "pred": pred,
        "truth": truth,
        "Vw_pred": Vw_np,
        "Vw_truth": Vw_truth,
        "Vww_pred": Vww_np,
        "Vww_truth": Vww_truth,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mape": float(rel_err.mean()),
        "v_w_mae": float(v_w_abs_err.mean()),
        "v_w_norm": float(np.sqrt(np.mean(Vw_np ** 2))),
        "v_w_norm_true": float(np.sqrt(np.mean(Vw_truth ** 2))),
        "hjb_rmse": float(np.sqrt(np.mean(hjb ** 2))),
        "params": asdict(params),
        "policy": asdict(policy),
    }

    if dt is not None:
        # Analytic per-sample noise variance of the dTD residual (leading order):
        #   Var(Delta W * V_w) = V_w^2 * pi^2 sigma^2 w^2 * dt
        noise_floor = (Vw_np ** 2) * (policy.pi ** 2) * (params.sigma ** 2) * (grid ** 2) * dt
        out["dtd_noise_floor"] = float(noise_floor.mean())

    return out
