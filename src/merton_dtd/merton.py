from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from .config import MertonParams, PolicyParams


def utility(consumption: torch.Tensor | np.ndarray | float, gamma: float):
    """CRRA utility U(c) = c^(1-gamma)/(1-gamma), gamma != 1."""
    if isinstance(consumption, np.ndarray):
        c = np.maximum(consumption, 1e-12)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    if isinstance(consumption, torch.Tensor):
        c = torch.clamp(consumption, min=1e-12)
        return torch.pow(c, 1.0 - gamma) / (1.0 - gamma)
    c = max(float(consumption), 1e-12)
    return c ** (1.0 - gamma) / (1.0 - gamma)


def exact_value_coefficient(params: MertonParams, policy: PolicyParams) -> float:
    """Closed-form value coefficient for a fixed constant policy.

    For a constant policy (pi, kappa), the exact value is
        V(w) = A * w^(1-gamma) / (1-gamma),
    with
        A = kappa^(1-gamma) / D,
        D = rho - (1-gamma) * [r + pi (mu-r) - kappa - 0.5 * gamma * pi^2 * sigma^2].

    D must be strictly positive for the infinite-horizon discounted value to be finite.
    """
    g = params.gamma
    drift_term = (
        params.r
        + policy.pi * (params.mu - params.r)
        - policy.kappa
        - 0.5 * g * (policy.pi ** 2) * (params.sigma ** 2)
    )
    denom = params.rho - (1.0 - g) * drift_term
    if denom <= 0.0:
        raise ValueError(
            "This policy is not admissible for the chosen parameters: the infinite-horizon "
            "value is not finite because the denominator is non-positive."
        )
    return policy.kappa ** (1.0 - g) / denom


def exact_value(wealth: torch.Tensor | np.ndarray | float, params: MertonParams, policy: PolicyParams):
    coeff = exact_value_coefficient(params, policy)
    g = params.gamma
    if isinstance(wealth, np.ndarray):
        w = np.maximum(wealth, 1e-12)
        return coeff * np.power(w, 1.0 - g) / (1.0 - g)
    if isinstance(wealth, torch.Tensor):
        w = torch.clamp(wealth, min=1e-12)
        return coeff * torch.pow(w, 1.0 - g) / (1.0 - g)
    w = max(float(wealth), 1e-12)
    return coeff * (w ** (1.0 - g)) / (1.0 - g)


def optimal_policy_closed_form(params: MertonParams) -> Tuple[PolicyParams, float]:
    """Closed-form optimal constant policy and its value coefficient.

    The infinite-horizon CRRA Merton solution is:
        pi* = (mu-r) / (gamma sigma^2)
        kappa* = [rho - (1-gamma)(r + 0.5 * (mu-r)^2 / (gamma sigma^2))] / gamma
    The policy is admissible only if kappa* > 0.
    """
    g = params.gamma
    pi_star = (params.mu - params.r) / (g * params.sigma ** 2)
    sharpe_term = 0.5 * ((params.mu - params.r) ** 2) / (g * params.sigma ** 2)
    kappa_star = (params.rho - (1.0 - g) * (params.r + sharpe_term)) / g
    if kappa_star <= 0.0:
        raise ValueError(
            "The chosen parameters do not produce a positive optimal consumption rate. "
            "Increase rho or modify (mu, r, sigma, gamma)."
        )
    policy = PolicyParams(pi=pi_star, kappa=kappa_star)
    coeff = exact_value_coefficient(params, policy)
    return policy, coeff


def is_policy_admissible(params: MertonParams, policy: PolicyParams) -> bool:
    try:
        _ = exact_value_coefficient(params, policy)
        return True
    except ValueError:
        return False


def risky_weight_to_sharpe_ratio(params: MertonParams, pi: float) -> float:
    return pi * params.sigma


def exact_step(
    wealth: torch.Tensor,
    params: MertonParams,
    policy: PolicyParams,
    dt: float,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """One exact step under a constant policy.

    Since dW/W is affine in dt and dB under a constant policy, the wealth process is geometric:
        W_{t+dt} = W_t exp((a - 0.5 b^2)dt + b sqrt(dt) eps)
    with a = r + pi(mu-r) - kappa and b = pi sigma.
    """
    if noise is None:
        noise = torch.randn_like(wealth)
    a = params.r + policy.pi * (params.mu - params.r) - policy.kappa
    b = policy.pi * params.sigma
    return wealth * torch.exp((a - 0.5 * b * b) * dt + b * math.sqrt(dt) * noise)


def reward_rate(wealth: torch.Tensor, params: MertonParams, policy: PolicyParams) -> torch.Tensor:
    consumption = policy.kappa * wealth
    return utility(consumption, params.gamma)
