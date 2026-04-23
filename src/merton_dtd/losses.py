from __future__ import annotations

import math
from typing import Literal

import torch

from .config import MertonParams, PolicyParams
from .merton import exact_step, reward_rate

LossName = Literal["td", "dtd", "beta_dtd"]


def make_batch(
    wealth: torch.Tensor,
    params: MertonParams,
    policy: PolicyParams,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample one exact step under the fixed constant policy.

    Returns
    -------
    wealth      : W_t
    wealth_next : W_{t+dt}
    reward      : instantaneous reward rate r_t = U(c_t)
    """
    noise = torch.randn_like(wealth)
    wealth_next = exact_step(wealth, params, policy, dt, noise)
    reward = reward_rate(wealth, params, policy)  # reward rate, not yet multiplied by dt
    return wealth, wealth_next, reward


def td_residual(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
) -> torch.Tensor:
    """
    Standard one-step TD residual:
        delta_TD = reward_step + gamma_disc * V(W_{t+dt}) - V(W_t)

    where
        reward_step = reward_rate * dt
        gamma_disc  = exp(-rho * dt)
    """
    V = critic.value(wealth)
    reward_step = reward * dt
    gamma_disc = math.exp(-params.rho * dt)

    with torch.no_grad():
        V_next = critic.value(wealth_next)

    return reward_step + gamma_disc * V_next - V


def dtd_prediction_and_target(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Practical dTD decomposition matching the paper's useful form.

    For one-dimensional state W, define:
        prediction = ΔW * V_w(W_t) + 0.5 * (ΔW)^2 * V_ww(W_t)

        target = - reward_step + rho * dt * V(W_{t+dt})

    so the dTD error is:
        prediction - target

    Notes
    -----
    - reward passed in is an instantaneous reward rate, so we multiply by dt here.
    - V(W_{t+dt}) is treated as a target (detached / no-grad).
    - This is deliberately different from the old "raw residual" implementation.
    """
    _, Vw, Vww = critic.value_and_derivatives(wealth)
    delta_w = wealth_next - wealth
    reward_step = reward * dt

    # Prediction side: derivative/local-dynamics terms at the current state
    prediction = delta_w * Vw + 0.5 * delta_w.square() * Vww

    # Target side: value/reward terms
    with torch.no_grad():
        V_next = critic.value(wealth_next)

    # Since gamma_disc = exp(-rho dt), we have -log(gamma_disc) = rho dt
    target = -reward_step + (params.rho * dt) * V_next

    return prediction, target


def dtd_residual(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
) -> torch.Tensor:
    """
    Practical dTD error:
        delta_dTD = prediction - target
    """
    pred, target = dtd_prediction_and_target(
        critic=critic,
        wealth=wealth,
        wealth_next=wealth_next,
        reward=reward,
        params=params,
        dt=dt,
    )
    return pred - target


def compute_loss(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
    loss_name: LossName,
    beta: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Losses:
      - td       : mean(delta_TD^2)
      - dtd      : mean(delta_dTD^2)
      - beta_dtd : (1-beta) * mean(delta_TD^2) + beta * mean(delta_dTD^2)

    We keep the metric key name 'dtd_scaled_mse' only so the rest of the repo
    does not break, even though it is now just the dTD MSE (not a scaled raw residual).
    """
    td = td_residual(
        critic=critic,
        wealth=wealth,
        wealth_next=wealth_next,
        reward=reward,
        params=params,
        dt=dt,
    )

    dtd = dtd_residual(
        critic=critic,
        wealth=wealth,
        wealth_next=wealth_next,
        reward=reward,
        params=params,
        dt=dt,
    )

    td_mse = torch.mean(td.square())
    dtd_mse = torch.mean(dtd.square())

    if loss_name == "td":
        loss = td_mse
    elif loss_name == "dtd":
        loss = dtd_mse
    elif loss_name == "beta_dtd":
        loss = (1.0 - beta) * td_mse + beta * dtd_mse
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    metrics = {
        "td_mse": float(td_mse.detach().cpu()),
        # keep old key name for compatibility with training.py / plotting.py
        "dtd_scaled_mse": float(dtd_mse.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    return loss, metrics