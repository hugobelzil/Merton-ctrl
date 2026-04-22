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
    noise = torch.randn_like(wealth)
    wealth_next = exact_step(wealth, params, policy, dt, noise)
    reward = reward_rate(wealth, params, policy)
    return wealth, wealth_next, reward


def td_residual(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
) -> torch.Tensor:
    V = critic.value(wealth)
    with torch.no_grad():
        V_next = critic.value(wealth_next)
    return reward * dt + math.exp(-params.rho * dt) * V_next - V


def dtd_residual(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
) -> torch.Tensor:
    V, Vw, Vww = critic.value_and_derivatives(wealth)
    delta_w = wealth_next - wealth
    return reward + (delta_w / dt) * Vw + 0.5 * (delta_w.square() / dt) * Vww - params.rho * V


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
    td = td_residual(critic, wealth, wealth_next, reward, params, dt)
    dtd = dtd_residual(critic, wealth, wealth_next, reward, params, dt)

    if loss_name == "td":
        loss = torch.mean(td.square())
    elif loss_name == "dtd":
        loss = torch.mean((dt * dtd).square())
    elif loss_name == "beta_dtd":
        loss = (1.0 - beta) * torch.mean(td.square()) + beta * torch.mean((dt * dtd).square())
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    metrics = {
        "td_mse": float(torch.mean(td.square()).detach().cpu()),
        "dtd_scaled_mse": float(torch.mean((dt * dtd).square()).detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    return loss, metrics
