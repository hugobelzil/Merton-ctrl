from __future__ import annotations

import math
from typing import Literal

import torch

from .config import MertonParams, PolicyParams
from .merton import exact_step, reward_rate

LossName = Literal["td", "dtd", "beta_dtd", "rl_pinn"]


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
    t: torch.Tensor | None = None,
    t_next: torch.Tensor | None = None,
    terminal_value_next: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Standard one-step TD residual:
        delta_TD = reward_step + gamma_disc * V(W_{t+dt}) - V(W_t)

    where
        reward_step = reward_rate * dt
        gamma_disc  = exp(-rho * dt)
    """
    V = critic.value(wealth, t)
    reward_step = reward * dt
    gamma_disc = math.exp(-params.rho * dt)

    with torch.no_grad():
        if terminal_value_next is None:
            V_next = critic.value(wealth_next, t_next)
        else:
            V_next = terminal_value_next

    return reward_step + gamma_disc * V_next - V


def dtd_prediction_and_target(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
    t: torch.Tensor | None = None,
    t_next: torch.Tensor | None = None,
    terminal_value_next: torch.Tensor | None = None,
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
    if t is None:
        _, Vw, Vww = critic.value_and_derivatives(wealth)
        time_prediction = 0.0
    else:
        _, Vt, Vw, Vww = critic.value_and_derivatives(wealth, t)
        time_prediction = dt * Vt
    delta_w = wealth_next - wealth
    reward_step = reward * dt

    # Prediction side: derivative/local-dynamics terms at the current state
    prediction = time_prediction + delta_w * Vw + 0.5 * delta_w.square() * Vww

    # Target side: value/reward terms
    with torch.no_grad():
        if terminal_value_next is None:
            V_next = critic.value(wealth_next, t_next)
        else:
            V_next = terminal_value_next

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
    t: torch.Tensor | None = None,
    t_next: torch.Tensor | None = None,
    terminal_value_next: torch.Tensor | None = None,
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
        t=t,
        t_next=t_next,
        terminal_value_next=terminal_value_next,
    )
    return pred - target


def rl_pinn_residual(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
    t: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    PINN-style Bellman residual from the RL_PINN note.

        R_theta(W_t) = V_theta(W_t)
                       - (1/rho) [ U(c_t)
                                   + (DeltaW/dt)        * V_w(W_t)
                                   + 0.5 * (DeltaW)^2/dt * V_ww(W_t) ]

    For finite horizon (critic built with `time_horizon`), pass `t` and the
    Ito expansion of V(t+dt, W_{t+dt}) contributes a V_t term:

        R = V(t,W) - (1/rho)[ U + V_t + (DeltaW/dt) V_w + 0.5 (DeltaW)^2/dt V_ww ].
    """
    delta = wealth_next - wealth
    if t is None:
        V, Vw, Vww = critic.value_and_derivatives(wealth)
        rhs = reward + (delta / dt) * Vw + 0.5 * (delta * delta / dt) * Vww
    else:
        V, Vt, Vw, Vww = critic.value_and_derivatives(wealth, t)
        rhs = reward + Vt + (delta / dt) * Vw + 0.5 * (delta * delta / dt) * Vww
    return V - rhs / params.rho


def compute_loss(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
    loss_name: LossName,
    beta: float = 0.5,
    t: torch.Tensor | None = None,
    t_next: torch.Tensor | None = None,
    terminal_value_next: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Losses:
      - td       : mean(delta_TD^2)
      - dtd      : mean(delta_dTD^2)
      - beta_dtd : (1-beta) * mean(delta_TD^2) + beta * mean(delta_dTD^2)
    """
    td = td_residual(
        critic=critic,
        wealth=wealth,
        wealth_next=wealth_next,
        reward=reward,
        params=params,
        dt=dt,
        t=t,
        t_next=t_next,
        terminal_value_next=terminal_value_next,
    )

    dtd = dtd_residual(
        critic=critic,
        wealth=wealth,
        wealth_next=wealth_next,
        reward=reward,
        params=params,
        dt=dt,
        t=t,
        t_next=t_next,
        terminal_value_next=terminal_value_next,
    )

    td_mse = torch.mean(td.square())
    dtd_mse = torch.mean(dtd.square())
    pinn_mse = torch.tensor(float("nan"))

    if loss_name == "rl_pinn":
        pinn = rl_pinn_residual(
            critic=critic,
            wealth=wealth,
            wealth_next=wealth_next,
            reward=reward,
            params=params,
            dt=dt,
            t=t,
        )
        pinn_mse = torch.mean(pinn.square())
        loss = pinn_mse
    elif loss_name == "td":
        loss = td_mse
    elif loss_name == "dtd":
        loss = dtd_mse
    elif loss_name == "beta_dtd":
        loss = (1.0 - beta) * td_mse + beta * dtd_mse
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    metrics = {
        "td_mse": float(td_mse.detach().cpu()),
        "dtd_mse": float(dtd_mse.detach().cpu()),
        "pinn_mse": float(pinn_mse.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    return loss, metrics
