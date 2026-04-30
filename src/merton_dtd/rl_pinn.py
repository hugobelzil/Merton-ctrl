"""
PINN-style Bellman residual training, following the RL_PINN note.

Differences from streaming dTD:
  * A frozen dataset is sampled up front: m1 initial states, each rolled out
    for m2 trajectories of length N under the fixed policy. The training
    loop then iterates SGD epochs over this dataset, like a PINN.
  * The residual is the full infinitesimal HJB consistency,
        R = V_theta(W) - (1/rho)[ U(c) + (DeltaW/dt) V_w + 0.5 (DeltaW)^2/dt V_ww ],
    with V_theta and its derivatives all evaluated at the current state.
    There is no detached bootstrap target and no V_theta(W_{t+dt}) term.

When a `HorizonConfig` is passed, the trainer switches to the finite-horizon
variant: the critic is V_theta(t, W) (same `VanillaMLPCritic` with
`time_horizon=T`), the residual picks up a V_t term, and an optional terminal
MSE | V_theta(T, W_N) - g(W_N) |^2 is added.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.optim import Adam
from tqdm import trange

from .config import HorizonConfig, MertonParams, PolicyParams, TrainConfig
from .critic import VanillaMLPCritic
from .eval import evaluate_critic_on_grid
from .losses import rl_pinn_residual
from .merton import exact_step, reward_rate, terminal_value_fn
from .sampling import sample_log_uniform


@dataclass(frozen=True)
class RLPinnConfig:
    """Dataset and optimization knobs for PINN-style training.

    m1, m2, N follow the notation of the RL_PINN note:
      * m1 initial states sampled from rho_0 (here: log-uniform on [w_min, w_max])
      * m2 trajectories per initial state, each of length N steps
      * total of m1 * m2 * N transitions
    """

    m1: int = 256
    m2: int = 4
    N: int = 64
    minibatch_size: int = 4096
    num_epochs: int = 40
    learning_rate: float = 2e-3
    seed: int = 0
    device: str = "cpu"
    log_every_epochs: int = 1
    # Finite-horizon only: weight on the optional terminal MSE.
    use_terminal_loss: bool = True
    terminal_weight: float = 1.0


@dataclass(frozen=True)
class TrajectoryDataset:
    wealth: torch.Tensor
    wealth_next: torch.Tensor
    reward: torch.Tensor
    terminal_wealth: torch.Tensor
    dt: float
    t: torch.Tensor | None = None


def sample_trajectory_dataset(
    params: MertonParams,
    policy: PolicyParams,
    cfg: RLPinnConfig,
    dt: float,
    wealth_min: float,
    wealth_max: float,
    horizon: HorizonConfig | None = None,
) -> TrajectoryDataset:
    """
    Sample m1 initial states and roll out m2 trajectories of length N each
    under the constant policy. When `horizon` is given, dt is overridden to
    horizon.T / cfg.N and the dataset includes per-step time tensors.
    """
    g = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    if horizon is not None:
        dt = horizon.T / cfg.N

    s0 = sample_log_uniform(
        batch_size=cfg.m1, low=wealth_min, high=wealth_max, device=cfg.device
    )
    w = s0.repeat_interleave(cfg.m2)

    t_list: list[torch.Tensor] = []
    w_t_list: list[torch.Tensor] = []
    w_tp_list: list[torch.Tensor] = []

    for n in range(cfg.N):
        if horizon is not None:
            t_list.append(torch.full_like(w, n * dt))
        noise = torch.randn(w.shape, generator=g, device=cfg.device)
        w_next = exact_step(w, params, policy, dt, noise)
        w_t_list.append(w)
        w_tp_list.append(w_next)
        w = w_next

    wealth = torch.cat(w_t_list, dim=0).detach()
    time_grid = torch.cat(t_list, dim=0).detach() if horizon is not None else None
    return TrajectoryDataset(
        wealth=wealth,
        wealth_next=torch.cat(w_tp_list, dim=0).detach(),
        reward=reward_rate(wealth, params, policy).detach(),
        terminal_wealth=w.detach(),
        dt=dt,
        t=time_grid,
    )


def _empty_history() -> dict[str, list[float]]:
    return {
        "step": [],
        "loss": [],
        "td_mse": [],
        "dtd_mse": [],
        "pinn_mse": [],
        "terminal_mse": [],
        "mae": [],
        "rmse": [],
        "mape": [],
        "v_w_mae": [],
        "v_w_norm": [],
        "hjb_rmse": [],
        "dtd_noise_floor": [],
    }


def _evaluate_rl_pinn_critic(
    critic: VanillaMLPCritic,
    params: MertonParams,
    policy: PolicyParams,
    train_cfg: TrainConfig,
    dt: float,
    device: str,
    horizon: HorizonConfig | None,
) -> dict[str, Any]:
    return evaluate_critic_on_grid(
        critic=critic,
        params=params,
        policy=policy,
        low=train_cfg.wealth_min,
        high=train_cfg.wealth_max,
        num=train_cfg.eval_points,
        dt=dt if horizon is None else None,
        device=device,
        horizon=horizon,
        num_t=21,
    )


def _append_history_row(
    history: dict[str, list[float]],
    metrics: dict[str, Any],
    global_step: int,
    loss: float,
    pinn_mse: float,
    terminal_mse: float,
) -> None:
    history["step"].append(global_step)
    history["loss"].append(loss)
    history["td_mse"].append(float("nan"))
    history["dtd_mse"].append(float("nan"))
    history["pinn_mse"].append(pinn_mse)
    history["terminal_mse"].append(terminal_mse)
    history["mae"].append(float(metrics["mae"]))
    history["rmse"].append(float(metrics["rmse"]))
    history["mape"].append(float(metrics["mape"]))

    for key in ("v_w_mae", "v_w_norm", "hjb_rmse", "dtd_noise_floor"):
        if key in metrics:
            history[key].append(float(metrics[key]))


def train_fixed_policy_critic_rl_pinn(
    params: MertonParams,
    policy: PolicyParams,
    train_cfg: TrainConfig,
    pinn_cfg: RLPinnConfig,
    horizon: HorizonConfig | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """PINN-style training. Stationary by default; finite-horizon when `horizon` is given."""
    torch.manual_seed(pinn_cfg.seed)

    finite_horizon = horizon is not None
    time_horizon = horizon.T if horizon is not None else None
    critic = VanillaMLPCritic(params, time_horizon=time_horizon).to(pinn_cfg.device)
    optimizer = Adam(critic.parameters(), lr=pinn_cfg.learning_rate)

    dataset = sample_trajectory_dataset(
        params=params,
        policy=policy,
        cfg=pinn_cfg,
        dt=train_cfg.dt,
        wealth_min=train_cfg.wealth_min,
        wealth_max=train_cfg.wealth_max,
        horizon=horizon,
    )
    wealth = dataset.wealth
    wealth_next = dataset.wealth_next
    reward = dataset.reward
    time_grid = dataset.t
    dt = dataset.dt
    terminal_wealth = dataset.terminal_wealth
    g_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    terminal_time: torch.Tensor | None = None
    if finite_horizon:
        assert horizon is not None
        g_fn = terminal_value_fn(params, horizon)
        terminal_time = torch.full_like(terminal_wealth, horizon.T)

    n_total = wealth.shape[0]
    bs = min(pinn_cfg.minibatch_size, n_total)

    history = _empty_history()

    global_step = 0
    label = "rl_pinn-finite" if finite_horizon else "rl_pinn"
    iterator = trange(pinn_cfg.num_epochs, desc=f"train-{label}", leave=False)
    for epoch in iterator:
        perm = torch.randperm(n_total, device=pinn_cfg.device)
        epoch_loss = 0.0
        epoch_pinn = 0.0
        epoch_term = 0.0
        n_batches = 0
        for start in range(0, n_total, bs):
            idx = perm[start : start + bs]
            residual = rl_pinn_residual(
                critic=critic,
                wealth=wealth[idx],
                wealth_next=wealth_next[idx],
                reward=reward[idx],
                params=params,
                dt=dt,
                t=time_grid[idx] if time_grid is not None else None,
            )

            optimizer.zero_grad(set_to_none=True)
            pinn_mse = torch.mean(residual.square())
            loss = pinn_mse

            term_mse_val = 0.0
            if finite_horizon and pinn_cfg.use_terminal_loss:
                assert g_fn is not None
                assert terminal_time is not None
                v_term = critic.value(terminal_wealth, terminal_time)
                term_mse = torch.mean((v_term - g_fn(terminal_wealth)).square())
                loss = loss + pinn_cfg.terminal_weight * term_mse
                term_mse_val = float(term_mse.detach().cpu())

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_pinn += float(pinn_mse.detach().cpu())
            epoch_term += term_mse_val
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_pinn = epoch_pinn / max(n_batches, 1)
        avg_term = epoch_term / max(n_batches, 1)

        should_log = (
            epoch % pinn_cfg.log_every_epochs == 0
            or epoch == pinn_cfg.num_epochs - 1
        )
        if should_log:
            metrics = _evaluate_rl_pinn_critic(
                critic, params, policy, train_cfg, dt, pinn_cfg.device, horizon
            )
            _append_history_row(
                history, metrics, global_step, avg_loss, avg_pinn, avg_term
            )
            iterator.set_postfix(loss=f"{avg_loss:.3e}", mae=f"{metrics['mae']:.3e}")

    summary = _evaluate_rl_pinn_critic(
        critic, params, policy, train_cfg, dt, pinn_cfg.device, horizon
    )

    meta: dict[str, Any] = {
        "params": asdict(params),
        "policy": asdict(policy),
        "train_cfg": asdict(train_cfg),
        "pinn_cfg": asdict(pinn_cfg),
        "loss_name": "rl_pinn_finite" if horizon is not None else "rl_pinn",
        "dt": dt,
    }
    if horizon is not None:
        meta["horizon"] = asdict(horizon)
    return critic, {"history": history, "summary": summary, "meta": meta}


def save_checkpoint(critic, result: dict[str, Any], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": critic.state_dict(), "result": result},
        out_path / "checkpoint.pt",
    )
