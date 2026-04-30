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
  * For finite-horizon problems the paper adds an optional terminal loss
    |V_theta(s_N) - g(s_N)|^2. The infinite-horizon Merton problem here
    has no terminal condition, so by default we do not use it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.optim import Adam
from tqdm import trange

from .config import MertonParams, PolicyParams, TrainConfig
from .critic import VanillaMLPCritic
from .eval import evaluate_critic_on_grid
from .losses import rl_pinn_residual
from .merton import exact_step, reward_rate
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
    # Terminal loss is irrelevant for infinite-horizon Merton; kept off by default.
    use_terminal_loss: bool = False
    terminal_weight: float = 1.0


def sample_trajectory_dataset(
    params: MertonParams,
    policy: PolicyParams,
    cfg: RLPinnConfig,
    dt: float,
    wealth_min: float,
    wealth_max: float,
) -> dict[str, torch.Tensor]:
    """
    Sample m1 initial states and roll out m2 trajectories of length N each
    under the constant policy.

    Returns flat tensors of shape (m1*m2*N,) for one-step transitions,
    plus the terminal wealth tensor (m1*m2,) for an optional terminal loss.
    """
    g = torch.Generator(device=cfg.device).manual_seed(cfg.seed)

    # m1 initial states ~ rho_0 (log-uniform)
    s0 = sample_log_uniform(
        batch_size=cfg.m1, low=wealth_min, high=wealth_max, device=cfg.device
    )
    # Replicate each initial state m2 times -> shape (m1*m2,)
    w0 = s0.repeat_interleave(cfg.m2)

    # Roll out N steps. Store all (W_n, W_{n+1}) pairs.
    w_t_list: list[torch.Tensor] = []
    w_tp_list: list[torch.Tensor] = []

    w = w0
    for _ in range(cfg.N):
        noise = torch.randn(w.shape, generator=g, device=cfg.device)
        w_next = exact_step(w, params, policy, dt, noise)
        w_t_list.append(w)
        w_tp_list.append(w_next)
        w = w_next

    wealth = torch.cat(w_t_list, dim=0)        # (m1*m2*N,)
    wealth_next = torch.cat(w_tp_list, dim=0)  # (m1*m2*N,)
    reward = reward_rate(wealth, params, policy)
    terminal_wealth = w  # (m1*m2,)

    return {
        "wealth": wealth.detach(),
        "wealth_next": wealth_next.detach(),
        "reward": reward.detach(),
        "terminal_wealth": terminal_wealth.detach(),
    }


def train_fixed_policy_critic_rl_pinn(
    params: MertonParams,
    policy: PolicyParams,
    train_cfg: TrainConfig,
    pinn_cfg: RLPinnConfig,
    terminal_value_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    PINN-style training over a frozen trajectory dataset.

    `terminal_value_fn`, if given, maps a wealth tensor to terminal values g(W).
    It is only used when `pinn_cfg.use_terminal_loss` is True.
    """
    torch.manual_seed(pinn_cfg.seed)

    critic = VanillaMLPCritic(params).to(pinn_cfg.device)
    optimizer = Adam(critic.parameters(), lr=pinn_cfg.learning_rate)

    dataset = sample_trajectory_dataset(
        params=params,
        policy=policy,
        cfg=pinn_cfg,
        dt=train_cfg.dt,
        wealth_min=train_cfg.wealth_min,
        wealth_max=train_cfg.wealth_max,
    )
    wealth = dataset["wealth"]
    wealth_next = dataset["wealth_next"]
    reward = dataset["reward"]
    terminal_wealth = dataset["terminal_wealth"]

    n_total = wealth.shape[0]
    bs = min(pinn_cfg.minibatch_size, n_total)

    history: dict[str, list[float]] = {
        "step": [],
        "loss": [],
        "td_mse": [],
        "dtd_mse": [],
        "pinn_mse": [],
        "mae": [],
        "rmse": [],
        "mape": [],
        "v_w_mae": [],
        "v_w_norm": [],
        "hjb_rmse": [],
        "dtd_noise_floor": [],
    }

    global_step = 0
    iterator = trange(pinn_cfg.num_epochs, desc="train-rl_pinn", leave=False)
    for epoch in iterator:
        perm = torch.randperm(n_total, device=pinn_cfg.device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_total, bs):
            idx = perm[start : start + bs]
            w_b = wealth[idx]
            wn_b = wealth_next[idx]
            r_b = reward[idx]

            optimizer.zero_grad(set_to_none=True)
            res = rl_pinn_residual(
                critic=critic,
                wealth=w_b,
                wealth_next=wn_b,
                reward=r_b,
                params=params,
                dt=train_cfg.dt,
            )
            loss = torch.mean(res.square())

            if pinn_cfg.use_terminal_loss and terminal_value_fn is not None:
                v_term = critic.value(terminal_wealth)
                g_term = terminal_value_fn(terminal_wealth)
                loss = loss + pinn_cfg.terminal_weight * torch.mean(
                    (v_term - g_term).square()
                )

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch % pinn_cfg.log_every_epochs == 0) or (epoch == pinn_cfg.num_epochs - 1):
            eval_metrics = evaluate_critic_on_grid(
                critic=critic,
                params=params,
                policy=policy,
                low=train_cfg.wealth_min,
                high=train_cfg.wealth_max,
                num=train_cfg.eval_points,
                dt=train_cfg.dt,
                device=pinn_cfg.device,
            )
            history["step"].append(global_step)
            history["loss"].append(avg_loss)
            history["td_mse"].append(float("nan"))
            history["dtd_mse"].append(float("nan"))
            history["pinn_mse"].append(avg_loss)
            history["mae"].append(float(eval_metrics["mae"]))
            history["rmse"].append(float(eval_metrics["rmse"]))
            history["mape"].append(float(eval_metrics["mape"]))
            history["v_w_mae"].append(float(eval_metrics["v_w_mae"]))
            history["v_w_norm"].append(float(eval_metrics["v_w_norm"]))
            history["hjb_rmse"].append(float(eval_metrics["hjb_rmse"]))
            history["dtd_noise_floor"].append(float(eval_metrics["dtd_noise_floor"]))
            iterator.set_postfix(
                loss=f"{avg_loss:.3e}", mae=f"{eval_metrics['mae']:.3e}"
            )

    summary = evaluate_critic_on_grid(
        critic=critic,
        params=params,
        policy=policy,
        low=train_cfg.wealth_min,
        high=train_cfg.wealth_max,
        num=train_cfg.eval_points,
        dt=train_cfg.dt,
        device=pinn_cfg.device,
    )

    meta = {
        "params": asdict(params),
        "policy": asdict(policy),
        "train_cfg": asdict(train_cfg),
        "pinn_cfg": asdict(pinn_cfg),
        "loss_name": "rl_pinn",
    }
    result = {"history": history, "summary": summary, "meta": meta}
    return critic, result


def save_checkpoint(critic, result: dict[str, Any], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": critic.state_dict(), "result": result},
        out_path / "checkpoint.pt",
    )
