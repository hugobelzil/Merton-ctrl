from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from tqdm import trange

from .config import MertonParams, PolicyParams, TrainConfig
from .critic import VanillaMLPCritic
from .eval import evaluate_critic_on_grid
from .losses import compute_loss, make_batch
from .sampling import sample_log_uniform


def build_critic(params: MertonParams, device: str = "cpu") -> VanillaMLPCritic:
    return VanillaMLPCritic(params).to(device)


def train_fixed_policy_critic(
    params: MertonParams,
    policy: PolicyParams,
    train_cfg: TrainConfig,
    loss_name: str = "beta_dtd",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(train_cfg.seed)

    critic = build_critic(params, device=train_cfg.device)
    optimizer = Adam(critic.parameters(), lr=train_cfg.learning_rate)

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

    wealth = sample_log_uniform(
        batch_size=train_cfg.batch_size,
        low=train_cfg.wealth_min,
        high=train_cfg.wealth_max,
        device=train_cfg.device,
    )

    iterator = trange(train_cfg.num_steps, desc=f"train-{loss_name}", leave=False)
    for step in iterator:
        wealth, wealth_next, reward = make_batch(wealth, params, policy, train_cfg.dt)

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_loss(
            critic=critic,
            wealth=wealth,
            wealth_next=wealth_next,
            reward=reward,
            params=params,
            dt=train_cfg.dt,
            loss_name=loss_name,
            beta=train_cfg.beta,
        )
        loss.backward()
        optimizer.step()

        wealth = wealth_next.detach()

        if (step % train_cfg.log_every == 0) or (step == train_cfg.num_steps - 1):
            eval_metrics = evaluate_critic_on_grid(
                critic=critic,
                params=params,
                policy=policy,
                low=train_cfg.wealth_min,
                high=train_cfg.wealth_max,
                num=train_cfg.eval_points,
                dt=train_cfg.dt,
                device=train_cfg.device,
            )
            history["step"].append(step)
            history["loss"].append(metrics["loss"])
            history["td_mse"].append(metrics["td_mse"])
            history["dtd_mse"].append(metrics["dtd_mse"])
            history["pinn_mse"].append(metrics.get("pinn_mse", float("nan")))
            history["mae"].append(float(eval_metrics["mae"]))
            history["rmse"].append(float(eval_metrics["rmse"]))
            history["mape"].append(float(eval_metrics["mape"]))
            history["v_w_mae"].append(float(eval_metrics["v_w_mae"]))
            history["v_w_norm"].append(float(eval_metrics["v_w_norm"]))
            history["hjb_rmse"].append(float(eval_metrics["hjb_rmse"]))
            history["dtd_noise_floor"].append(float(eval_metrics["dtd_noise_floor"]))
            iterator.set_postfix(loss=f"{metrics['loss']:.3e}", mae=f"{eval_metrics['mae']:.3e}")

    summary = evaluate_critic_on_grid(
        critic=critic,
        params=params,
        policy=policy,
        low=train_cfg.wealth_min,
        high=train_cfg.wealth_max,
        num=train_cfg.eval_points,
        dt=train_cfg.dt,
        device=train_cfg.device,
    )

    meta = {
        "params": asdict(params),
        "policy": asdict(policy),
        "train_cfg": asdict(train_cfg),
        "loss_name": loss_name,
    }
    result = {"history": history, "summary": summary, "meta": meta}
    return critic, result


def save_checkpoint(critic, result: dict[str, Any], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": critic.state_dict(),
            "result": result,
        },
        out_path / "checkpoint.pt",
    )
