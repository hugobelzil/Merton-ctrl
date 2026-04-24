from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from tqdm import trange

from .config import MertonParams, PolicyParams, TrainConfig
from .critic import MLPCRRACritic, ScalarCRRACritic, VanillaMLPCritic
from .eval import evaluate_critic_on_grid
from .losses import compute_loss, make_batch
from .sampling import sample_log_uniform
from .merton import exact_value_coefficient


CRITIC_TYPES = {
    "scalar": ScalarCRRACritic,
    "mlp": MLPCRRACritic,
    "vanilla_mlp": VanillaMLPCritic,
}


def build_critic(name: str, params: MertonParams, policy: PolicyParams, device: str = "cpu"):
    exact_A = exact_value_coefficient(params, policy)
    init_log_A = float(torch.log(torch.tensor(exact_A)).item())

    if name == "scalar":
        critic = ScalarCRRACritic(params, init_log_A=init_log_A)
    elif name == "mlp":
        critic = MLPCRRACritic(params, init_log_A=init_log_A)
    elif name == "vanilla_mlp":
        critic = VanillaMLPCritic(params)
    else:
        raise ValueError(f"Unknown critic type: {name}")

    return critic.to(device)

def train_fixed_policy_critic(
    params: MertonParams,
    policy: PolicyParams,
    train_cfg: TrainConfig,
    critic_name: str = "scalar",
    loss_name: str = "beta_dtd",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(train_cfg.seed)

    critic = build_critic(critic_name, params, policy, device=train_cfg.device)
    optimizer = Adam(critic.parameters(), lr=train_cfg.learning_rate)

    history: dict[str, list[float]] = {
        "step": [],
        "loss": [],
        "td_mse": [],
        "dtd_scaled_mse": [],
        "mae": [],
        "rmse": [],
        "mape": [],
    }

    iterator = trange(train_cfg.num_steps, desc=f"train-{critic_name}-{loss_name}", leave=False)
    for step in iterator:
        wealth = sample_log_uniform(
            batch_size=train_cfg.batch_size,
            low=train_cfg.wealth_min,
            high=train_cfg.wealth_max,
            device=train_cfg.device,
        )
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

        if (step % train_cfg.log_every == 0) or (step == train_cfg.num_steps - 1):
            eval_metrics = evaluate_critic_on_grid(
                critic=critic,
                params=params,
                policy=policy,
                low=train_cfg.wealth_min,
                high=train_cfg.wealth_max,
                num=train_cfg.eval_points,
                device=train_cfg.device,
            )
            history["step"].append(step)
            history["loss"].append(metrics["loss"])
            history["td_mse"].append(metrics["td_mse"])
            history["dtd_scaled_mse"].append(metrics["dtd_scaled_mse"])
            history["mae"].append(float(eval_metrics["mae"]))
            history["rmse"].append(float(eval_metrics["rmse"]))
            history["mape"].append(float(eval_metrics["mape"]))
            iterator.set_postfix(loss=f"{metrics['loss']:.3e}", mae=f"{eval_metrics['mae']:.3e}")

    summary = evaluate_critic_on_grid(
        critic=critic,
        params=params,
        policy=policy,
        low=train_cfg.wealth_min,
        high=train_cfg.wealth_max,
        num=train_cfg.eval_points,
        device=train_cfg.device,
    )

    meta = {
        "params": asdict(params),
        "policy": asdict(policy),
        "train_cfg": asdict(train_cfg),
        "critic_name": critic_name,
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
