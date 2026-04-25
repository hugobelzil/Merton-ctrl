from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from merton_dtd.config import MertonParams, PolicyParams, TrainConfig
from merton_dtd.plotting import plot_training_curves, plot_value_fit
from merton_dtd.training import save_checkpoint, train_fixed_policy_critic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TD / dTD critic on a fixed Merton policy.")

    parser.add_argument("--r", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=0.08)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--rho", type=float, default=0.08)

    parser.add_argument("--pi", type=float, default=0.75)
    parser.add_argument("--kappa", type=float, default=0.06125)

    parser.add_argument("--loss", type=str, default="beta_dtd", choices=["td", "dtd", "beta_dtd"])
    parser.add_argument("--beta", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dt", type=float, default=1.0 / 252.0)
    parser.add_argument("--wealth-min", type=float, default=0.3)
    parser.add_argument("--wealth-max", type=float, default=3.0)
    parser.add_argument("--eval-points", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="results/train_critic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    params = MertonParams(r=args.r, mu=args.mu, sigma=args.sigma, gamma=args.gamma, rho=args.rho)
    policy = PolicyParams(pi=args.pi, kappa=args.kappa)
    train_cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        dt=args.dt,
        wealth_min=args.wealth_min,
        wealth_max=args.wealth_max,
        eval_points=args.eval_points,
        beta=args.beta,
        device=args.device,
        log_every=args.log_every,
    )

    critic, result = train_fixed_policy_critic(
        params=params,
        policy=policy,
        train_cfg=train_cfg,
        loss_name=args.loss,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(critic, result, out_dir)
    plot_training_curves(result, out_dir / "training_curves.png")
    plot_value_fit(result, out_dir / "value_fit.png")

    # numpy arrays inside result["summary"] are not JSON serializable; keep only scalars in summary.json
    serializable = {
        "params": asdict(params),
        "policy": asdict(policy),
        "train_cfg": asdict(train_cfg),
        "loss": args.loss,
        "mae": result["summary"]["mae"],
        "rmse": result["summary"]["rmse"],
        "mape": result["summary"]["mape"],
    }
    (out_dir / "summary.json").write_text(json.dumps(serializable, indent=2))

    print("Training finished.")
    print(f"Final MAE  : {result['summary']['mae']:.6e}")
    print(f"Final RMSE : {result['summary']['rmse']:.6e}")
    print(f"Final MAPE : {result['summary']['mape']:.6e}")
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
