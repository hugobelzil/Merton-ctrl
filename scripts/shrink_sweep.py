"""
Sweep over the shrinkage intensity lambda for dTD with the per-sample
drift estimator shrunk toward 0:

    ΔW_lambda  = (1 - lambda) * ΔW
    ΔW²_lambda = (1 - lambda) * (ΔW)²

lambda = 0 recovers vanilla (beta-)dTD; lambda = 1 zeros out the prediction.

Infinite-horizon Merton, fixed-policy evaluation. We train one critic per
lambda (optionally across seeds), then plot regret (MAE vs the closed-form
value) as a function of lambda.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from merton_dtd.config import MertonParams, PolicyParams, TrainConfig
from merton_dtd.training import train_fixed_policy_critic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=0.20)
    p.add_argument("--pi", type=float, default=0.75)
    p.add_argument("--kappa", type=float, default=0.06125)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--mu", type=float, default=0.08)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--rho", type=float, default=0.08)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--num-steps", type=int, default=12000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dt", type=float, default=1.0 / 252.0)
    p.add_argument("--wealth-min", type=float, default=0.3)
    p.add_argument("--wealth-max", type=float, default=3.0)
    p.add_argument("--eval-points", type=int, default=200)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--beta", type=float, default=0.5,
                   help="beta in beta_dtd mixture; set 1.0 for pure dtd")
    p.add_argument("--loss", type=str, default="beta_dtd",
                   choices=["dtd", "beta_dtd"],
                   help="dTD variant; shrink_lambda applies to the dTD prediction term in both")
    p.add_argument("--lambdas", type=str,
                   default="0.0,0.1,0.2,0.3,0.4,0.5,0.7,0.9,1.0")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="results/shrink_sweep")
    args = p.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    loss_name = args.loss

    params = MertonParams(r=args.r, mu=args.mu, sigma=args.sigma,
                          gamma=args.gamma, rho=args.rho)
    policy = PolicyParams(pi=args.pi, kappa=args.kappa)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # rows: lambda, cols: seed
    best_mae = np.zeros((len(lambdas), len(seeds)))
    final_mae = np.zeros((len(lambdas), len(seeds)))
    histories: dict[float, list[dict]] = {lam: [] for lam in lambdas}

    for i, lam in enumerate(lambdas):
        for j, seed in enumerate(seeds):
            train_cfg = TrainConfig(
                seed=seed, batch_size=args.batch_size, num_steps=args.num_steps,
                learning_rate=args.lr, dt=args.dt,
                wealth_min=args.wealth_min, wealth_max=args.wealth_max,
                eval_points=args.eval_points, beta=args.beta,
                device=args.device, log_every=args.log_every,
                shrink_lambda=lam,
            )
            print(f"=== lambda={lam:g} seed={seed} ({loss_name}) ===")
            _, result = train_fixed_policy_critic(
                params=params, policy=policy, train_cfg=train_cfg,
                loss_name=loss_name,
            )
            h = result["history"]
            mae = np.asarray(h["mae"])
            best_mae[i, j] = float(mae.min())
            final_mae[i, j] = float(mae[-1])
            histories[lam].append(h)

    lam_arr = np.array(lambdas)
    best_mean = best_mae.mean(axis=1)
    best_std = best_mae.std(axis=1)
    final_mean = final_mae.mean(axis=1)
    final_std = final_mae.std(axis=1)

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.errorbar(lam_arr, best_mean, yerr=best_std, fmt="o-",
                color="tab:red", capsize=3, label="best MAE (early stop)")
    ax.errorbar(lam_arr, final_mean, yerr=final_std, fmt="s--",
                color="tab:blue", capsize=3, label="final MAE")
    ax.axvline(0.0, color="k", ls=":", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"shrinkage intensity $\lambda$")
    ax.set_ylabel("regret = MAE vs $V^\\pi$")
    ax.set_title(f"Regret vs $\\lambda$  ({loss_name}, β={args.beta}, "
                 f"σ={args.sigma}, {len(seeds)} seed{'s' if len(seeds)>1 else ''})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    cmap = plt.cm.viridis(np.linspace(0.0, 0.9, len(lambdas)))
    for lam, color in zip(lambdas, cmap):
        # use the first seed for the training-curve panel
        h = histories[lam][0]
        ax.plot(h["step"], h["mae"], color=color, linewidth=1.3,
                label=rf"$\lambda$={lam:g}")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("MAE vs $V^\\pi$")
    ax.set_title("MAE over training (seed 0)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "shrink_sweep.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    summary = {
        "lambdas": lambdas,
        "seeds": seeds,
        "loss_name": loss_name,
        "beta": args.beta,
        "best_mae_mean": best_mean.tolist(),
        "best_mae_std": best_std.tolist(),
        "final_mae_mean": final_mean.tolist(),
        "final_mae_std": final_std.tolist(),
        "best_mae_per_seed": best_mae.tolist(),
        "final_mae_per_seed": final_mae.tolist(),
        "argmin_lambda_best": float(lam_arr[int(np.argmin(best_mean))]),
        "argmin_lambda_final": float(lam_arr[int(np.argmin(final_mean))]),
        "meta": {
            "params": asdict(params), "policy": asdict(policy),
            "num_steps": args.num_steps, "lr": args.lr, "dt": args.dt,
            "batch_size": args.batch_size, "sigma": args.sigma,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
