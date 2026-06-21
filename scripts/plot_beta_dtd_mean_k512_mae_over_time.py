from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_mae(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing history file: {hist_path}")

    history = json.loads(hist_path.read_text())
    return np.asarray(history["step"], dtype=float), np.asarray(history["mae"], dtype=float)


def best_run_by_min_mae(
    runs_dir: Path,
    run_name_template: str,
    betas: list[float],
    seed: int,
) -> tuple[float, str, np.ndarray, np.ndarray, float, float]:
    best = None
    for beta in betas:
        beta_text = f"{beta:g}"
        run_name = run_name_template.format(beta=beta_text, seed=seed)
        steps, mae = load_mae(runs_dir / run_name)
        best_i = int(np.argmin(mae))
        candidate = (
            float(mae[best_i]),
            beta,
            run_name,
            steps,
            mae,
            float(steps[best_i]),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise ValueError("No beta runs were available to select from.")

    best_mae, beta, run_name, steps, mae, best_step = best
    return beta, run_name, steps, mae, best_mae, best_step


def plot_curve(
    ax,
    runs_dir: Path,
    run_name: str,
    label: str,
    *,
    color,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> None:
    steps, mae = load_mae(runs_dir / run_name)
    plot_loaded_curve(
        ax,
        steps,
        mae,
        label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )


def plot_loaded_curve(
    ax,
    steps: np.ndarray,
    mae: np.ndarray,
    label: str,
    *,
    color,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> None:
    best_i = int(np.argmin(mae))
    ax.plot(
        steps,
        mae,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=f"{label} (best {mae[best_i]:.2f}, final {mae[-1]:.2f})",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="results/beta_dtd_mean_K512_B200_12k",
        help="Result folder containing the runs/ directory.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. Defaults inside --root.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    runs_dir = root / "runs"
    out_path = Path(args.out or root / "mae_over_time_best_beta_K512_and_K1_with_td.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    betas = [0.25, 0.5, 0.75, 0.9]
    seed = args.seed

    k512_beta, k512_run, k512_steps, k512_mae, k512_best_mae, k512_best_step = (
        best_run_by_min_mae(
            runs_dir,
            "sig0.05_lr0.005_B200_beta_dtd_mean_K512_beta{beta}_s{seed}",
            betas,
            seed,
        )
    )
    k1_beta, k1_run, k1_steps, k1_mae, k1_best_mae, k1_best_step = best_run_by_min_mae(
        runs_dir,
        "sig0.05_lr0.005_B200_beta_dtd_K1_beta{beta}_s{seed}",
        betas,
        seed,
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.7))

    plot_curve(
        ax,
        runs_dir,
        f"sig0.05_lr0.005_B200_td_mean_K512_s{seed}",
        "TD-mean, K=512",
        color="C3",
        linewidth=2.6,
    )
    plot_curve(
        ax,
        runs_dir,
        f"sig0.05_lr0.005_B200_td_K1_s{seed}",
        "TD, K=1",
        color="C3",
        linestyle="--",
        linewidth=1.8,
    )

    plot_loaded_curve(
        ax,
        k512_steps,
        k512_mae,
        rf"Best $\beta$-dTD-mean K=512: $\beta={k512_beta:g}$",
        color="C0",
        linewidth=2.1,
    )
    plot_loaded_curve(
        ax,
        k1_steps,
        k1_mae,
        rf"Best $\beta$-dTD K=1: $\beta={k1_beta:g}$",
        color="C2",
        linestyle="--",
        linewidth=2.0,
    )

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"MAE vs. closed-form $V^\pi$ (log scale)")
    ax.set_title(
        r"MAE over training, $\sigma=0.05$, lr=$0.005$, $B=200$, 12k steps, seed 0"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    print(
        f"best K=512 beta run: {k512_run} "
        f"(min MAE {k512_best_mae:.6g} at step {k512_best_step:g})"
    )
    print(
        f"best K=1 beta run: {k1_run} "
        f"(min MAE {k1_best_mae:.6g} at step {k1_best_step:g})"
    )


if __name__ == "__main__":
    main()
