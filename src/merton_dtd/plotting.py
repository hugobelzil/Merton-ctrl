from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _nice_log_ticks(xmin: float, xmax: float) -> np.ndarray:
    """
    Return log-scale ticks spanning the full range [xmin, xmax].
    """
    xmin = max(float(xmin), 1e-12)
    xmax = max(float(xmax), xmin * 1.000001)

    e_min = int(np.floor(np.log10(xmin)))
    e_max = int(np.ceil(np.log10(xmax)))
    ticks = 10.0 ** np.arange(e_min, e_max + 1)

    # Keep only ticks inside the visible range, but ensure endpoints are covered
    ticks = ticks[(ticks >= xmin * 0.999) & (ticks <= xmax * 1.001)]
    if ticks.size == 0:
        ticks = np.array([xmin, xmax])
    return ticks


def plot_value_fit(result: dict, out_file: str | Path) -> None:
    summary = result["summary"]
    wealth = np.asarray(summary["wealth"], dtype=float)
    truth = np.asarray(summary["truth"], dtype=float)
    pred = np.asarray(summary["pred"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(wealth, truth, label="Exact closed-form value", linewidth=2.2)
    ax.plot(wealth, pred, label="Neural critic prediction", linestyle="--", linewidth=2.0)

    ax.set_xscale("log")
    ax.set_xlim(wealth.min(), wealth.max())

    ticks = _nice_log_ticks(wealth.min(), wealth.max())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])

    ax.set_xlabel("Current wealth $W_t$")
    ax.set_ylabel("Value $V(W_t)$")
    ax.set_title("Fixed-policy Merton problem:\nExact value function vs learned critic")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(frameon=True)
    fig.tight_layout()

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(result: dict, out_file: str | Path) -> None:
    history = result["history"]
    steps = np.asarray(history["step"], dtype=float)
    loss = np.asarray(history["loss"], dtype=float)
    td_mse = np.asarray(history["td_mse"], dtype=float)
    dtd_mse = np.asarray(history["dtd_mse"], dtype=float)
    mae = np.asarray(history["mae"], dtype=float)
    rmse = np.asarray(history["rmse"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left panel: training losses
    axes[0].plot(steps, loss, label="Total training loss", linewidth=2.0)
    axes[0].plot(steps, td_mse, label="TD loss component", linestyle="--", linewidth=1.8)
    axes[0].plot(steps, dtd_mse, label="dTD loss component", linestyle=":", linewidth=2.0)

    axes[0].set_yscale("log")
    axes[0].set_xlim(steps.min(), steps.max())
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Training losses over optimization")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[0].legend(frameon=True)

    # Right panel: error against exact solution
    axes[1].plot(steps, mae, label="MAE", linewidth=2.0)
    axes[1].plot(steps, rmse, label="RMSE", linestyle="--", linewidth=1.8)

    axes[1].set_yscale("log")
    axes[1].set_xlim(steps.min(), steps.max())
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Error vs exact value (log scale)")
    axes[1].set_title("Critic accuracy against closed-form benchmark")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[1].legend(frameon=True)

    fig.suptitle("Merton fixed-policy critic training summary", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


