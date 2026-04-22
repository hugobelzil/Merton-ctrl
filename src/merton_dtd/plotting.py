from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_value_fit(result: dict, out_file: str | Path) -> None:
    summary = result["summary"]
    wealth = np.asarray(summary["wealth"])
    truth = np.asarray(summary["truth"])
    pred = np.asarray(summary["pred"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(wealth, truth, label="exact value")
    ax.plot(wealth, pred, label="critic prediction", linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("wealth")
    ax.set_ylabel("value")
    ax.set_title("Merton fixed-policy value fit")
    ax.legend()
    fig.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_training_curves(result: dict, out_file: str | Path) -> None:
    history = result["history"]
    steps = np.asarray(history["step"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(steps, history["loss"], label="train loss")
    axes[0].plot(steps, history["td_mse"], label="TD MSE", linestyle="--")
    axes[0].plot(steps, history["dtd_scaled_mse"], label="scaled dTD MSE", linestyle=":")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("training step")
    axes[0].set_title("critic losses")
    axes[0].legend()

    axes[1].plot(steps, history["mae"], label="MAE")
    axes[1].plot(steps, history["rmse"], label="RMSE", linestyle="--")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("training step")
    axes[1].set_title("error vs exact value")
    axes[1].legend()

    fig.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_policy_heatmap(
    values: np.ndarray,
    pi_grid: np.ndarray,
    kappa_grid: np.ndarray,
    out_file: str | Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[pi_grid[0], pi_grid[-1], kappa_grid[0], kappa_grid[-1]],
    )
    ax.set_xlabel("risky weight pi")
    ax.set_ylabel("consumption rate kappa")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="value at wealth=1")
    fig.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
