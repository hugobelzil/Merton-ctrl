from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from merton_dtd.config import MertonParams
from merton_dtd.plotting import plot_policy_heatmap
from merton_dtd.sweep import sweep_constant_policies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-sweep constant policies for the Merton problem.")
    parser.add_argument("--r", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=0.08)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--rho", type=float, default=0.08)
    parser.add_argument("--pi-min", type=float, default=0.0)
    parser.add_argument("--pi-max", type=float, default=1.5)
    parser.add_argument("--pi-num", type=int, default=121)
    parser.add_argument("--kappa-min", type=float, default=0.01)
    parser.add_argument("--kappa-max", type=float, default=0.15)
    parser.add_argument("--kappa-num", type=int, default=121)
    parser.add_argument("--wealth-ref", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="results/policy_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = MertonParams(r=args.r, mu=args.mu, sigma=args.sigma, gamma=args.gamma, rho=args.rho)
    pi_grid = np.linspace(args.pi_min, args.pi_max, args.pi_num)
    kappa_grid = np.linspace(args.kappa_min, args.kappa_max, args.kappa_num)

    result = sweep_constant_policies(params, pi_grid, kappa_grid, wealth_ref=args.wealth_ref)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    values = result["values"]
    plot_policy_heatmap(
        values=values,
        pi_grid=pi_grid,
        kappa_grid=kappa_grid,
        out_file=out_dir / "value_heatmap.png",
        title="Exact value over constant Merton policies",
    )

    # Store scalar summary as JSON.
    serializable = {
        "params": result["params"],
        "wealth_ref": result["wealth_ref"],
        "best_grid_value": result["best_grid_value"],
        "best_grid_policy": result["best_grid_policy"],
        "closed_form_policy": result["closed_form_policy"],
        "closed_form_value": result["closed_form_value"],
    }
    (out_dir / "summary.json").write_text(json.dumps(serializable, indent=2))
    np.save(out_dir / "value_grid.npy", values)
    np.save(out_dir / "pi_grid.npy", pi_grid)
    np.save(out_dir / "kappa_grid.npy", kappa_grid)

    print("Policy sweep finished.")
    print(f"Best grid policy     : {result['best_grid_policy']}")
    print(f"Closed-form optimum  : {result['closed_form_policy']}")
    print(f"Saved artifacts to   : {out_dir}")


if __name__ == "__main__":
    main()
