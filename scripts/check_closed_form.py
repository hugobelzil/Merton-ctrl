from __future__ import annotations

import argparse
import json
from pathlib import Path

from merton_dtd.config import MertonParams, PolicyParams
from merton_dtd.merton import exact_value, optimal_policy_closed_form


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the closed-form Merton solution.")
    parser.add_argument("--r", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=0.08)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--rho", type=float, default=0.08)
    parser.add_argument("--wealth", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="results/closed_form_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = MertonParams(r=args.r, mu=args.mu, sigma=args.sigma, gamma=args.gamma, rho=args.rho)
    policy, coeff = optimal_policy_closed_form(params)
    value = float(exact_value(args.wealth, params, policy))

    payload = {
        "params": vars(args),
        "optimal_policy": {"pi": policy.pi, "kappa": policy.kappa},
        "value_coefficient": coeff,
        "value_at_wealth": value,
    }

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2))

    print("Closed-form optimal policy")
    print(f"  pi*    = {policy.pi:.6f}")
    print(f"  kappa* = {policy.kappa:.6f}")
    print(f"  V({args.wealth:.4f}) = {value:.6f}")
    print(f"Saved summary to {out_file}")


if __name__ == "__main__":
    main()
