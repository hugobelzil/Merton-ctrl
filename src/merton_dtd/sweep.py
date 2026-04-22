from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .config import MertonParams, PolicyParams
from .merton import exact_value, is_policy_admissible, optimal_policy_closed_form


def sweep_constant_policies(
    params: MertonParams,
    pi_grid: np.ndarray,
    kappa_grid: np.ndarray,
    wealth_ref: float = 1.0,
) -> dict:
    values = np.full((len(kappa_grid), len(pi_grid)), np.nan, dtype=float)
    best_val = -np.inf
    best_policy: PolicyParams | None = None

    for i, kappa in enumerate(kappa_grid):
        for j, pi in enumerate(pi_grid):
            policy = PolicyParams(pi=float(pi), kappa=float(kappa))
            if not is_policy_admissible(params, policy):
                continue
            val = float(exact_value(wealth_ref, params, policy))
            values[i, j] = val
            if val > best_val:
                best_val = val
                best_policy = policy

    closed_form_policy, _ = optimal_policy_closed_form(params)
    closed_form_value = float(exact_value(wealth_ref, params, closed_form_policy))

    return {
        "params": asdict(params),
        "wealth_ref": wealth_ref,
        "pi_grid": pi_grid,
        "kappa_grid": kappa_grid,
        "values": values,
        "best_grid_value": best_val,
        "best_grid_policy": asdict(best_policy) if best_policy is not None else None,
        "closed_form_policy": asdict(closed_form_policy),
        "closed_form_value": closed_form_value,
    }
