"""Minimal local starter repo for the infinite-horizon Merton problem.

This package is intentionally small. It focuses on:
1. exact closed-form formulas for the infinite-horizon CRRA Merton problem,
2. simulation under constant policies,
3. critic training with TD / dTD / beta-dTD for fixed-policy evaluation.
"""

from .config import MertonParams, PolicyParams, TrainConfig
from .merton import (
    utility,
    optimal_policy_closed_form,
    exact_value_coefficient,
    exact_value,
    is_policy_admissible,
)

__all__ = [
    "MertonParams",
    "PolicyParams",
    "TrainConfig",
    "utility",
    "optimal_policy_closed_form",
    "exact_value_coefficient",
    "exact_value",
    "is_policy_admissible",
]
