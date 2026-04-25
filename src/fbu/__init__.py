"""Fairness Bonded Utility (FBU) — public API."""

from .fbu import FBU, REGION_LABELS
from .metrics import (
    balanced_accuracy,
    cumulative_eod,
    cumulative_spd,
    imbalance_ratio,
)
from .visualization import plot_fbu

__all__ = [
    "FBU",
    "REGION_LABELS",
    "plot_fbu",
    "cumulative_spd",
    "cumulative_eod",
    "balanced_accuracy",
    "imbalance_ratio",
]
