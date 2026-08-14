"""Metric families used by FBU (spec §2.2)."""

from .fairness import (
    CUMULATIVE_FAIRNESS_METRICS,
    DEFAULT_LAMBDA,
    FAIRNESS_METRICS,
    cum_eod,
    cum_eod_final,
    cum_spd,
    cum_spd_final,
    eod,
    fairness_score,
    get_fairness_metric,
    spd,
)
from .performance import (
    PERFORMANCE_METRICS,
    balanced_accuracy,
    get_performance_metric,
    imbalance_ratio,
    recall,
    true_negative_rate,
    true_positive_rate,
)

__all__ = [
    "balanced_accuracy",
    "recall",
    "true_positive_rate",
    "true_negative_rate",
    "imbalance_ratio",
    "PERFORMANCE_METRICS",
    "get_performance_metric",
    "spd",
    "eod",
    "cum_spd",
    "cum_eod",
    "cum_spd_final",
    "cum_eod_final",
    "fairness_score",
    "DEFAULT_LAMBDA",
    "FAIRNESS_METRICS",
    "CUMULATIVE_FAIRNESS_METRICS",
    "get_fairness_metric",
]
