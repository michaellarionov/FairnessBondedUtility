"""Fairness Bonded Utility (FBU) — public API.

Wang et al., "Preventing Discriminatory Decision-making in Evolving Data
Streams", ACM FAccT 2023, §5.1. Implementation spec: docs/FBU_IMPLEMENTATION.md.
"""

from .area import EnclosedArea, PerpendicularDistance, enclosed_area, max_area, normalised_area
from .baseline import BaselineCurve, build_baseline, original_point
from .fbu import (
    FBUReport,
    assert_aligned,
    control_predictions,
    evaluate,
    seed_sensitivity,
)
from .metrics import (
    balanced_accuracy,
    cum_eod,
    cum_spd,
    eod,
    fairness_score,
    recall,
    spd,
)
from .plotting import plot_region_distribution, plot_replacement_ratio, plot_tradeoff
from .pseudo import PseudoPoint, majority_label, pseudo_points, pseudo_predictions
from .regions import classify
from .types import (
    REGION_LABELS,
    BinaryScorer,
    CaseResult,
    FBUResult,
    Predictions,
    SplitData,
    TradeoffArea,
)

__all__ = [
    # contracts
    "Predictions",
    "SplitData",
    "CaseResult",
    "FBUResult",
    "BinaryScorer",
    "TradeoffArea",
    "REGION_LABELS",
    # metrics
    "balanced_accuracy",
    "recall",
    "spd",
    "eod",
    "cum_spd",
    "cum_eod",
    "fairness_score",
    # FBU core
    "majority_label",
    "pseudo_predictions",
    "pseudo_points",
    "PseudoPoint",
    "BaselineCurve",
    "build_baseline",
    "original_point",
    "classify",
    "enclosed_area",
    "max_area",
    "normalised_area",
    "EnclosedArea",
    "PerpendicularDistance",
    # orchestration
    "evaluate",
    "FBUReport",
    "assert_aligned",
    "control_predictions",
    "seed_sensitivity",
    # plots
    "plot_replacement_ratio",
    "plot_tradeoff",
    "plot_region_distribution",
]
