"""Performance metrics for FBU (spec §2.2). Higher is better.

Only ``balanced_accuracy`` and ``recall`` are exported for use on the
performance axis. Raw accuracy, F1 and precision are excluded on purpose
(`[D8]`): accuracy is majority-class dominated on Adult, and F1/precision
divide by predicted positives, which is exactly zero at the ``M_100``
constant-negative pseudo-model.
"""

from __future__ import annotations

import numpy as np

from ..types import IntArray, PerformanceMetric


def _rate(numerator: float, denominator: float) -> float:
    """Ratio with the 0/0 case pinned to 0.0 rather than NaN."""
    return float(numerator / denominator) if denominator > 0 else 0.0


def true_positive_rate(y_true: IntArray, y_pred: IntArray) -> float:
    """TPR = TP / (TP + FN). Returns 0.0 when there are no true positives."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    positives = y_true == 1
    return _rate(float(np.count_nonzero(y_pred[positives] == 1)), float(np.count_nonzero(positives)))


def true_negative_rate(y_true: IntArray, y_pred: IntArray) -> float:
    """TNR = TN / (TN + FP). Returns 0.0 when there are no true negatives."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    negatives = y_true == 0
    return _rate(float(np.count_nonzero(y_pred[negatives] == 0)), float(np.count_nonzero(negatives)))


def balanced_accuracy(y_true: IntArray, y_pred: IntArray) -> float:
    """(TPR + TNR) / 2.

    Computed directly rather than via sklearn so the constant-classifier
    anchors of spec §2.3 hold exactly (0.5) without warnings on degenerate
    inputs.
    """
    return 0.5 * (true_positive_rate(y_true, y_pred) + true_negative_rate(y_true, y_pred))


def recall(y_true: IntArray, y_pred: IntArray) -> float:
    """TPR. Safe at ``M_100``: the denominator counts true, not predicted, positives."""
    return true_positive_rate(y_true, y_pred)


def imbalance_ratio(y: IntArray) -> float:
    """IR = S_min / (S_maj + S_min), paper Equation 1.

    Not part of the FBU metric; retained for the out-of-scope streaming phase.
    """
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        return 0.0
    return float(counts.min() / (counts.max() + counts.min()))


PERFORMANCE_METRICS: dict[str, PerformanceMetric] = {
    "balanced_accuracy": balanced_accuracy,
    "recall": recall,
}


def get_performance_metric(name: str) -> PerformanceMetric:
    """Look up a performance metric by name, rejecting the excluded ones."""
    if name in {"accuracy", "f1", "precision"}:
        raise ValueError(
            f"'{name}' is excluded from the performance axis by spec §2.2 [D8]"
        )
    try:
        return PERFORMANCE_METRICS[name]
    except KeyError:
        raise ValueError(
            f"unknown performance metric '{name}'; choose from {sorted(PERFORMANCE_METRICS)}"
        ) from None


__all__ = [
    "balanced_accuracy",
    "recall",
    "true_positive_rate",
    "true_negative_rate",
    "imbalance_ratio",
    "PERFORMANCE_METRICS",
    "get_performance_metric",
]
