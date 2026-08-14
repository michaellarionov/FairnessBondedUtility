"""Group-wise thresholds: post-processing that equalises TPR across groups.

One threshold per group, each chosen on **train** so that both groups reach the
same true-positive rate. This targets EOD directly, so it should be the area
leader on the EOD metric pair; SPD is not its objective.
"""

from __future__ import annotations

import numpy as np

from ..metrics.performance import true_positive_rate
from ..models.scorers import LogitScorer
from ..types import FloatArray, IntArray, Predictions, ScorerFactory, SplitData

NAME = "group_threshold"


def group_thresholds(
    y: IntArray,
    s: IntArray,
    scores: FloatArray,
    target_tpr: float | None = None,
) -> dict[int, float]:
    """Per-group score thresholds attaining ``target_tpr`` on ``y``/``scores``.

    The default target is the pooled TPR of the 0.5-thresholded model, which
    keeps overall recall close to the original while removing the gap between
    groups. For a group with positive scores ``P``, the threshold reaching TPR
    ``t`` is the (1 − t) quantile of ``P``.
    """
    y = np.asarray(y)
    s = np.asarray(s)
    scores = np.asarray(scores, dtype=np.float64)

    if target_tpr is None:
        target_tpr = true_positive_rate(y, (scores >= 0.5).astype(np.int64))

    thresholds: dict[int, float] = {}
    for group in (0, 1):
        positives = scores[(s == group) & (y == 1)]
        if positives.size == 0 or target_tpr <= 0.0:
            thresholds[group] = float(np.inf)  # never predict favorable
        elif target_tpr >= 1.0:
            thresholds[group] = float(-np.inf)
        else:
            thresholds[group] = float(np.quantile(positives, 1.0 - target_tpr))
    return thresholds


def apply_group_thresholds(
    scores: FloatArray, s: IntArray, thresholds: dict[int, float]
) -> IntArray:
    scores = np.asarray(scores, dtype=np.float64)
    s = np.asarray(s)
    out = np.zeros(scores.shape[0], dtype=np.int64)
    for group, tau in thresholds.items():
        mask = s == group
        out[mask] = (scores[mask] >= tau).astype(np.int64)
    return out


def group_threshold(
    data: SplitData,
    scorer_factory: ScorerFactory = LogitScorer,
    target_tpr: float | None = None,
) -> Predictions:
    """Fit once, then apply TPR-equalising thresholds fitted on train."""
    model = scorer_factory().fit(data.X_train, data.y_train)
    thresholds = group_thresholds(
        data.y_train, data.s_train, model.score(data.X_train), target_tpr
    )
    scores = model.score(data.X_test)
    return Predictions(
        y_true=data.y_test,
        y_pred=apply_group_thresholds(scores, data.s_test, thresholds),
        s=data.s_test,
        score=scores,
        name=NAME,
    )


__all__ = ["group_threshold", "group_thresholds", "apply_group_thresholds", "NAME"]
