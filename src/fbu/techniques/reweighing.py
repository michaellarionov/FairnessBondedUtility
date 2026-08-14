"""Reweighing (Kamiran & Calders 2012, paper ref [26]).

Pre-processing: leave the labels alone and reweight the training rows so that
group membership and label become statistically independent,

    w(s, y) = P(S = s) · P(Y = y) / P(S = s, Y = y)

then pass the weights as ``sample_weight``.
"""

from __future__ import annotations

import numpy as np

from ..models.scorers import LogitScorer
from ..types import FloatArray, IntArray, Predictions, ScorerFactory, SplitData

NAME = "reweighing"


def reweighing_weights(y: IntArray, s: IntArray) -> FloatArray:
    """Per-row weights ``w(s, y)``. Cells with no rows cannot contribute, so
    their weight is irrelevant and set to 1.0."""
    y = np.asarray(y)
    s = np.asarray(s)
    n = y.shape[0]
    weights = np.ones(n, dtype=np.float64)
    for s_value in (0, 1):
        for y_value in (0, 1):
            cell = (s == s_value) & (y == y_value)
            n_cell = int(np.count_nonzero(cell))
            if n_cell == 0:
                continue
            p_s = np.count_nonzero(s == s_value) / n
            p_y = np.count_nonzero(y == y_value) / n
            weights[cell] = p_s * p_y / (n_cell / n)
    return weights


def reweighing(
    data: SplitData,
    scorer_factory: ScorerFactory = LogitScorer,
    threshold: float = 0.5,
) -> Predictions:
    """Fit the base model with reweighing weights on the training split."""
    weights = reweighing_weights(data.y_train, data.s_train)
    model = scorer_factory().fit(data.X_train, data.y_train, sample_weight=weights)
    scores = model.score(data.X_test)
    return Predictions(
        y_true=data.y_test,
        y_pred=(scores >= threshold).astype(int),
        s=data.s_test,
        score=scores,
        name=NAME,
    )


__all__ = ["reweighing", "reweighing_weights", "NAME"]
