"""Massaging (Kamiran & Calders 2009, paper ref [25]).

Pre-processing that relabels the training rows nearest the decision boundary:
promote the highest-scoring unprivileged negatives, demote the lowest-scoring
privileged positives, in equal numbers, until the training-label SPD is ≈ 0.
Then refit on the massaged labels. The ranking uses the continuous score the
base model already provides.
"""

from __future__ import annotations

import numpy as np

from ..models.scorers import LogitScorer
from ..types import FloatArray, IntArray, Predictions, ScorerFactory, SplitData

NAME = "massaging"


def n_promotions(y: IntArray, s: IntArray) -> int:
    """Number of promote/demote pairs needed to erase the label SPD.

    ``M = disc · n_priv · n_unpriv / n`` (Kamiran & Calders): flipping M
    unprivileged negatives up and M privileged positives down moves the two
    group rates toward each other by exactly the observed discrimination.
    """
    y = np.asarray(y)
    s = np.asarray(s)
    n = y.shape[0]
    n_priv = int(np.count_nonzero(s == 1))
    n_unpriv = n - n_priv
    if n_priv == 0 or n_unpriv == 0:
        return 0
    disc = float(np.mean(y[s == 1])) - float(np.mean(y[s == 0]))
    if disc <= 0:
        return 0
    return int(round(disc * n_priv * n_unpriv / n))


def massage_labels(y: IntArray, s: IntArray, scores: FloatArray) -> IntArray:
    """Return relabelled training targets. The number of flips is capped by the
    smaller of the promotable and demotable pools."""
    y = np.asarray(y).astype(np.int64, copy=True)
    s = np.asarray(s)
    scores = np.asarray(scores, dtype=np.float64)

    promotable = np.flatnonzero((s == 0) & (y == 0))
    demotable = np.flatnonzero((s == 1) & (y == 1))
    m = min(n_promotions(y, s), promotable.size, demotable.size)
    if m == 0:
        return y

    # Closest to the boundary from below / above respectively.
    promote = promotable[np.argsort(-scores[promotable], kind="stable")[:m]]
    demote = demotable[np.argsort(scores[demotable], kind="stable")[:m]]
    y[promote] = 1
    y[demote] = 0
    return y


def massaging(
    data: SplitData,
    scorer_factory: ScorerFactory = LogitScorer,
    threshold: float = 0.5,
) -> Predictions:
    """Rank with the base model, massage the training labels, refit, predict test."""
    ranker = scorer_factory().fit(data.X_train, data.y_train)
    train_scores = ranker.score(data.X_train)
    y_massaged = massage_labels(data.y_train, data.s_train, train_scores)

    model = scorer_factory().fit(data.X_train, y_massaged)
    scores = model.score(data.X_test)
    return Predictions(
        y_true=data.y_test,
        y_pred=(scores >= threshold).astype(int),
        s=data.s_test,
        score=scores,
        name=NAME,
    )


__all__ = ["massaging", "massage_labels", "n_promotions", "NAME"]
