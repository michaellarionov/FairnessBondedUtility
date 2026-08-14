"""Hand-constructed curves, points and prediction arrays with known geometry.

Everything here is built so that expected values can be derived on paper; the
derivations live in the docstrings of the tests that use them.
"""

from __future__ import annotations

import numpy as np

from fbu.baseline import BaselineCurve
from fbu.pseudo import PseudoPoint
from fbu.types import Predictions


def point(ratio: float, fairness: float, performance: float) -> PseudoPoint:
    """A pseudo-model knot with zero seed spread."""
    return PseudoPoint(
        ratio=ratio,
        fairness=fairness,
        performance=performance,
        fairness_std=0.0,
        performance_std=0.0,
        n_seeds=1,
    )


def line_curve() -> BaselineCurve:
    """Straight baseline from M_ori (0.8, 0.75) to M_100 (1.0, 0.5).

    g(f) = 0.75 − 1.25 (f − 0.8), slope −1.25.
    A_max = area of the triangle with base 0.2 and height 0.25 = 0.025.
    """
    return BaselineCurve.from_points([point(0.0, 0.8, 0.75), point(1.0, 1.0, 0.5)])


def two_segment_curve() -> BaselineCurve:
    """M_ori (0.6, 0.9) → (0.8, 0.8) → M_100 (1.0, 0.5).

    Segment slopes are −0.5 then −1.5.
    ∫ g over [0.6, 1.0] = 0.2·(0.9+0.8)/2 + 0.2·(0.8+0.5)/2 = 0.17 + 0.13 = 0.30.
    A_max = 0.9·0.4 − 0.30 = 0.36 − 0.30 = 0.06.
    """
    return BaselineCurve.from_points(
        [point(0.0, 0.6, 0.9), point(0.5, 0.8, 0.8), point(1.0, 1.0, 0.5)]
    )


def synthetic_original(n: int = 4000, seed: int = 0) -> Predictions:
    """A biased original model on synthetic data, for tests that avoid Adult.

    Half the rows are privileged. Base rates are 40% positive (privileged) and
    20% positive (unprivileged). The model is correct except that it

    * suppresses half the unprivileged positives and a fifth of the privileged
      ones, leaving false negatives available in *both* groups, and
    * invents false positives for a tenth of the privileged negatives,

    so SPD and EOD are both clearly positive while balanced accuracy stays well
    above 0.5.
    """
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    s[: n // 2] = 1
    y_true = np.where(
        s == 1, rng.binomial(1, 0.4, n), rng.binomial(1, 0.2, n)
    ).astype(int)

    y_pred = y_true.copy()
    for group, share in ((0, 0.5), (1, 0.2)):
        positives = np.flatnonzero((s == group) & (y_true == 1))
        y_pred[rng.choice(positives, size=int(positives.size * share), replace=False)] = 0
    priv_neg = np.flatnonzero((s == 1) & (y_true == 0))
    y_pred[rng.choice(priv_neg, size=priv_neg.size // 10, replace=False)] = 1

    # A monotone stand-in for a model score, only used by score-consuming code.
    score = y_pred.astype(float) * 0.6 + rng.uniform(0.0, 0.4, n)
    return Predictions(y_true=y_true, y_pred=y_pred, s=s, score=score, name="original")


def _flip(
    prediction: Predictions, candidates: np.ndarray, count: int, value: int, name: str
) -> Predictions:
    """Flip the first ``count`` candidates to ``value``.

    Runs out loudly: a silent no-op would make a construction test pass for the
    wrong reason (the technique would simply equal the original).
    """
    if candidates.size < count:
        raise AssertionError(
            f"only {candidates.size} candidate rows available, needed {count}"
        )
    y_pred = prediction.y_pred.copy()
    y_pred[candidates[:count]] = value
    return prediction.with_predictions(y_pred, name=name)


def fix_false_negatives(
    prediction: Predictions, group: int, count: int, name: str
) -> Predictions:
    """Correct ``count`` false negatives inside one group.

    Raises TPR (so both balanced accuracy and recall rise) and raises that
    group's favorable rate. Applied to the unprivileged group this improves
    fairness too (region 1); applied to the privileged group it worsens fairness
    while still improving performance (region 3). The construction therefore
    behaves the same way for every SPD/EOD × balanced-accuracy/recall pair.
    """
    candidates = np.flatnonzero(
        (prediction.s == group) & (prediction.y_true == 1) & (prediction.y_pred == 0)
    )
    return _flip(prediction, candidates, count, 1, name)


def add_false_positives(
    prediction: Predictions, group: int, count: int, name: str
) -> Predictions:
    """Invent ``count`` false positives inside one group: TNR falls, that group's
    favorable rate rises."""
    candidates = np.flatnonzero(
        (prediction.s == group) & (prediction.y_true == 0) & (prediction.y_pred == 0)
    )
    return _flip(prediction, candidates, count, 1, name)


def drop_true_positives(
    prediction: Predictions, group: int, count: int, name: str
) -> Predictions:
    """Discard ``count`` correct positives inside one group: TPR falls."""
    candidates = np.flatnonzero(
        (prediction.s == group) & (prediction.y_true == 1) & (prediction.y_pred == 1)
    )
    return _flip(prediction, candidates, count, 0, name)


__all__ = [
    "point",
    "line_curve",
    "two_segment_curve",
    "synthetic_original",
    "fix_false_negatives",
    "add_false_positives",
    "drop_true_positives",
]
