"""Pseudo-model construction (spec §2.3).

``M_p`` replaces a uniformly random ``p%`` of the original model's predictions
with the majority label ``L*`` of those predictions. ``M_100`` is therefore the
constant classifier ``L*``. Because the replacement is random, each ratio is
averaged over ``n_seeds`` seeds (`[D3]`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics.fairness import fairness_score
from .types import FairnessMetric, IntArray, PerformanceMetric, Predictions

#: p ∈ {10, 20, …, 100}% as fractions.
DEFAULT_RATIOS: tuple[float, ...] = tuple(round(0.1 * k, 2) for k in range(1, 11))

#: Repetitions per ratio (`[D3]`).
DEFAULT_N_SEEDS = 20


def majority_label(y_pred: IntArray) -> int:
    """L* — the majority label among the *original model's* predictions.

    Ties resolve to 0; on Adult L* is 0 by a wide margin (asserted in tests
    rather than assumed).
    """
    y_pred = np.asarray(y_pred)
    n_ones = int(np.count_nonzero(y_pred == 1))
    return 1 if n_ones > y_pred.shape[0] - n_ones else 0


def _rng(seed: int, seed_index: int, ratio: float) -> np.random.Generator:
    """Independent generator per (base seed, repetition, ratio).

    Ratios draw independently rather than nesting M_p's replaced indices inside
    M_{p+10}'s: the spec asks for a uniformly random p% at each ratio, and the
    mean over ``n_seeds`` supplies the smoothness that nesting would fake.
    """
    return np.random.default_rng([seed, seed_index, int(round(ratio * 100))])


def pseudo_predictions(
    y_pred: IntArray,
    ratio: float,
    seed: int = 0,
    seed_index: int = 0,
    label: int | None = None,
) -> IntArray:
    """Predictions of ``M_p``: ``ratio`` of ``y_pred`` overwritten with ``L*``."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must lie in [0, 1], got {ratio}")
    y_pred = np.asarray(y_pred)
    if label is None:
        label = majority_label(y_pred)
    n = y_pred.shape[0]
    n_replace = int(round(ratio * n))
    out = y_pred.astype(np.int64, copy=True)
    if n_replace == n:
        out[:] = label
        return out
    if n_replace > 0:
        idx = _rng(seed, seed_index, ratio).choice(n, size=n_replace, replace=False)
        out[idx] = label
    return out


@dataclass(frozen=True)
class PseudoPoint:
    """One baseline knot: the mean over seeds of a ratio's (fairness, performance)."""

    ratio: float
    fairness: float
    performance: float
    fairness_std: float
    performance_std: float
    n_seeds: int

    @property
    def label(self) -> str:
        return "M_ori" if self.ratio == 0.0 else f"M_{int(round(self.ratio * 100))}"


def pseudo_point(
    original: Predictions,
    performance: PerformanceMetric,
    fairness: FairnessMetric,
    ratio: float,
    n_seeds: int = DEFAULT_N_SEEDS,
    seed: int = 0,
    label: int | None = None,
) -> PseudoPoint:
    """Mean (fairness_score, performance) of ``M_p`` across ``n_seeds`` seeds `[D3]`."""
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    if label is None:
        label = majority_label(original.y_pred)

    # M_100 is deterministic, so one draw suffices; keeping the loop uniform
    # would only repeat identical work.
    repeats = 1 if ratio in (0.0, 1.0) else n_seeds
    fair_vals = np.empty(repeats)
    perf_vals = np.empty(repeats)
    for i in range(repeats):
        preds = (
            original.y_pred
            if ratio == 0.0
            else pseudo_predictions(original.y_pred, ratio, seed, i, label)
        )
        fair_vals[i] = fairness_score(fairness(original.y_true, preds, original.s))
        perf_vals[i] = performance(original.y_true, preds)

    return PseudoPoint(
        ratio=ratio,
        fairness=float(fair_vals.mean()),
        performance=float(perf_vals.mean()),
        fairness_std=float(fair_vals.std(ddof=0)),
        performance_std=float(perf_vals.std(ddof=0)),
        n_seeds=repeats,
    )


def pseudo_points(
    original: Predictions,
    performance: PerformanceMetric,
    fairness: FairnessMetric,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    n_seeds: int = DEFAULT_N_SEEDS,
    seed: int = 0,
) -> list[PseudoPoint]:
    """``M_ori`` followed by one :class:`PseudoPoint` per ratio, in ratio order."""
    label = majority_label(original.y_pred)
    return [
        pseudo_point(original, performance, fairness, ratio, n_seeds, seed, label)
        for ratio in (0.0, *ratios)
    ]


__all__ = [
    "DEFAULT_RATIOS",
    "DEFAULT_N_SEEDS",
    "majority_label",
    "pseudo_predictions",
    "PseudoPoint",
    "pseudo_point",
    "pseudo_points",
]
