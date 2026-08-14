"""Fairness metrics for FBU (spec §2.2).

Two families live here:

* **Static** ``spd`` / ``eod`` — signed, 0 == fair, lower-is-better. These are
  what this phase uses (`[D7]`: Adult has no temporal order, so the decay in
  paper Equations 2–3 would run over an arbitrary shuffle).
* **Cumulative** ``cum_spd`` / ``cum_eod`` — paper Equations 2–3, implemented
  and tested for the future streaming phase but not used here.

``fairness_score`` maps either family onto the higher-is-better axis the
baseline curve needs (`[D1]`).
"""

from __future__ import annotations

import numpy as np

from ..types import FloatArray, IntArray

#: Paper §6.3.2 finds λ ≥ 0.5 stable.
DEFAULT_LAMBDA = 0.5


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _positive_rate(y_pred: IntArray, mask: np.ndarray) -> float:
    return _rate(float(np.count_nonzero(y_pred[mask] == 1)), float(np.count_nonzero(mask)))


def spd(y_true: IntArray, y_pred: IntArray, s: IntArray) -> float:
    """SPD = P(ŷ=1 | S=1) − P(ŷ=1 | S=0).

    Privileged minus unprivileged (paper Eq. 2 conditions both terms on S=s;
    that is a typo, see spec §1.3 G7). ``y_true`` is unused but kept in the
    signature so all fairness metrics share one call shape.
    """
    y_pred = np.asarray(y_pred)
    s = np.asarray(s)
    return _positive_rate(y_pred, s == 1) - _positive_rate(y_pred, s == 0)


def eod(y_true: IntArray, y_pred: IntArray, s: IntArray) -> float:
    """EOD = TPR(S=1) − TPR(S=0)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    s = np.asarray(s)
    return _positive_rate(y_pred, (s == 1) & (y_true == 1)) - _positive_rate(
        y_pred, (s == 0) & (y_true == 1)
    )


def _cumulative(
    y_pred: IntArray,
    s: IntArray,
    eligible: np.ndarray,
    lambda_decay: float,
) -> FloatArray:
    """Shared decay recursion for Equations 2–3.

    ``eligible`` selects the rows that contribute to the running rates (all
    rows for CSPD, ``y_true == 1`` rows for CEOD). A row with a zero
    denominator carries the previous value forward.
    """
    if not 0.0 <= lambda_decay <= 1.0:
        raise ValueError(f"lambda_decay must lie in [0, 1], got {lambda_decay}")
    y_pred = np.asarray(y_pred)
    s = np.asarray(s)
    n = y_pred.shape[0]
    out = np.zeros(n, dtype=np.float64)

    priv_fav = priv_tot = unpriv_fav = unpriv_tot = 0.0
    prev = 0.0
    for t in range(n):
        if eligible[t]:
            if s[t] == 1:
                priv_tot += 1.0
                priv_fav += float(y_pred[t] == 1)
            else:
                unpriv_tot += 1.0
                unpriv_fav += float(y_pred[t] == 1)
        if priv_tot > 0.0 and unpriv_tot > 0.0:
            raw = priv_fav / priv_tot - unpriv_fav / unpriv_tot
            prev = (1.0 - lambda_decay) * raw + lambda_decay * prev
        # else: denominator empty — carry the previous value forward (spec §2.2)
        out[t] = prev
    return out


def cum_spd(
    y_true: IntArray,
    y_pred: IntArray,
    s: IntArray,
    lambda_decay: float = DEFAULT_LAMBDA,
) -> FloatArray:
    """Running CSPD (paper Eq. 2). Not used this phase; see `[D7]`."""
    y_pred = np.asarray(y_pred)
    return _cumulative(y_pred, s, np.ones(y_pred.shape[0], dtype=bool), lambda_decay)


def cum_eod(
    y_true: IntArray,
    y_pred: IntArray,
    s: IntArray,
    lambda_decay: float = DEFAULT_LAMBDA,
) -> FloatArray:
    """Running CEOD (paper Eq. 3): CSPD with every count conditioned on y_true == 1."""
    return _cumulative(y_pred, s, np.asarray(y_true) == 1, lambda_decay)


def cum_spd_final(
    y_true: IntArray,
    y_pred: IntArray,
    s: IntArray,
    lambda_decay: float = DEFAULT_LAMBDA,
) -> float:
    """Last value of :func:`cum_spd`, matching the ``FairnessMetric`` shape."""
    return float(cum_spd(y_true, y_pred, s, lambda_decay)[-1])


def cum_eod_final(
    y_true: IntArray,
    y_pred: IntArray,
    s: IntArray,
    lambda_decay: float = DEFAULT_LAMBDA,
) -> float:
    """Last value of :func:`cum_eod`, matching the ``FairnessMetric`` shape."""
    return float(cum_eod(y_true, y_pred, s, lambda_decay)[-1])


def fairness_score(metric_value: float) -> float:
    """1 − |metric|, the quantity on the fairness axis (`[D1]`).

    Range [0, 1], higher is fairer. Putting a raw signed SPD/EOD on that axis
    is a bug: the baseline curve construction assumes higher-is-better.
    """
    value = float(metric_value)
    if not np.isfinite(value):
        raise ValueError(f"fairness metric must be finite, got {value}")
    return 1.0 - abs(value)


FAIRNESS_METRICS: dict[str, object] = {"spd": spd, "eod": eod}

#: Cumulative variants, registered separately so this phase cannot pick them up
#: by accident (`[D7]`).
CUMULATIVE_FAIRNESS_METRICS: dict[str, object] = {
    "cum_spd": cum_spd_final,
    "cum_eod": cum_eod_final,
}


def get_fairness_metric(name: str):
    """Look up a static fairness metric by name."""
    try:
        return FAIRNESS_METRICS[name]
    except KeyError:
        if name in CUMULATIVE_FAIRNESS_METRICS:
            raise ValueError(
                f"'{name}' is a streaming metric and is out of scope this phase [D7]"
            ) from None
        raise ValueError(
            f"unknown fairness metric '{name}'; choose from {sorted(FAIRNESS_METRICS)}"
        ) from None


__all__ = [
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
