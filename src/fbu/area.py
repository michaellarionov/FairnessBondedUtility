"""Trade-off area for region-2 points (spec §2.5, `[D2]`).

The paper names "the area enclosed by the fairness-performance points and the
baseline" but never states the enclosing boundary. Here the region is bounded
above by ``y = y_p``, on the right by ``x = f_p``, and below-left by the curve:

    f_a = the leftmost f in [f_o, f_p] with g(f) = y_p
    A   = ∫_{f_a}^{f_p} (y_p − g(f)) df

Areas are normalised by ``A_max``, the same integral taken at the original
model's performance across the full fairness range, so the score is comparable
across datasets and metric pairs. Larger area = better trade-off.
"""

from __future__ import annotations

import numpy as np

from .baseline import BaselineCurve
from .regions import EPS


def enclosed_area(
    curve: BaselineCurve,
    f_p: float,
    y_p: float,
    f_o: float,
    y_o: float,
    eps: float = EPS,
) -> float:
    """Raw enclosed area ``A`` for a point in the trade-off quadrant `[D2]`.

    Returns 0.0 for points that enclose nothing: at or left of ``f_o``, or below
    the curve (region 4). ``y_p`` is lifted to ``g(f_p)`` when it sits below it
    by less than ``eps``, so a point classified region 2 on the tolerance never
    fails to produce an area.
    """
    del y_o  # the upper bound is y_p; y_o only enters A_max
    f_p = float(f_p)
    y_p = float(y_p)
    if f_p <= f_o:
        return 0.0
    g_at_p = curve.evaluate(f_p)
    if y_p < g_at_p - eps:
        return 0.0
    y_eff = max(y_p, g_at_p)
    f_a = curve.solve_fairness(y_eff, f_lo=f_o)
    return float(y_eff * (f_p - f_a) - curve.integrate(f_a, f_p))


def max_area(curve: BaselineCurve, f_o: float, y_o: float) -> float:
    """``A_max`` = ∫_{f_o}^{f_max} (y_o − g(f)) df, the normaliser of spec §2.5.

    ``f_max`` is the curve's right-hand knot, which is the ``M_100`` fairness
    score — exactly 1.0 whenever the constant classifier is perfectly fair.
    """
    f_hi = curve.f_max
    if f_hi <= f_o:
        return 0.0
    return float(y_o * (f_hi - f_o) - curve.integrate(f_o, f_hi))


def normalised_area(
    curve: BaselineCurve,
    f_p: float,
    y_p: float,
    f_o: float,
    y_o: float,
    eps: float = EPS,
) -> tuple[float, float]:
    """Return ``(A, A_norm)``. ``A_norm`` is 0.0 when ``A_max`` degenerates to 0."""
    area = enclosed_area(curve, f_p, y_p, f_o, y_o, eps)
    denominator = max_area(curve, f_o, y_o)
    return area, (area / denominator if denominator > 0.0 else 0.0)


class EnclosedArea:
    """Default :class:`~fbu.types.TradeoffArea`: the closed-region integral `[D2]`."""

    name = "enclosed_integral"

    def __call__(
        self,
        curve: BaselineCurve,
        f_p: float,
        y_p: float,
        f_o: float,
        y_o: float,
    ) -> float:
        return enclosed_area(curve, f_p, y_p, f_o, y_o)


class PerpendicularDistance:
    """Alternative quantifier: Euclidean distance from P to the baseline polyline.

    Provided for the sensitivity check spec §2.5 asks for. Not normalised, so it
    is not comparable with :class:`EnclosedArea` values.
    """

    name = "perpendicular_distance"

    def __call__(
        self,
        curve: BaselineCurve,
        f_p: float,
        y_p: float,
        f_o: float,
        y_o: float,
    ) -> float:
        del f_o, y_o
        point = np.array([float(f_p), float(y_p)])
        best = np.inf
        knots = list(curve.knots())
        for (ax, ay), (bx, by) in zip(knots[:-1], knots[1:]):
            a = np.array([ax, ay])
            b = np.array([bx, by])
            ab = b - a
            t = float(np.clip(np.dot(point - a, ab) / np.dot(ab, ab), 0.0, 1.0))
            best = min(best, float(np.linalg.norm(point - (a + t * ab))))
        return best


__all__ = [
    "enclosed_area",
    "max_area",
    "normalised_area",
    "EnclosedArea",
    "PerpendicularDistance",
]
