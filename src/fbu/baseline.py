"""The FBU baseline curve g (spec §2.3).

``g`` is the piecewise-linear interpolation through the 11 points
``M_ori, M_10, …, M_100`` sorted ascending by ``fairness_score``. It exists for
exactly one purpose: splitting the trade-off quadrant into regions 2 and 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from .pseudo import (
    DEFAULT_N_SEEDS,
    DEFAULT_RATIOS,
    PseudoPoint,
    pseudo_points,
)
from .types import FairnessMetric, FloatArray, PerformanceMetric, Predictions

#: Fairness values closer than this are treated as one knot.
KNOT_TOL = 1e-12


@dataclass(frozen=True)
class BaselineCurve:
    """A non-increasing piecewise-linear curve over the fairness axis.

    Attributes
    ----------
    fairness, performance:
        Knot coordinates, ``fairness`` strictly ascending.
    points:
        The originating pseudo-model points in ratio order (``M_ori`` first),
        kept for plotting and diagnostics.
    """

    fairness: FloatArray
    performance: FloatArray
    points: tuple[PseudoPoint, ...] = ()

    def __post_init__(self) -> None:
        if self.fairness.shape != self.performance.shape:
            raise ValueError("fairness and performance must have equal shape")
        if self.fairness.shape[0] < 2:
            raise ValueError("a curve needs at least two distinct knots")
        if np.any(np.diff(self.fairness) <= 0):
            raise ValueError("fairness knots must be strictly ascending")

    # -- construction -------------------------------------------------------

    @classmethod
    def from_points(
        cls, points: Sequence[PseudoPoint], envelope: bool = False
    ) -> "BaselineCurve":
        """Build the curve from pseudo-model points.

        Knots sharing a fairness value (which happens when a metric saturates)
        are merged by keeping the larger performance, so ``g`` stays a function
        of fairness and remains the *upper* boundary of the swept points `[D9]`.
        """
        fair = np.array([p.fairness for p in points], dtype=np.float64)
        perf = np.array([p.performance for p in points], dtype=np.float64)
        order = np.argsort(fair, kind="stable")
        fair, perf = fair[order], perf[order]

        merged_f: list[float] = []
        merged_y: list[float] = []
        for f, y in zip(fair, perf):
            if merged_f and f - merged_f[-1] <= KNOT_TOL:
                merged_y[-1] = max(merged_y[-1], y)
            else:
                merged_f.append(float(f))
                merged_y.append(float(y))

        curve = cls(
            fairness=np.array(merged_f),
            performance=np.array(merged_y),
            points=tuple(points),
        )
        return curve.monotone_envelope() if envelope else curve

    # -- geometry -----------------------------------------------------------

    @property
    def f_min(self) -> float:
        return float(self.fairness[0])

    @property
    def f_max(self) -> float:
        return float(self.fairness[-1])

    def knots(self) -> Iterator[tuple[float, float]]:
        yield from zip(map(float, self.fairness), map(float, self.performance))

    def evaluate(self, f: float) -> float:
        """g(f). Fairness values outside the knot range clamp to the end knots."""
        return float(np.interp(float(f), self.fairness, self.performance))

    def evaluate_many(self, f: FloatArray) -> FloatArray:
        return np.interp(np.asarray(f, dtype=np.float64), self.fairness, self.performance)

    def integrate(self, f_lo: float, f_hi: float) -> float:
        """∫ g df over [f_lo, f_hi], analytically over the linear segments.

        No numerical quadrature: each segment contributes the exact trapezoid
        of its clipped portion (spec §2.5).
        """
        f_lo = float(np.clip(f_lo, self.f_min, self.f_max))
        f_hi = float(np.clip(f_hi, self.f_min, self.f_max))
        if f_hi <= f_lo:
            return 0.0
        total = 0.0
        for (a, ya), (b, yb) in zip(list(self.knots())[:-1], list(self.knots())[1:]):
            lo, hi = max(a, f_lo), min(b, f_hi)
            if hi <= lo:
                continue
            slope = (yb - ya) / (b - a)
            y_lo = ya + slope * (lo - a)
            y_hi = ya + slope * (hi - a)
            total += 0.5 * (y_lo + y_hi) * (hi - lo)
        return float(total)

    def solve_fairness(self, y: float, f_lo: float | None = None) -> float:
        """Leftmost f ≥ ``f_lo`` with g(f) = y.

        ``g`` is non-increasing, so the crossing is unique; taking the leftmost
        one keeps the result well defined when a flat segment sits exactly at
        ``y`` (spec §2.5).
        """
        y = float(y)
        start = self.f_min if f_lo is None else float(np.clip(f_lo, self.f_min, self.f_max))
        if self.evaluate(start) <= y:
            return start

        knots = [(f, self.evaluate(f)) for f in self.fairness if f > start]
        prev_f, prev_y = start, self.evaluate(start)
        for f, y_f in knots:
            if y_f <= y:
                if prev_y == y_f:  # flat segment already at y
                    return prev_f
                t = (prev_y - y) / (prev_y - y_f)
                return float(prev_f + t * (f - prev_f))
            prev_f, prev_y = f, y_f
        raise ValueError(
            f"g never reaches y={y} on [{start}, {self.f_max}]; the point lies "
            "below the whole curve (region 4, no area defined)"
        )

    # -- monotonicity diagnostics ------------------------------------------

    def max_inversion(self) -> float:
        """Largest increase in performance between consecutive knots (0 if monotone)."""
        diffs = np.diff(self.performance)
        return float(max(diffs.max(initial=0.0), 0.0))

    def is_non_increasing(self, tol: float = 0.0) -> bool:
        return self.max_inversion() <= tol

    def monotone_envelope(self) -> "BaselineCurve":
        """Upper non-increasing envelope: a right-to-left running maximum `[D5]`.

        Off by default. Raw curves are reported; this exists only for
        sensitivity checks.
        """
        envelope = np.maximum.accumulate(self.performance[::-1])[::-1]
        return BaselineCurve(
            fairness=self.fairness.copy(), performance=envelope, points=self.points
        )


def build_baseline(
    original: Predictions,
    performance: PerformanceMetric,
    fairness: FairnessMetric,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    n_seeds: int = DEFAULT_N_SEEDS,
    seed: int = 0,
    envelope: bool = False,
) -> BaselineCurve:
    """Construct g for one (performance, fairness) metric pair."""
    points = pseudo_points(original, performance, fairness, ratios, n_seeds, seed)
    return BaselineCurve.from_points(points, envelope=envelope)


def original_point(curve: BaselineCurve) -> tuple[float, float]:
    """The ``M_ori`` coordinates (f_o, y_o) that regions 1/3/5 are measured against."""
    if not curve.points:
        raise ValueError("curve carries no pseudo-model points")
    ori = curve.points[0]
    if ori.ratio != 0.0:
        raise ValueError("first pseudo-model point must be M_ori (ratio 0)")
    return ori.fairness, ori.performance


__all__ = ["BaselineCurve", "build_baseline", "original_point", "KNOT_TOL"]
