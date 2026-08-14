"""M3 acceptance tests — baseline curve (spec §2.3, §5.2 M3)."""

from __future__ import annotations

import numpy as np
import pytest

from fbu.baseline import BaselineCurve, build_baseline, original_point
from fbu.metrics import balanced_accuracy, recall, spd
from fbu.pseudo import pseudo_points

from .fixtures.golden_cases import line_curve, point, synthetic_original, two_segment_curve


def test_curve_passes_through_every_knot():
    """g(f_i) == y_i at all 11 knots — the interpolation must not smooth them."""
    original = synthetic_original()
    curve = build_baseline(original, balanced_accuracy, spd, n_seeds=5)
    assert len(list(curve.knots())) == 11
    for f, y in curve.knots():
        assert curve.evaluate(f) == pytest.approx(y)


def test_curve_is_sorted_ascending_by_fairness():
    original = synthetic_original()
    curve = build_baseline(original, balanced_accuracy, spd, n_seeds=5)
    assert np.all(np.diff(curve.fairness) > 0)
    assert curve.f_max == pytest.approx(1.0)
    assert curve.points[0].ratio == 0.0  # M_ori stays first for region reference


def test_curve_endpoints_are_m_ori_and_m100():
    """The curve runs from M_ori up to the (1.0, 0.5) anchor of M_100."""
    original = synthetic_original()
    curve = build_baseline(original, balanced_accuracy, spd, n_seeds=5)
    f_o, y_o = original_point(curve)
    assert (curve.f_min, curve.performance[0]) == pytest.approx((f_o, y_o))
    assert (curve.f_max, float(curve.performance[-1])) == pytest.approx((1.0, 0.5))


def test_curve_is_non_increasing_with_the_default_seed_budget():
    """Reported raw, not enveloped [D5]: with 20 seeds the curve should already
    be non-increasing to within 0.01."""
    original = synthetic_original()
    for performance in (balanced_accuracy, recall):
        curve = build_baseline(original, performance, spd, n_seeds=20)
        assert curve.max_inversion() <= 0.01


def test_recall_curve_ends_at_zero():
    """With recall on the performance axis, M_100 sits at (1.0, 0.0)."""
    original = synthetic_original()
    curve = build_baseline(original, recall, spd, n_seeds=5)
    assert float(curve.performance[-1]) == pytest.approx(0.0)


def test_integrate_line_curve_by_hand():
    """Straight curve from (0.8, 0.75) to (1.0, 0.5).

    ∫ g over the whole domain is the trapezoid 0.2 · (0.75 + 0.5)/2 = 0.125.
    Over the right half [0.9, 1.0], g runs 0.625 → 0.5, giving
    0.1 · (0.625 + 0.5)/2 = 0.05625.
    """
    curve = line_curve()
    assert curve.integrate(0.8, 1.0) == pytest.approx(0.125)
    assert curve.integrate(0.9, 1.0) == pytest.approx(0.05625)
    assert curve.integrate(0.9, 0.9) == 0.0
    assert curve.integrate(1.0, 0.8) == 0.0  # reversed bounds enclose nothing


def test_integrate_spans_multiple_segments():
    """Two-segment curve: 0.2·(0.9+0.8)/2 + 0.2·(0.8+0.5)/2 = 0.17 + 0.13 = 0.30."""
    curve = two_segment_curve()
    assert curve.integrate(0.6, 1.0) == pytest.approx(0.30)
    assert curve.integrate(0.6, 0.8) == pytest.approx(0.17)
    assert curve.integrate(0.8, 1.0) == pytest.approx(0.13)


def test_integrate_clamps_outside_the_domain():
    """Values beyond the knots clamp, so integrating [0.0, 2.0] equals [0.8, 1.0]."""
    curve = line_curve()
    assert curve.integrate(0.0, 2.0) == pytest.approx(0.125)


def test_solve_fairness_by_hand():
    """On g(f) = 0.75 − 1.25 (f − 0.8): g = 0.625 at f = 0.9, g = 0.5 at f = 1.0.

    A target at or above the left endpoint returns the left endpoint itself.
    """
    curve = line_curve()
    assert curve.solve_fairness(0.625) == pytest.approx(0.9)
    assert curve.solve_fairness(0.5) == pytest.approx(1.0)
    assert curve.solve_fairness(0.75) == pytest.approx(0.8)
    assert curve.solve_fairness(0.9) == pytest.approx(0.8)


def test_solve_fairness_respects_the_lower_bound():
    curve = two_segment_curve()
    assert curve.solve_fairness(0.8, f_lo=0.6) == pytest.approx(0.8)
    assert curve.solve_fairness(0.65, f_lo=0.8) == pytest.approx(0.9)


def test_solve_fairness_rejects_points_below_the_curve():
    """A performance below g(f_max) has no crossing — that is a region-4 point."""
    curve = line_curve()
    with pytest.raises(ValueError, match="never reaches"):
        curve.solve_fairness(0.4)


def test_duplicate_fairness_knots_merge_to_the_upper_boundary():
    """Two knots at the same fairness collapse to the larger performance [D9],
    keeping g a function and the upper boundary of the swept points."""
    curve = BaselineCurve.from_points(
        [point(0.0, 0.8, 0.75), point(0.5, 0.9, 0.60), point(0.6, 0.9, 0.65), point(1.0, 1.0, 0.5)]
    )
    assert list(curve.fairness) == pytest.approx([0.8, 0.9, 1.0])
    assert curve.evaluate(0.9) == pytest.approx(0.65)


def test_monotone_envelope_is_off_by_default_but_available():
    """The envelope is a right-to-left running maximum [D5]."""
    points = [point(0.0, 0.8, 0.70), point(0.5, 0.9, 0.75), point(1.0, 1.0, 0.5)]
    raw = BaselineCurve.from_points(points)
    assert raw.max_inversion() == pytest.approx(0.05)
    assert not raw.is_non_increasing()

    enveloped = BaselineCurve.from_points(points, envelope=True)
    assert enveloped.is_non_increasing()
    assert list(enveloped.performance) == pytest.approx([0.75, 0.75, 0.5])


def test_curve_requires_two_distinct_knots():
    with pytest.raises(ValueError, match="two distinct knots"):
        BaselineCurve.from_points([point(0.0, 0.8, 0.75)])


def test_original_point_requires_m_ori_first():
    curve = BaselineCurve.from_points(
        [point(0.1, 0.8, 0.75), point(1.0, 1.0, 0.5)]
    )
    with pytest.raises(ValueError, match="M_ori"):
        original_point(curve)


def test_more_seeds_shrink_the_knot_spread():
    """The randomised baseline is only stable in the mean [D3]; more seeds must
    tighten the curve between independent constructions."""
    original = synthetic_original()
    fairness = {
        budget: np.array(
            [
                p.fairness
                for p in pseudo_points(original, balanced_accuracy, spd, n_seeds=budget, seed=s)
            ]
        )
        for budget in (2, 50)
        for s in (0,)
    }
    spread_small = np.abs(
        fairness[2]
        - np.array(
            [p.fairness for p in pseudo_points(original, balanced_accuracy, spd, n_seeds=2, seed=7)]
        )
    ).max()
    spread_large = np.abs(
        fairness[50]
        - np.array(
            [p.fairness for p in pseudo_points(original, balanced_accuracy, spd, n_seeds=50, seed=7)]
        )
    ).max()
    assert spread_large < spread_small
