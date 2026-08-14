"""M4 acceptance tests — region classifier (spec §2.4, §5.2 M4).

All cases sit on the hand-built straight baseline of
``fixtures.golden_cases.line_curve``: M_ori = (0.8, 0.75), M_100 = (1.0, 0.5),
g(f) = 0.75 − 1.25 (f − 0.8). So g(0.9) = 0.625.
"""

from __future__ import annotations

import pytest

from fbu.regions import classify, region_label

from .fixtures.golden_cases import line_curve

F_O, Y_O = 0.8, 0.75


@pytest.fixture()
def curve():
    return line_curve()


@pytest.mark.parametrize(
    ("f_p", "y_p", "expected", "why"),
    [
        (0.90, 0.80, 1, "fairer and more accurate than M_ori"),
        (0.90, 0.70, 2, "fairer, less accurate, above g(0.9)=0.625"),
        (0.70, 0.80, 3, "less fair, more accurate"),
        (0.90, 0.60, 4, "fairer, less accurate, below g(0.9)=0.625"),
        (0.70, 0.70, 5, "less fair and less accurate"),
    ],
)
def test_one_point_per_region(curve, f_p, y_p, expected, why):
    assert classify(f_p, y_p, F_O, Y_O, curve) == expected, why


def test_boundary_on_f_o(curve):
    """f_p exactly at f_o with lower performance falls in the trade-off quadrant.

    g(0.8) = 0.75 and y_p = 0.70 < 0.75, so the point is below the curve → 4.
    """
    assert classify(0.80, 0.70, F_O, Y_O, curve) == 4


def test_boundary_on_y_o(curve):
    """y_p exactly at y_o counts as "not worse": left of f_o that is region 3,
    right of f_o it is region 1."""
    assert classify(0.70, 0.75, F_O, Y_O, curve) == 3
    assert classify(0.90, 0.75, F_O, Y_O, curve) == 1


def test_boundary_on_the_curve(curve):
    """A point exactly on g is effective: "above the line" includes the line."""
    assert classify(0.90, 0.625, F_O, Y_O, curve) == 2


def test_m_ori_classifies_as_region_1(curve):
    """M_ori ties both thresholds — a harmless degenerate case (spec §2.4)."""
    assert classify(F_O, Y_O, F_O, Y_O, curve) == 1


def test_tolerance_absorbs_float_noise(curve):
    """A point a femtometre below both thresholds is still region 1, not 5."""
    assert classify(F_O - 1e-15, Y_O - 1e-15, F_O, Y_O, curve) == 1


def test_regions_1_3_5_ignore_the_curve():
    """Regions 1/3/5 are quadrants of M_ori: swapping the curve cannot move them.

    Only the region-2/4 split may consult g, so a deliberately absurd curve must
    leave those three assignments untouched.
    """
    from fbu.baseline import BaselineCurve

    from .fixtures.golden_cases import point

    absurd = BaselineCurve.from_points([point(0.0, 0.8, 0.05), point(1.0, 1.0, 0.0)])
    assert classify(0.90, 0.80, F_O, Y_O, absurd) == 1
    assert classify(0.70, 0.80, F_O, Y_O, absurd) == 3
    assert classify(0.70, 0.70, F_O, Y_O, absurd) == 5
    # ...while the trade-off quadrant does move, because that is the curve's job.
    assert classify(0.90, 0.60, F_O, Y_O, absurd) == 2


def test_region_labels():
    assert region_label(1) == "Jointly advantageous"
    assert region_label(2) == "Impressive"
    assert region_label(3) == "Reversed"
    assert region_label(4) == "Deficient"
    assert region_label(5) == "Jointly disadvantageous"
