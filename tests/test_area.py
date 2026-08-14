"""M5 acceptance tests — trade-off area (spec §2.5, §5.2 M5).

Geometry, on the straight baseline g(f) = 0.75 − 1.25 (f − 0.8) between
M_ori (0.8, 0.75) and M_100 (1.0, 0.5):

    A_max = ½ · 0.2 · 0.25 = 0.025
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fbu.area import EnclosedArea, PerpendicularDistance, enclosed_area, max_area, normalised_area

from .fixtures.golden_cases import line_curve, two_segment_curve

F_O, Y_O = 0.8, 0.75


@pytest.fixture()
def curve():
    return line_curve()


def test_max_area_is_the_hand_triangle(curve):
    """A_max = 0.75·0.2 − ∫_{0.8}^{1} g = 0.15 − 0.125 = 0.025."""
    assert max_area(curve, F_O, Y_O) == pytest.approx(0.025)


def test_point_on_the_curve_has_zero_area(curve):
    """f_a coincides with f_p, so the enclosed region is degenerate."""
    assert enclosed_area(curve, 0.9, 0.625, F_O, Y_O) == pytest.approx(0.0)
    assert enclosed_area(curve, 1.0, 0.5, F_O, Y_O) == pytest.approx(0.0)


def test_corner_point_normalises_to_one(curve):
    """P = (1.0, y_o) encloses exactly A_max, so A_norm = 1.0."""
    area, area_norm = normalised_area(curve, 1.0, Y_O, F_O, Y_O)
    assert area == pytest.approx(0.025)
    assert area_norm == pytest.approx(1.0)


def test_hand_computable_trapezoid(curve):
    """P = (1.0, 0.625).

    f_a solves g(f) = 0.625 → f_a = 0.9. On [0.9, 1.0] the gap y_p − g(f) grows
    linearly from 0 to 0.625 − 0.5 = 0.125, so A is the triangle
    ½ · 0.1 · 0.125 = 0.00625 and A_norm = 0.00625 / 0.025 = 0.25.
    """
    area, area_norm = normalised_area(curve, 1.0, 0.625, F_O, Y_O)
    assert area == pytest.approx(0.00625)
    assert area_norm == pytest.approx(0.25)


def test_area_across_two_segments():
    """On the two-segment curve, P = (1.0, 0.8).

    f_a = 0.8 because g(0.8) = 0.8 exactly. Then
    A = 0.8·0.2 − ∫_{0.8}^{1} g = 0.16 − 0.13 = 0.03, and with A_max = 0.06 the
    normalised area is 0.5.
    """
    curve = two_segment_curve()
    area, area_norm = normalised_area(curve, 1.0, 0.8, 0.6, 0.9)
    assert area == pytest.approx(0.03)
    assert area_norm == pytest.approx(0.5)


def test_region_4_and_left_of_origin_enclose_nothing(curve):
    """Below the curve, or at/left of f_o, there is nothing to integrate."""
    assert enclosed_area(curve, 0.9, 0.60, F_O, Y_O) == 0.0
    assert enclosed_area(curve, 0.8, 0.70, F_O, Y_O) == 0.0
    assert enclosed_area(curve, 0.7, 0.70, F_O, Y_O) == 0.0


def test_tolerance_lifts_points_a_hair_below_the_curve(curve):
    """A point classified region 2 on the ε tolerance still yields an area."""
    assert enclosed_area(curve, 0.9, 0.625 - 1e-12, F_O, Y_O) == pytest.approx(0.0, abs=1e-9)


@settings(max_examples=100, deadline=None)
@given(
    f_p=st.floats(min_value=0.82, max_value=0.99),
    y_p=st.floats(min_value=0.51, max_value=0.74),
    df=st.floats(min_value=0.005, max_value=0.01),
    dy=st.floats(min_value=0.005, max_value=0.01),
)
def test_area_is_monotone_in_both_coordinates(f_p, y_p, df, dy):
    """A grows when the point moves right or up: the enclosed set only gains area.

    Points below the curve are skipped — they enclose nothing by construction,
    so monotonicity is asserted only where an area exists.
    """
    curve = line_curve()
    base = enclosed_area(curve, f_p, y_p, F_O, Y_O)
    if base <= 0.0:
        return
    assert enclosed_area(curve, f_p + df, y_p, F_O, Y_O) >= base
    assert enclosed_area(curve, f_p, y_p + dy, F_O, Y_O) >= base


def test_enclosed_area_protocol_matches_the_function(curve):
    quantifier = EnclosedArea()
    assert quantifier.name == "enclosed_integral"
    assert quantifier(curve, 1.0, 0.625, F_O, Y_O) == pytest.approx(0.00625)


def test_perpendicular_distance_alternative(curve):
    """The swappable sensitivity-check quantifier: zero on the curve, positive above.

    From (1.0, 0.625) the nearest point of the segment through (0.8, 0.75) and
    (1.0, 0.5) is found by projection; the distance must be strictly positive and
    below the vertical gap 0.125.
    """
    quantifier = PerpendicularDistance()
    assert quantifier.name == "perpendicular_distance"
    assert quantifier(curve, 0.9, 0.625, F_O, Y_O) == pytest.approx(0.0, abs=1e-12)
    distance = quantifier(curve, 1.0, 0.625, F_O, Y_O)
    assert 0.0 < distance < 0.125
