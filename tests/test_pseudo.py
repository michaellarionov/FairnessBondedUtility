"""M3 acceptance tests — pseudo-models (spec §2.3, §5.2 M3)."""

from __future__ import annotations

import numpy as np
import pytest

from fbu.metrics import balanced_accuracy, eod, fairness_score, recall, spd
from fbu.pseudo import (
    DEFAULT_RATIOS,
    majority_label,
    pseudo_point,
    pseudo_points,
    pseudo_predictions,
)

from .fixtures.golden_cases import synthetic_original


def test_majority_label_hand_cases():
    """L* is the majority of the ORIGINAL MODEL's predictions, ties resolving to 0."""
    assert majority_label(np.array([0, 0, 0, 1])) == 0
    assert majority_label(np.array([1, 1, 1, 0])) == 1
    assert majority_label(np.array([0, 1])) == 0


def test_m100_is_the_constant_classifier_for_every_seed():
    """M_100 replaces everything, so it equals the constant-L* array exactly."""
    original = synthetic_original()
    label = majority_label(original.y_pred)
    assert label == 0
    for seed_index in range(10):
        preds = pseudo_predictions(original.y_pred, 1.0, seed=0, seed_index=seed_index)
        assert np.array_equal(preds, np.full(len(original), label))


def test_replacement_count_is_exact():
    """A ratio of 0.3 over 1000 rows overwrites exactly 300 positions.

    The original is all-ones with the replacement label forced to 0, so every
    replaced position is observable.
    """
    y_pred = np.ones(1000, dtype=int)
    preds = pseudo_predictions(y_pred, 0.3, seed=0, seed_index=0, label=0)
    assert int(np.count_nonzero(preds == 0)) == 300


def test_ratio_zero_is_a_no_op_and_bounds_are_checked():
    y_pred = np.array([1, 0, 1, 1])
    assert np.array_equal(pseudo_predictions(y_pred, 0.0), y_pred)
    with pytest.raises(ValueError, match="ratio"):
        pseudo_predictions(y_pred, 1.5)


def test_seeds_draw_independently():
    """Different repetitions must replace different positions, or averaging over
    ``n_seeds`` would be pointless [D3]."""
    y_pred = np.ones(1000, dtype=int)
    first = pseudo_predictions(y_pred, 0.3, seed=0, seed_index=0, label=0)
    second = pseudo_predictions(y_pred, 0.3, seed=0, seed_index=1, label=0)
    assert not np.array_equal(first, second)
    # ...and the same (seed, repetition, ratio) is reproducible.
    assert np.array_equal(
        first, pseudo_predictions(y_pred, 0.3, seed=0, seed_index=0, label=0)
    )


def test_m100_hits_the_three_anchors():
    """Spec §2.3 anchors for L* = 0: SPD = EOD = 0 → fairness 1.0;
    balanced accuracy exactly 0.5; recall exactly 0.0."""
    original = synthetic_original()
    constant = pseudo_predictions(original.y_pred, 1.0)
    assert spd(original.y_true, constant, original.s) == pytest.approx(0.0)
    assert eod(original.y_true, constant, original.s) == pytest.approx(0.0)
    assert fairness_score(spd(original.y_true, constant, original.s)) == pytest.approx(1.0)
    assert balanced_accuracy(original.y_true, constant) == pytest.approx(0.5)
    assert recall(original.y_true, constant) == pytest.approx(0.0)


def test_pseudo_point_m100_is_deterministic():
    """M_100 needs no averaging: one draw, zero spread."""
    original = synthetic_original()
    knot = pseudo_point(original, balanced_accuracy, spd, ratio=1.0, n_seeds=20)
    assert knot.n_seeds == 1
    assert knot.fairness == pytest.approx(1.0)
    assert knot.performance == pytest.approx(0.5)
    assert knot.fairness_std == 0.0
    assert knot.label == "M_100"


def test_pseudo_points_shape_and_labels():
    original = synthetic_original()
    points = pseudo_points(original, balanced_accuracy, spd, n_seeds=5)
    assert len(points) == 11
    assert [p.label for p in points][:3] == ["M_ori", "M_10", "M_20"]
    assert points[0].ratio == 0.0
    assert points[0].fairness == pytest.approx(
        fairness_score(spd(original.y_true, original.y_pred, original.s))
    )
    assert tuple(p.ratio for p in points[1:]) == DEFAULT_RATIOS


def test_fairness_is_monotone_non_decreasing_in_p():
    """Replacing more predictions with one constant label can only shrink the
    gap between groups, so fairness rises with p.

    One inversion of at most 0.01 is tolerated as seed noise; more than that
    means a bug or too few seeds (spec §5.2 M3).
    """
    original = synthetic_original()
    points = pseudo_points(original, balanced_accuracy, spd, n_seeds=20)
    fairness = np.array([p.fairness for p in points])
    inversions = -np.diff(fairness)
    assert inversions.max(initial=0.0) <= 0.01
    assert fairness[-1] == pytest.approx(1.0)
    assert fairness[0] < fairness[-1]


def test_performance_is_non_increasing_in_p():
    """Balanced accuracy falls toward the 0.5 anchor as p rises."""
    original = synthetic_original()
    points = pseudo_points(original, balanced_accuracy, spd, n_seeds=20)
    performance = np.array([p.performance for p in points])
    assert np.diff(performance).max(initial=0.0) <= 0.01
    assert performance[-1] == pytest.approx(0.5)
