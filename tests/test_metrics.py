"""Tests for fbu.metrics."""

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from fbu.metrics import (
    balanced_accuracy,
    cumulative_eod,
    cumulative_spd,
    imbalance_ratio,
)


# ---------------------------------------------------------------------------
# imbalance_ratio
# ---------------------------------------------------------------------------

def test_imbalance_ratio_balanced():
    y = np.array([0] * 50 + [1] * 50)
    assert imbalance_ratio(y) == pytest.approx(0.5)


def test_imbalance_ratio_minority():
    y = np.array([1] * 10 + [0] * 90)
    assert imbalance_ratio(y) == pytest.approx(0.1)


def test_imbalance_ratio_single_class():
    y = np.ones(100, dtype=int)
    assert imbalance_ratio(y) == 0.0


# ---------------------------------------------------------------------------
# cumulative_spd
# ---------------------------------------------------------------------------

def test_cumulative_spd_perfect_fairness():
    """When both groups receive identical positive rates, SPD converges to 0.

    The very first timestep may have SPD ≠ 0 because only one group has been
    seen; once both groups appear the running average stabilises at 0.
    """
    n = 500
    sensitive = np.tile([0, 1], n // 2)
    predictions = np.tile([1, 1], n // 2)
    y_true = np.ones(n, dtype=int)
    cspd = cumulative_spd(predictions, sensitive, y_true)
    # After the first pair (t≥1) SPD should be 0 since rates are equal
    assert np.allclose(cspd[1:], 0.0, atol=1e-10)


def test_cumulative_spd_biased():
    """Privileged always gets 1, unprivileged always gets 0 → large SPD."""
    n = 200
    sensitive = np.array([1, 0] * (n // 2))
    predictions = sensitive.copy()   # pred == group membership
    y_true = np.ones(n, dtype=int)
    cspd = cumulative_spd(predictions, sensitive, y_true)
    # After enough samples, SPD should be close to 1.0
    assert cspd[-1] > 0.5


def test_cumulative_spd_length():
    n = 100
    sensitive = np.zeros(n, dtype=int)
    sensitive[:50] = 1
    predictions = np.zeros(n, dtype=int)
    y_true = np.zeros(n, dtype=int)
    cspd = cumulative_spd(predictions, sensitive, y_true)
    assert len(cspd) == n


def test_cumulative_spd_lambda_decay():
    """Higher lambda means more historical weight — last value should differ."""
    n = 200
    rng = np.random.default_rng(7)
    sensitive = rng.integers(0, 2, n)
    predictions = rng.integers(0, 2, n)
    y_true = rng.integers(0, 2, n)
    cspd_0 = cumulative_spd(predictions, sensitive, y_true, lambda_decay=0.0)
    cspd_5 = cumulative_spd(predictions, sensitive, y_true, lambda_decay=0.5)
    # Values should differ with different decay factors
    assert not np.allclose(cspd_0, cspd_5)


# ---------------------------------------------------------------------------
# cumulative_eod
# ---------------------------------------------------------------------------

def test_cumulative_eod_perfect_tpr():
    """Both groups have equal TPR → CEOD ≈ 0."""
    n = 400
    sensitive = np.tile([0, 1], n // 2)
    y_true = np.tile([1, 1], n // 2)
    predictions = np.tile([1, 1], n // 2)
    ceod = cumulative_eod(predictions, sensitive, y_true)
    assert np.allclose(ceod[-20:], 0.0, atol=1e-9)


def test_cumulative_eod_biased():
    """Privileged TP rate = 1, unprivileged TP rate = 0 → EOD = 1."""
    n = 200
    sensitive = np.array([1, 0] * (n // 2))
    y_true = np.ones(n, dtype=int)
    predictions = sensitive.copy()
    ceod = cumulative_eod(predictions, sensitive, y_true)
    assert ceod[-1] > 0.5


def test_cumulative_eod_length():
    n = 50
    sensitive = np.zeros(n, dtype=int)
    sensitive[:25] = 1
    y_true = np.zeros(n, dtype=int)
    predictions = np.zeros(n, dtype=int)
    ceod = cumulative_eod(predictions, sensitive, y_true)
    assert len(ceod) == n


# ---------------------------------------------------------------------------
# balanced_accuracy
# ---------------------------------------------------------------------------

def test_balanced_accuracy_perfect():
    y = np.array([0, 0, 1, 1])
    assert balanced_accuracy(y, y) == pytest.approx(1.0)


def test_balanced_accuracy_matches_sklearn():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    expected = balanced_accuracy_score(y_true, y_pred)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(expected)
