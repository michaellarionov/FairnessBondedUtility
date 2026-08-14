"""M1 acceptance tests — metrics (spec §2.2, §5.2 M1).

Every expected value in this file is derived by hand in the test docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from fbu.metrics import (
    balanced_accuracy,
    cum_eod,
    cum_spd,
    eod,
    fairness_score,
    get_fairness_metric,
    get_performance_metric,
    imbalance_ratio,
    recall,
    spd,
)

# A hand-tabulated 8-row example reused across the static-metric tests.
#
#   row : y_true y_pred s
#     0 :   1      1    1
#     1 :   1      0    1
#     2 :   0      1    1
#     3 :   0      0    1
#     4 :   1      1    0
#     5 :   1      0    0
#     6 :   0      0    0
#     7 :   0      0    0
HAND_Y_TRUE = np.array([1, 1, 0, 0, 1, 1, 0, 0])
HAND_Y_PRED = np.array([1, 0, 1, 0, 1, 0, 0, 0])
HAND_S = np.array([1, 1, 1, 1, 0, 0, 0, 0])


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def test_perfect_classifier():
    """Perfect predictions: TPR = 1, TNR = 1 → balanced accuracy 1.0, recall 1.0."""
    y = np.array([0, 0, 1, 1, 1])
    assert balanced_accuracy(y, y) == pytest.approx(1.0)
    assert recall(y, y) == pytest.approx(1.0)


def test_constant_positive_performance():
    """All-ones on 2 positives / 3 negatives: TPR = 2/2 = 1, TNR = 0/3 = 0.

    balanced accuracy = (1 + 0) / 2 = 0.5; recall = 1.0.
    """
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.ones(5, dtype=int)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)
    assert recall(y_true, y_pred) == pytest.approx(1.0)


def test_constant_negative_performance():
    """All-zeros: TPR = 0/2 = 0, TNR = 3/3 = 1 → balanced accuracy 0.5, recall 0.0.

    These are the ``M_100`` anchors of spec §2.3 for Adult, where L* = 0.
    """
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.zeros(5, dtype=int)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)
    assert recall(y_true, y_pred) == pytest.approx(0.0)


def test_balanced_accuracy_hand_value():
    """y_true = [1,1,0,0,0], y_pred = [1,0,1,0,0].

    TPR = 1/2 = 0.5, TNR = 2/3 = 0.666...; balanced accuracy = 0.5833333...
    """
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(7.0 / 12.0)
    assert recall(y_true, y_pred) == pytest.approx(0.5)


def test_excluded_performance_metrics_rejected():
    """Accuracy / F1 / precision must not be reachable through the registry [D8]."""
    for name in ("accuracy", "f1", "precision"):
        with pytest.raises(ValueError, match="excluded"):
            get_performance_metric(name)
    assert get_performance_metric("recall") is recall


def test_imbalance_ratio():
    """10 positives, 90 negatives → 10 / (90 + 10) = 0.1 (paper Eq. 1)."""
    y = np.array([1] * 10 + [0] * 90)
    assert imbalance_ratio(y) == pytest.approx(0.1)
    assert imbalance_ratio(np.ones(5, dtype=int)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Static fairness metrics
# ---------------------------------------------------------------------------


def test_spd_hand_value():
    """Hand table: privileged ŷ=1 on rows 0,2 → 2/4 = 0.5; unprivileged row 4 → 1/4 = 0.25.

    SPD = 0.5 − 0.25 = 0.25.
    """
    assert spd(HAND_Y_TRUE, HAND_Y_PRED, HAND_S) == pytest.approx(0.25)


def test_eod_hand_value():
    """Hand table, restricted to y_true = 1.

    Privileged positives are rows 0,1 → TPR = 1/2 = 0.5.
    Unprivileged positives are rows 4,5 → TPR = 1/2 = 0.5.  EOD = 0.
    """
    assert eod(HAND_Y_TRUE, HAND_Y_PRED, HAND_S) == pytest.approx(0.0)


def test_eod_nonzero_hand_value():
    """y_true = [1,1,1,1], y_pred = [1,1,1,0], s = [1,1,0,0].

    TPR(S=1) = 2/2 = 1.0, TPR(S=0) = 1/2 = 0.5 → EOD = 0.5.
    Positive rates are the same sets here, so SPD = 1.0 − 0.5 = 0.5 as well.
    """
    y_true = np.ones(4, dtype=int)
    y_pred = np.array([1, 1, 1, 0])
    s = np.array([1, 1, 0, 0])
    assert eod(y_true, y_pred, s) == pytest.approx(0.5)
    assert spd(y_true, y_pred, s) == pytest.approx(0.5)


def test_constant_classifiers_are_perfectly_fair():
    """Both groups get the same rate under a constant classifier → SPD = EOD = 0."""
    y_true = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([1, 1, 1, 0, 0, 0])
    for constant in (np.zeros(6, dtype=int), np.ones(6, dtype=int)):
        assert spd(y_true, constant, s) == pytest.approx(0.0)
        assert eod(y_true, constant, s) == pytest.approx(0.0)


def test_fully_segregated_spd_is_one():
    """Privileged all ŷ=1, unprivileged all ŷ=0 → SPD = 1 − 0 = 1.0 (EOD likewise)."""
    s = np.array([1, 1, 1, 0, 0, 0])
    y_pred = s.copy()
    y_true = np.ones(6, dtype=int)
    assert spd(y_true, y_pred, s) == pytest.approx(1.0)
    assert eod(y_true, y_pred, s) == pytest.approx(1.0)


def test_spd_sign_is_privileged_minus_unprivileged():
    """Favouring the unprivileged group must give a negative SPD (spec §1.3 G7)."""
    s = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    y_true = np.ones(4, dtype=int)
    assert spd(y_true, y_pred, s) == pytest.approx(-1.0)


def test_empty_group_rate_is_zero_not_nan():
    """A group with no members contributes rate 0.0 rather than NaN."""
    s = np.ones(4, dtype=int)
    y_pred = np.array([1, 1, 0, 0])
    y_true = np.ones(4, dtype=int)
    assert spd(y_true, y_pred, s) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# fairness_score  [D1]
# ---------------------------------------------------------------------------


def test_fairness_score_maps_to_higher_is_better():
    """1 − |metric|: 0 → 1.0, ±0.25 → 0.75, ±1 → 0.0."""
    assert fairness_score(0.0) == pytest.approx(1.0)
    assert fairness_score(0.25) == pytest.approx(0.75)
    assert fairness_score(-0.25) == pytest.approx(0.75)
    assert fairness_score(1.0) == pytest.approx(0.0)
    assert fairness_score(-1.0) == pytest.approx(0.0)


def test_fairness_score_of_hand_table():
    """SPD = 0.25 → fairness_score = 0.75."""
    assert fairness_score(spd(HAND_Y_TRUE, HAND_Y_PRED, HAND_S)) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Cumulative fairness metrics (implemented, not used this phase)  [D7]
# ---------------------------------------------------------------------------


def test_cum_spd_lambda_zero_is_instantaneous_rate_difference():
    """With λ=0 the recursion collapses to p_t − u_t, recomputed here by prefix sums.

    Before the first unprivileged row appears the denominator is empty, so the
    value carries forward from CSPD_0 = 0.
    """
    rng = np.random.default_rng(0)
    n = 200
    s = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    y_true = rng.integers(0, 2, n)

    priv_fav = np.cumsum((s == 1) & (y_pred == 1))
    priv_tot = np.cumsum(s == 1)
    unpriv_fav = np.cumsum((s == 0) & (y_pred == 1))
    unpriv_tot = np.cumsum(s == 0)
    expected = np.zeros(n)
    prev = 0.0
    for t in range(n):
        if priv_tot[t] > 0 and unpriv_tot[t] > 0:
            prev = priv_fav[t] / priv_tot[t] - unpriv_fav[t] / unpriv_tot[t]
        expected[t] = prev

    assert np.allclose(cum_spd(y_true, y_pred, s, lambda_decay=0.0), expected)


def test_cum_spd_hand_recursion():
    """s = [1,0,1,0], ŷ = [1,0,1,1], λ = 0.5, all y_true = 1.

    t=0: only the privileged group has been seen → carry CSPD_0 = 0.
    t=1: raw = 1/1 − 0/1 = 1     → 0.5·1 + 0.5·0    = 0.5
    t=2: raw = 2/2 − 0/1 = 1     → 0.5·1 + 0.5·0.5  = 0.75
    t=3: raw = 2/2 − 1/2 = 0.5   → 0.5·0.5 + 0.5·0.75 = 0.625
    """
    s = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1])
    y_true = np.ones(4, dtype=int)
    got = cum_spd(y_true, y_pred, s, lambda_decay=0.5)
    assert np.allclose(got, [0.0, 0.5, 0.75, 0.625])


def test_cum_spd_lambda_one_stays_at_zero():
    """λ = 1 keeps the zero-initialised prefix forever: every value is 0.

    On this stream (privileged always ŷ=1, unprivileged always ŷ=0) the raw rate
    difference is 1 from the third row onward, so the recursion is a geometric
    sum: CSPD_n = 1 − λ^(n−2). With n = 100, λ = 0.999 gives
    1 − 0.999^98 ≈ 0.093, i.e. λ → 1 drives the value toward 0 — which is why a
    λ tuned for a stream cannot be transplanted onto an unordered dataset [D7].
    """
    s = np.array([1, 1, 0, 0] * 25)
    y_pred = np.array([1, 1, 0, 0] * 25)
    y_true = np.ones(100, dtype=int)
    assert np.allclose(cum_spd(y_true, y_pred, s, lambda_decay=1.0), 0.0)

    final_no_decay = cum_spd(y_true, y_pred, s, lambda_decay=0.0)[-1]
    assert final_no_decay == pytest.approx(1.0)
    assert cum_spd(y_true, y_pred, s, lambda_decay=0.999)[-1] == pytest.approx(
        1.0 - 0.999**98, abs=1e-9
    )
    assert cum_spd(y_true, y_pred, s, lambda_decay=0.999)[-1] < 0.2 * final_no_decay


def test_cum_eod_conditions_on_true_positives():
    """Negative-label rows must not move CEOD.

    Rows: (y_true, ŷ, s) = (1,1,1), (0,0,1), (1,0,0), (0,0,0).
    CEOD sees only rows 0 and 2 → TPR(S=1) = 1/1, TPR(S=0) = 0/1 → 1.0.
    CSPD counts all four → 1/2 − 0/2 = 0.5, so the two must differ here.
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    s = np.array([1, 1, 0, 0])
    assert cum_eod(y_true, y_pred, s, lambda_decay=0.0)[-1] == pytest.approx(1.0)
    assert cum_spd(y_true, y_pred, s, lambda_decay=0.0)[-1] == pytest.approx(0.5)


def test_cum_metrics_length_and_lambda_validation():
    y_true = np.ones(10, dtype=int)
    y_pred = np.ones(10, dtype=int)
    s = np.array([1, 0] * 5)
    assert cum_spd(y_true, y_pred, s).shape == (10,)
    assert cum_eod(y_true, y_pred, s).shape == (10,)
    with pytest.raises(ValueError, match="lambda_decay"):
        cum_spd(y_true, y_pred, s, lambda_decay=1.5)


def test_streaming_metrics_not_selectable_this_phase():
    """get_fairness_metric must refuse the cumulative forms [D7]."""
    with pytest.raises(ValueError, match="out of scope"):
        get_fairness_metric("cum_spd")
    assert get_fairness_metric("spd") is spd
