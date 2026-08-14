"""M7 acceptance tests — orchestration (spec §2.6, §5.2 M7, §6.1)."""

from __future__ import annotations

import numpy as np
import pytest

from fbu.baseline import build_baseline
from fbu.fbu import assert_aligned, control_predictions, evaluate, seed_sensitivity
from fbu.metrics import balanced_accuracy, spd
from fbu.pseudo import pseudo_predictions

from .conftest import requires_adult
from .fixtures.golden_cases import (
    add_false_positives,
    drop_true_positives,
    fix_false_negatives,
    synthetic_original,
)

ALL_PAIRS = 4  # 2 performance metrics × 2 fairness metrics


@pytest.fixture(scope="module")
def original():
    return synthetic_original()


# ---------------------------------------------------------------------------
# §6.1.2 — baseline membership, the strongest single end-to-end check
# ---------------------------------------------------------------------------


def test_m30_control_on_its_own_baseline_has_exactly_zero_area(original):
    """With ``n_seeds=1`` the M_30 knot *is* the control's own draw.

    The point then coincides with a curve knot, so it must classify as region 2
    ("above the line" includes the line) with an area of exactly 0. Any sign
    error, axis inversion or interpolation error breaks this.
    """
    control = control_predictions(original, ratio=0.3, seed=0, seed_index=0)
    report = evaluate(original, [control], n_seeds=1, seed=0)
    assert report[control.name].region_counts[1] == ALL_PAIRS
    for case in report[control.name].cases:
        assert case.region == 2
        assert case.area == pytest.approx(0.0, abs=1e-12)
        assert case.area_norm == pytest.approx(0.0, abs=1e-12)


def test_m30_control_hugs_the_averaged_baseline(original):
    """Against the default 20-seed mean curve the control sits *on* the curve.

    It therefore lands in the trade-off quadrant with a negligible area, but
    whether that reads as region 2 or region 4 is decided by seed noise — the
    honest illustration of the first known weakness in spec §6.2.

    Closeness is judged against the knot's own seed spread (four standard
    deviations of a single draw) rather than a fixed tolerance: EOD conditions on
    true positives, so its knots are several times noisier than the SPD ones and
    a single absolute threshold would either be vacuous or flaky.
    """
    control = control_predictions(original, ratio=0.3, seed=0, seed_index=3)
    report = evaluate(original, [control], n_seeds=20, seed=0)
    result = report[control.name]
    assert result.region_counts[1] + result.region_counts[3] == ALL_PAIRS
    for case in result.cases:
        curve = report.curve(case.performance_metric, case.fairness_metric)
        knot = next(p for p in curve.points if p.ratio == pytest.approx(0.3))
        assert abs(case.fairness - knot.fairness) <= 4 * knot.fairness_std + 1e-9
        assert abs(case.performance - knot.performance) <= 4 * knot.performance_std + 1e-9
        if case.region == 2:
            assert case.area_norm is not None and case.area_norm < 0.02


# ---------------------------------------------------------------------------
# §6.1.3 — Pareto dominance and its mirror image
# ---------------------------------------------------------------------------


def test_pareto_dominating_technique_is_region_1(original):
    """Correcting unprivileged false negatives raises TPR *and* closes the gap.

    Both axes improve, for either fairness metric and either performance metric,
    so every case must be region 1 with no area.
    """
    better = fix_false_negatives(original, group=0, count=150, name="pareto")
    report = evaluate(original, [better], n_seeds=20, seed=0)
    result = report[better.name]
    assert result.region_counts[0] == result.n_cases == ALL_PAIRS
    assert result.percentage(1) == pytest.approx(100.0)
    assert result.area_mean == 0.0
    assert all(case.area is None for case in result.cases)


def test_accuracy_up_fairness_down_is_region_3(original):
    """Correcting *privileged* false negatives raises TPR while widening the gap."""
    reversed_technique = fix_false_negatives(original, group=1, count=120, name="reversed")
    report = evaluate(original, [reversed_technique], n_seeds=20, seed=0)
    result = report[reversed_technique.name]
    assert result.region_counts[2] == result.n_cases == ALL_PAIRS
    for case in result.cases:
        f_o, y_o = report.original(case.performance_metric, case.fairness_metric)
        assert case.fairness < f_o and case.performance >= y_o


def test_both_axes_down_is_region_5(original):
    """Privileged false positives plus discarded unprivileged true positives.

    The first costs TNR and raises the privileged favorable rate; the second
    costs TPR and lowers the unprivileged one. Both SPD and EOD widen and both
    performance metrics fall, so all four cases are region 5. (False positives
    alone would leave EOD untouched, since EOD only sees ``y_true == 1`` rows,
    and the point would drift into the trade-off quadrant instead.)
    """
    worse = add_false_positives(original, group=1, count=200, name="both_down")
    worse = drop_true_positives(worse, group=0, count=100, name="both_down")
    report = evaluate(original, [worse], n_seeds=20, seed=0)
    result = report[worse.name]
    assert result.region_counts[4] == result.n_cases == ALL_PAIRS
    for case in result.cases:
        f_o, y_o = report.original(case.performance_metric, case.fairness_metric)
        assert case.fairness < f_o and case.performance < y_o


def test_below_the_curve_is_region_4(original):
    """A near-constant classifier with a few unprivileged false positives.

    Almost perfectly fair, but balanced accuracy sits just under the 0.5 anchor,
    which puts it below the curve. Only balanced accuracy is used: with recall on
    the performance axis the point coincides with the M_100 knot at (1.0, 0.0)
    and legitimately reads as region 2 with zero area.
    """
    constant = original.with_predictions(
        pseudo_predictions(original.y_pred, 1.0), name="deficient"
    )
    deficient = add_false_positives(constant, group=0, count=40, name="deficient")
    report = evaluate(
        original,
        [deficient],
        performance_metrics=("balanced_accuracy",),
        n_seeds=20,
        seed=0,
    )
    result = report[deficient.name]
    assert result.region_counts[3] == result.n_cases == 2
    assert result.area_mean == 0.0


# ---------------------------------------------------------------------------
# Case grid bookkeeping (§2.6, [D4])
# ---------------------------------------------------------------------------


def test_case_count_and_percentages(original):
    """cases = n_runs × n_technique_runs × n_fairness × n_performance [D4].

    Three runs of the curve, one prediction array, 2 × 2 metrics = 12 cases; the
    competitor count never enters, so both techniques get the same total.
    """
    first = fix_false_negatives(original, group=0, count=150, name="a")
    second = fix_false_negatives(original, group=1, count=150, name="b")
    report = evaluate(original, [first, second], n_runs=3, n_seeds=5, seed=0)
    for name in ("a", "b"):
        result = report[name]
        assert result.n_cases == 3 * ALL_PAIRS == 12
        assert sum(result.region_counts) == 12
        assert sum(result.region_percentages) == pytest.approx(100.0)


def test_multiple_runs_per_technique_multiply_the_grid(original):
    """Several prediction arrays under one name are runs, not separate techniques."""
    runs = [
        fix_false_negatives(original, group=0, count=count, name="varied")
        for count in (60, 120, 180)
    ]
    report = evaluate(original, {"varied": runs}, n_runs=2, n_seeds=5, seed=0)
    assert report["varied"].n_cases == 2 * 3 * ALL_PAIRS


def test_dataframes(original):
    technique = fix_false_negatives(original, group=0, count=150, name="pareto")
    report = evaluate(original, [technique], n_seeds=5, seed=0)

    cases = report.to_dataframe()
    assert len(cases) == ALL_PAIRS
    assert set(cases["region"]) == {1}
    assert set(cases.columns) >= {
        "technique", "run", "dataset", "performance_metric", "fairness_metric",
        "fairness", "performance", "region", "region_label", "area", "area_norm",
    }

    summary = report.summary()
    assert len(summary) == 1
    assert summary.loc[0, "region_1_pct"] == pytest.approx(100.0)
    assert summary.loc[0, "n_cases"] == ALL_PAIRS


def test_reported_curve_matches_a_standalone_build(original):
    """The report's baseline is reproducible from ``build_baseline`` with the same
    seed — a percentage is only meaningful alongside the curve that produced it."""
    technique = fix_false_negatives(original, group=0, count=150, name="pareto")
    report = evaluate(original, [technique], n_seeds=5, seed=0)
    standalone = build_baseline(original, balanced_accuracy, spd, n_seeds=5, seed=0)
    reported = report.curve("balanced_accuracy", "spd")
    assert np.allclose(reported.fairness, standalone.fairness)
    assert np.allclose(reported.performance, standalone.performance)


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_misaligned_predictions_are_rejected(original):
    """A silent misalignment produces plausible-looking garbage (spec §2.1)."""
    shuffled = original.with_predictions(original.y_pred, name="shuffled")
    scrambled = type(original)(
        y_true=original.y_true[::-1],
        y_pred=original.y_pred,
        s=original.s,
        name="scrambled",
    )
    assert_aligned(original, [shuffled])
    with pytest.raises(ValueError, match="y_true differs"):
        assert_aligned(original, [scrambled])
    with pytest.raises(ValueError, match="rows"):
        evaluate(
            original,
            [
                type(original)(
                    y_true=original.y_true[:100],
                    y_pred=original.y_pred[:100],
                    s=original.s[:100],
                    name="short",
                )
            ],
            n_seeds=2,
        )


def test_empty_comparison_set_is_rejected(original):
    """FBU is a comparison metric: a base model alone has nothing plotted against it."""
    with pytest.raises(ValueError, match="at least one competing technique"):
        evaluate(original, [], n_seeds=2)


def test_excluded_metrics_cannot_reach_the_orchestrator(original):
    technique = fix_false_negatives(original, group=0, count=50, name="t")
    with pytest.raises(ValueError, match="excluded"):
        evaluate(original, [technique], performance_metrics=("accuracy",), n_seeds=2)


# ---------------------------------------------------------------------------
# §6.1.4 — seed sensitivity
# ---------------------------------------------------------------------------


@requires_adult
@pytest.mark.parametrize("scorer", ["logit", "lpm"])
def test_area_ordering_survives_the_alternative_quantifier(adult, scorer):
    """§2.5 sensitivity check: swapping the enclosed integral for perpendicular
    distance must not reorder the techniques.

    On the SPD pairs the ranking is massaging > reweighing > group thresholds >
    fairness-unaware under both quantifiers and under either base model; the
    absolute values differ (the distance is not normalised), the ordering does not.
    """
    from fbu.area import EnclosedArea, PerpendicularDistance
    from fbu.models.scorers import get_scorer_factory
    from fbu.techniques import ALL_TECHNIQUES
    from fbu.techniques.base import fit_original

    factory = get_scorer_factory(scorer)
    base = fit_original(adult, factory)
    techniques = [technique(adult, factory) for technique in ALL_TECHNIQUES.values()]
    expected = ["massaging", "reweighing", "group_threshold", "fairness_unaware"]

    for quantifier in (EnclosedArea(), PerpendicularDistance()):
        report = evaluate(
            base,
            techniques,
            fairness_metrics=("spd",),
            n_runs=1,
            n_seeds=20,
            area_fn=quantifier,
        )
        ranking = sorted(
            report.results.values(), key=lambda result: result.area_mean, reverse=True
        )
        assert [result.name for result in ranking] == expected, quantifier.name


def test_seed_sensitivity_reports_every_budget(original):
    """Region percentages are reported per ``n_seeds`` budget so the spread of a
    randomised baseline is visible rather than hidden."""
    technique = fix_false_negatives(original, group=0, count=150, name="pareto")
    frame = seed_sensitivity(original, [technique], seed_budgets=(2, 5), n_runs=2)
    assert sorted(frame["n_seeds"].unique()) == [2, 5]
    assert (frame["region_1_pct"] == 100.0).all()  # Pareto dominance is seed-proof
