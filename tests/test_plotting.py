"""M8 checks — figures (spec §5.2 M8).

The acceptance criterion for M8 is visual review; these tests only guard the
structural claims that a reviewer would otherwise have to eyeball, above all the
monotonicity of Figure 3(a).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fbu.baseline import build_baseline  # noqa: E402
from fbu.fbu import control_predictions, evaluate  # noqa: E402
from fbu.metrics import balanced_accuracy, spd  # noqa: E402
from fbu.plotting import (  # noqa: E402
    plot_region_distribution,
    plot_replacement_ratio,
    plot_tradeoff,
)

from .fixtures.golden_cases import fix_false_negatives, synthetic_original  # noqa: E402


@pytest.fixture(scope="module")
def original():
    return synthetic_original()


@pytest.fixture(scope="module")
def report(original):
    techniques = [
        fix_false_negatives(original, group=0, count=150, name="pareto"),
        fix_false_negatives(original, group=1, count=120, name="reversed"),
        control_predictions(original, ratio=0.3),
    ]
    return evaluate(original, techniques, n_seeds=20, seed=0)


def test_fig3a_series_are_monotone(original):
    """Figure 3(a) must show fairness rising and performance falling in p."""
    curve = build_baseline(original, balanced_accuracy, spd, n_seeds=20)
    axes = plot_replacement_ratio(curve.points)
    # errorbar labels its container, not the underlying line, and also adds cap
    # lines — so the two series are read off the containers.
    series = {
        container.get_label(): container.lines[0].get_ydata()
        for container in axes.containers
    }
    fairness = next(v for k, v in series.items() if "fairness" in k)
    performance = next(v for k, v in series.items() if "accuracy" in k)
    assert np.all(np.diff(fairness) > 0)
    assert np.all(np.diff(performance) < 0)
    assert axes.get_xlabel().startswith("Replacement ratio")
    plt.close(axes.figure)


def test_fig3b_axes_are_fairness_by_performance(report):
    """Spec §1.3 G1: Figure 3(b) is x = fairness, y = performance, x higher-is-better."""
    axes = plot_tradeoff(report, "balanced_accuracy", "spd")
    assert "Fairness score" in axes.get_xlabel()
    assert "higher = fairer" in axes.get_xlabel()
    assert "Performance" in axes.get_ylabel()
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert any("pareto" in label for label in labels)
    assert any("M_ori" == annotation.get_text() for annotation in axes.texts)
    plt.close(axes.figure)


def test_fig3b_draws_one_shaded_area_per_region_2_case(report):
    """Region-2 areas are shaded; other regions contribute no filled patch."""
    n_region_2 = sum(
        1
        for result in report.results.values()
        for case in result.cases
        if case.region == 2
        and case.performance_metric == "balanced_accuracy"
        and case.fairness_metric == "spd"
        and case.run == 0
    )
    axes = plot_tradeoff(report, "balanced_accuracy", "spd", shade_areas=True)
    # Five region backdrops are always drawn; the rest are the per-case areas.
    assert len(axes.collections) - n_region_2 == len(
        plot_tradeoff(
            report, "balanced_accuracy", "spd", shade_areas=False, ax=plt.subplots()[1]
        ).collections
    )
    plt.close("all")


def test_fig6_bars_sum_to_100(report):
    axes = plot_region_distribution(report)
    totals: dict[float, float] = {}
    for patch in axes.patches:
        totals[patch.get_x()] = totals.get(patch.get_x(), 0.0) + patch.get_height()
    assert len(totals) == len(report.results)
    for total in totals.values():
        assert total == pytest.approx(100.0)
    assert axes.get_ylim() == (0, 100)
    plt.close(axes.figure)
