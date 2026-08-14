"""FBU figures: paper Figure 3(a), Figure 3(b) and Figure 6.

Figure 3(a) is the diagnostic with the **replacement ratio** on the x-axis.
Figure 3(b) is the FBU coordinate space itself: x = fairness, y = performance
(spec §1.3 G1). The fairness axis is always ``fairness_score = 1 − |metric|``.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .baseline import BaselineCurve
from .fbu import FBUReport
from .pseudo import PseudoPoint
from .types import REGION_LABELS

REGION_COLOURS: dict[int, str] = {
    1: "#d4edda",  # green  — jointly advantageous
    2: "#cce5ff",  # blue   — impressive
    3: "#fff3cd",  # yellow — reversed
    4: "#f8d7da",  # pink   — deficient
    5: "#e2e3e5",  # grey   — jointly disadvantageous
}

_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def _new_axes(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    if ax is not None:
        return ax
    _, created = plt.subplots(figsize=figsize)
    return created


def plot_replacement_ratio(
    points: Sequence[PseudoPoint],
    performance_label: str = "balanced accuracy",
    fairness_label: str = "fairness score (1 − |SPD|)",
    ax: Axes | None = None,
) -> Axes:
    """Figure 3(a): fairness and performance against the replacement ratio.

    Fairness should rise and performance should fall monotonically as more
    predictions are replaced by the majority label.
    """
    ax = _new_axes(ax, (7.0, 4.5))
    ratios = np.array([p.ratio * 100 for p in points])
    fairness = np.array([p.fairness for p in points])
    performance = np.array([p.performance for p in points])
    fairness_err = np.array([p.fairness_std for p in points])
    performance_err = np.array([p.performance_std for p in points])

    ax.errorbar(
        ratios, fairness, yerr=fairness_err, marker="o", color="#1f77b4",
        capsize=3, label=fairness_label,
    )
    ax.errorbar(
        ratios, performance, yerr=performance_err, marker="s", color="#d62728",
        capsize=3, label=performance_label,
    )
    ax.set_xlabel("Replacement ratio p (%)")
    ax.set_ylabel("Score")
    ax.set_title("Pseudo-model sweep (Fig. 3a)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9)
    return ax


def _shade_regions(
    ax: Axes,
    curve: BaselineCurve,
    f_o: float,
    y_o: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Fill the five regions. 1/3/5 are quadrants of M_ori; only 2/4 use the curve."""
    ax.fill_between([f_o, xlim[1]], y_o, ylim[1], color=REGION_COLOURS[1], zorder=0)
    ax.fill_between([xlim[0], f_o], y_o, ylim[1], color=REGION_COLOURS[3], zorder=0)
    ax.fill_between([xlim[0], f_o], ylim[0], y_o, color=REGION_COLOURS[5], zorder=0)

    grid = np.linspace(f_o, xlim[1], 400)
    baseline = np.clip(curve.evaluate_many(grid), ylim[0], y_o)
    ax.fill_between(grid, ylim[0], baseline, color=REGION_COLOURS[4], zorder=0)
    ax.fill_between(grid, baseline, y_o, color=REGION_COLOURS[2], zorder=0)


def plot_tradeoff(
    report: FBUReport,
    performance_metric: str = "balanced_accuracy",
    fairness_metric: str = "spd",
    run: int = 0,
    shade_areas: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Figure 3(b): baseline curve, technique points, shaded region-2 areas."""
    ax = _new_axes(ax, (8.5, 6.0))
    curve = report.curve(performance_metric, fairness_metric, run)
    f_o, y_o = report.original(performance_metric, fairness_metric, run)

    cases = {
        result.name: case
        for result in report.results.values()
        for case in result.cases
        if case.performance_metric == performance_metric
        and case.fairness_metric == fairness_metric
        and case.run == run
    }

    fair_all = [*curve.fairness, f_o, *(c.fairness for c in cases.values())]
    perf_all = [*curve.performance, y_o, *(c.performance for c in cases.values())]
    pad_f = max((max(fair_all) - min(fair_all)) * 0.12, 0.02)
    pad_p = max((max(perf_all) - min(perf_all)) * 0.12, 0.02)
    xlim = (min(fair_all) - pad_f, max(fair_all) + pad_f)
    ylim = (min(perf_all) - pad_p, max(perf_all) + pad_p)

    _shade_regions(ax, curve, f_o, y_o, xlim, ylim)

    ax.plot(
        curve.fairness, curve.performance, color="black", linewidth=2,
        label="FBU baseline g", zorder=3,
    )
    for point in curve.points:
        ax.scatter(point.fairness, point.performance, s=28, color="black", zorder=4)
        ax.annotate(
            point.label, (point.fairness, point.performance),
            textcoords="offset points", xytext=(4, 4), fontsize=7,
        )

    for index, (name, case) in enumerate(sorted(cases.items())):
        if shade_areas and case.region == 2:
            f_a = curve.solve_fairness(
                max(case.performance, curve.evaluate(case.fairness)), f_lo=f_o
            )
            grid = np.linspace(f_a, case.fairness, 200)
            ax.fill_between(
                grid, curve.evaluate_many(grid), case.performance,
                color="#3182bd", alpha=0.35, zorder=2,
            )
        ax.scatter(
            case.fairness, case.performance,
            marker=_MARKERS[index % len(_MARKERS)], s=110, zorder=5,
            edgecolors="black", linewidths=0.8, color="white",
            label=f"{name} (R{case.region}"
            + (f", A={case.area_norm:.3f})" if case.area_norm is not None else ")"),
        )

    ax.axvline(f_o, color="grey", linestyle=":", linewidth=1, zorder=1)
    ax.axhline(y_o, color="grey", linestyle=":", linewidth=1, zorder=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(f"Fairness score  1 − |{fairness_metric.upper()}|  (higher = fairer)")
    ax.set_ylabel(f"Performance ({performance_metric.replace('_', ' ')})")
    ax.set_title("Fairness Bonded Utility (Fig. 3b)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0)
    return ax


def plot_region_distribution(report: FBUReport, ax: Axes | None = None) -> Axes:
    """Figure 6: stacked region percentages, one bar per technique."""
    ax = _new_axes(ax, (8.0, 5.0))
    names = list(report.results)
    bottom = np.zeros(len(names))
    for region in range(1, 6):
        heights = np.array([report[name].percentage(region) for name in names])
        ax.bar(
            names, heights, bottom=bottom,
            color=REGION_COLOURS[region], edgecolor="black", linewidth=0.6,
            label=f"R{region}: {REGION_LABELS[region]}",
        )
        bottom += heights
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of cases (%)")
    ax.set_title(
        f"FBU region distribution ({report.n_runs} run(s) × {report.n_seeds} seeds)"
    )
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0)
    return ax


__all__ = [
    "plot_replacement_ratio",
    "plot_tradeoff",
    "plot_region_distribution",
    "REGION_COLOURS",
]
