"""FBU trade-off plot (Figure 3 from Wang et al., FAccT 2023)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .fbu import FBU

# Colours for the five regions (light fills)
_REGION_COLOURS = {
    1: "#d4edda",  # green  — Jointly advantageous
    2: "#cce5ff",  # blue   — Impressive
    3: "#fff3cd",  # yellow — Reversed
    4: "#f8d7da",  # pink   — Deficient
    5: "#e2e3e5",  # grey   — Jointly disadvantageous
}

_REGION_NAMES = {
    1: "Jointly advantageous",
    2: "Impressive",
    3: "Reversed",
    4: "Deficient",
    5: "Jointly disadvantageous",
}

# Scatter markers for techniques (cycles if > 7 techniques)
_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def plot_fbu(
    fbu_instance: "FBU",
    techniques_results: dict[str, dict[str, Any]],
    perf_metric: str = "balanced_accuracy",
    fair_metric: str = "spd",
    ax: "Axes | None" = None,
) -> "Axes":
    """Render the FBU fairness–performance coordinate space.

    Parameters
    ----------
    fbu_instance:       a fitted FBU object
    techniques_results: output of FBU.evaluate()
    perf_metric:        which performance metric to plot on the y-axis
    fair_metric:        which fairness metric to plot on the x-axis
    ax:                 existing Axes to draw into; creates one if None

    Returns
    -------
    The matplotlib Axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    points = fbu_instance.baseline_points(perf_metric, fair_metric)
    fair_vals = np.array([p[0] for p in points])
    perf_vals = np.array([p[1] for p in points])

    ori_fair, ori_perf = points[0]

    # --- Determine axis limits with some padding ---
    tech_fairs: list[float] = []
    tech_perfs: list[float] = []
    for name, res in techniques_results.items():
        detail = res["details"].get((perf_metric, fair_metric))
        if detail is None:
            continue
        y_pred = None  # we'll re-compute coords from fbu internals below

    # Collect technique coords directly from fbu
    _tech_points: dict[str, tuple[float, float]] = {}
    for name, y_pred in _get_tech_preds(fbu_instance, techniques_results, perf_metric, fair_metric):
        _tech_points[name] = y_pred

    all_fairs = list(fair_vals) + [p[0] for p in _tech_points.values()]
    all_perfs = list(perf_vals) + [p[1] for p in _tech_points.values()]
    pad_f = (max(all_fairs) - min(all_fairs)) * 0.15 or 0.05
    pad_p = (max(all_perfs) - min(all_perfs)) * 0.15 or 0.05
    xlim = (min(all_fairs) - pad_f, max(all_fairs) + pad_f)
    ylim = (min(all_perfs) - pad_p, max(all_perfs) + pad_p)

    # --- Shade the five regions ---
    _shade_regions(ax, ori_fair, ori_perf, fair_vals, perf_vals, xlim, ylim)

    # --- Draw the baseline curve ---
    order = np.argsort(fair_vals)
    ax.plot(
        fair_vals[order],
        perf_vals[order],
        color="red",
        linewidth=2,
        label="FBU baseline",
        zorder=3,
    )
    # Mark M_ori and every pseudo-model
    labels_shown = {"M_ori", "M_10", "M_20", "M_30", "M_40", "M_50",
                    "M_60", "M_70", "M_80", "M_90", "M_100"}
    for i, (f, p) in enumerate(points):
        lbl = "M_ori" if i == 0 else f"M_{i * 10}"
        ax.scatter(f, p, color="red", s=40, zorder=4)
        ax.annotate(
            lbl,
            (f, p),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="red",
        )

    # --- Plot each technique ---
    for idx, (name, (tf, tp)) in enumerate(_tech_points.items()):
        marker = _MARKERS[idx % len(_MARKERS)]
        region = techniques_results[name]["details"].get((perf_metric, fair_metric), {}).get("region", 0)
        colour = _REGION_COLOURS.get(region, "black")
        ax.scatter(tf, tp, marker=marker, s=100, zorder=5,
                   edgecolors="black", linewidths=0.8,
                   color=colour, label=f"{name} (R{region})")
        ax.annotate(
            name,
            (tf, tp),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=8,
        )

    # --- Legend for regions (placed outside the axes to avoid blocking the curve) ---
    region_patches = [
        mpatches.Patch(facecolor=_REGION_COLOURS[r], edgecolor="grey",
                       label=f"Region {r}: {_REGION_NAMES[r]}")
        for r in range(1, 6)
    ]
    ax.legend(handles=region_patches, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              fontsize=8, title="FBU Regions", title_fontsize=8)

    ax.set_xlabel(f"Fairness ({fair_metric.upper()}, negated — higher = more fair)", fontsize=10)
    ax.set_ylabel(f"Performance ({perf_metric})", fontsize=10)
    ax.set_title("Fairness Bonded Utility (FBU) Trade-off", fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, linestyle="--", alpha=0.4)

    return ax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tech_preds(
    fbu: "FBU",
    results: dict[str, dict[str, Any]],
    perf_metric: str,
    fair_metric: str,
):
    """Reconstruct (fairness_signed, performance) from stored fbu internals.

    We store the raw values in classify(); here we re-derive them by
    calling fbu's internal methods.  The technique predictions are not
    stored on FBU, so we read them from the stashed _last_tech_preds dict
    if available, otherwise fall back to dummy coords from the details dict.
    """
    # We need the actual (fair, perf) coords.  Since FBU.classify() doesn't
    # cache them we re-read from a private stash set by evaluate().  As a
    # clean public alternative we store them in results["details"].
    tech_coords: list[tuple[str, tuple[float, float]]] = []
    for name, res in results.items():
        detail = res["details"].get((perf_metric, fair_metric))
        if detail is None:
            continue
        # detail only stores region+area; we need to recompute coords.
        # Use the _raw_coords stash if present (set by evaluate).
        raw = res.get("_raw_coords", {}).get((perf_metric, fair_metric))
        if raw is not None:
            tech_coords.append((name, raw))
    return tech_coords


def _shade_regions(
    ax,
    ori_fair: float,
    ori_perf: float,
    fair_vals: np.ndarray,
    perf_vals: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Fill the five FBU regions with light colour."""
    # Build a sorted baseline (x ascending) for the curve boundary
    order = np.argsort(fair_vals)
    fs = fair_vals[order]
    ps = perf_vals[order]

    # Region 1 (Jointly advantageous): right of M_ori AND above M_ori
    ax.fill_between(
        [ori_fair, xlim[1]], [ori_perf, ori_perf], ylim[1],
        color=_REGION_COLOURS[1], alpha=0.5, zorder=0,
    )

    # Region 5 (Jointly disadvantageous): left of M_ori AND below M_ori
    ax.fill_between(
        [xlim[0], ori_fair], ylim[0], [ori_perf, ori_perf],
        color=_REGION_COLOURS[5], alpha=0.5, zorder=0,
    )

    # Region 3 (Reversed): left of M_ori AND above M_ori
    ax.fill_between(
        [xlim[0], ori_fair], [ori_perf, ori_perf], ylim[1],
        color=_REGION_COLOURS[3], alpha=0.5, zorder=0,
    )

    # Regions 2 & 4 live to the right of M_ori below M_ori performance.
    # Shade the whole stripe first as region 4, then overlay region 2 above
    # the baseline curve.
    x_right = np.linspace(ori_fair, xlim[1], 300)
    from scipy.interpolate import interp1d as _interp1d
    if len(fs) >= 2:
        interp = _interp1d(fs, ps, kind="linear", bounds_error=False,
                           fill_value=(ps[0], ps[-1]))
        y_baseline = interp(x_right)
    else:
        y_baseline = np.full_like(x_right, ori_perf)

    # Region 4: below baseline curve, to the right of M_ori, below M_ori perf
    ax.fill_between(
        x_right, ylim[0], y_baseline,
        color=_REGION_COLOURS[4], alpha=0.5, zorder=0,
    )

    # Region 2: above baseline curve, to the right of M_ori, below M_ori perf
    ax.fill_between(
        x_right, y_baseline, ori_perf,
        color=_REGION_COLOURS[2], alpha=0.5, zorder=0,
    )
