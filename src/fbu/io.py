"""Persistence for FBU artefacts: case tables, summaries, figures."""

from __future__ import annotations

import json
from pathlib import Path

from matplotlib.figure import Figure

from .fbu import FBUReport

DEFAULT_OUTPUT_DIR = Path("outputs")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_cases_csv(report: FBUReport, path: Path) -> Path:
    """Write one row per case of the grid."""
    ensure_dir(path.parent)
    report.to_dataframe().to_csv(path, index=False)
    return path


def save_summary_csv(report: FBUReport, path: Path) -> Path:
    """Write one row per technique: region percentages and area statistics."""
    ensure_dir(path.parent)
    report.summary().to_csv(path, index=False)
    return path


def save_report_json(report: FBUReport, path: Path) -> Path:
    """Write region percentages, area statistics and the baseline knots.

    The knots are included because region assignments near the curve depend on
    them, so a reported percentage is only reproducible alongside its baseline.
    """
    payload = {
        "original": report.original_name,
        "n_runs": report.n_runs,
        "n_seeds": report.n_seeds,
        "area_quantifier": report.area_name,
        "techniques": {
            name: {
                "region_percentages": list(result.region_percentages),
                "region_counts": list(result.region_counts),
                "n_cases": result.n_cases,
                "area_mean": result.area_mean,
                "area_median": result.area_median,
                "area_std": result.area_std,
            }
            for name, result in report.results.items()
        },
        "baselines": {
            f"run{run}|{perf}|{fair}": {
                "original_point": list(report.originals[(run, perf, fair)]),
                "knots": [list(knot) for knot in curve.knots()],
                "max_inversion": curve.max_inversion(),
            }
            for (run, perf, fair), curve in report.curves.items()
        },
    }
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))
    return path


def save_figure(figure: Figure, path: Path, dpi: int = 150) -> Path:
    ensure_dir(path.parent)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "ensure_dir",
    "save_cases_csv",
    "save_summary_csv",
    "save_report_json",
    "save_figure",
]
