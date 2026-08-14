"""Baseline curve on Adult: the pseudo-model sweep and Figure 3(a).

    python examples/01_adult_baseline_curve.py --scorer logit

Prints the 11 baseline knots and writes the replacement-ratio diagnostic. FBU
proper needs competing techniques; see ``02_adult_full_comparison.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fbu.baseline import build_baseline  # noqa: E402
from fbu.data.adult import load_adult  # noqa: E402
from fbu.io import save_figure  # noqa: E402
from fbu.metrics import balanced_accuracy, eod, recall, spd  # noqa: E402
from fbu.models.scorers import get_scorer_factory  # noqa: E402
from fbu.plotting import plot_replacement_ratio  # noqa: E402
from fbu.pseudo import majority_label  # noqa: E402
from fbu.techniques.base import fit_original  # noqa: E402

METRICS = {
    ("balanced_accuracy", "spd"): (balanced_accuracy, spd),
    ("balanced_accuracy", "eod"): (balanced_accuracy, eod),
    ("recall", "spd"): (recall, spd),
    ("recall", "eod"): (recall, eod),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", choices=("logit", "lpm"), default="logit")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    data = load_adult()
    original = fit_original(data, get_scorer_factory(args.scorer))
    print(f"base model: {args.scorer}   L* = {majority_label(original.y_pred)}")

    for (perf_name, fair_name), (perf_fn, fair_fn) in METRICS.items():
        curve = build_baseline(original, perf_fn, fair_fn, n_seeds=args.n_seeds)
        print(f"\n{perf_name} × {fair_name}   (max inversion {curve.max_inversion():.4f})")
        for point in curve.points:
            print(
                f"  {point.label:>6}  fairness {point.fairness:.4f} ±{point.fairness_std:.4f}"
                f"   performance {point.performance:.4f} ±{point.performance_std:.4f}"
            )

        axes = plot_replacement_ratio(
            curve.points,
            performance_label=perf_name.replace("_", " "),
            fairness_label=f"fairness score (1 − |{fair_name.upper()}|)",
        )
        path = args.output_dir / args.scorer / f"fig3a_{perf_name}_{fair_name}.png"
        save_figure(axes.figure, path)
        plt.close(axes.figure)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
