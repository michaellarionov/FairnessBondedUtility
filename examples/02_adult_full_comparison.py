"""Full FBU comparison on Adult: four techniques plus the M_30 control.

    python examples/02_adult_full_comparison.py --scorer logit --n-runs 5

Writes Figure 3(b), Figure 6, the case table, the summary table, the JSON report
and, with ``--seed-sweep``, a seed-sensitivity table under ``outputs/<scorer>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fbu.data.adult import load_adult  # noqa: E402
from fbu.fbu import control_predictions, evaluate, seed_sensitivity  # noqa: E402
from fbu.io import save_cases_csv, save_figure, save_report_json, save_summary_csv  # noqa: E402
from fbu.metrics import balanced_accuracy, eod, recall, spd  # noqa: E402
from fbu.models.scorers import get_scorer_factory  # noqa: E402
from fbu.plotting import plot_region_distribution, plot_tradeoff  # noqa: E402
from fbu.techniques import ALL_TECHNIQUES  # noqa: E402
from fbu.techniques.base import fit_original  # noqa: E402

CONTROL_NAME = "M_30_control"

SUMMARY_COLUMNS = [
    "technique", "n_cases", "region_1_pct", "region_2_pct", "region_3_pct",
    "region_4_pct", "region_5_pct", "area_mean", "area_median", "area_std",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", choices=("logit", "lpm"), default="logit")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-controls", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed-sweep", action="store_true")
    args = parser.parse_args()

    out = args.output_dir / args.scorer
    factory = get_scorer_factory(args.scorer)
    data = load_adult()
    original = fit_original(data, factory)
    mitigations = {
        name: technique(data, factory) for name, technique in ALL_TECHNIQUES.items()
    }

    # The control goes in as several draws rather than one. A single M_30 draw
    # sits roughly one standard deviation off the 20-seed mean curve, so its
    # region is decided by which draw you happen to pick; several runs show that
    # spread instead of hiding it (spec §6.2).
    controls = [
        control_predictions(original, ratio=0.3, seed=0, seed_index=index)
        for index in range(args.n_controls)
    ]

    print("point estimates on the Adult test split")
    for prediction in [original, *mitigations.values(), controls[0]]:
        print(
            f"  {prediction.name:>18}  BA {balanced_accuracy(prediction.y_true, prediction.y_pred):.4f}"
            f"  recall {recall(prediction.y_true, prediction.y_pred):.4f}"
            f"  SPD {spd(prediction.y_true, prediction.y_pred, prediction.s):+.4f}"
            f"  EOD {eod(prediction.y_true, prediction.y_pred, prediction.s):+.4f}"
        )

    techniques = {name: [prediction] for name, prediction in mitigations.items()}
    techniques[CONTROL_NAME] = controls

    report = evaluate(
        original, techniques, n_runs=args.n_runs, n_seeds=args.n_seeds, seed=0
    )

    pd.set_option("display.width", 220)
    print("\nFBU region distribution (% of cases)")
    print(
        report.summary()[SUMMARY_COLUMNS].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
    )

    cases = report.to_dataframe()
    print("\nper-metric-pair detail (first run)")
    first = cases[cases["run"] == 0][
        ["technique", "performance_metric", "fairness_metric", "fairness",
         "performance", "region", "area_norm"]
    ]
    print(first.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    save_cases_csv(report, out / "cases.csv")
    save_summary_csv(report, out / "summary.csv")
    save_report_json(report, out / "report.json")

    for fair_name in ("spd", "eod"):
        axes = plot_tradeoff(report, "balanced_accuracy", fair_name)
        save_figure(axes.figure, out / f"fig3b_balanced_accuracy_{fair_name}.png")
        plt.close(axes.figure)

    axes = plot_region_distribution(report)
    save_figure(axes.figure, out / "fig6_region_distribution.png")
    plt.close(axes.figure)

    if args.seed_sweep:
        sweep = seed_sensitivity(
            original, techniques, seed_budgets=(5, 20, 100), n_runs=args.n_runs
        )
        sweep.to_csv(out / "seed_sensitivity.csv", index=False)
        print("\nseed sensitivity (region 2 share and mean normalised area)")
        print(
            sweep[["technique", "n_seeds", "region_2_pct", "region_4_pct", "area_mean"]]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )

    print(f"\nartefacts written to {out}")


if __name__ == "__main__":
    main()
