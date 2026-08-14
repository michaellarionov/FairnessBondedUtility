"""FBU orchestration: prediction arrays in, region distribution out (spec §2.6).

FBU is a post-hoc *comparison* metric. It has no dependency on any particular
model, on FS², or on streaming infrastructure: it consumes the prediction arrays
of a set of competing techniques and reports, per technique, a distribution over
the five effectiveness regions plus a trade-off area score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .area import EnclosedArea
from .baseline import BaselineCurve, build_baseline, original_point
from .metrics.fairness import fairness_score, get_fairness_metric
from .metrics.performance import get_performance_metric
from .pseudo import DEFAULT_N_SEEDS, DEFAULT_RATIOS, pseudo_predictions
from .regions import classify
from .types import (
    REGION_LABELS,
    CaseResult,
    FairnessMetric,
    FBUResult,
    PerformanceMetric,
    Predictions,
    TradeoffArea,
    unique_name,
)

DEFAULT_PERFORMANCE_METRICS: tuple[str, ...] = ("balanced_accuracy", "recall")
DEFAULT_FAIRNESS_METRICS: tuple[str, ...] = ("spd", "eod")

MetricSpec = str | tuple[str, PerformanceMetric] | tuple[str, FairnessMetric]
TechniqueInput = (
    Sequence[Predictions] | Mapping[str, Predictions | Sequence[Predictions]]
)


def assert_aligned(reference: Predictions, others: Iterable[Predictions]) -> None:
    """Every technique must predict the same rows in the same order (spec §2.1).

    A silent misalignment produces plausible-looking garbage, so the check
    compares the ground truth and sensitive arrays element-wise rather than
    trusting shapes alone.
    """
    for other in others:
        if len(other) != len(reference):
            raise ValueError(
                f"'{other.name}' has {len(other)} rows, expected {len(reference)}"
            )
        if not np.array_equal(other.y_true, reference.y_true):
            raise ValueError(f"'{other.name}' y_true differs from '{reference.name}'")
        if not np.array_equal(other.s, reference.s):
            raise ValueError(f"'{other.name}' sensitive attribute differs from '{reference.name}'")


def control_predictions(
    original: Predictions, ratio: float = 0.3, seed: int = 0, seed_index: int = 0
) -> Predictions:
    """A pseudo-model fed back as a fake technique (spec Part IV control).

    ``M_30`` must land in region 2 with ``A_norm ≈ 0``: it *is* a point on the
    baseline curve, up to seed noise. The check catches sign errors,
    axis-direction errors and interpolation errors simultaneously.
    """
    label = int(round(ratio * 100))
    return original.with_predictions(
        pseudo_predictions(original.y_pred, ratio, seed, seed_index),
        name=f"M_{label}_control",
    )


def _resolve(specs: Sequence[MetricSpec], kind: str) -> list[tuple[str, object]]:
    resolved: list[tuple[str, object]] = []
    for spec in specs:
        if isinstance(spec, str):
            getter = get_performance_metric if kind == "performance" else get_fairness_metric
            resolved.append((spec, getter(spec)))
        else:
            name, fn = spec
            resolved.append((name, fn))
    if not resolved:
        raise ValueError(f"at least one {kind} metric is required")
    unique_name([name for name, _ in resolved])
    return resolved


def _as_runs(techniques: TechniqueInput) -> dict[str, list[Predictions]]:
    """Normalise the technique argument to ``name -> list of per-run predictions``."""
    runs: dict[str, list[Predictions]] = {}
    if isinstance(techniques, Mapping):
        for name, value in techniques.items():
            runs[name] = [value] if isinstance(value, Predictions) else list(value)
    else:
        for prediction in techniques:
            runs.setdefault(prediction.name, []).append(prediction)
    if not runs:
        raise ValueError(
            "FBU needs at least one competing technique; a base model alone "
            "yields only the origin point and the baseline curve (spec §1.1)"
        )
    for name, values in runs.items():
        if not values:
            raise ValueError(f"technique '{name}' has no prediction runs")
    return runs


@dataclass(frozen=True)
class FBUReport:
    """Per-technique results plus the baselines they were measured against."""

    results: dict[str, FBUResult]
    curves: dict[tuple[int, str, str], BaselineCurve]
    originals: dict[tuple[int, str, str], tuple[float, float]]
    original_name: str
    n_runs: int
    n_seeds: int
    area_name: str

    def __getitem__(self, name: str) -> FBUResult:
        return self.results[name]

    def __iter__(self):
        return iter(self.results)

    def curve(
        self,
        performance_metric: str = "balanced_accuracy",
        fairness_metric: str = "spd",
        run: int = 0,
    ) -> BaselineCurve:
        return self.curves[(run, performance_metric, fairness_metric)]

    def original(
        self,
        performance_metric: str = "balanced_accuracy",
        fairness_metric: str = "spd",
        run: int = 0,
    ) -> tuple[float, float]:
        return self.originals[(run, performance_metric, fairness_metric)]

    def to_dataframe(self) -> pd.DataFrame:
        """One row per case of the grid (spec §2.6)."""
        records = [
            {
                "technique": case.technique,
                "run": case.run,
                "dataset": case.dataset,
                "performance_metric": case.performance_metric,
                "fairness_metric": case.fairness_metric,
                "fairness": case.fairness,
                "performance": case.performance,
                "region": case.region,
                "region_label": case.region_label,
                "area": case.area,
                "area_norm": case.area_norm,
            }
            for result in self.results.values()
            for case in result.cases
        ]
        return pd.DataFrame.from_records(records)

    def summary(self) -> pd.DataFrame:
        """One row per technique: region percentages and region-2 area statistics."""
        records = []
        for result in self.results.values():
            row = {"technique": result.name, "n_cases": result.n_cases}
            for region in range(1, 6):
                row[f"region_{region}_pct"] = result.percentage(region)
                row[f"region_{region}_label"] = REGION_LABELS[region]
            row["area_mean"] = result.area_mean
            row["area_median"] = result.area_median
            row["area_std"] = result.area_std
            records.append(row)
        return pd.DataFrame.from_records(records)


def evaluate(
    original: Predictions,
    techniques: TechniqueInput,
    performance_metrics: Sequence[MetricSpec] = DEFAULT_PERFORMANCE_METRICS,
    fairness_metrics: Sequence[MetricSpec] = DEFAULT_FAIRNESS_METRICS,
    n_runs: int = 1,
    n_seeds: int = DEFAULT_N_SEEDS,
    seed: int = 0,
    dataset: str = "adult",
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    envelope: bool = False,
    area_fn: TradeoffArea | None = None,
) -> FBUReport:
    """Classify every technique over the full case grid.

    ``cases = n_runs × n_technique_runs × n_fairness × n_performance × n_datasets``
    (`[D4]`: the paper's ``n_t`` factor is dropped — a technique's own case count
    cannot depend on how many competitors it faces).

    A *run* is an independent construction of the baseline curve, each with its
    own block of ``n_seeds`` pseudo-model draws; running more than one exposes
    the seed-variance band that §6.2 requires. Techniques may also supply
    several prediction arrays, which multiply into the same grid.
    """
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")
    area_fn = area_fn or EnclosedArea()

    perf_metrics = _resolve(performance_metrics, "performance")
    fair_metrics = _resolve(fairness_metrics, "fairness")
    technique_runs = _as_runs(techniques)
    assert_aligned(original, [p for runs in technique_runs.values() for p in runs])

    curves: dict[tuple[int, str, str], BaselineCurve] = {}
    originals: dict[tuple[int, str, str], tuple[float, float]] = {}
    normalisers: dict[tuple[int, str, str], float] = {}
    for run in range(n_runs):
        for perf_name, perf_fn in perf_metrics:
            for fair_name, fair_fn in fair_metrics:
                key = (run, perf_name, fair_name)
                curve = build_baseline(
                    original,
                    perf_fn,  # type: ignore[arg-type]
                    fair_fn,  # type: ignore[arg-type]
                    ratios=ratios,
                    n_seeds=n_seeds,
                    seed=seed + run,
                    envelope=envelope,
                )
                curves[key] = curve
                f_o, y_o = original_point(curve)
                originals[key] = (f_o, y_o)
                # Scale for A_norm: the same quantifier evaluated at the corner
                # (f_max, y_o), which for the default area is exactly A_max.
                normalisers[key] = area_fn(curve, curve.f_max, y_o, f_o, y_o)

    results: dict[str, FBUResult] = {}
    for name, runs in technique_runs.items():
        cases: list[CaseResult] = []
        for run in range(n_runs):
            for run_index, prediction in enumerate(runs):
                for perf_name, perf_fn in perf_metrics:
                    for fair_name, fair_fn in fair_metrics:
                        key = (run, perf_name, fair_name)
                        curve = curves[key]
                        f_o, y_o = originals[key]
                        y_p = float(perf_fn(prediction.y_true, prediction.y_pred))  # type: ignore[operator]
                        f_p = fairness_score(
                            fair_fn(prediction.y_true, prediction.y_pred, prediction.s)  # type: ignore[operator]
                        )
                        region = classify(f_p, y_p, f_o, y_o, curve)
                        area: float | None = None
                        area_norm: float | None = None
                        if region == 2:
                            area = area_fn(curve, f_p, y_p, f_o, y_o)
                            scale = normalisers[key]
                            area_norm = area / scale if scale > 0.0 else 0.0
                        cases.append(
                            CaseResult(
                                technique=name,
                                run=run * len(runs) + run_index,
                                dataset=dataset,
                                performance_metric=perf_name,
                                fairness_metric=fair_name,
                                fairness=f_p,
                                performance=y_p,
                                region=region,
                                area=area,
                                area_norm=area_norm,
                            )
                        )
        results[name] = _summarise(name, cases)

    return FBUReport(
        results=results,
        curves=curves,
        originals=originals,
        original_name=original.name,
        n_runs=n_runs,
        n_seeds=n_seeds,
        area_name=getattr(area_fn, "name", type(area_fn).__name__),
    )


def _summarise(name: str, cases: Sequence[CaseResult]) -> FBUResult:
    counts = [0, 0, 0, 0, 0]
    for case in cases:
        counts[case.region - 1] += 1
    n_cases = len(cases)
    percentages = tuple(100.0 * c / n_cases for c in counts)

    areas = np.array(
        [case.area_norm for case in cases if case.region == 2 and case.area_norm is not None],
        dtype=np.float64,
    )
    return FBUResult(
        name=name,
        region_percentages=percentages,  # type: ignore[arg-type]
        region_counts=tuple(counts),  # type: ignore[arg-type]
        n_cases=n_cases,
        area_mean=float(areas.mean()) if areas.size else 0.0,
        area_median=float(np.median(areas)) if areas.size else 0.0,
        area_std=float(areas.std(ddof=0)) if areas.size else 0.0,
        area_norms=tuple(float(a) for a in areas),
        cases=tuple(cases),
    )


def seed_sensitivity(
    original: Predictions,
    techniques: TechniqueInput,
    seed_budgets: Sequence[int] = (5, 20, 100),
    n_runs: int = 3,
    **kwargs: object,
) -> pd.DataFrame:
    """Region percentages and area means under several ``n_seeds`` budgets (§6.1.4).

    The baseline is randomised, so two FBU runs on identical predictions can
    assign different regions to a point near the curve. If assignments flip
    between budgets, the default is too low.
    """
    frames = []
    for budget in seed_budgets:
        report = evaluate(
            original, techniques, n_seeds=budget, n_runs=n_runs, **kwargs  # type: ignore[arg-type]
        )
        frame = report.summary()
        frame.insert(1, "n_seeds", budget)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "evaluate",
    "FBUReport",
    "assert_aligned",
    "control_predictions",
    "seed_sensitivity",
    "DEFAULT_PERFORMANCE_METRICS",
    "DEFAULT_FAIRNESS_METRICS",
]
