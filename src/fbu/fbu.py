"""Fairness Bonded Utility (FBU) — core implementation.

Reference: Wang et al., "Preventing Discriminatory Decision-making in Evolving
Data Streams", ACM FAccT 2023, Section 5.1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from .metrics import balanced_accuracy, cumulative_eod, cumulative_spd

# Supported metric names and their callables
_PERF_METRICS: dict[str, Any] = {
    "balanced_accuracy": balanced_accuracy,
    "recall": lambda yt, yp: (
        np.sum((yp == 1) & (yt == 1)) / np.sum(yt == 1)
        if np.sum(yt == 1) > 0
        else 0.0
    ),
}

_FAIR_METRICS: dict[str, Any] = {
    "spd": cumulative_spd,
    "eod": cumulative_eod,
}

# Five region labels (index 0 = region 1)
REGION_LABELS = [
    "Jointly advantageous",
    "Impressive",
    "Reversed",
    "Deficient",
    "Jointly disadvantageous",
]


class FBU:
    """Fairness Bonded Utility evaluator.

    Parameters
    ----------
    y_true:               ground-truth labels, shape (n,)
    y_pred_original:      original (unfair) model predictions, shape (n,)
    sensitive:            binary sensitive attribute (1=privileged), shape (n,)
    performance_metrics:  list of "balanced_accuracy" and/or "recall"
    fairness_metrics:     list of "spd" and/or "eod"
    lambda_decay:         decay factor λ for streaming fairness metrics
    random_state:         seed for reproducible pseudo-model generation
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred_original: np.ndarray,
        sensitive: np.ndarray,
        performance_metrics: list[str] | None = None,
        fairness_metrics: list[str] | None = None,
        lambda_decay: float = 0.0,
        random_state: int = 42,
    ) -> None:
        self.y_true = np.asarray(y_true)
        self.y_pred_original = np.asarray(y_pred_original)
        self.sensitive = np.asarray(sensitive)
        self.lambda_decay = lambda_decay
        self.rng = np.random.default_rng(random_state)

        self.performance_metrics = performance_metrics or ["balanced_accuracy"]
        self.fairness_metrics = fairness_metrics or ["spd"]

        for pm in self.performance_metrics:
            if pm not in _PERF_METRICS:
                raise ValueError(f"Unknown performance metric '{pm}'. Choose from {list(_PERF_METRICS)}")
        for fm in self.fairness_metrics:
            if fm not in _FAIR_METRICS:
                raise ValueError(f"Unknown fairness metric '{fm}'. Choose from {list(_FAIR_METRICS)}")

        self._majority_label = self._compute_majority_label()
        self._pseudo_preds = self._generate_pseudo_models()
        # baseline[(pm, fm)] = list of 11 (fair_sign, perf) tuples
        # fair_sign is negated so larger = more fair (right on x-axis)
        self.baseline: dict[tuple[str, str], list[tuple[float, float]]] = (
            self._build_baseline()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_majority_label(self) -> int:
        vals, counts = np.unique(self.y_pred_original, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _generate_pseudo_models(self) -> list[np.ndarray]:
        """Return 10 pseudo-prediction arrays for p ∈ {10%, 20%, ..., 100%}.

        Uses a single fixed permutation sliced cumulatively so that the set of
        replaced indices for M_{p+10} is a strict superset of those for M_p.
        This prevents non-monotonic performance/fairness jumps caused by
        independent random draws landing on different subsets.
        """
        n = len(self.y_pred_original)
        order = self.rng.permutation(n)
        pseudo_list = []
        for p in range(1, 11):
            n_replace = math.floor(p / 10.0 * n)
            preds = self.y_pred_original.copy()
            preds[order[:n_replace]] = self._majority_label
            pseudo_list.append(preds)
        return pseudo_list

    def _compute_perf(self, y_pred: np.ndarray, pm: str) -> float:
        fn = _PERF_METRICS[pm]
        return float(fn(self.y_true, y_pred))

    def _compute_fair(self, y_pred: np.ndarray, fm: str) -> float:
        """Return the final value of the cumulative fairness metric (signed)."""
        fn = _FAIR_METRICS[fm]
        arr = fn(y_pred, self.sensitive, self.y_true, self.lambda_decay)
        return float(arr[-1])

    def _signed_fair(self, raw_fair: float) -> float:
        """Negate absolute value so larger (closer to 0 and above) = more fair."""
        return -abs(raw_fair)

    def _build_baseline(self) -> dict[tuple[str, str], list[tuple[float, float]]]:
        baseline: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for pm in self.performance_metrics:
            for fm in self.fairness_metrics:
                points: list[tuple[float, float]] = []
                # M_ori
                ori_perf = self._compute_perf(self.y_pred_original, pm)
                ori_fair = self._signed_fair(self._compute_fair(self.y_pred_original, fm))
                points.append((ori_fair, ori_perf))
                # M_10 … M_100
                for pseudo in self._pseudo_preds:
                    perf = self._compute_perf(pseudo, pm)
                    fair = self._signed_fair(self._compute_fair(pseudo, fm))
                    points.append((fair, perf))
                baseline[(pm, fm)] = points
        return baseline

    def _interpolate_baseline_perf(
        self, points: list[tuple[float, float]], f_tech: float
    ) -> float:
        """Return expected performance on the baseline curve at fairness value f_tech.

        Uses linear interpolation over the sorted baseline; extrapolates with
        nearest value if outside the baseline's fairness range.
        """
        fair_vals = np.array([p[0] for p in points])
        perf_vals = np.array([p[1] for p in points])
        # Sort by fairness (x-axis) for interpolation
        order = np.argsort(fair_vals)
        fair_sorted = fair_vals[order]
        perf_sorted = perf_vals[order]

        if len(fair_sorted) < 2:
            return float(perf_sorted[0])

        # Clamp to the range to handle extrapolation gracefully
        f_clamped = float(np.clip(f_tech, fair_sorted[0], fair_sorted[-1]))
        interp = interp1d(fair_sorted, perf_sorted, kind="linear", assume_sorted=True)
        return float(interp(f_clamped))

    def _area_above_baseline(
        self,
        points: list[tuple[float, float]],
        f_tech: float,
        p_tech: float,
    ) -> float:
        """Area of the triangle formed by the technique point and the nearest
        baseline segment.  Returns 0.0 for non-Region-2 points."""
        fair_vals = [p[0] for p in points]
        perf_vals = [p[1] for p in points]
        order = np.argsort(fair_vals)
        fs = np.array(fair_vals)[order]
        ps = np.array(perf_vals)[order]

        # Find the baseline segment that brackets f_tech
        idx = np.searchsorted(fs, f_tech)
        if idx == 0:
            f_lo, p_lo = fs[0], ps[0]
            f_hi, p_hi = fs[min(1, len(fs) - 1)], ps[min(1, len(ps) - 1)]
        elif idx >= len(fs):
            f_lo, p_lo = fs[-2], ps[-2]
            f_hi, p_hi = fs[-1], ps[-1]
        else:
            f_lo, p_lo = fs[idx - 1], ps[idx - 1]
            f_hi, p_hi = fs[idx], ps[idx]

        # Shoelace formula for the triangle (f_tech, p_tech), (f_lo, p_lo), (f_hi, p_hi)
        area = 0.5 * abs(
            (f_lo - f_tech) * (p_hi - p_tech)
            - (f_hi - f_tech) * (p_lo - p_tech)
        )
        return float(area)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self, y_pred_technique: np.ndarray
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Classify a fairness technique into one of the 5 FBU regions.

        Parameters
        ----------
        y_pred_technique: predicted labels from the fairness technique

        Returns
        -------
        Mapping of (perf_metric, fair_metric) → {"region": 1-5, "area": float}
        """
        y_pred_technique = np.asarray(y_pred_technique)
        result: dict[tuple[str, str], dict[str, Any]] = {}

        for pm in self.performance_metrics:
            for fm in self.fairness_metrics:
                points = self.baseline[(pm, fm)]
                ori_fair, ori_perf = points[0]

                tech_perf = self._compute_perf(y_pred_technique, pm)
                tech_fair = self._signed_fair(self._compute_fair(y_pred_technique, fm))

                better_perf = tech_perf >= ori_perf
                better_fair = tech_fair >= ori_fair

                if better_perf and better_fair:
                    region = 1
                elif not better_perf and not better_fair:
                    region = 5
                elif better_perf and not better_fair:
                    region = 3
                else:
                    # better fairness, worse performance — compare to baseline curve
                    baseline_perf = self._interpolate_baseline_perf(points, tech_fair)
                    if tech_perf >= baseline_perf:
                        region = 2
                    else:
                        region = 4

                area = (
                    self._area_above_baseline(points, tech_fair, tech_perf)
                    if region == 2
                    else 0.0
                )
                result[(pm, fm)] = {"region": region, "area": area}

        return result

    def evaluate(
        self, techniques: dict[str, np.ndarray]
    ) -> dict[str, dict[str, Any]]:
        """Evaluate multiple fairness techniques and return region percentages.

        Parameters
        ----------
        techniques: mapping of technique name → prediction array

        Returns
        -------
        {name: {
            "region_percentages": [r1%, r2%, r3%, r4%, r5%],  # sum to 100
            "region_areas": {2: float},                        # area for region 2 cases
            "details": {(pm, fm): {"region": int, "area": float}, ...},
            "_raw_coords": {(pm, fm): (fairness_signed, performance)}
        }}
        """
        output: dict[str, dict[str, Any]] = {}
        for name, y_pred in techniques.items():
            y_pred = np.asarray(y_pred)
            details = self.classify(y_pred)

            # Store raw (fair_signed, perf) coordinates for visualization
            raw_coords: dict[tuple[str, str], tuple[float, float]] = {}
            for pm in self.performance_metrics:
                for fm in self.fairness_metrics:
                    perf = self._compute_perf(y_pred, pm)
                    fair = self._signed_fair(self._compute_fair(y_pred, fm))
                    raw_coords[(pm, fm)] = (fair, perf)

            counts = [0, 0, 0, 0, 0]
            total_area_r2 = 0.0
            for info in details.values():
                counts[info["region"] - 1] += 1
                if info["region"] == 2:
                    total_area_r2 += info["area"]
            n_cases = sum(counts)
            percentages = [
                100.0 * c / n_cases if n_cases > 0 else 0.0 for c in counts
            ]
            output[name] = {
                "region_percentages": percentages,
                "region_areas": {2: total_area_r2},
                "region2_mean_area": (total_area_r2 / counts[1]) if counts[1] > 0 else 0.0,
                "details": details,
                "_raw_coords": raw_coords,
            }
        return output

    def evaluate_runs(
        self,
        techniques_runs: dict[str, list[np.ndarray]],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate FBU outcomes across multiple runs per technique.

        This matches the paper's reporting style where region percentages are
        computed over all cases:
        n_runs x n_fair_metrics x n_perf_metrics (for each technique).

        Parameters
        ----------
        techniques_runs:
            Mapping of technique name to a list of prediction arrays, one per run.

        Returns
        -------
        {name: {
            "region_percentages": [r1%, r2%, r3%, r4%, r5%],
            "region_areas": {2: float},           # summed area for Region 2
            "region2_mean_area": float,           # average area per Region 2 case
            "n_runs": int,
            "n_cases": int,
            "per_run": [evaluate(...) output for that run]
        }}
        """
        aggregated: dict[str, dict[str, Any]] = {}
        for name, runs in techniques_runs.items():
            if len(runs) == 0:
                raise ValueError(f"Technique '{name}' has no runs.")

            per_run: list[dict[str, Any]] = []
            counts = [0, 0, 0, 0, 0]
            total_area_r2 = 0.0
            total_cases = 0

            for y_pred in runs:
                run_result = self.evaluate({name: y_pred})[name]
                per_run.append(run_result)
                run_counts = [
                    int(round((pct / 100.0) * len(run_result["details"])))
                    for pct in run_result["region_percentages"]
                ]
                # Guard against any floating-rounding drift from percentage conversion.
                drift = len(run_result["details"]) - sum(run_counts)
                if drift != 0:
                    run_counts[run_result["region_percentages"].index(max(run_result["region_percentages"]))] += drift

                counts = [a + b for a, b in zip(counts, run_counts)]
                total_area_r2 += run_result["region_areas"][2]
                total_cases += len(run_result["details"])

            percentages = [100.0 * c / total_cases if total_cases > 0 else 0.0 for c in counts]
            aggregated[name] = {
                "region_percentages": percentages,
                "region_areas": {2: total_area_r2},
                "region2_mean_area": (total_area_r2 / counts[1]) if counts[1] > 0 else 0.0,
                "n_runs": len(runs),
                "n_cases": total_cases,
                "per_run": per_run,
            }
        return aggregated

    def baseline_points(
        self, perf_metric: str = "balanced_accuracy", fair_metric: str = "spd"
    ) -> list[tuple[float, float]]:
        """Return the 11 baseline (fairness_signed, performance) tuples for plotting.

        Index 0 = M_ori, indices 1–10 = M_10 … M_100.
        """
        return self.baseline[(perf_metric, fair_metric)]
