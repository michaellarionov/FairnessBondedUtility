"""Data contracts and protocols for FBU.

Spec: docs/FBU_IMPLEMENTATION.md §2.1, §2.2, §2.5, §2.6, §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float64]

#: Performance metric: (y_true, y_pred) -> float, higher is better.
PerformanceMetric = Callable[[IntArray, IntArray], float]

#: Fairness metric: (y_true, y_pred, s) -> float, signed, 0 == fair.
#: Argument order mirrors ``PerformanceMetric`` so the two families compose.
FairnessMetric = Callable[[IntArray, IntArray, IntArray], float]

REGION_LABELS: dict[int, str] = {
    1: "Jointly advantageous",
    2: "Impressive",
    3: "Reversed",
    4: "Deficient",
    5: "Jointly disadvantageous",
}


def _as_binary(array: npt.ArrayLike, name: str, n: int | None = None) -> IntArray:
    arr = np.asarray(array)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} has length {arr.shape[0]}, expected {n}")
    out = arr.astype(np.int64, copy=True)
    if not np.array_equal(out, arr):
        raise ValueError(f"{name} must contain integers, got {arr.dtype}")
    bad = np.setdiff1d(np.unique(out), np.array([0, 1]))
    if bad.size:
        raise ValueError(f"{name} must be binary {{0,1}}, found {bad.tolist()}")
    return out


@dataclass(frozen=True)
class Predictions:
    """Predictions of one model/technique over one evaluation set.

    ``1`` is the favorable outcome in ``y_true``/``y_pred`` and the privileged
    group in ``s``. Every technique in a comparison must produce predictions
    over the *same* evaluation rows in the *same* order; see
    :func:`fbu.fbu.assert_aligned`.
    """

    y_true: IntArray
    y_pred: IntArray
    s: IntArray
    score: FloatArray | None = None
    name: str = "model"

    def __post_init__(self) -> None:
        y_true = _as_binary(self.y_true, "y_true")
        n = y_true.shape[0]
        if n == 0:
            raise ValueError("Predictions must be non-empty")
        object.__setattr__(self, "y_true", y_true)
        object.__setattr__(self, "y_pred", _as_binary(self.y_pred, "y_pred", n))
        object.__setattr__(self, "s", _as_binary(self.s, "s", n))
        if self.score is not None:
            score = np.asarray(self.score, dtype=np.float64)
            if score.shape != (n,):
                raise ValueError(f"score must have shape ({n},), got {score.shape}")
            object.__setattr__(self, "score", score)

    def __len__(self) -> int:
        return int(self.y_true.shape[0])

    def with_predictions(self, y_pred: npt.ArrayLike, name: str) -> "Predictions":
        """Return a copy carrying different labels over the same rows."""
        return Predictions(
            y_true=self.y_true, y_pred=y_pred, s=self.s, score=self.score, name=name
        )

    def require_score(self) -> FloatArray:
        if self.score is None:
            raise ValueError(f"technique '{self.name}' requires continuous scores")
        return self.score


@dataclass(frozen=True)
class SplitData:
    """A fixed train/test split with the sensitive attribute carried alongside.

    ``X`` keeps the sensitive attribute as a feature (spec §3.1.5); the
    fairness-unaware technique drops ``sensitive_columns`` to build its contrast.
    """

    X_train: FloatArray
    y_train: IntArray
    s_train: IntArray
    X_test: FloatArray
    y_test: IntArray
    s_test: IntArray
    feature_names: tuple[str, ...]
    sensitive_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("train/test feature counts differ")
        if self.X_train.shape[1] != len(self.feature_names):
            raise ValueError("feature_names length does not match X width")
        missing = set(self.sensitive_columns) - set(self.feature_names)
        if missing:
            raise ValueError(f"sensitive columns absent from features: {missing}")

    @property
    def sensitive_indices(self) -> tuple[int, ...]:
        lookup = {name: i for i, name in enumerate(self.feature_names)}
        return tuple(lookup[name] for name in self.sensitive_columns)

    def drop_sensitive(self) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
        """Return (X_train, X_test, feature_names) without the sensitive columns."""
        keep = [
            i for i in range(len(self.feature_names)) if i not in set(self.sensitive_indices)
        ]
        names = tuple(self.feature_names[i] for i in keep)
        return self.X_train[:, keep], self.X_test[:, keep], names


@runtime_checkable
class BinaryScorer(Protocol):
    """A model that emits a continuous score and a thresholded label."""

    def fit(
        self, X: FloatArray, y: IntArray, sample_weight: FloatArray | None = None
    ) -> "BinaryScorer": ...

    def score(self, X: FloatArray) -> FloatArray:
        """Continuous score, any range; higher means more likely favorable."""
        ...

    def predict(self, X: FloatArray, threshold: float = 0.5) -> IntArray: ...


#: Factory producing a fresh, unfitted scorer. Techniques refit from scratch.
ScorerFactory = Callable[[], BinaryScorer]


@dataclass(frozen=True)
class CaseResult:
    """One cell of the FBU case grid (spec §2.6)."""

    technique: str
    run: int
    dataset: str
    performance_metric: str
    fairness_metric: str
    fairness: float
    performance: float
    region: int
    area: float | None = None
    area_norm: float | None = None

    @property
    def region_label(self) -> str:
        return REGION_LABELS[self.region]


@dataclass(frozen=True)
class FBUResult:
    """FBU outcome for one technique over the full case grid (spec §2.6)."""

    name: str
    region_percentages: tuple[float, float, float, float, float]
    region_counts: tuple[int, int, int, int, int]
    n_cases: int
    area_mean: float
    area_median: float
    area_std: float
    area_norms: tuple[float, ...] = field(default_factory=tuple)
    cases: tuple[CaseResult, ...] = field(default_factory=tuple)

    def percentage(self, region: int) -> float:
        return self.region_percentages[region - 1]


class TradeoffArea(Protocol):
    """Quantifier for a region-2 point's distance from the baseline curve.

    The default implementation is the closed-region integral of spec §2.5
    (`[D2]`); alternatives exist for the sensitivity check demanded there.
    """

    name: str

    def __call__(
        self,
        curve: "BaselineCurveLike",
        f_p: float,
        y_p: float,
        f_o: float,
        y_o: float,
    ) -> float: ...


class BaselineCurveLike(Protocol):
    """The piecewise-linear baseline curve g (spec §2.3)."""

    @property
    def f_min(self) -> float: ...

    @property
    def f_max(self) -> float: ...

    def evaluate(self, f: float) -> float: ...

    def integrate(self, f_lo: float, f_hi: float) -> float: ...

    def solve_fairness(self, y: float, f_lo: float | None = None) -> float: ...

    def knots(self) -> Iterable[tuple[float, float]]: ...


def unique_name(names: Sequence[str]) -> None:
    """Raise if technique names collide, which would silently merge results."""
    seen = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate technique name '{name}'")
        seen.add(name)
