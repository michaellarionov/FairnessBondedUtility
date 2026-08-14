"""Base models (spec §3.2).

FBU consumes only the thresholded {0,1} array, so the linear-vs-logistic choice
does not touch the FBU spec at all — but it does move the reported numbers.
**Logistic regression is the resolved choice for this study** (spec §6.4) and the
default everywhere; the linear probability model stays available behind the flag
for comparison.

* :class:`LPMScorer` — Linear Probability Model: OLS on {0,1} labels, thresholded
  at 0.5. Scores are unbounded; values outside [0,1] are expected and fine.
  ``lstsq`` absorbs the rank deficiency left by one-hot encoding.
* :class:`LogitScorer` — logistic regression; score is ``predict_proba[:, 1]``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from ..types import BinaryScorer, FloatArray, IntArray, ScorerFactory


class LPMScorer:
    """Linear Probability Model: ``LinearRegression`` thresholded at 0.5."""

    def __init__(self) -> None:
        self._model = LinearRegression()

    def fit(
        self, X: FloatArray, y: IntArray, sample_weight: FloatArray | None = None
    ) -> "LPMScorer":
        self._model.fit(X, np.asarray(y, dtype=np.float64), sample_weight=sample_weight)
        return self

    def score(self, X: FloatArray) -> FloatArray:
        return np.asarray(self._model.predict(X), dtype=np.float64)

    def predict(self, X: FloatArray, threshold: float = 0.5) -> IntArray:
        return (self.score(X) >= threshold).astype(np.int64)

    @property
    def coef_(self) -> FloatArray:
        return np.asarray(self._model.coef_, dtype=np.float64)


class LogitScorer:
    """Logistic regression; ``score`` is the predicted probability of class 1."""

    def __init__(self, max_iter: int = 1000, C: float = 1.0) -> None:
        self._model = LogisticRegression(max_iter=max_iter, C=C)

    def fit(
        self, X: FloatArray, y: IntArray, sample_weight: FloatArray | None = None
    ) -> "LogitScorer":
        self._model.fit(X, np.asarray(y, dtype=np.int64), sample_weight=sample_weight)
        return self

    def score(self, X: FloatArray) -> FloatArray:
        return np.asarray(self._model.predict_proba(X)[:, 1], dtype=np.float64)

    def predict(self, X: FloatArray, threshold: float = 0.5) -> IntArray:
        return (self.score(X) >= threshold).astype(np.int64)

    @property
    def coef_(self) -> FloatArray:
        return np.asarray(self._model.coef_, dtype=np.float64).ravel()


ScorerName = Literal["lpm", "logit"]

SCORERS: dict[str, ScorerFactory] = {"logit": LogitScorer, "lpm": LPMScorer}

DEFAULT_SCORER: ScorerName = "logit"


def get_scorer_factory(name: ScorerName = DEFAULT_SCORER) -> ScorerFactory:
    """Factory for a named base model. Techniques refit from scratch, hence a factory."""
    try:
        return SCORERS[name]
    except KeyError:
        raise ValueError(
            f"unknown scorer '{name}'; choose from {sorted(SCORERS)}"
        ) from None


__all__ = [
    "BinaryScorer",
    "LPMScorer",
    "LogitScorer",
    "SCORERS",
    "ScorerName",
    "DEFAULT_SCORER",
    "get_scorer_factory",
]
