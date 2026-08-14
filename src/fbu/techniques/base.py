"""Shared plumbing for the comparison set (spec Part IV)."""

from __future__ import annotations

from typing import Protocol

from ..models.scorers import LogitScorer
from ..types import Predictions, ScorerFactory, SplitData


class Technique(Protocol):
    """A bias-mitigation technique: fit on train, predict on test.

    Every technique returns predictions over the same test rows in the same
    order; the orchestrator asserts this.
    """

    name: str

    def __call__(self, data: SplitData, scorer_factory: ScorerFactory) -> Predictions: ...


def fit_original(
    data: SplitData,
    scorer_factory: ScorerFactory = LogitScorer,
    threshold: float = 0.5,
    name: str = "original",
) -> Predictions:
    """Fit the unmitigated base model and predict on test."""
    model = scorer_factory().fit(data.X_train, data.y_train)
    scores = model.score(data.X_test)
    return Predictions(
        y_true=data.y_test,
        y_pred=(scores >= threshold).astype(int),
        s=data.s_test,
        score=scores,
        name=name,
    )


__all__ = ["Technique", "fit_original"]
