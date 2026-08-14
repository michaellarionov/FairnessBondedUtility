"""Fairness-unaware baseline: drop the sensitive attribute and refit.

The classic negative result. SPD barely moves because proxies (``relationship``,
``marital-status``, ``hours-per-week``) carry the same information. If |SPD|
collapses to ~0 here, ``sex`` was not the driver and something upstream is wrong.
"""

from __future__ import annotations

from ..models.scorers import LogitScorer
from ..types import Predictions, ScorerFactory, SplitData

NAME = "fairness_unaware"


def fairness_unaware(
    data: SplitData,
    scorer_factory: ScorerFactory = LogitScorer,
    threshold: float = 0.5,
) -> Predictions:
    """Refit the base model on the feature matrix without the sensitive columns."""
    X_train, X_test, _ = data.drop_sensitive()
    model = scorer_factory().fit(X_train, data.y_train)
    scores = model.score(X_test)
    return Predictions(
        y_true=data.y_test,
        y_pred=(scores >= threshold).astype(int),
        s=data.s_test,
        score=scores,
        name=NAME,
    )


__all__ = ["fairness_unaware", "NAME"]
