"""The comparison set (spec Part IV).

FBU is a comparison metric: a base model alone yields only the origin point and
the baseline curve, with nothing plotted against it. These four techniques are
the minimum set that exercises it.
"""

from .base import Technique, fit_original
from .group_threshold import group_threshold
from .massaging import massaging
from .reweighing import reweighing, reweighing_weights
from .unaware import fairness_unaware

#: Name -> callable(data, scorer_factory) -> Predictions
ALL_TECHNIQUES = {
    "fairness_unaware": fairness_unaware,
    "reweighing": reweighing,
    "massaging": massaging,
    "group_threshold": group_threshold,
}

__all__ = [
    "Technique",
    "fit_original",
    "fairness_unaware",
    "reweighing",
    "reweighing_weights",
    "massaging",
    "group_threshold",
    "ALL_TECHNIQUES",
]
