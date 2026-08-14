"""The five-region classifier (spec §2.4).

Regions 1, 3 and 5 are quadrants around the **original model** point. Only the
region-2/4 split consults the **baseline curve**. These two reference objects
must not be collapsed into one.
"""

from __future__ import annotations

from .baseline import BaselineCurve
from .types import REGION_LABELS

EPS = 1e-9


def classify(
    f_p: float,
    y_p: float,
    f_o: float,
    y_o: float,
    curve: BaselineCurve,
    eps: float = EPS,
) -> int:
    """Region of technique point ``P = (f_p, y_p)`` given ``M_ori = (f_o, y_o)``.

    ``M_ori`` itself ties both thresholds and so classifies as region 1; that
    degenerate case is harmless and documented in spec §2.4.
    """
    if f_p >= f_o - eps and y_p >= y_o - eps:
        return 1  # jointly advantageous
    if f_p < f_o and y_p >= y_o - eps:
        return 3  # reversed: more accurate, less fair
    if f_p < f_o and y_p < y_o:
        return 5  # jointly disadvantageous
    # Fairer but less accurate — the trade-off quadrant, split by the curve.
    return 2 if y_p >= curve.evaluate(f_p) - eps else 4


def region_label(region: int) -> str:
    return REGION_LABELS[region]


__all__ = ["classify", "region_label", "REGION_LABELS", "EPS"]
