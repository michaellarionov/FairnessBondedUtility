"""Streaming fairness and performance metrics for FBU."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def imbalance_ratio(y: np.ndarray) -> float:
    """Return IR = S_min / (S_maj + S_min) for a label array (Equation 1)."""
    y = np.asarray(y)
    values, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        return 0.0 if counts[0] == len(y) else 1.0
    s_min = counts.min()
    s_maj = counts.max()
    return s_min / (s_maj + s_min)


def cumulative_spd(
    predictions: np.ndarray,
    sensitive: np.ndarray,
    y_true: np.ndarray,
    lambda_decay: float = 0.0,
) -> np.ndarray:
    """Compute running Cumulative Statistical Parity Difference (Equation 2).

    Parameters
    ----------
    predictions:  predicted labels, shape (n,)
    sensitive:    binary sensitive attribute (1 = privileged, 0 = unprivileged)
    y_true:       true labels (used only to define the favourable label = 1)
    lambda_decay: decay factor λ ∈ [0, 1]; higher → more historical weight

    Returns
    -------
    cspd: shape (n,), running CSPD at each timestep (value near 0 = fair)
    """
    predictions = np.asarray(predictions)
    sensitive = np.asarray(sensitive)
    n = len(predictions)
    cspd = np.zeros(n)

    # Running counts: favourable predictions per group and group totals
    priv_fav = 0.0   # F(d) ∧ Y=1 | S=privileged
    priv_tot = 0.0   # d | S=privileged
    unpriv_fav = 0.0
    unpriv_tot = 0.0

    prev = 0.0
    for t in range(n):
        s = sensitive[t]
        pred = predictions[t]
        if s == 1:
            priv_tot += 1
            if pred == 1:
                priv_fav += 1
        else:
            unpriv_tot += 1
            if pred == 1:
                unpriv_fav += 1

        p_priv = priv_fav / priv_tot if priv_tot > 0 else 0.0
        p_unpriv = unpriv_fav / unpriv_tot if unpriv_tot > 0 else 0.0
        raw = p_priv - p_unpriv
        prev = (1.0 - lambda_decay) * raw + lambda_decay * prev
        cspd[t] = prev

    return cspd


def cumulative_eod(
    predictions: np.ndarray,
    sensitive: np.ndarray,
    y_true: np.ndarray,
    lambda_decay: float = 0.0,
) -> np.ndarray:
    """Compute running Cumulative Equal Opportunity Difference (Equation 3).

    Conditions on true positives (y_true == 1), measuring TPR gap between
    privileged and unprivileged groups.

    Returns
    -------
    ceod: shape (n,), running CEOD at each timestep
    """
    predictions = np.asarray(predictions)
    sensitive = np.asarray(sensitive)
    y_true = np.asarray(y_true)
    n = len(predictions)
    ceod = np.zeros(n)

    priv_tp = 0.0    # F(d) ∧ Y=1 | S=priv, Y=1
    priv_pos = 0.0   # d | S=priv, Y=1
    unpriv_tp = 0.0
    unpriv_pos = 0.0

    prev = 0.0
    for t in range(n):
        s = sensitive[t]
        pred = predictions[t]
        yt = y_true[t]
        if yt == 1:
            if s == 1:
                priv_pos += 1
                if pred == 1:
                    priv_tp += 1
            else:
                unpriv_pos += 1
                if pred == 1:
                    unpriv_tp += 1

        tpr_priv = priv_tp / priv_pos if priv_pos > 0 else 0.0
        tpr_unpriv = unpriv_tp / unpriv_pos if unpriv_pos > 0 else 0.0
        raw = tpr_priv - tpr_unpriv
        prev = (1.0 - lambda_decay) * raw + lambda_decay * prev
        ceod[t] = prev

    return ceod


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return balanced accuracy = (TPR + TNR) / 2."""
    return float(balanced_accuracy_score(y_true, y_pred))
