"""M2 acceptance tests — Adult data and base models (spec §3.1–§3.3, §5.2 M2)."""

from __future__ import annotations

import numpy as np
import pytest

from fbu.data.adult import CONTINUOUS, N_TEST, N_TRAIN, base_rates, load_raw
from fbu.metrics import balanced_accuracy, eod, recall, spd
from fbu.models.scorers import LogitScorer, LPMScorer
from fbu.pseudo import majority_label
from fbu.techniques.base import fit_original

from .conftest import requires_adult

pytestmark = requires_adult


def test_row_counts():
    """32,561 train + 16,281 test = 48,842 rows, canonical split preserved."""
    raw = load_raw()
    assert len(raw) == N_TRAIN + N_TEST == 48842
    assert int((raw["split"] == "train").sum()) == N_TRAIN
    assert int((raw["split"] == "test").sum()) == N_TEST


def test_test_split_labels_are_cleaned():
    """adult.test carries a stray header row and a trailing period on every label."""
    raw = load_raw()
    assert set(raw["income"].unique()) == {"<=50K", ">50K"}
    assert not raw["income"].str.endswith(".").any()


def test_missing_values_are_kept_as_a_category():
    """'?' stays in the data as its own level [D6] — dropping ~7% of rows would
    itself be an intervention with a differential rate across groups."""
    raw = load_raw()
    marked = (raw == "?").sum().sum()
    assert marked > 0
    assert not raw.isna().any().any()
    for column in ("workclass", "occupation", "native-country"):
        assert "?" in set(raw[column].unique())


def test_base_rates(adult):
    """~24% positive overall; ~30% of men and ~11% of women, a gap of ~0.19."""
    rates = base_rates(adult)
    assert rates["overall"] == pytest.approx(0.24, abs=0.005)
    assert rates["privileged"] == pytest.approx(0.30, abs=0.02)
    assert rates["unprivileged"] == pytest.approx(0.11, abs=0.02)
    gap = rates["privileged"] - rates["unprivileged"]
    assert 0.18 <= gap <= 0.22


def test_dropped_and_kept_columns(adult):
    """fnlwgt is a sampling weight and education duplicates education-num; both go.
    ``sex`` stays in the matrix so the fairness-unaware contrast is possible."""
    names = adult.feature_names
    assert not any(name == "fnlwgt" or name.startswith("education_") for name in names)
    assert "education-num" in names
    assert adult.sensitive_columns == ("sex_Male",)
    column = adult.X_train[:, adult.sensitive_indices[0]]
    assert np.array_equal(column.astype(int), adult.s_train)


def test_continuous_features_are_standardised_on_train(adult):
    """Train means ≈ 0 and standard deviations ≈ 1; test uses the train statistics."""
    for name in CONTINUOUS:
        index = adult.feature_names.index(name)
        assert adult.X_train[:, index].mean() == pytest.approx(0.0, abs=1e-9)
        assert adult.X_train[:, index].std() == pytest.approx(1.0, abs=1e-9)
    test_means = [
        adult.X_test[:, adult.feature_names.index(name)].mean() for name in CONTINUOUS
    ]
    assert max(abs(m) for m in test_means) < 0.1


def test_train_and_test_share_the_design_matrix(adult):
    """One-hot encoding runs on both splits together, so column counts match and
    no categorical level is silently missing from one side."""
    assert adult.X_train.shape[1] == adult.X_test.shape[1] == len(adult.feature_names)
    assert adult.X_train.shape[0] == N_TRAIN
    assert adult.X_test.shape[0] == N_TEST


def test_drop_sensitive_removes_exactly_one_column(adult):
    X_train, X_test, names = adult.drop_sensitive()
    assert X_train.shape[1] == adult.X_train.shape[1] - 1
    assert X_test.shape[1] == adult.X_test.shape[1] - 1
    assert "sex_Male" not in names


@pytest.mark.parametrize("factory", [LPMScorer, LogitScorer])
def test_base_model_quality_and_bias(adult, factory):
    """Both scorers must clear accuracy ≥ 0.80 and balanced accuracy ≥ 0.70.

    SPD near 0 or above 0.35 would mean the group coding or threshold is wrong
    (spec §3.3), so the band is asserted rather than merely printed.
    """
    prediction = fit_original(adult, factory)
    accuracy = float((prediction.y_pred == prediction.y_true).mean())
    assert accuracy >= 0.80
    assert balanced_accuracy(prediction.y_true, prediction.y_pred) >= 0.70
    assert 0.10 <= spd(prediction.y_true, prediction.y_pred, prediction.s) <= 0.25
    assert 0.05 <= eod(prediction.y_true, prediction.y_pred, prediction.s) <= 0.15
    assert recall(prediction.y_true, prediction.y_pred) >= 0.40


@pytest.mark.parametrize("factory", [LPMScorer, LogitScorer])
def test_majority_label_is_zero_on_adult(adult, factory):
    """L* = 0 ('≤50K'): the base rate is ~24% and a 0.5-thresholded linear or
    logistic model predicts positive less often still. Asserted, not assumed."""
    prediction = fit_original(adult, factory)
    assert majority_label(prediction.y_pred) == 0
    assert prediction.y_pred.mean() < 0.5


def test_lpm_scores_leave_the_unit_interval(adult):
    """The Linear Probability Model is OLS on {0,1}: scores outside [0,1] are
    expected and harmless, since only the thresholded array reaches FBU."""
    prediction = fit_original(adult, LPMScorer)
    scores = prediction.require_score()
    assert scores.min() < 0.0 or scores.max() > 1.0


def test_logit_scores_are_probabilities(adult):
    scores = fit_original(adult, LogitScorer).require_score()
    assert 0.0 <= scores.min() and scores.max() <= 1.0


@pytest.mark.parametrize("factory", [LPMScorer, LogitScorer])
def test_sample_weight_changes_the_fit(adult, factory):
    """Reweighing depends on this: weights must reach the underlying estimator."""
    rng = np.random.default_rng(0)
    weights = rng.uniform(0.1, 2.0, adult.y_train.shape[0])
    unweighted = factory().fit(adult.X_train, adult.y_train)
    weighted = factory().fit(adult.X_train, adult.y_train, sample_weight=weights)
    assert not np.allclose(unweighted.coef_, weighted.coef_)
    assert not np.array_equal(
        unweighted.predict(adult.X_test), weighted.predict(adult.X_test)
    )
