"""Hand-computed metrics on invented grouped observations."""

import numpy as np
import pandas as pd
import pytest

from seqfidelity.metrics import (
    EvaluationThresholds, absolute_error_decomposition, evaluate_predictions,
    fit_thresholds, paired_comparison,
)


def grouped_predictions():
    return pd.DataFrame({"delta_true": [2., 2., -4.], "delta_pred": [0., 0., -4.],
                         "locus_id": ["group_a", "group_a", "group_b"],
                         "element": ["group_a", "group_a", "group_b"]})


def test_macro_error_and_bootstrap_weight_groups_equally():
    frame = grouped_predictions()
    thresholds = EvaluationThresholds(2.)
    result = evaluate_predictions(frame, thresholds, n_bootstrap=128, seed=0)
    assert result["macro_delta_mse"] == 2.
    assert result["micro_delta_mse"] == pytest.approx(8 / 3)
    duplicated = pd.concat([frame, frame.iloc[:2]], ignore_index=True)
    repeated = evaluate_predictions(duplicated, thresholds, n_bootstrap=128, seed=0)
    assert repeated["macro_delta_mse"] == 2.
    assert repeated["all"]["confidence_intervals"] == result["all"]["confidence_intervals"]
    assert result["per_element"]["group_a"]["confidence_intervals"]["macro_delta_mse"]["low"] is None


def test_screening_uses_ceil_budget_and_quality_requires_explicit_evidence():
    frame = grouped_predictions()
    thresholds = EvaluationThresholds(2., screening_fraction=1 / 4)
    unknown = evaluate_predictions(frame, thresholds, n_bootstrap=0, seed=0)
    assert unknown["all"]["reliable_large"] is None
    frame["quality_pass"] = [True, np.nan, False]
    frame["p_value"] = [.001, .5, .001]
    result = evaluate_predictions(frame, thresholds, n_bootstrap=0, seed=0)
    assert result["all"]["screening_budget"] == 1
    assert result["all"]["large_effect"]["recall_at_budget"] == pytest.approx(1 / 3)
    assert result["all"]["reliable_large"]["n"] == 1
    assert result["high_quality"]["n"] == 1


def test_threshold_fit_uses_validation_quantiles_only():
    frame = grouped_predictions().assign(split="validation")
    thresholds = fit_thresholds(frame, strong_quantile=.75)
    assert thresholds.strong_abs_delta == 3.
    assert thresholds.validation_rows == 3
    assert EvaluationThresholds.from_dict(thresholds.to_dict()) == thresholds
    test_frame = frame.assign(split="test")
    with pytest.raises(ValueError, match="validation"):
        fit_thresholds(test_frame)
    result = evaluate_predictions(test_frame, thresholds, n_bootstrap=0, seed=0)
    assert result["thresholds"]["strong_abs_delta"] == 3.


@pytest.mark.parametrize("kwargs", ({"screening_fraction": 0}, {"screening_fraction": 2},
                                    {"tail_quantile": 0}, {"tail_quantile": 1}))
def test_invalid_threshold_fit_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_thresholds(grouped_predictions(), **kwargs)


def test_threshold_fit_requires_finite_labels():
    frame = grouped_predictions()
    frame["delta_true"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_thresholds(frame)


def test_paired_error_difference_is_candidate_minus_reference():
    frame = grouped_predictions()
    frame["candidate"] = frame.delta_true
    result = paired_comparison(frame, "candidate", n_bootstrap=128, seed=0)
    assert result["macro_delta_mse_difference"] == -2.
    assert result["macro_delta_mae_difference"] == -1.
    assert result["n_loci"] == 2
    assert result["n"] == 3


def test_paired_error_covariance_identity_keeps_shared_error():
    truth_ref, truth_alt = np.array([0., 2., 4.]), np.array([1., 4., 7.])
    common_error = np.array([1., -2., 3.])
    result = absolute_error_decomposition(truth_ref, truth_alt,
                                           truth_ref + common_error, truth_alt + common_error)
    assert result["delta_mse"] == 0.
    assert result["decomposed_delta_mse"] == pytest.approx(0.)
    unequal = absolute_error_decomposition(truth_ref, truth_alt,
                                            truth_ref + common_error, truth_alt - common_error)
    assert unequal["delta_mse"] == pytest.approx(56 / 3)
    assert unequal["decomposed_delta_mse"] == pytest.approx(56 / 3)


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_nonfinite_predictions_cannot_silently_drop_eligible_rows(value):
    frame = grouped_predictions()
    frame.loc[0, "delta_pred"] = value
    with pytest.raises(ValueError, match="Nonfinite model predictions"):
        evaluate_predictions(frame, EvaluationThresholds(2.), n_bootstrap=0, seed=0)


def test_missing_group_identifier_is_rejected():
    frame = grouped_predictions()
    frame.loc[0, "locus_id"] = None
    with pytest.raises(ValueError, match="identifiers"):
        evaluate_predictions(frame, EvaluationThresholds(2.), n_bootstrap=0, seed=0)
