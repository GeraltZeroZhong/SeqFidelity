"""Exact matching of independently constructed paired predictions."""

import numpy as np
import pandas as pd
import pytest

from seqfidelity.comparisons import align_predictions, compare_predictions
from seqfidelity.metrics import EvaluationThresholds


def predictions(error=0.):
    truth = np.array([-2., -1., 1., 2.])
    return pd.DataFrame({"pair_id": [f"pair_{index}" for index in range(4)],
                         "locus_id": ["group_a", "group_a", "group_b", "group_b"],
                         "delta_true": truth, "delta_pred": truth + error,
                         "quality_pass": [True] * 4, "p_value": [.001] * 4,
                         "se": [.25] * 4})


def test_reordered_pairs_align_without_discarding_rows():
    candidate, reference = predictions(), predictions(2.).iloc[::-1]
    aligned = align_predictions(candidate, reference)
    assert aligned.pair_id.tolist() == candidate.pair_id.tolist()
    np.testing.assert_array_equal(aligned.reference_prediction, candidate.delta_true + 2.)
    result = compare_predictions(candidate, reference, EvaluationThresholds(1.), n_bootstrap=64, seed=0)
    assert result["macro_delta_mse_difference"] == -4.
    assert result["relative_macro_mse_improvement"] == 1.
    assert result["all_pairs_retained"]
    assert result["n"] == 4
    assert result["n_loci"] == 2


def test_mismatched_pair_sets_raise_instead_of_joining_subset():
    with pytest.raises(ValueError, match="no rows discarded"):
        align_predictions(predictions().iloc[:-1], predictions())


@pytest.mark.parametrize("value", (None, "pair_1"))
def test_missing_or_duplicate_pair_identifier_is_rejected(value):
    reference = predictions()
    reference.loc[0, "pair_id"] = value
    with pytest.raises(ValueError, match="nonmissing and unique"):
        align_predictions(predictions(), reference)


@pytest.mark.parametrize("column,value,message", (
    ("locus_id", "group_other", "locus assignments"),
    ("delta_true", 3., "effect labels"),
    ("quality_pass", False, "measurement field"),
    ("p_value", .5, "measurement field"),
    ("se", .5, "measurement field"),
    ("delta_pred", np.nan, "finite"),
))
def test_matched_pairs_require_identical_labels_and_finite_predictions(column, value, message):
    reference = predictions()
    reference.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        align_predictions(predictions(), reference)
