"""DNA symbols and variable-length collation from invented strings."""

import numpy as np
import pandas as pd
import pytest
import torch

from seqfidelity.encoding import PairDataset, collate_pairs, device_batch, encode_sequence, loader


def pairs():
    return pd.DataFrame({"ref_seq": ["ACGN", "TA"], "alt_seq": ["ATGN", "TC"],
                         "delta": [1., -1.], "y_ref": [2., 4.], "y_alt": [3., 3.]})


def test_dna_symbols_have_explicit_codes_in_both_letter_cases():
    expected = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    np.testing.assert_array_equal(encode_sequence("ACGTN"), expected)
    np.testing.assert_array_equal(encode_sequence("acgtn"), expected)


@pytest.mark.parametrize("value", ("", None, 12, "ACGU", "A C", "ACGTé"))
def test_invalid_sequences_are_rejected(value):
    with pytest.raises(ValueError):
        encode_sequence(value)


def test_collation_preserves_lengths_labels_and_original_indices():
    dataset = PairDataset(pairs())
    batch = collate_pairs([dataset[1], dataset[0]])
    assert torch.equal(batch["lengths"], torch.tensor([2, 4]))
    assert torch.equal(batch["ref"], torch.tensor([[3, 0, 4, 4], [0, 1, 2, 4]]))
    assert torch.equal(batch["alt"], torch.tensor([[3, 1, 4, 4], [0, 3, 2, 4]]))
    assert torch.equal(batch["index"], torch.tensor([1, 0]))
    torch.testing.assert_close(batch["delta"], torch.tensor([-1., 1.]))
    torch.testing.assert_close(batch["activity"], torch.tensor([[4., 3.], [2., 3.]]))
    moved = device_batch(batch, "cpu")
    assert "index" not in moved
    for key, value in moved.items():
        assert torch.equal(value, batch[key])


def test_inference_pairs_can_omit_targets():
    dataset = PairDataset(pairs()[["ref_seq", "alt_seq"]])
    batch = collate_pairs([dataset[0], dataset[1]])
    assert torch.isnan(batch["delta"]).all()
    assert torch.isnan(batch["activity"]).all()


def test_unequal_pair_lengths_and_empty_batches_are_rejected():
    frame = pairs()
    frame.loc[0, "alt_seq"] = "AC"
    with pytest.raises(ValueError, match="equal lengths"):
        PairDataset(frame)
    with pytest.raises(ValueError, match="empty batch"):
        collate_pairs([])


def test_seeded_shuffle_visits_each_pair_once_and_is_repeatable():
    frame = pd.concat([pairs()] * 4, ignore_index=True)
    first = torch.cat([batch["index"] for batch in loader(frame, 3, shuffle=True, seed=0)])
    second = torch.cat([batch["index"] for batch in loader(frame, 3, shuffle=True, seed=0)])
    assert torch.equal(first, second)
    assert torch.equal(first.sort().values, torch.arange(len(frame)))
