"""Independent analytic fixtures for coordinates and feature normalization."""

import numpy as np
import pandas as pd
import pytest
import torch

from seqfidelity.probes import PairProbe, _moments, _tensor_batch
from seqfidelity.representations import TargetProbe, pool_state, target_values
from seqfidelity.token_probes import TokenDeltaProbe, compressed_input_identity, extract_compressed_prefix, token_pool


def coordinate_state():
    hidden = torch.tensor([[[2.], [4.], [900.]]])
    coordinates = torch.tensor([[0, 1, 900]])
    return hidden, coordinates, torch.tensor([2]), torch.tensor([5])


def test_coordinate_pooling_uses_original_positions_and_ignores_padding():
    state = coordinate_state()
    torch.testing.assert_close(pool_state(*state, pooling="moment"), torch.tensor([[3., 4., .5]]))
    torch.testing.assert_close(pool_state(*state, pooling="fourier"), torch.tensor([[3., 2., 1.]]))
    torch.testing.assert_close(token_pool(*state), torch.tensor([[3., 4., 2., 1.]]))
    with pytest.raises(ValueError):
        pool_state(*state, pooling="unknown")


def test_constant_training_features_keep_native_scale_for_new_values():
    features = np.array([[[0., 7.], [2., 7.]], [[4., 7.], [6., 7.]]], dtype=np.float32)
    mean, scale = _moments(features, batch_size=1)
    np.testing.assert_array_equal(mean, [3., 7.])
    np.testing.assert_allclose(scale, [np.sqrt(5), 1.])
    other_mean, other_scale = _moments(features, batch_size=3)
    np.testing.assert_array_equal(mean, other_mean)
    np.testing.assert_array_equal(scale, other_scale)
    future = np.array([[[3., 9.], [3., 5.]]], dtype=np.float32)
    normalized = _tensor_batch(future, [0], mean, scale, "cpu")
    torch.testing.assert_close(normalized, torch.tensor([[[0., 2.], [0., -2.]]]))


@pytest.mark.parametrize("probe_class", (PairProbe, TargetProbe))
def test_shared_delta_readout_is_antisymmetric_and_zero_for_identical_inputs(probe_class):
    torch.manual_seed(0)
    probe = probe_class(feature_dim=3, hidden=4)
    assert sum(parameter.numel() for parameter in probe.parameters()) == 21
    values = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3) / 4
    torch.testing.assert_close(probe(values), -probe(values.flip(1)), atol=0, rtol=0)
    identical = values[:, :1].expand(-1, 2, -1)
    torch.testing.assert_close(probe(identical), torch.zeros(2), atol=0, rtol=0)


def test_background_readout_averages_allele_scores_and_targets():
    torch.manual_seed(0)
    probe = TargetProbe(feature_dim=3, hidden=4, target="background")
    assert sum(parameter.numel() for parameter in probe.parameters()) == 21
    values = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3) / 4
    expected = probe.head(values).squeeze(-1).mean(-1)
    torch.testing.assert_close(probe(values), expected)
    torch.testing.assert_close(probe(values), probe(values.flip(1)), rtol=0, atol=0)
    frame = pd.DataFrame({"delta": [1., -2.], "y_ref": [2., 6.], "y_alt": [4., 2.]})
    np.testing.assert_array_equal(target_values(frame, "delta"), [1., -2.])
    np.testing.assert_array_equal(target_values(frame, "background"), [3., 4.])


def identity_prefix():
    return {"hidden": torch.tensor([[[1., 2.], [3., 4.], [100., 100.]],
                                     [[1., 2.], [3., 4.], [200., 200.]]]),
            "coords": torch.tensor([[0, 2, 100], [0, 2, 200]]),
            "lengths": torch.tensor([2, 2]), "original_lengths": torch.tensor([5, 5]),
            "pair_count": 1}


def test_token_identity_ignores_padding_and_requires_coordinates_and_lengths():
    prefix = identity_prefix()
    assert compressed_input_identity(prefix)["compressed_input_identical"].item()
    prefix["coords"][1, 1] = 3
    changed = compressed_input_identity(prefix)
    assert changed["compressed_values_identical"].item()
    assert not changed["compressed_coords_identical"].item()
    assert not changed["compressed_input_identical"].item()
    prefix = identity_prefix()
    prefix["original_lengths"][1] = 6
    assert not compressed_input_identity(prefix)["compressed_input_identical"].item()
    prefix = identity_prefix()
    prefix["lengths"][1] = 1
    assert not compressed_input_identity(prefix)["compressed_input_identical"].item()


def test_token_readout_is_antisymmetric_and_ignores_padded_features():
    torch.manual_seed(0)
    probe = TokenDeltaProbe(mean=torch.zeros(2), std=torch.ones(2), conv_channels=4, hidden=8)
    assert sum(parameter.numel() for parameter in probe.parameters()) == 165
    prefix = identity_prefix()
    torch.testing.assert_close(probe(prefix), torch.zeros(1), rtol=0, atol=1e-7)
    prefix["hidden"][1, 0, 0] += 1
    forward = probe(prefix)
    reverse = {name: value.flip(0) if isinstance(value, torch.Tensor) else value
               for name, value in prefix.items()}
    torch.testing.assert_close(forward, -probe(reverse), rtol=1e-5, atol=1e-7)
    assert forward.abs().item() > 1e-6
    prefix["hidden"][:, 2] *= 100
    torch.testing.assert_close(probe(prefix), forward, rtol=0, atol=0)

    cropped = probe.allele_scores(prefix["hidden"][:1, :2], prefix["coords"][:1, :2],
                                  prefix["lengths"][:1], prefix["original_lengths"][:1])
    padded = probe.allele_scores(prefix["hidden"][:1], prefix["coords"][:1],
                                 prefix["lengths"][:1], prefix["original_lengths"][:1])
    torch.testing.assert_close(cropped, padded, rtol=1e-6, atol=1e-6)
    probe(prefix).square().sum().backward()
    assert probe.conv.weight.grad is not None
    assert probe.conv.weight.grad.abs().sum().item() > 0


@pytest.mark.parametrize("mode", ("none", "fixed", "learned"))
def test_compressed_prefix_matches_full_features_without_evaluating_backbone(mode):
    from seqfidelity.model import ModelConfig, PairedSequenceModel

    torch.manual_seed(0)
    model = PairedSequenceModel(ModelConfig(mode=mode, dim=8, heads=2, depth=1,
                                 encoder_depth=1, decoder_depth=1, compression=3)).eval()
    ref = torch.arange(2 * 11).reshape(2, 11) % 4
    alt = ref.clone()
    alt[:, 3] = (alt[:, 3] + 1) % 4
    lengths = torch.tensor([8, 11])
    with torch.no_grad():
        full = model.forward_pair(ref, alt, lengths, return_features=True)

    def reject_backbone(*args):
        raise AssertionError("Prefix extraction evaluated the backbone")

    hooks = [block.register_forward_pre_hook(reject_backbone) for block in model.backbone]
    try:
        prefix = extract_compressed_prefix(model, ref, alt, lengths, bf16=False)
    finally:
        for hook in hooks:
            hook.remove()
    for key, suffix in (("hidden", ""), ("coords", "_coords"), ("lengths", "_lengths")):
        expected = torch.cat([full["features_" + allele]["compressed" + suffix] for allele in ("ref", "alt")])
        torch.testing.assert_close(prefix[key], expected, rtol=0, atol=0)
    assert not prefix["hidden"].requires_grad
    assert not torch.is_inference(prefix["hidden"])
    assert prefix["original_lengths"].tolist() == [8, 11, 8, 11]
