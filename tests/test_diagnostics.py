"""Route intervention controls on independent token tensors."""

import torch

from seqfidelity.diagnostics import _matched_random_route, diagnose_batch, extract_pair_features
from seqfidelity.model import ModelConfig, PairedSequenceModel


def model_and_pairs(mode="learned"):
    torch.manual_seed(0)
    model = PairedSequenceModel(ModelConfig(mode=mode, dim=8, heads=2, depth=1,
                                 encoder_depth=1, decoder_depth=1, expansion=2, compression=3))
    ref = torch.arange(2 * 12).reshape(2, 12) % 4
    alt = ref.clone()
    alt[:, 4] = (alt[:, 4] + 1) % 4
    return model, ref, alt, torch.tensor([12, 10])


def test_diagnostics_preserve_additive_delta_and_model_training_state():
    model, ref, alt, lengths = model_and_pairs()
    model.train()
    result = diagnose_batch(model, ref, alt, lengths, seed=0, n_random=1)
    assert model.training
    torch.testing.assert_close(result["delta_pred"], result["phi_input"] + result["phi_route"], atol=1e-7, rtol=1e-5)
    assert result["same_k"].all()
    assert result["random_0_matched"].shape == (2,)
    assert all(not value.requires_grad for value in result.values())
    features = extract_pair_features(model, ref, alt, lengths, length_scale=1.)
    assert all(value.shape == (2, 26) for value in features.values())
    assert all(not value.requires_grad for value in features.values())


def test_random_control_preserves_probability_histogram_and_boundary_count():
    model, _, _, _ = model_and_pairs()
    lengths = torch.tensor([12])
    probabilities = torch.linspace(.1, .3, 12)[None]
    probabilities[:, 0] = 1.
    reference = model.route_from_probs(probabilities, lengths, indices=torch.tensor([[0, 4, 6, 9]]))
    alternate = model.route_from_probs(probabilities, lengths, indices=torch.tensor([[0, 3, 6, 9]]))
    route, matched, active = _matched_random_route(model, reference, alternate, seed=0)
    assert matched.item() and active.item()
    assert torch.equal(torch.sort(route.probs).values, torch.sort(reference.probs).values)
    assert route.selected_lengths.item() == 4
    assert (route.mask != alternate.mask).sum().item() == 2
    assert not (torch.equal(route.indices, reference.indices)
                and torch.allclose(route.probs, reference.probs, atol=1e-8, rtol=0))
    repeated, repeated_match, repeated_active = _matched_random_route(model, reference, alternate, seed=0)
    assert torch.equal(repeated.indices, route.indices)
    assert torch.equal(repeated.probs, route.probs)
    assert torch.equal(repeated_match, matched)
    assert torch.equal(repeated_active, active)


def test_equal_routes_have_zero_route_contribution_with_identical_batch_shapes():
    model, ref, alt, lengths = model_and_pairs(mode="fixed")
    shapes = []
    original_encode = model.encode

    def record_encode(tokens, input_lengths):
        shapes.append(tuple(tokens.shape))
        return original_encode(tokens, input_lengths)

    model.encode = record_encode
    result = diagnose_batch(model, ref, alt, lengths, seed=0, n_random=1, zero_residual=True)
    assert shapes == [(4, 12)] * 4
    assert torch.equal(result["phi_route"], torch.zeros(2))
