"""Analytic routing, pooling, and paired-scoring invariants."""

from dataclasses import replace
import math

import pytest
import torch

from seqfidelity.model import ModelConfig, PairedSequenceModel, RouteState, length_mask, masked_mean


MODES = ("none", "fixed", "aggregate", "learned", "shared", "coupled", "no_backbone", "cnn")


def small_config(**kwargs):
    return ModelConfig(dim=16, heads=4, depth=1, encoder_depth=1,
                       decoder_depth=1, expansion=2, compression=3,
                       cnn_dim=8, cnn_depth=2, **kwargs)


def paired_tokens():
    ref = torch.arange(3 * 14).reshape(3, 14) % 4
    alt = ref.clone()
    alt[:, 3] = (alt[:, 3] + 1) % 4
    return ref, alt, torch.tensor([14, 11, 7])


@pytest.mark.parametrize("mode", MODES)
def test_pair_delta_is_antisymmetric_and_identity_is_zero(mode):
    torch.manual_seed(0)
    model = PairedSequenceModel(small_config(mode=mode)).eval()
    ref, alt, lengths = paired_tokens()
    with torch.no_grad():
        forward = model.forward_pair(ref, alt, lengths)
        reverse = model.forward_pair(alt, ref, lengths)
        identity = model.forward_pair(ref, ref, lengths)
    torch.testing.assert_close(forward["delta"], -reverse["delta"], atol=2e-7, rtol=1e-5)
    torch.testing.assert_close(identity["delta"], torch.zeros(3), atol=1e-7, rtol=0)
    torch.testing.assert_close(forward["delta"], forward["y_alt"] - forward["y_ref"])
    if mode == "shared":
        assert torch.equal(forward["route_ref"].indices, forward["route_alt"].indices)
        torch.testing.assert_close(forward["route_ref"].probs, forward["route_alt"].probs)


@pytest.mark.parametrize("mode", MODES)
def test_padding_values_and_extra_width_do_not_change_scores(mode):
    torch.manual_seed(0)
    model = PairedSequenceModel(small_config(mode=mode)).eval()
    ref, alt, lengths = paired_tokens()
    padded_ref = torch.nn.functional.pad(ref, (0, 5), value=3)
    padded_alt = torch.nn.functional.pad(alt, (0, 5), value=1)
    for row, length in enumerate(lengths.tolist()):
        padded_ref[row, length:] = 3
        padded_alt[row, length:] = 1
    with torch.no_grad():
        before = model.forward_pair(ref, alt, lengths)
        after = model.forward_pair(padded_ref, padded_alt, lengths)
    for key in ("y_ref", "y_alt", "delta"):
        torch.testing.assert_close(before[key], after[key], atol=4e-7, rtol=1e-5)


@pytest.mark.parametrize("mode", ("fixed", "aggregate", "learned", "shared", "coupled"))
def test_route_budget_masks_and_coordinate_maps_are_consistent(mode):
    torch.manual_seed(0)
    model = PairedSequenceModel(small_config(mode=mode))
    ref, _, lengths = paired_tokens()
    route = model.make_route(model.encode(ref, lengths), lengths)
    expected_budget = (lengths + 2) // 3
    assert torch.equal(route.mask.sum(1), expected_budget)
    assert torch.equal(route.selected_lengths, expected_budget)
    assert torch.equal(route.selected_mask.sum(1), expected_budget)
    assert torch.equal(route.indices[:, 0], torch.zeros(3, dtype=torch.long))
    assert not route.mask[~route.valid_mask].any()
    assert torch.equal(route.selected_probs, route.probs.gather(1, route.indices) * route.selected_mask)
    assert torch.equal(route.dechunk_map, (route.mask.long().cumsum(1) - 1).clamp_min(0))
    for row, length in enumerate(lengths.tolist()):
        selected = route.indices[row, route.selected_mask[row]]
        assert (selected < length).all()
        assert (selected[1:] > selected[:-1]).all()


def test_route_split_and_concatenation_preserve_every_field():
    torch.manual_seed(0)
    model = PairedSequenceModel(small_config())
    ref, _, lengths = paired_tokens()
    route = model.make_route(model.encode(ref, lengths), lengths)
    restored = RouteState.concatenate(route.split(1))
    detached = route.detached()
    for name in route.__dataclass_fields__:
        assert torch.equal(getattr(restored, name), getattr(route, name))
        assert torch.equal(getattr(detached, name), getattr(route, name))
        assert not getattr(detached, name).requires_grad


@pytest.mark.parametrize("mode", ("none", "fixed", "aggregate", "learned", "shared", "coupled"))
def test_route_exchange_additively_recovers_pair_delta(mode):
    torch.manual_seed(0)
    model = PairedSequenceModel(small_config(mode=mode)).eval()
    ref, alt, lengths = paired_tokens()
    with torch.no_grad():
        out = model.forward_pair(ref, alt, lengths)
        encoded_ref, encoded_alt = model.encode(ref, lengths), model.encode(alt, lengths)
        cross_alt = model.forward_with_route(encoded_alt, lengths, out["route_ref"])
        cross_ref = model.forward_with_route(encoded_ref, lengths, out["route_alt"])
        input_effect = ((cross_alt - out["y_ref"]) + (out["y_alt"] - cross_ref)) / 2
        route_effect = ((cross_ref - out["y_ref"]) + (out["y_alt"] - cross_alt)) / 2
    torch.testing.assert_close(input_effect + route_effect, out["delta"], atol=1e-7, rtol=1e-6)
    if mode in {"none", "fixed", "aggregate", "shared"}:
        torch.testing.assert_close(route_effect, torch.zeros(3), atol=1e-7, rtol=0)


def test_route_uses_recipient_features_and_continuous_probabilities():
    torch.manual_seed(0)
    model = PairedSequenceModel(small_config()).eval()
    ref, _, lengths = paired_tokens()
    encoded = model.encode(ref, lengths)
    route = model.make_route(encoded, lengths)
    changed_content = encoded.clone()
    changed_content[:, 0, 0] += 2
    before = model.forward_with_route(encoded, lengths, route)
    after = model.forward_with_route(changed_content, lengths, route)
    assert (after - before).abs().max().item() > 1e-5
    changed_probs = route.probs * 0.5
    changed_probs[:, 0] = 1
    changed_route = model.route_from_probs(changed_probs, lengths, route.indices)
    different = model.forward_with_route(encoded, lengths, changed_route)
    assert (different - before).abs().max().item() > 1e-6


def test_router_receives_finite_nonzero_gradients():
    torch.manual_seed(0)
    model = PairedSequenceModel(small_config())
    ref, alt, lengths = paired_tokens()
    output = model.forward_pair(ref, alt, lengths)
    (output["delta"] - torch.tensor([1., -1., 2.])).square().mean().backward()
    gradients = [parameter.grad for parameter in model.router.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def test_masked_pooling_ignores_padding_and_handles_empty_mask():
    values = torch.tensor([[[2., 4.], [4., 8.], [900., -900.]],
                           [[8., 16.], [500., 500.], [500., 500.]],
                           [[500., 500.], [500., 500.], [500., 500.]]])
    mask = length_mask(torch.tensor([2, 1, 0]), 3)
    expected = torch.tensor([[3., 6.], [8., 16.], [0., 0.]])
    torch.testing.assert_close(masked_mean(values, mask), expected, rtol=0, atol=0)


def test_interval_mean_uses_partial_interval_denominator_and_valid_gradients():
    torch.manual_seed(0)
    config = small_config(mode="aggregate")
    model = PairedSequenceModel(config)
    lengths = torch.tensor([9, 7, 4])
    encoded = torch.arange(3 * 9 * 16, dtype=torch.float32).reshape(3, 9, 16)
    valid = length_mask(lengths, 9)
    encoded[~valid] = 1e8
    encoded.requires_grad_()
    route = model.make_route(encoded, lengths)
    fixed_route = PairedSequenceModel(replace(config, mode="fixed")).make_route(encoded, lengths)
    for name in route.__dataclass_fields__:
        assert torch.equal(getattr(route, name), getattr(fixed_route, name))
    compressed = model.forward_with_route(encoded, lengths, route, return_features=True)["compressed"]
    for row, length in enumerate(lengths.tolist()):
        for slot, start in enumerate(range(0, length, 3)):
            torch.testing.assert_close(compressed[row, slot], encoded[row, start:min(start + 3, length)].mean(0), rtol=0, atol=0)
    compressed.sum().backward()
    for row, length in enumerate(lengths.tolist()):
        for position in range(length):
            denominator = min(3, length - (position // 3) * 3)
            torch.testing.assert_close(encoded.grad[row, position], torch.full((16,), 1 / denominator), rtol=0, atol=0)
    assert not encoded.grad[~valid].any()
    assert not compressed[~route.selected_mask].any()


def test_parallel_exponential_smoothing_matches_scalar_recurrence():
    torch.manual_seed(0)
    model = PairedSequenceModel(small_config())
    lengths = torch.tensor([8, 5])
    probabilities = torch.tensor([[1., .2, .3, .4, .5, .6, .7, .8],
                                  [1., .3, .4, .5, .6, 0., 0., 0.]])
    indices = torch.tensor([[0, 3, 6], [0, 3, 0]])
    route = model.route_from_probs(probabilities, lengths, indices)
    hidden = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(2, 3, 16) / 16
    coefficients = route.selected_probs.clamp(1e-4, 1 - 1e-4)
    coefficients[:, 0] = 1
    coefficients *= route.selected_mask
    current = torch.zeros(2, 16)
    states = []
    for slot in range(hidden.shape[1]):
        coefficient = coefficients[:, slot, None]
        current = coefficient * hidden[:, slot] + (1 - coefficient) * current
        states.append(current)
    expected = torch.stack(states, dim=1).gather(1, route.dechunk_map[..., None].expand(-1, -1, 16))
    expected *= route.valid_mask[..., None]
    torch.testing.assert_close(model._dechunk(hidden, route), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("external_skip", (True, False))
def test_rms_branch_fusion_matches_formula_and_preserves_gradients(external_skip):
    torch.manual_seed(0)
    model = PairedSequenceModel(small_config(mode="fixed", branch_fusion="rms_balanced", external_skip=external_skip))
    left = torch.arange(2 * 5 * 16, dtype=torch.float32).reshape(2, 5, 16) / 16
    right = torch.flip(left + 1, dims=[-1])
    normalize = lambda value: value / torch.sqrt(value.square().mean(-1, keepdim=True) + 1e-6)
    expected = (normalize(left) + normalize(right)) / math.sqrt(2) if external_skip else normalize(left)
    torch.testing.assert_close(model.fuse_branches(left, right), expected)
    ref, alt, lengths = paired_tokens()
    model.forward_pair(ref, alt, lengths)["y_ref"].square().sum().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.backbone.parameters())
    if not external_skip:
        assert all(parameter.grad is None for parameter in model.residual_proj.parameters())


@pytest.mark.parametrize("mode", ("none", "fixed"))
def test_balanced_initialization_keeps_shared_parameters_and_rng(mode):
    config = small_config(mode=mode)
    torch.manual_seed(0)
    summed = PairedSequenceModel(config)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(0)
    balanced = PairedSequenceModel(replace(config, branch_fusion="rms_balanced"))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in summed.state_dict().items():
        if not name.startswith("residual_proj."):
            assert torch.equal(value, balanced.state_dict()[name])
    assert torch.equal(balanced.residual_proj.weight, torch.eye(config.dim))
    assert not balanced.residual_proj.bias.any()
    torch.manual_seed(0)
    without_skip = PairedSequenceModel(replace(config, branch_fusion="rms_balanced", external_skip=False))
    assert all(torch.equal(value, without_skip.state_dict()[name]) for name, value in balanced.state_dict().items())


@pytest.mark.parametrize("kwargs", ({"mode": "unknown"}, {"dim": 15}, {"compression": 0},
                                    {"kernel_size": 2}, {"branch_fusion": "unknown"}))
def test_invalid_model_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_branch_ablation_feature_shapes_and_router_only_training():
    torch.manual_seed(0)
    model = PairedSequenceModel(small_config()).eval()
    ref, alt, lengths = paired_tokens()
    base = model.forward_pair(ref, alt, lengths, return_features=True)
    ablated = model.forward_pair(ref, alt, lengths, zero_residual=True)
    assert (base["y_ref"] - ablated["y_ref"]).abs().max().item() > 1e-5
    assert base["features_ref"]["backbone"].shape == (3, 5, 16)
    assert base["features_ref"]["full"].shape == (3, 14, 16)
    without_backbone = PairedSequenceModel(small_config(mode="no_backbone")).eval()
    disabled = without_backbone.forward_pair(ref, alt, lengths, zero_residual=True)
    torch.testing.assert_close(disabled["delta"], torch.zeros(3), atol=1e-7, rtol=0)
    model.freeze_except_router()
    assert all(parameter.requires_grad == name.startswith("router.")
               for name, parameter in model.named_parameters())


@pytest.mark.parametrize("mode,fusion", (("aggregate", "sum"), ("none", "rms_balanced"), ("fixed", "rms_balanced")))
@pytest.mark.parametrize("external_skip", (True, False))
def test_branch_policies_preserve_pair_order_and_padding(mode, fusion, external_skip):
    torch.manual_seed(0)
    model = PairedSequenceModel(small_config(mode=mode, branch_fusion=fusion, external_skip=external_skip)).eval()
    ref, alt, lengths = paired_tokens()
    padded_ref = torch.nn.functional.pad(ref, (0, 5), value=3)
    padded_alt = torch.nn.functional.pad(alt, (0, 5), value=1)
    for row, length in enumerate(lengths.tolist()):
        padded_ref[row, length:] = 3
        padded_alt[row, length:] = 1
    with torch.no_grad():
        forward = model.forward_pair(ref, alt, lengths)
        reverse = model.forward_pair(alt, ref, lengths)
        padded = model.forward_pair(padded_ref, padded_alt, lengths)
    torch.testing.assert_close(forward["delta"], -reverse["delta"], rtol=1e-5, atol=2e-7)
    for key in ("y_ref", "y_alt", "delta"):
        torch.testing.assert_close(forward[key], padded[key], rtol=1e-5, atol=4e-7)
