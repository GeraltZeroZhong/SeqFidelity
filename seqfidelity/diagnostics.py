"""Complete-state route interventions, branch probes, and measured pair costs."""
from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from .metrics import evaluate_predictions, paired_comparison


def _route_changes(left, right) -> dict[str, torch.Tensor]:
    valid = left.valid_mask & right.valid_mask
    length = valid.sum(1).clamp_min(1)
    same_k = left.selected_lengths == right.selected_lengths
    hamming = ((left.mask != right.mask) & valid).sum(1)
    probability_l1 = ((left.probs - right.probs).abs() * valid).sum(1) / length
    # Sorted boundaries define an absolute-coordinate displacement for equal K.
    movement = torch.full_like(probability_l1, float("nan"))
    for i in torch.nonzero(same_k, as_tuple=False).flatten().tolist():
        k = int(left.selected_lengths[i])
        movement[i] = (left.indices[i, :k] - right.indices[i, :k]).float().abs().mean()
    return {"k_ref": left.selected_lengths, "k_alt": right.selected_lengths,
            "same_k": same_k, "boundary_hamming": hamming,
            "boundary_change_fraction": hamming / length,
            "probability_l1": probability_l1, "boundary_move_bp": movement}


def _matched_random_route(model, ref_route, alt_route, seed: int):
    """Label-free random intervention with prespecified matching tolerances.

    Retains K, the number of removed/added boundaries, and the donor probability
    histogram exactly. Random moves match natural mean sorted-boundary movement
    within max(1 bp, 20%). Probability L1 change must be within max(.01, 2x)
    the natural change. Active controls must differ from the reference route;
    donor probability ranks use seeded perturbations with scale equal to the
    natural probability L1 change. Failed matching is explicitly flagged.
    """
    rng = np.random.default_rng(seed)
    ref_prob = ref_route.probs.detach().cpu().numpy()
    alt_prob = alt_route.probs.detach().cpu().numpy()
    ref_indices = ref_route.indices.detach().cpu().numpy()
    alt_indices = alt_route.indices.detach().cpu().numpy()
    lengths = ref_route.lengths.detach().cpu().numpy()
    kr = ref_route.selected_lengths.detach().cpu().numpy()
    ka = alt_route.selected_lengths.detach().cpu().numpy()
    random_indices = alt_indices.copy()
    random_prob = alt_prob.copy()
    matched = np.zeros(len(lengths), dtype=bool)
    active = np.zeros(len(lengths), dtype=bool)
    for i, length in enumerate(lengths):
        length, k = int(length), int(ka[i])
        if int(kr[i]) != k:
            continue
        a = alt_indices[i, :k]
        r = ref_indices[i, :k]
        removed = np.setdiff1d(a, r)
        added = np.setdiff1d(r, a)
        target_move = float(np.mean(np.abs(a - r)))
        natural_l1 = float(np.abs(ref_prob[i, :length] - alt_prob[i, :length]).mean())
        active[i] = bool(len(removed) or natural_l1 > 1e-8)
        offsets = added - removed
        for attempt in range(64):
            candidate = set(map(int, a))
            if len(removed):
                changed_from: set[int] = set()
                changed_to: set[int] = set()
                for shift in rng.permutation(offsets):
                    origins = np.array([v for v in candidate if v != 0 and v not in changed_to])
                    rng.shuffle(origins)
                    for origin in origins:
                        destination = int(origin + shift)
                        if 0 < destination < length and destination not in candidate and destination not in changed_from:
                            candidate.remove(int(origin))
                            candidate.add(destination)
                            changed_from.add(int(origin))
                            changed_to.add(destination)
                            break
                    else:
                        break
                if len(changed_from) != len(removed):
                    continue
            c = np.array(sorted(candidate), dtype=int)
            if len(np.setdiff1d(a, c)) != len(removed):
                continue
            movement = float(np.mean(np.abs(a - c)))
            if abs(movement - target_move) > max(1., .2 * target_move):
                continue
            # Reassign selected/unselected donor probabilities by recipient rank.
            # This preserves the entire probability distribution and its relation
            # to the selected boundary class. Position 0 is always anchored.
            p = np.empty(length, dtype=ref_prob.dtype)
            p[0] = ref_prob[i, 0]
            universe = np.arange(1, length)
            for target, donor in [(c[c != 0], r[r != 0]),
                                  (np.setdiff1d(universe, c), np.setdiff1d(universe, r))]:
                ranking_scores = alt_prob[i, target] + rng.normal(0., natural_l1, size=len(target))
                ranking = target[np.argsort(ranking_scores, kind="stable")]
                values = np.sort(ref_prob[i, donor], kind="stable")
                p[ranking] = values
            probability_l1 = float(np.abs(p - alt_prob[i, :length]).mean())
            if probability_l1 > max(.01, 2. * natural_l1):
                continue
            if active[i] and np.array_equal(c, r) and np.allclose(p, ref_prob[i, :length], atol=1e-8, rtol=0):
                continue
            random_indices[i, :k] = c
            random_prob[i, :length] = p
            matched[i] = True
            break
    route = model.route_from_probs(
        torch.as_tensor(random_prob, device=alt_route.probs.device, dtype=alt_route.probs.dtype),
        alt_route.lengths,
        indices=torch.as_tensor(random_indices, device=alt_route.indices.device, dtype=alt_route.indices.dtype),
    )
    return route, torch.as_tensor(matched, device=alt_route.probs.device), torch.as_tensor(active, device=alt_route.probs.device)


@torch.no_grad()
def four_forward(model, ref: torch.Tensor, alt: torch.Tensor, lengths: torch.Tensor):
    """Four actual sequence forwards, including recipient re-encoding for swaps."""
    natural = model.forward_pair(ref, alt, lengths, return_features=False)
    rr, aa = natural["y_ref"], natural["y_alt"]
    ref_route, alt_route = natural["route_ref"], natural["route_alt"]
    batch = len(ref)
    both_lengths = torch.cat((lengths, lengths))
    cross_encoded = model.encode(torch.cat((ref, alt)), both_lengths)
    cross_route = type(ref_route).concatenate((alt_route, ref_route))
    cross_predictions = model.forward_with_route(cross_encoded, both_lengths, cross_route)
    ra, ar = cross_predictions.split(batch)
    input_effect = .5 * ((ar - rr) + (aa - ra))
    route_effect = .5 * ((ra - rr) + (aa - ar))
    values = {"f_rr": rr, "f_aa": aa, "f_ar": ar, "f_ra": ra,
              "delta_pred": aa - rr, "phi_input": input_effect, "phi_route": route_effect,
              "decomposition_residual": aa - rr - input_effect - route_effect,
              "delta_reference_route": ar - rr, "delta_alternate_route": aa - ra}
    values.update(_route_changes(ref_route, alt_route))
    return values, natural


@torch.no_grad()
def diagnose_batch(model, ref: torch.Tensor, alt: torch.Tensor, lengths: torch.Tensor,
                   seed: int = 0, n_random: int = 1, zero_residual: bool = True) -> dict[str, torch.Tensor]:
    """Return per-pair interventions; no labels enter route selection or matching."""
    previous_training = model.training
    model.eval()
    try:
        values, natural = four_forward(model, ref, alt, lengths)
        rr, aa = values["f_rr"], values["f_aa"]
        ref_route, alt_route = natural["route_ref"], natural["route_alt"]
        # Keep every receiving-content batch at 2B, as in natural paired scoring.
        # Different GPU matrix shapes can change reduced-precision hard routes.
        # Every extra intervention pays for its own full paired re-encoding.
        both_lengths = torch.cat((lengths, lengths))
        both_tokens = torch.cat((ref, alt))
        for index in range(n_random):
            route, matched, active = _matched_random_route(model, ref_route, alt_route, seed + index)
            encoded = model.encode(both_tokens, both_lengths)
            intervention_route = type(ref_route).concatenate((ref_route, route))
            prediction = model.forward_with_route(encoded, both_lengths, intervention_route)
            random_ref, random_alt = prediction.split(len(ref))
            prefix = f"random_{index}"
            values[f"delta_{prefix}"] = random_alt - random_ref
            values[f"{prefix}_matched"] = matched
            values[f"{prefix}_active"] = active
            changes = _route_changes(alt_route, route)
            for name in ["boundary_hamming", "boundary_move_bp", "probability_l1"]:
                values[f"{prefix}_{name}"] = changes[name]
        if zero_residual:
            encoded = model.encode(both_tokens, both_lengths)
            natural_route = type(ref_route).concatenate((ref_route, alt_route))
            no_residual = model.forward_with_route(encoded, both_lengths, natural_route, zero_residual=True)
            ref_no_residual, alt_no_residual = no_residual.split(len(ref))
            values["delta_zero_residual"] = alt_no_residual - ref_no_residual
        return {name: value.detach() for name, value in values.items()}
    finally:
        model.train(previous_training)


def diagnostics_frame(values: dict[str, torch.Tensor], metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({key: val.detach().cpu().numpy() for key, val in values.items()})
    if metadata is not None:
        for key in metadata:
            if key not in frame:
                frame[key] = metadata[key].to_numpy()
    return frame


def summarize_diagnostics(frame: pd.DataFrame, thresholds, n_bootstrap: int = 1000,
                          seed: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n": len(frame), "core_sequence_forwards_per_pair": 4,
        "total_sequence_forwards_per_pair": 4 + 2 * sum(name.startswith("delta_random_") for name in frame) + 2 * int("delta_zero_residual" in frame),
        "inference_batch_shape": "All natural, swapped, random and zero-residual forwards use 2B concatenated ref/alt batches.",
        "n_same_k": int(frame.same_k.sum()),
        "n_different_k": int((~frame.same_k.astype(bool)).sum()),
        "max_decomposition_residual": float(frame.decomposition_residual.abs().max()),
        "mean_abs_phi_input": float(frame.phi_input.abs().mean()),
        "mean_abs_phi_route": float(frame.phi_route.abs().mean()),
        "mean_boundary_change_fraction": float(frame.boundary_change_fraction.mean()),
        "mean_probability_l1": float(frame.probability_l1.mean()),
        "interpretation": "phi_route is a computational intervention contribution, not a biological causal effect. Test-label error improvements, not algebraically coupled correlations, assess interventions.",
        "interventions": {}, "strata": {},
    }
    candidates = ["delta_reference_route", "delta_alternate_route", "delta_zero_residual"]
    candidates += [name for name in frame if name.startswith("delta_random_")]
    for name in candidates:
        if name in frame:
            result["interventions"][name] = paired_comparison(frame, name, n_bootstrap=n_bootstrap, seed=seed)
    for same, label in [(True, "same_k"), (False, "different_k")]:
        subset = frame.loc[frame.same_k == same]
        result["strata"][label] = evaluate_predictions(subset, thresholds, n_bootstrap, seed) if len(subset) else {"n": 0, "status": "no_rows"}
    for name in frame:
        if name.startswith("random_") and name.endswith("_matched"):
            prefix = name.removesuffix("_matched")
            subset = frame.loc[frame[name].astype(bool)]
            active = frame.get(f"{prefix}_active", pd.Series(True, index=frame.index)).astype(bool)
            result[f"{prefix}_matching"] = {"matched_n": len(subset), "total_n": len(frame),
                                            "matched_fraction": float(frame[name].mean()),
                                            "active_n": int(active.sum()),
                                            "active_matched_n": int((active & frame[name].astype(bool)).sum()),
                                            "failed_control_behavior": "unchanged alt route; retained in overall results"}
            if len(subset):
                result[f"reference_vs_{prefix}_matched"] = paired_comparison(
                    subset, "delta_reference_route", f"delta_{prefix}", n_bootstrap=n_bootstrap, seed=seed)
    return result


def _pool(hidden: torch.Tensor, positions: torch.Tensor, lengths: torch.Tensor,
          sequence_lengths: torch.Tensor, length_scale: float = 1.0) -> torch.Tensor:
    if length_scale <= 0:
        raise ValueError('length_scale must be positive')
    valid = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]
    weights = valid.to(hidden.dtype)
    denominator = lengths.clamp_min(1).to(hidden.dtype)[:, None]
    mean = (hidden * weights[..., None]).sum(1) / denominator
    maximum = hidden.masked_fill(~valid[..., None], -torch.inf).amax(1)
    coordinates = positions.to(hidden.dtype) / (sequence_lengths - 1).clamp_min(1)[:, None]
    coordinate_moment = (hidden * coordinates[..., None] * weights[..., None]).sum(1) / denominator
    scalars = torch.stack([lengths / sequence_lengths, sequence_lengths / length_scale], dim=1).to(hidden.dtype)
    return torch.cat([mean, maximum, coordinate_moment, scalars], dim=1)


def pooled_branch_features(features: dict[str, torch.Tensor], length_scale: float = 1.0) -> dict[str, torch.Tensor]:
    lengths = features["lengths"]
    coordinates = torch.arange(features["encoder"].shape[1], device=lengths.device)[None, :].expand(len(lengths), -1)
    return {
        "full": _pool(features["full"], coordinates, lengths, lengths, length_scale),
        "encoder": _pool(features["encoder"], coordinates, lengths, lengths, length_scale),
        "backbone": _pool(features["backbone"], features["backbone_coords"], features["backbone_lengths"], lengths, length_scale),
    }


@torch.no_grad()
def extract_pair_features(model, ref, alt, lengths, length_scale=1.0) -> dict[str, torch.Tensor]:
    previous_training = model.training
    model.eval()
    try:
        pair = model.forward_pair(ref, alt, lengths, return_features=True)
        left = pooled_branch_features(pair["features_ref"], length_scale)
        right = pooled_branch_features(pair["features_alt"], length_scale)
        return {name: right[name] - left[name] for name in left}
    finally:
        model.train(previous_training)


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_pair(model, ref: torch.Tensor, alt: torch.Tensor, lengths: torch.Tensor,
                   warmup: int = 5, repeats: int = 20, profile_flops: bool = False) -> dict[str, Any]:
    """Measure complete paired scoring and four-sequence diagnostic forward cost."""
    if warmup < 1 or repeats < 1:
        raise ValueError('warmup and repeats must be positive')
    device = ref.device
    previous_training = model.training
    model.eval()
    tasks = {"pair_score": lambda: model.forward_pair(ref, alt, lengths),
             "four_forward": lambda: four_forward(model, ref, alt, lengths)}
    if not hasattr(model, "forward_with_route"):
        tasks.pop("four_forward")
    result: dict[str, Any] = {
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "batch_pairs": len(ref), "padded_sequence_length": ref.shape[1],
        "actual_sequence_lengths": [int(lengths.min()), int(lengths.max())],
        "warmup": warmup, "repeats": repeats, "includes": "both inputs, encoding, routing, trunk, decoding, residual and readout; existing device tensors, no file I/O",
        "parameter_dtype": str(next(model.parameters()).dtype),
        "autocast_enabled": torch.is_autocast_enabled(device.type),
        "autocast_dtype": str(torch.get_autocast_dtype(device.type)),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "measurements": {},
    }
    try:
        for name, task in tasks.items():
            for _ in range(warmup):
                output = task()
            del output
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            elapsed = []
            for _ in range(repeats):
                _sync(device)
                start = perf_counter()
                output = task()
                _sync(device)
                elapsed.append(perf_counter() - start)
                del output
            total = float(np.sum(elapsed))
            result["measurements"][name] = {
                "sequence_forwards_per_pair": 2 if name == "pair_score" else 4,
                "latency_batch_ms_median": float(np.median(elapsed) * 1000),
                "latency_batch_ms_p95": float(np.quantile(elapsed, .95) * 1000),
                "latency_pair_ms_mean": total / (repeats * len(ref)) * 1000,
                "throughput_pairs_per_second": repeats * len(ref) / total,
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            }
            if profile_flops:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities, with_flops=True) as profile:
                    task()
                    _sync(device)
                reported = sum(event.flops for event in profile.key_averages())
                result["measurements"][name]["profiler_counted_flops_per_pair"] = reported / len(ref)
                result["measurements"][name]["flops_scope"] = "Profiler-supported operators only; not a complete FLOP count or a speed claim."
        return result
    finally:
        model.train(previous_training)
