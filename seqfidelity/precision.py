"""Compare BF16/autocast execution with IEEE FP32 using unchanged weights."""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
from .metrics import evaluate_predictions, paired_comparison
from .encoding import device_batch, loader


@contextmanager
def execution_precision(device_type: str, bf16: bool):
    """Temporarily select a documented policy and restore it even after failure."""
    prior_matmul = torch.get_float32_matmul_precision()
    prior_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision('high' if bf16 else 'highest')
        torch.backends.cuda.matmul.allow_tf32 = bool(bf16)
        torch.backends.cudnn.allow_tf32 = bool(bf16)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=bf16):
            yield
    finally:
        torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = prior_cuda_tf32
        torch.set_float32_matmul_precision(prior_matmul)



def route_differences(out16, out32, lengths):
    """Compare two execution policies at identical unpadded input coordinates."""
    columns = {}
    total_changed = torch.zeros_like(lengths)
    total_probability_l1 = torch.zeros_like(lengths, dtype=torch.float32)
    for allele in ('ref', 'alt'):
        first, second = out16[f'route_{allele}'], out32[f'route_{allele}']
        valid = torch.arange(first.mask.shape[1], device=lengths.device)[None] < lengths[:, None]
        changed = ((first.mask != second.mask) & valid).sum(1)
        probability_l1 = ((first.probs.float() - second.probs.float()).abs() * valid).sum(1)
        columns[f'hardmask_changed_fraction_{allele}'] = changed.float() / lengths
        columns[f'route_probability_l1_{allele}'] = probability_l1
        columns[f'k16_{allele}'] = first.selected_lengths
        columns[f'k32_{allele}'] = second.selected_lengths
        total_changed += changed
        total_probability_l1 += probability_l1
    columns['hardmask_changed_count'] = total_changed
    columns['hardmask_changed_fraction'] = total_changed.float() / (2 * lengths)
    columns['hardmask_any_changed'] = total_changed > 0
    columns['route_probability_l1'] = total_probability_l1
    columns['route_probability_mean_abs'] = total_probability_l1 / (2 * lengths)
    return columns



@torch.inference_mode()
def predict_precision(model, frame, device='cpu', *, batch_size):
    """Return every input pair in original order, using one unchanged model."""
    if frame.empty:
        raise ValueError('Precision sensitivity requires at least one pair')
    if not np.isfinite(frame.delta.to_numpy(float)).all():
        raise ValueError('Precision sensitivity requires finite truth for every supplied pair; rows are never dropped')
    model.eval()
    target_device = torch.device(device)
    pieces = []
    for step, raw in enumerate(loader(frame, batch_size, shuffle=False)):
        batch = device_batch(raw, target_device)
        with execution_precision(target_device.type, bf16=True):
            out16 = model.forward_pair(batch['ref'], batch['alt'], batch['lengths'])
        with execution_precision(target_device.type, bf16=False):
            out32 = model.forward_pair(batch['ref'], batch['alt'], batch['lengths'])
        values = {'pred16': out16['delta'].float(), 'pred32': out32['delta'].float(),
                  'y_ref_pred16': out16['y_ref'].float(), 'y_alt_pred16': out16['y_alt'].float(),
                  'y_ref_pred32': out32['y_ref'].float(), 'y_alt_pred32': out32['y_alt'].float(),
                  **route_differences(out16, out32, batch['lengths'])}
        numeric = torch.stack([value.float() for value in values.values()])
        if not torch.isfinite(numeric).all():
            raise FloatingPointError('A precision prediction or routing comparison is nonfinite; evaluation cannot omit it')
        part = frame.iloc[raw['index'].numpy()].drop(columns=['ref_seq', 'alt_seq'], errors='ignore').reset_index(drop=True)
        part['delta_true'] = part.delta
        for name, tensor in values.items():
            part[name] = tensor.cpu().numpy()
        part['delta_shift'] = part.pred32 - part.pred16
        part['absolute_delta_shift'] = part.delta_shift.abs()
        part['squared_error_change'] = (part.pred32 - part.delta_true)**2 - (part.pred16 - part.delta_true)**2
        pieces.append(part)
        if (step + 1) % 100 == 0:
            print(f'precision pairs: {min((step + 1) * batch_size, len(frame))}/{len(frame)}', flush=True)
    result = pd.concat(pieces, ignore_index=True)
    if len(result) != len(frame) or result.pair_id.tolist() != frame.pair_id.tolist():
        raise RuntimeError('Precision output must contain every pair exactly once in the input order')
    return result



def summarize_precision(frame, thresholds, n_bootstrap=1000, seed=0):
    """Use one locked endpoint definition and paired locus resampling."""
    truth_and_predictions = frame[['delta_true', 'pred16', 'pred32']].to_numpy(float)
    if not np.isfinite(truth_and_predictions).all() or frame.locus_id.isna().any():
        raise ValueError('All pairs need finite predictions/truth and a locus; no finite-row filtering is allowed')
    metrics = {}
    for name, column in [('bf16', 'pred16'), ('fp32_ieee', 'pred32')]:
        comparison = frame.assign(delta_pred=frame[column])
        metrics[name] = evaluate_predictions(comparison, thresholds, n_bootstrap=n_bootstrap, seed=seed)
        if metrics[name]['excluded_nonfinite_rows'] != 0:
            raise RuntimeError('A precision endpoint unexpectedly dropped rows')
    paired = paired_comparison(frame, candidate='pred32', reference='pred16', n_bootstrap=n_bootstrap, seed=seed)
    if paired['n'] != len(frame):
        raise RuntimeError('Paired precision comparison unexpectedly dropped rows')
    change = frame.delta_shift.to_numpy(float)
    return {'n_pairs': len(frame), 'n_loci': frame.locus_id.nunique(),
            'metrics': metrics, 'paired_fp32_minus_bf16': paired,
            'delta_shift': {'mean': float(change.mean()), 'rms': float(np.sqrt(np.mean(change**2))),
                            'median_absolute': float(np.median(np.abs(change))),
                            'p95_absolute': float(np.quantile(np.abs(change), .95)),
                            'p99_absolute': float(np.quantile(np.abs(change), .99)),
                            'maximum_absolute': float(np.max(np.abs(change)))},
            'routing': {'pairs_with_changed_hardmask': int(frame.hardmask_any_changed.sum()),
                        'pairs_with_changed_hardmask_fraction': float(frame.hardmask_any_changed.mean()),
                        'mean_hardmask_changed_fraction': float(frame.hardmask_changed_fraction.mean()),
                        'mean_probability_l1': float(frame.route_probability_l1.mean()),
                        'mean_probability_absolute_difference_per_position': float(frame.route_probability_mean_abs.mean())}}

