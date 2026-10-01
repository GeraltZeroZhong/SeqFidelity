"""Input-gradient times one-hot attribution for a fixed sequence model.

Attribution differentiates a continuous one-hot relaxation along the selected
routing path. Scores describe computational sensitivity to that relaxation.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F
from .precision import execution_precision

SUPPORTED_ROUTING_MODES = {'none', 'fixed', 'learned', 'no_backbone', 'cnn'}


def input_gradient_times_onehot(model, tokens, lengths, *, bf16):
    """Return FP32 A,C,G,T input gradients and positional signed scores.

    The routed model's embedding hook keeps lookup values exactly unchanged. Chain-rule
    multiplication by the fixed embedding weights recovers d(output)/d(one-hot).
    LegNet's hook differentiates its original one-hot stem input directly.
    Model parameters and their .grad fields are left untouched.
    """
    if len(torch.unique(lengths))!=1 or int(lengths[0])!=tokens.shape[1]:
        raise ValueError('Attribution batches use one exact actual construct length')
    if bool((tokens>3).any()):
        raise ValueError('Attribution requires canonical A,C,G,T sequences')
    architecture = model.config.architecture
    if architecture=='routed' and model.config.mode not in SUPPORTED_ROUTING_MODES:
        raise ValueError('Pair-aware routes have no defined single-sequence attribution for this attribution method')
    if architecture not in {'routed', 'legnet'}:
        raise ValueError('Architecture has no frozen input-gradient mapping')
    previous_training = model.training
    flags = [p.requires_grad for p in model.parameters()]
    captured = []
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    def embedding_hook(module, inputs, output):
        leaf = output.detach().requires_grad_(True)
        captured.append(leaf)
        return leaf
    def stem_hook(module, inputs):
        leaf = inputs[0].detach().requires_grad_(True)
        captured.append(leaf)
        return (leaf,)
    hook = (model.embedding.register_forward_hook(embedding_hook) if architecture=='routed'
            else model.net.stem.register_forward_pre_hook(stem_hook))
    try:
        with torch.enable_grad(), execution_precision(tokens.device.type, bf16=bf16):
            prediction = model(tokens, lengths).float()
            if len(captured)!=1:
                raise RuntimeError('Actual single-sequence forward must visit the input hook exactly once')
            gradient = torch.autograd.grad(prediction.sum(), captured[0], create_graph=False)[0].float()
        if architecture=='routed':
            weights = model.embedding.weight.detach()[:4].float()
            # Elementwise multiplication/reduction avoids a TF32/BF16 matmul in
            # this FP32 chain-rule conversion; positional encodings stay constant.
            input_gradient = (gradient[:, :, None, :]*weights[None, None, :, :]).sum(-1)
        else:
            # Author channels A,G,C,T -> shared artifact convention A,C,G,T.
            input_gradient = gradient.transpose(1, 2)[..., [0, 2, 1, 3]]
        onehot = F.one_hot(tokens, num_classes=4).float()
        score = (input_gradient.float()*onehot).sum(-1)
        if not torch.isfinite(score).all() or not torch.isfinite(input_gradient).all() or not torch.isfinite(prediction).all():
            raise FloatingPointError('Attribution requires finite scalar outputs and every position gradient')
        return {'prediction': prediction.detach().cpu().numpy().astype(np.float32),
                'gradient': input_gradient.detach().cpu().numpy().astype(np.float32),
                'score': score.detach().cpu().numpy().astype(np.float32)}
    finally:
        hook.remove()
        for p, flag in zip(model.parameters(), flags): p.requires_grad_(flag)
        model.train(previous_training)



def position_summary(score, fraction=.1):
    score = np.asarray(score, dtype=np.float32)
    absolute = np.abs(score)
    maximum = float(absolute.max())
    budget = int(np.ceil(fraction*len(score)))
    if maximum==0:
        return {'peak_available': False, 'primary_peak': None, 'peak_positions': [], 'peak_tie_count': 0,
                'maximum_absolute_score': 0., 'top_budget': budget, 'top_positions': [],
                'top_cutoff_tie_count': None, 'top_cutoff_ties_selected': None}
    peaks = np.flatnonzero(absolute==maximum)
    order = np.argsort(-absolute, kind='stable')
    selected = order[:budget]
    cutoff = absolute[selected[-1]]
    return {'peak_available': True, 'primary_peak': int(peaks[0]), 'peak_positions': peaks.tolist(),
            'peak_tie_count': len(peaks), 'maximum_absolute_score': maximum, 'top_budget': budget,
            'top_positions': sorted(selected.tolist()), 'top_cutoff_tie_count': int((absolute==cutoff).sum()),
            'top_cutoff_ties_selected': int((absolute[selected]==cutoff).sum())}



def compare_attributions(score16, score32):
    a, b = np.asarray(score16, dtype=np.float32), np.asarray(score32, dtype=np.float32)
    if a.shape!=b.shape or a.ndim!=1 or not np.isfinite([a,b]).all():
        raise ValueError('Attribution vectors must share the same finite native coordinates')
    sa, sb = position_summary(a), position_summary(b)
    comparable = sa['peak_available'] and sb['peak_available']
    pa, pb = set(sa['peak_positions']), set(sb['peak_positions'])
    ta, tb = set(sa['top_positions']), set(sb['top_positions'])
    da, db = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.linalg.norm(da), np.linalg.norm(db)
    result = {'length': len(a), **{key+'_bf16': val for key,val in sa.items()},
              **{key+'_fp32': val for key,val in sb.items()}, 'peak_comparable': comparable,
              'primary_peak_same': sa['primary_peak']==sb['primary_peak'] if comparable else None,
              'primary_peak_distance': abs(sa['primary_peak']-sb['primary_peak']) if comparable else None,
              'peak_set_intersection': sorted(pa & pb) if comparable else None,
              'peak_set_intersection_count': len(pa & pb) if comparable else None,
              'peak_sets_intersect': bool(pa & pb) if comparable else None,
              'peak_set_jaccard': len(pa & pb)/len(pa | pb) if comparable else None,
              'peak_set_minimum_distance': min(abs(x-y) for x in pa for y in pb) if comparable else None,
              'top10_intersection_count': len(ta & tb) if comparable else None,
              'top10_overlap_fraction': len(ta & tb)/sa['top_budget'] if comparable else None,
              'top10_set_jaccard': len(ta & tb)/len(ta | tb) if comparable else None,
              'signed_vector_cosine': float(np.dot(da,db)/(na*nb)) if na and nb else None,
              'signed_vector_pearson': float(np.corrcoef(da,db)[0,1]) if np.ptp(da)>0 and np.ptp(db)>0 else None,
              'absolute_score_rank_spearman': float(spearmanr(np.abs(da),np.abs(db)).statistic)
                    if np.ptp(np.abs(da))>0 and np.ptp(np.abs(db))>0 else None,
              'score_rms_bf16': float(np.sqrt(np.mean(da**2))), 'score_rms_fp32': float(np.sqrt(np.mean(db**2))),
              'score_difference_rms': float(np.sqrt(np.mean((da-db)**2))),
              'relative_vector_l2_to_fp32': float(np.linalg.norm(da-db)/nb) if nb else None}
    return result

