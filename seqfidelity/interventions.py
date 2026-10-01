"""Gradient, branch-path, routing and prediction-decision diagnostics."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from .comparisons import align_predictions
from .metrics import _reliability_mask
from .training import loss_for


def _number(value):
    value = float(value)
    return value if np.isfinite(value) else None



def _correlation(x, y, rank=False):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return _number(spearmanr(x, y).statistic if rank else np.corrcoef(x, y)[0, 1])



def gradient_snapshot(model, batch, cfg, delta_scale, activity_scale, stage):
    """Observe both standardized gradients without altering RNG or .grad.

    Caller-selected snapshots preserve the random state and existing parameter
    gradients while measuring each supervision term.
    """
    device = batch['ref'].device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == 'cuda'):
            out = model.forward_pair(batch['ref'], batch['alt'], batch['lengths'], return_features=True)
            _, dloss, aloss = loss_for(out, batch, cfg, delta_scale, activity_scale)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        params = [p for _, p in named]
        dg = torch.autograd.grad(dloss, params, retain_graph=True, allow_unused=True)
        ag = torch.autograd.grad(aloss, params, allow_unused=True)
        grouped = defaultdict(list)
        for (name, p), d, a in zip(named, dg, ag):
            grouped[name.split('.')[0]].append((p, d, a))
        grouped['all'] = [(p, d, a) for (_, p), d, a in zip(named, dg, ag)]
        wd, wa = (0., 1.) if cfg.objective == 'activity' else (1., cfg.absolute_weight)
        blocks = {}
        for key, values in grouped.items():
            d2 = sum(float((d.detach().float()**2).sum()) for _, d, _ in values if d is not None)
            a2 = sum(float((a.detach().float()**2).sum()) for _, _, a in values if a is not None)
            dot = sum(float((d.detach().float()*a.detach().float()).sum()) for _, d, a in values if d is not None and a is not None)
            dnorm, anorm = np.sqrt(d2), np.sqrt(a2)
            joint2 = d2 + cfg.absolute_weight**2*a2 + 2*cfg.absolute_weight*dot
            active2 = wd**2*d2 + wa**2*a2 + 2*wd*wa*dot
            blocks[key] = {'delta_gradient_norm': dnorm, 'activity_gradient_norm': anorm,
                'cosine': dot/(dnorm*anorm) if dnorm and anorm else None,
                'delta_gradient_missing_parameters': sum(p.numel() for p, d, _ in values if d is None),
                'activity_gradient_missing_parameters': sum(p.numel() for p, _, a in values if a is None),
                'active_delta_weighted_norm': wd*dnorm, 'active_activity_weighted_norm': wa*anorm,
                'active_total_gradient_norm': np.sqrt(max(0., active2)),
                'joint_reference_activity_weight': cfg.absolute_weight,
                'joint_reference_activity_weighted_norm': cfg.absolute_weight*anorm,
                'joint_reference_activity_to_delta_norm': cfg.absolute_weight*anorm/dnorm if dnorm else None,
                'joint_reference_total_norm': np.sqrt(max(0., joint2))}
        branch_norms = {}
        if 'features_ref' in out:
            for stage_name in ['encoder', 'compressed', 'backbone', 'dechunked', 'bypass', 'full']:
                features = out['features_ref']
                if stage_name not in features:
                    continue
                stage_lengths = features.get(f'{stage_name}_lengths', features['lengths'])
                vals = []
                for allele in ['ref', 'alt']:
                    h = out[f'features_{allele}'][stage_name].detach().float()
                    mask = torch.arange(h.shape[1], device=h.device)[None] < stage_lengths[:, None]
                    vals.append(float((h.square().sum(-1)*mask).sum()/(mask.sum()*h.shape[-1])))
                branch_norms[stage_name] = {'rms_ref': np.sqrt(vals[0]), 'rms_alt': np.sqrt(vals[1])}
        return {'stage': stage, 'objective': cfg.objective, 'batch_pairs': len(batch['ref']),
            'delta_scale': delta_scale, 'activity_scale': activity_scale, 'normalized_delta_mse': float(dloss.detach()),
            'normalized_activity_mse': float(aloss.detach()), 'parameter_blocks': blocks, 'branch_norms': branch_norms,
            'interpretation': 'Unclipped loss gradients; descriptive training division, not evidence of causal gradient stealing. Joint-reference weight is counterfactual for activity-only training.'}



@torch.no_grad()
def path_swap_batch(model, ref, alt, lengths):
    """Score all four combinations of reference/alternate path and bypass states."""
    was_training = model.training
    model.eval()
    try:
        natural = model.forward_pair(ref, alt, lengths, return_features=True)
        r, a = natural['features_ref'], natural['features_alt']
        # Keep decoder tensors in identical 2B order/shape. First crossed result
        # uses the alternate compressed path and reference bypass; the second reverses them.
        predictions = model.decode_branches(torch.cat((a['dechunked'], r['dechunked'])),
                                             torch.cat((r['bypass'], a['bypass'])), torch.cat((lengths, lengths)))
        ar, ra = predictions.split(len(ref))
        rr, aa = natural['y_ref'], natural['y_alt']
        phi_u, phi_s = .5*((ar-rr)+(aa-ra)), .5*((ra-rr)+(aa-ar))
        out = {'f_rr': rr, 'f_aa': aa, 'f_ar': ar, 'f_ra': ra, 'delta_pred': aa-rr,
               'phi_compressed_path': phi_u, 'phi_bypass': phi_s, 'interaction': aa-ar-ra+rr,
               'decomposition_residual': aa-rr-phi_u-phi_s,
               'delta_compressed_path_at_reference_bypass': ar-rr, 'delta_bypass_at_reference_compressed_path': ra-rr}
        # Position consequences are evaluated on actual SNVs; no construct shifts.
        differences = (ref != alt) & (torch.arange(ref.shape[1], device=ref.device)[None] < lengths[:, None])
        if not torch.all(differences.sum(1) == 1):
            raise ValueError('SNV accessibility diagnostics require exactly one actual substitution per pair')
        variant = differences.long().argmax(1)
        radius = model.config.encoder_depth*(model.config.kernel_size//2)
        out['snv_offset'] = variant
        out['sequence_length'] = lengths
        out['encoder_rf_radius_bp'] = torch.full_like(lengths, radius)
        for allele in ['ref', 'alt']:
            route = natural[f'route_{allele}']
            distances = (route.indices-variant[:, None]).abs().masked_fill(~route.selected_mask, ref.shape[1])
            out[f'snv_nearest_selected_{allele}'] = distances.min(1).values
            out[f'selected_states_covering_snv_{allele}'] = ((distances <= radius) & route.selected_mask).sum(1)
        return out
    finally:
        model.train(was_training)



def summarize_snv_accessibility(frame):
    """Direct local encoder-content reachability, not total compressed-path use."""
    strata={'all':np.ones(len(frame),dtype=bool)}
    left=frame.snv_offset.to_numpy()<=frame.encoder_rf_radius_bp.to_numpy()
    right=(frame.sequence_length-frame.snv_offset-1).to_numpy()<=frame.encoder_rf_radius_bp.to_numpy()
    strata.update({'left_edge':left,'right_edge':right,'interior':~(left|right)})
    if 'element' in frame:
        strata.update({f'element:{name}':frame.element.eq(name).to_numpy() for name in frame.element.dropna().unique()})
    result={}
    for name,mask in strata.items():
        f=frame.loc[mask]
        if f.empty:continue
        missing_ref=f.selected_states_covering_snv_ref.eq(0).to_numpy()
        missing_alt=f.selected_states_covering_snv_alt.eq(0).to_numpy()
        row={'n_pairs':len(f),'encoder_rf_radius_bp':sorted(f.encoder_rf_radius_bp.unique().tolist()),
             'snv_offset_min':int(f.snv_offset.min()),'snv_offset_max':int(f.snv_offset.max()),
             'either_allele_no_direct_coverage_fraction':float((missing_ref|missing_alt).mean()),
             'both_alleles_no_direct_coverage_fraction':float((missing_ref&missing_alt).mean()),'alleles':{}}
        for allele in ['ref','alt']:
            distance=f[f'snv_nearest_selected_{allele}']
            count=f[f'selected_states_covering_snv_{allele}']
            row['alleles'][allele]={
                'nearest_selected_distance_bp':{'mean':float(distance.mean()),'median':float(distance.median()),
                    'p95':float(distance.quantile(.95)),'minimum':int(distance.min()),'maximum':int(distance.max()),
                    'distribution':{str(int(k)):int(v) for k,v in distance.value_counts().sort_index().items()}},
                'selected_states_in_encoder_rf':{'mean':float(count.mean()),'median':float(count.median()),
                    'minimum':int(count.min()),'maximum':int(count.max()),
                    'distribution':{str(int(k)):int(v) for k,v in count.value_counts().sort_index().items()}},
                'no_direct_coverage_fraction':float(count.eq(0).mean())}
        result[name]=row
    return {'strata':result,'interpretation':'Direct encoder-content reachability only. Zero coverage does not imply the compressed path has no variant information: the router uses adjacent encoded states, route changes and EMA coefficients can also transmit information. Offsets are diagnostics only and are never probe inputs.'}



def fixed_grid_slot_route(model, natural_route):
    """Force fixed positions while assigning natural EMA coefficients by slot.

    Slot j's coefficient is the natural j-th selected probability, even when its
    recipient coordinate changes. Remaining probabilities stay at natural input
    coordinates; the complete route is rebuilt before the selected override.
    """
    grid = torch.arange(natural_route.indices.shape[1], device=natural_route.indices.device)[None].expand_as(natural_route.indices)*model.config.compression
    grid = grid.clamp_max(natural_route.probs.shape[1]-1)
    probs = natural_route.probs.clone()
    # Masked padding must not overwrite a valid coefficient at its clipped index.
    rows, slots = torch.where(natural_route.selected_mask)
    probs[rows, grid[rows, slots]] = natural_route.selected_probs[rows, slots]
    return model.route_from_probs(probs, natural_route.lengths, indices=grid)



@torch.no_grad()
def routing_factorial_batch(model, ref, alt, lengths):
    if model.config.mode != 'learned':
        raise ValueError('Hard-position and EMA intervention is defined for learned checkpoints')
    was_training = model.training
    model.eval()
    try:
        both_lengths = torch.cat((lengths, lengths))
        encoded = model.encode(torch.cat((ref, alt)), both_lengths)
        natural = model.make_route(encoded, both_lengths)
        fixed = fixed_grid_slot_route(model, natural)
        result = {}
        for h, route in [('natural', natural), ('fixed', fixed)]:
            for ema in [True, False]:
                score = model.forward_with_route(encoded, both_lengths, route, ema_enabled=ema)
                r, a = score.split(len(ref))
                result[f'delta_positions_{h}_ema_{"on" if ema else "off"}'] = a-r
        result['delta_pred'] = result['delta_positions_natural_ema_on']
        result['prediction_interaction'] = (result['delta_positions_fixed_ema_off']-result['delta_positions_natural_ema_off']
                                             -result['delta_positions_fixed_ema_on']+result['delta_positions_natural_ema_on'])
        return result
    finally:
        model.train(was_training)



def bootstrap_columns(values, n_bootstrap=1000, seed=0):
    """The same component indices feed every column and every contrast."""
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    if not np.isfinite(values).all():
        raise ValueError('All component contrasts must be finite')
    point = values.mean(0)
    if len(values) < 2 or n_bootstrap <= 0:
        return [{'estimate': float(x), 'ci95_low': None, 'ci95_high': None,
                 'one_sided95_lower': None, 'one_sided95_upper': None, 'n_components': len(values), 'n_resamples': 0} for x in point]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, values.shape[1]))
    for i in range(n_bootstrap):
        draws[i] = values[rng.integers(0, len(values), len(values))].mean(0)
    quantiles = np.quantile(draws, [.025, .975, .05, .95], axis=0)
    return [{'estimate': float(x), 'ci95_low': float(quantiles[0, i]), 'ci95_high': float(quantiles[1, i]),
             'one_sided95_lower': float(quantiles[2, i]), 'one_sided95_upper': float(quantiles[3, i]),
             'n_components': len(values), 'n_resamples': n_bootstrap} for i, x in enumerate(point)]



def factorial_comparison(cells, n_bootstrap=1000, seed=0):
    names = ['none_on', 'none_off', 'fixed_on', 'fixed_off', 'learned_on', 'learned_off']
    if set(cells) != set(names):
        raise ValueError('Factorial inference requires all six complete cells')
    anchor = cells['none_on']
    errors = pd.DataFrame({'locus_id': anchor.locus_id.to_numpy()})
    for name in names:
        aligned = align_predictions(anchor, cells[name])
        errors[name] = (aligned.reference_prediction-aligned.delta_true)**2
    errors['zero'] = np.asarray(anchor.delta_true if 'delta_true' in anchor else anchor.delta)**2
    table = errors.groupby('locus_id', sort=False).mean()
    for mode in ['none', 'fixed', 'learned']:
        table[f'{mode}_off_minus_on'] = table[f'{mode}_off']-table[f'{mode}_on']
    for mode in ['fixed', 'learned']:
        for skip in ['on', 'off']:
            table[f'compression_effect_{mode}_{skip}'] = table[f'{mode}_{skip}']-table[f'none_{skip}']
        table[f'skip_interaction_{mode}'] = table[f'compression_effect_{mode}_off']-table[f'compression_effect_{mode}_on']
    table['routing_skip_interaction'] = table.skip_interaction_learned-table.skip_interaction_fixed
    stats = dict(zip(table.columns, bootstrap_columns(table.to_numpy(), n_bootstrap, seed)))
    for name in names:
        stats[name]['relative_gain_over_zero'] = 1-stats[name]['estimate']/stats['zero']['estimate']
    return {'statistics': stats, 'n_pairs': len(anchor), 'contrasts': ['skip_interaction_fixed', 'skip_interaction_learned', 'routing_skip_interaction'],
            'uncertainty': 'Within-base-seed paired component percentile bootstrap; shared component draws for all cells and contrasts.',
            'interpretation': 'A positive skip interaction indicates changed relative compression cost. Inspect absolute risks and simple effects to interpret branch interactions.'}, table.reset_index()



def engineering_comparison(candidate, reference, thresholds, n_bootstrap=1000, seed=0, *, mse_margin, score_margin):
    f = align_predictions(candidate, reference)
    truth, c, r = (f[k].to_numpy(float) for k in ['delta_true', 'candidate_prediction', 'reference_prediction'])
    errors = pd.DataFrame({'locus': f.locus_id, 'difference': (c-truth)**2-(r-truth)**2,
                           'candidate': (c-truth)**2, 'reference': (r-truth)**2, 'zero': truth**2})
    by = errors.groupby('locus', sort=False).mean()
    mse = bootstrap_columns(by.difference, n_bootstrap, seed)[0]
    upper, lower, high = mse['one_sided95_upper'], mse['ci95_low'], mse['ci95_high']
    mse.update({'margin': mse_margin, 'noninferior': upper < mse_margin if upper is not None else None,
                'equivalent': lower > -mse_margin and high < mse_margin if lower is not None else None})
    reliable, known = _reliability_mask(f, thresholds)
    reliable &= (np.abs(truth) >= thresholds.strong_abs_delta) & (truth != 0)
    endpoints = {}
    budget = max(1, int(np.ceil(thresholds.screening_fraction*len(f))))
    cs, rs = np.zeros(len(f), bool), np.zeros(len(f), bool)
    cs[np.argsort(-np.abs(c), kind='stable')[:budget]] = True
    rs[np.argsort(-np.abs(r), kind='stable')[:budget]] = True
    for name, values in [('direction', (np.sign(c) == np.sign(truth)).astype(float)-(np.sign(r) == np.sign(truth)).astype(float)),
                         ('recall', cs.astype(float)-rs.astype(float))]:
        if not known or not reliable.any():
            endpoints[name] = {'n': 0, 'noninferior': None}
            continue
        by_locus = pd.Series(values[reliable]).groupby(f.locus_id.to_numpy()[reliable], sort=False).mean()
        stat = bootstrap_columns(by_locus, n_bootstrap, seed)[0]
        stat.update({'n': int(reliable.sum()), 'margin': score_margin,
                     'noninferior': stat['one_sided95_lower'] > -score_margin if stat['one_sided95_lower'] is not None else None})
        endpoints[name] = stat
    checks = [mse['noninferior'], *(x['noninferior'] for x in endpoints.values())]
    return {'mse': mse, 'reliable_large': endpoints, 'overall_noninferiority': all(checks) if all(v is not None for v in checks) else None,
            'macro_risks': by[['candidate', 'reference', 'zero']].mean().to_dict(),
            'relative_gain_over_zero': {name: 1-by[name].mean()/by.zero.mean() for name in ['candidate', 'reference']},
            'recall_interval_scope': 'Component-resampled conditional uncertainty for the fixed whole-split candidate sets, not a repeated-screening-population design.',
            'scale_scope': 'The caller supplies margins in the squared target units and score units.'}



def precision_decision_summary(frame, thresholds):
    """Reuse stored precision predictions and boundary counts without inference."""
    n = len(frame)
    p, q, truth = (frame[k].to_numpy(float) for k in ['pred16', 'pred32', 'delta_true'])
    budget = max(1, int(np.ceil(thresholds.screening_fraction*n)))
    op, oq = np.argsort(-np.abs(p), kind='stable'), np.argsort(-np.abs(q), kind='stable')
    a, b = set(op[:budget]), set(oq[:budget])
    rp, rq = np.empty(n, int), np.empty(n, int)
    rp[op], rq[oq] = np.arange(n), np.arange(n)
    reliable, known = _reliability_mask(frame, thresholds)
    reliable &= (np.abs(truth) >= thresholds.strong_abs_delta) & (truth != 0)
    sign_flip = np.sign(p) != np.sign(q)
    routing_available = 'hardmask_any_changed' in frame
    changed = frame.hardmask_any_changed.to_numpy(bool) if routing_available else None
    selected_xor = np.zeros(n, bool)
    selected_xor[list(a ^ b)] = True
    per_allele = {} if routing_available else None
    for allele in (['ref', 'alt'] if routing_available else []):
        # Counts suffice for exact set intersection/union because same K is stored.
        k16, k32 = frame[f'k16_{allele}'].to_numpy(float), frame[f'k32_{allele}'].to_numpy(float)
        if not np.array_equal(k16, k32):
            raise ValueError('Stored precision counts have unequal budgets')
        length = frame.get('length', None)
        if length is None:
            raise ValueError('A length column is required for boundary-set comparisons')
        difference = frame[f'hardmask_changed_fraction_{allele}'].to_numpy(float)*np.asarray(length)
        total = k16+k32
        per_allele[allele] = {'mean_boundary_set_jaccard': float(np.mean((total-difference)/(total+difference)))}
    return {'n': n, 'budget': budget, 'candidate_intersection': len(a & b), 'candidate_set_jaccard': len(a & b)/len(a | b),
            'candidate_overlap_fraction': len(a & b)/budget, 'candidate_symmetric_difference_n': len(a ^ b),
            'absolute_rank_shift_median': float(np.median(np.abs(rp-rq))), 'absolute_rank_shift_p95': float(np.quantile(np.abs(rp-rq), .95)),
            'score_rank_spearman': _correlation(np.abs(p), np.abs(q), True),
            'reliable_large_n': int(reliable.sum()), 'reliable_large_direction_flips': int((sign_flip & reliable).sum()),
            'reliable_large_flip_fraction': float(sign_flip[reliable].mean()) if reliable.any() else None,
            'flipped_reliable_max_absolute_score': float(np.max(np.maximum(np.abs(p), np.abs(q))[sign_flip & reliable])) if (sign_flip & reliable).any() else None,
            'routing_available': routing_available,
            'changed_route_pair_fraction': float(changed.mean()) if routing_available else None,
            'changed_positions_fraction': float(frame.hardmask_changed_fraction.mean()) if routing_available else None,
            'candidate_changes_with_changed_route_n': int((selected_xor & changed).sum()) if routing_available else None,
            'candidate_changes_without_changed_route_n': int((selected_xor & ~changed).sum()) if routing_available else None, 'boundary_sets': per_allele,
            'attribution_peak_locations': 'Position-wise attribution is required to compare attribution peak locations.',
            'policy': 'BF16/autocast + TF32 policy versus IEEE FP32; not isolated quantization.'}

