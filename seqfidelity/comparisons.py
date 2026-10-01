"""Align complete prediction sets and compare paired locus risks."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import EvaluationThresholds, _bootstrap_means, _reliability_mask, paired_comparison


def align_predictions(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Require every pair and its true locus/label; never inner-join away errors."""
    for name, frame in [('candidate', candidate), ('reference', reference)]:
        if frame.pair_id.isna().any() or frame.pair_id.duplicated().any():
            raise ValueError(f'{name} pair_id must be nonmissing and unique')
    left_ids, right_ids = set(candidate.pair_id), set(reference.pair_id)
    if left_ids != right_ids:
        raise ValueError(f'Pair sets differ: candidate_only={len(left_ids-right_ids)}, reference_only={len(right_ids-left_ids)}; no rows discarded')
    reference = reference.set_index('pair_id').loc[candidate.pair_id].reset_index()
    candidate = candidate.reset_index(drop=True).copy()
    if not candidate.locus_id.astype(str).equals(reference.locus_id.astype(str)):
        raise ValueError('Aligned pairs have different locus assignments')
    if candidate.locus_id.isna().any():
        raise ValueError('Missing locus assignment')
    if 'delta_true' not in candidate:
        candidate['delta_true'] = candidate.delta
    reference_truth = reference.delta_true if 'delta_true' in reference else reference.delta
    if not np.allclose(candidate.delta_true.to_numpy(float), reference_truth.to_numpy(float), atol=1e-12, rtol=1e-10, equal_nan=False):
        raise ValueError('Aligned pairs have different or missing measured effect labels')
    candidate['candidate_prediction'] = candidate.delta_pred
    candidate['reference_prediction'] = reference.delta_pred.to_numpy()
    if not np.isfinite(candidate[['delta_true','candidate_prediction','reference_prediction']].to_numpy(float)).all():
        raise ValueError('Every aligned measured label and model prediction must be finite')
    for column in ['quality_pass','q_value','p_value','se']:
        if column in candidate and column in reference:
            if column == 'quality_pass':
                left = candidate[column].astype(str).str.lower().replace({'1':'true','0':'false'})
                right = reference[column].astype(str).str.lower().replace({'1':'true','0':'false'})
                agrees = left.equals(right)
            else:
                agrees = np.allclose(candidate[column].to_numpy(float), reference[column].to_numpy(float), equal_nan=True)
            if not agrees:
                raise ValueError(f'Aligned labels disagree on measurement field {column}')
    return candidate



def compare_predictions(candidate, reference, thresholds: EvaluationThresholds,
                        n_bootstrap=1000, seed=0):
    frame = align_predictions(candidate, reference)
    result = paired_comparison(frame, 'candidate_prediction', 'reference_prediction',
                               n_bootstrap=n_bootstrap, seed=seed)
    truth = frame.delta_true.to_numpy(float)
    errors = pd.DataFrame({'locus_id': frame.locus_id,
                           'candidate': (frame.candidate_prediction.to_numpy()-truth)**2,
                           'reference': (frame.reference_prediction.to_numpy()-truth)**2})
    means = errors.groupby('locus_id', sort=False)[['candidate','reference']].mean().mean()
    result.update({'candidate_macro_delta_mse': float(means.candidate),
                   'reference_macro_delta_mse': float(means.reference),
                   'relative_macro_mse_improvement': float(1-means.candidate/means.reference) if means.reference > 0 else None,
                   'all_pairs_retained': True})
    reliable, evidence_known = _reliability_mask(frame, thresholds)
    reliable &= (np.abs(truth) >= thresholds.strong_abs_delta) & (truth != 0)
    result['reliable_large_n'] = int(reliable.sum())
    result['reliable_large_n_loci'] = int(frame.loc[reliable,'locus_id'].nunique())
    result['reliable_large_sign_accuracy_difference'] = None
    result['reliable_large_macro_sign_accuracy_difference'] = None
    result['reliable_large_macro_sign_difference_ci95'] = {'low':None,'high':None,'n_resamples':0}
    if evidence_known and reliable.any():
        candidate_correct = np.sign(frame.candidate_prediction.to_numpy()[reliable]) == np.sign(truth[reliable])
        reference_correct = np.sign(frame.reference_prediction.to_numpy()[reliable]) == np.sign(truth[reliable])
        difference = candidate_correct.astype(float)-reference_correct.astype(float)
        by_locus = pd.Series(difference).groupby(frame.loc[reliable,'locus_id'].to_numpy(),sort=False).mean().to_numpy()
        result.update({'reliable_large_candidate_sign_accuracy': float(candidate_correct.mean()),
                       'reliable_large_reference_sign_accuracy': float(reference_correct.mean()),
                       'reliable_large_sign_accuracy_difference': float(difference.mean()),
                       'reliable_large_macro_sign_accuracy_difference': float(by_locus.mean()),
                       'reliable_large_macro_sign_difference_ci95': _bootstrap_means(by_locus,n_bootstrap,seed)})
    result['thresholds'] = thresholds.to_dict()
    return result

