"""Compare masked computational states and inspect executed EMA coefficients."""
from __future__ import annotations
import torch

def state_difference(ref, alt, ref_mask, alt_mask):
    """Exact equality and continuous norms; masks are part of the state.

Norms zero-extend each masked tensor onto the union of valid slots. A mask
mismatch remains a mismatch even if the corresponding numeric values are zero.
"""
    if ref.ndim == 2:
        ref, alt = ref[..., None], alt[..., None]
    union = ref_mask | alt_mask
    r = torch.where(ref_mask[..., None], ref.float(), 0.)
    a = torch.where(alt_mask[..., None], alt.float(), 0.)
    difference = a - r
    if not torch.isfinite(difference).all():
        raise FloatingPointError('A valid state value is nonfinite')
    squared = difference.square().flatten(1).sum(1)
    maximum = difference.abs().flatten(1).amax(1)
    mask_equal = ref_mask.eq(alt_mask).all(1)
    return {'equal': mask_equal & maximum.eq(0), 'max_abs': maximum,
            'l2': squared.sqrt(), 'rms': (squared / (union.sum(1) * ref.shape[-1]).clamp_min(1)).sqrt(),
            'mask_equal': mask_equal, 'unmatched_slots': (ref_mask ^ alt_mask).sum(1)}


def executed_ema(model, route):
    """The coefficient actually used; fixed/none bypass EMA entirely."""
    if model.config.mode in {'none', 'fixed', 'aggregate', 'no_backbone', 'cnn'}:
        return route.selected_mask.float()
    return actual_ema_coefficients(route)


def masked_pair_difference(ref, alt, ref_mask, alt_mask, *, tolerance):
    """Compare computational slots; padding is excluded and mask mismatch explicit."""
    if ref.ndim == 2:
        ref, alt = ref[..., None], alt[..., None]
    common = ref_mask & alt_mask
    valid = common[..., None]
    difference = torch.where(valid, (alt.float()-ref.float()).abs(), 0)
    if not torch.isfinite(difference).all():
        raise FloatingPointError('Nonfinite paired values in valid computational slots')
    maximum = difference.flatten(1).amax(1)
    same_mask = (ref_mask == alt_mask).all(1)
    return {'max_abs_difference': maximum, 'exact_equal': same_mask & (maximum == 0),
            'near_equal': same_mask & (maximum <= tolerance), 'valid_masks_equal': same_mask,
            'common_valid_slots': common.sum(1), 'unmatched_valid_slots': (ref_mask ^ alt_mask).sum(1)}


def actual_ema_coefficients(route):
    """Match learned _dechunk's p before the triangular EMA weight calculation."""
    p = route.selected_probs.float().clamp(1e-4, 1-1e-4)
    p = torch.cat((torch.ones_like(p[:, :1]), p[:, 1:]), dim=1)
    return p*route.selected_mask

