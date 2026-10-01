"""Coordinate-aware representation pooling and shared target readouts."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import spearmanr
from .metrics import _bootstrap_means


def pool_state(hidden, coords, lengths, original_lengths, pooling):
    """All three blocks have D entries; padding never affects max or means."""
    h = hidden.float()
    valid = torch.arange(h.shape[1], device=h.device)[None] < lengths[:, None]
    mask = valid.unsqueeze(-1)
    h = h.masked_fill(~mask, 0)
    u = coords.float() / (original_lengths[:, None] - 1).clamp_min(1)
    count = lengths[:, None].float()
    mean = h.sum(1) / count
    if pooling == 'moment':
        parts = (mean, h.masked_fill(~mask, -torch.inf).max(1).values,
                 (h * u.unsqueeze(-1)).sum(1) / count)
    elif pooling == 'fourier':
        angle = 2 * torch.pi * u.unsqueeze(-1)
        parts = (mean, (h * angle.sin()).sum(1) / count,
                 (h * angle.cos()).sum(1) / count)
    else:
        raise ValueError(pooling)
    return torch.cat(parts, -1)



class TargetProbe(nn.Module):
    def __init__(self, feature_dim, hidden=128, target='delta'):
        super().__init__()
        self.target = target
        self.head = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, values):
        scores = self.head(values).squeeze(-1)
        return scores[:, 1] - scores[:, 0] if self.target == 'delta' else scores.mean(-1)



def target_values(frame, target):
    if target == 'delta':
        return frame.delta.to_numpy(dtype=np.float32)
    activity = frame.reindex(columns=['y_ref', 'y_alt']).to_numpy(dtype=np.float32)
    return activity.mean(1)  # A missing allele means this row is ineligible.



@torch.inference_mode()
def probe_predictions(probe, values, target_scale, batch_size):
    probe.eval()
    return torch.cat([probe(values[start:start+batch_size]) * target_scale
                      for start in range(0, len(values), batch_size)]).cpu().numpy()



def background_metrics(frame, truth, prediction, config):
    error = prediction - truth
    table = pd.DataFrame({'locus_id': frame.locus_id.to_numpy(), 'mse': error**2,
                          'mae': np.abs(error)}).groupby('locus_id', sort=False).mean()
    return {'n': len(frame), 'n_loci': len(table),
            'macro_background_mse': float(table.mse.mean()),
            'macro_background_mae': float(table.mae.mean()),
            'micro_background_mse': float(np.mean(error**2)),
            'spearman': float(spearmanr(truth, prediction).statistic)
                if np.ptp(truth) and np.ptp(prediction) else None,
            'macro_background_mse_ci95': _bootstrap_means(table.mse.to_numpy(), config.n_bootstrap, config.seed)}

