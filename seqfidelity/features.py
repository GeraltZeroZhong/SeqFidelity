"""Read paired features from a local cache and score a normalized readout."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn


class FeatureBank:
    def __init__(self, cache_root):
        root = Path(cache_root)
        progress = json.loads((root / 'progress.json').read_text())
        if progress['completed'] != progress['total']:
            raise ValueError('Feature extraction must complete before supervised training')
        self.features = np.load(root / 'features.npy', mmap_mode='r')
        self.pairs = pd.read_parquet(root / 'pair_index.parquet').set_index('pair_id')
        self.progress = progress

    def align(self, frame):
        # Pair IDs and literal allele sequences jointly protect cache row alignment.
        mapped = self.pairs.reindex(frame.pair_id.to_numpy())
        for side in ['ref', 'alt']:
            if not np.array_equal(mapped[f'{side}_seq'].to_numpy(), frame[f'{side}_seq'].to_numpy()):
                raise ValueError(f'Feature cache has missing or changed {side} sequences')
        return mapped[['ref_index', 'alt_index', 'ref_rc_index', 'alt_rc_index']].to_numpy(np.int64)

    def standardization(self, indices, include_rc=True):
        columns = 4 if include_rc else 2
        total = 0
        sums = np.zeros(self.features.shape[1], dtype=np.float64)
        squares = sums.copy()
        for begin in range(0, len(indices), 1024):
            values = np.asarray(self.features[indices[begin:begin + 1024, :columns].reshape(-1)], dtype=np.float64)
            sums += values.sum(0)
            squares += np.square(values).sum(0)
            total += len(values)
        mean = sums / total
        variance = np.maximum(squares / total - mean ** 2, 0)
        return mean.astype(np.float32), np.maximum(np.sqrt(variance), 1e-4).astype(np.float32)



class FrozenReadout(nn.Module):
    def __init__(self, feature_dim, mean=None, std=None, hidden=256):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(feature_dim) if mean is None else torch.as_tensor(mean))
        self.register_buffer('feature_std', torch.ones(feature_dim) if std is None else torch.as_tensor(std))
        self.head = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, features):
        return self.head((features.float() - self.feature_mean) / self.feature_std).squeeze(-1)

    def forward_pair(self, ref, alt):
        batch = len(ref)
        prediction = self(torch.cat((ref, alt)))
        y_ref, y_alt = prediction[:batch], prediction[batch:]
        return {'y_ref': y_ref, 'y_alt': y_alt, 'delta': y_alt - y_ref}



def _feature_batch(bank, indices, rows, device, reverse=False, rng=None):
    selected = indices[rows, :2].copy()
    if reverse:
        rc = rng.random(len(rows)) < .5
        selected[rc] = indices[rows[rc], 2:]
    ref = torch.from_numpy(np.asarray(bank.features[selected[:, 0]]).copy()).to(device)
    alt = torch.from_numpy(np.asarray(bank.features[selected[:, 1]]).copy()).to(device)
    return ref, alt



@torch.inference_mode()
def predict_readout(model, bank, frame, indices, cfg):
    model.eval()
    results = {key: [] for key in ['delta_pred', 'y_ref_pred', 'y_alt_pred']}
    for begin in range(0, len(frame), cfg.batch_size):
        rows = np.arange(begin, min(begin + cfg.batch_size, len(frame)))
        ref, alt = _feature_batch(bank, indices, rows, cfg.device)
        output = model.forward_pair(ref, alt)
        for destination, source in [('delta_pred', 'delta'), ('y_ref_pred', 'y_ref'), ('y_alt_pred', 'y_alt')]:
            results[destination].extend(output[source].cpu().numpy().tolist())
    prediction = frame.drop(columns=['ref_seq', 'alt_seq'], errors='ignore').copy()
    prediction['delta_true'] = prediction.delta
    for key, values in results.items():
        prediction[key] = values
    return prediction

