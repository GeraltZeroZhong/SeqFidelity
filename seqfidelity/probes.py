"""Frozen branch representations with identical shared nonlinear pair readouts."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn

from .diagnostics import pooled_branch_features
from .metrics import evaluate_predictions, fit_thresholds
from .precision import execution_precision
from .training import device_batch, loader, macro_mse, seed_everything, write_json

BRANCHES = ('full', 'backbone', 'encoder')


@dataclass
class ProbeConfig:
    seed: int = 0
    hidden: int = 128
    epochs: int = 20
    min_epochs: int = 5
    patience: int = 4
    batch_size: int = 4096
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    device: str = 'cpu'
    n_bootstrap: int = 1000


class PairProbe(nn.Module):
    """The same head sees each allele separately; the final score is their difference."""
    def __init__(self, feature_dim, hidden=128):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, paired_features):
        predictions = self.head(paired_features).squeeze(-1)
        return predictions[:, 1] - predictions[:, 0]


@torch.inference_mode()
def cache_representations(model, frames: dict[str, pd.DataFrame], directory,
                          checkpoint, batch_size=None, device='cpu', amp=True,
                          checkpoint_batch_size=None, length_scale=1.0):
    """Cache pooled representations at a caller-specified directory.

    No reverse-complement augmentation is used in any branch. This is a common
    frozen-representation comparison, not an exactly matched retraining of the
    end-to-end head. Float32 files retain the extracted activation precision.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint).resolve()
    requested_batch_size = batch_size
    if batch_size is None:
        if checkpoint_batch_size is None:
            checkpoint_batch_size = torch.load(checkpoint, map_location='cpu', weights_only=False)['train_config']['batch_size']
        batch_size = checkpoint_batch_size
    bf16 = bool(amp and torch.device(device).type == 'cuda')
    execution = {'autocast': 'bfloat16' if bf16 else False,
                 'float32_matmul_precision': 'high' if bf16 else 'highest',
                 'cuda_matmul_allow_tf32': bf16, 'cudnn_allow_tf32': bf16}
    marker = directory / 'cache_manifest.json'
    identity = {'checkpoint': str(checkpoint), 'checkpoint_mtime_ns': checkpoint.stat().st_mtime_ns,
                'split_rows': {key: len(value) for key, value in frames.items()},
                'pooling': 'mean,max,coordinate moment,retained fraction,scaled length',
                'length_scale': length_scale,
                'dtype': 'float32', 'reverse_complement': False, 'amp': amp,
                'batch_size': batch_size, 'checkpoint_batch_size': checkpoint_batch_size,
                'requested_batch_size': requested_batch_size,
                'batch_size_source': 'checkpoint' if requested_batch_size is None else 'explicit_cli_override',
                'execution': execution,
                'allele_order': ['ref', 'alt']}
    if marker.exists():
        previous = json.loads(marker.read_text())
        if all(previous.get(key) == value for key, value in identity.items()):
            for split, frame in frames.items():
                metadata = pd.read_parquet(directory / f'{split}_metadata.parquet')
                if not np.array_equal(metadata.pair_id.to_numpy(), frame.pair_id.to_numpy()):
                    raise ValueError('Cached row order differs from requested paired constructs')
            return previous
        raise ValueError('Existing representation cache belongs to another checkpoint or data selection; choose a fresh cache directory')
    previous_training = model.training
    model.eval()
    started = perf_counter()
    feature_dim = None
    try:
        for split, frame in frames.items():
            arrays = {}
            metadata = frame.drop(columns=['ref_seq', 'alt_seq'], errors='ignore').copy()
            metadata.to_parquet(directory / f'{split}_metadata.parquet', index=False)
            for raw in loader(frame, batch_size=batch_size):
                batch = device_batch(raw, torch.device(device))
                with execution_precision(torch.device(device).type, bf16=bf16):
                    outputs = model.forward_pair(batch['ref'], batch['alt'], batch['lengths'], return_features=True)
                    left = pooled_branch_features(outputs['features_ref'], length_scale)
                    right = pooled_branch_features(outputs['features_alt'], length_scale)
                if not arrays:
                    feature_dim = left['full'].shape[1]
                    for branch in BRANCHES:
                        arrays[branch] = np.lib.format.open_memmap(
                            directory / f'{split}_{branch}.npy', mode='w+', dtype=np.float32,
                            shape=(len(frame), 2, feature_dim))
                indices = raw['index'].numpy()
                for branch in BRANCHES:
                    arrays[branch][indices] = torch.stack((left[branch], right[branch]), 1).float().cpu().numpy()
            for array in arrays.values():
                array.flush()
            del arrays
            print(f'Frozen representations cached: {split}, {len(frame)} complete pairs', flush=True)
    finally:
        model.train(previous_training)
    manifest = {**identity, 'feature_dim': feature_dim, 'extraction_seconds': perf_counter() - started,
                'branches': list(BRANCHES), 'status': 'complete',
                'raw_bases_given_to_backbone_probe': False}
    write_json(marker, manifest)
    return manifest


def _moments(features, batch_size=8192):
    total = np.zeros(features.shape[-1], dtype=np.float64)
    square = np.zeros_like(total)
    count = 0
    for start in range(0, len(features), batch_size):
        values = np.asarray(features[start:start + batch_size], dtype=np.float64).reshape(-1, features.shape[-1])
        total += values.sum(0)
        square += np.square(values).sum(0)
        count += len(values)
    mean = total / count
    variance = np.maximum(square / count - mean ** 2, 0)
    # Keep constant training coordinates at their native scale; small numerical
    # variance must not amplify values in new inputs.
    scale = np.where(variance <= 1e-12, 1.0, np.maximum(np.sqrt(variance), 1e-4))
    return mean.astype(np.float32), scale.astype(np.float32)


def _tensor_batch(features, indices, mean, scale, device):
    values = torch.from_numpy(np.array(features[indices], copy=True)).to(device)
    mean = torch.as_tensor(mean, device=device)
    scale = torch.as_tensor(scale, device=device)
    return (values - mean) / scale


@torch.inference_mode()
def predict_probe(probe, features, mean, scale, delta_scale, config):
    probe.eval()
    prediction = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), config.batch_size):
        end = min(start + config.batch_size, len(features))
        values = _tensor_batch(features, slice(start, end), mean, scale, config.device)
        prediction[start:end] = (probe(values) * delta_scale).float().cpu().numpy()
    return prediction


def train_probe(cache_directory, branch: str, config: ProbeConfig, output_directory,
                checkpoint_directory=None):
    if branch not in BRANCHES:
        raise ValueError(f'Unknown branch {branch}')
    cache_directory = Path(cache_directory)
    representation_manifest = json.loads((cache_directory / 'cache_manifest.json').read_text())
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = (output_directory / 'checkpoints' if checkpoint_directory is None
                            else Path(checkpoint_directory) / output_directory.name)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    split_names = [split for split in ('train', 'validation', 'test', 'external')
                   if (cache_directory / f'{split}_{branch}.npy').exists()]
    arrays = {split: np.load(cache_directory / f'{split}_{branch}.npy', mmap_mode='r') for split in split_names}
    frames = {split: pd.read_parquet(cache_directory / f'{split}_metadata.parquet') for split in split_names}
    train, validation = frames['train'], frames['validation']
    mean_np, scale_np = _moments(arrays['train'])
    mean = torch.from_numpy(mean_np).to(config.device)
    scale = torch.from_numpy(scale_np).to(config.device)
    delta_scale = max(float(np.sqrt(np.mean(np.square(train.delta.to_numpy())))), .01)
    probe = PairProbe(arrays['train'].shape[-1], config.hidden).to(config.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    targets = train.delta.to_numpy(dtype=np.float32)
    history, best_metric, bad_epochs, best_state = [], float('inf'), 0, None
    started = perf_counter()
    for epoch in range(config.epochs):
        probe.train()
        order = np.random.default_rng(config.seed + epoch).permutation(len(train))
        total_loss = 0.
        epoch_start = perf_counter()
        for start in range(0, len(train), config.batch_size):
            indices = order[start:start + config.batch_size]
            features = _tensor_batch(arrays['train'], indices, mean, scale, config.device)
            target = torch.from_numpy(targets[indices]).to(config.device) / delta_scale
            optimizer.zero_grad(set_to_none=True)
            loss = (probe(features) - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite frozen-probe loss: {branch}, epoch {epoch + 1}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        prediction = predict_probe(probe, arrays['validation'], mean, scale, delta_scale, config)
        metric = macro_mse(validation, prediction)
        history.append({'epoch': epoch + 1, 'train_normalized_delta_mse': total_loss / len(train),
                        'validation_macro_delta_mse': metric, 'seconds': perf_counter() - epoch_start,
                        'pairs_seen': len(train)})
        if metric < best_metric:
            best_metric, bad_epochs = metric, 0
            best_state = copy.deepcopy(probe.state_dict())
            best_epoch = epoch + 1
        else:
            bad_epochs += 1
        write_json(output_directory / 'history.json', history)
        print(f'Probe {branch} seed {config.seed}: epoch {epoch + 1}, val macro MSE {metric:.7g}', flush=True)
        if epoch + 1 >= config.min_epochs and bad_epochs >= config.patience:
            break
    probe.load_state_dict(best_state)
    saved = {'state_dict': probe.state_dict(), 'probe_config': asdict(config), 'branch': branch,
             'input_dimension': arrays['train'].shape[-1], 'mean': mean_np, 'scale': scale_np,
             'delta_scale': delta_scale, 'best_epoch': best_epoch, 'cache_directory': str(cache_directory)}
    torch.save(saved, checkpoint_directory / 'best.pt')
    validation = validation.copy()
    validation['split'] = 'validation'
    thresholds = fit_thresholds(validation)
    result = {}
    for split in ('validation', 'test', 'external'):
        if split not in arrays:
            continue
        frame = frames[split].copy()
        frame['delta_true'] = frame.delta
        frame['delta_pred'] = predict_probe(probe, arrays[split], mean, scale, delta_scale, config)
        result[split] = evaluate_predictions(frame, thresholds, config.n_bootstrap, config.seed)
        frame.to_csv(output_directory / f'{split}_predictions.tsv', sep='\t', index=False)
        if split == 'test':
            frame.to_csv(output_directory / 'predictions.tsv', sep='\t', index=False)
    write_json(output_directory / 'metrics.json', result)
    write_json(output_directory / 'run_manifest.json', {
        'status': 'complete', 'branch': branch, 'seed': config.seed,
        'probe_config': asdict(config), 'parameters': sum(p.numel() for p in probe.parameters()),
        'representation_cache': str(cache_directory), 'checkpoint': str(checkpoint_directory / 'best.pt'),
        'representation_batch_size': representation_manifest['batch_size'],
        'representation_batch_size_source': representation_manifest['batch_size_source'],
        'representation_execution': representation_manifest['execution'],
        'training_seconds': perf_counter() - started, 'best_epoch': best_epoch,
        'epochs_completed': len(history), 'train_pairs': len(train), 'validation_pairs': len(validation),
        'selection': 'validation locus-macro delta MSE',
        'augmentation': 'none, identical for every frozen branch',
        'standardization': 'Training alleles only; variance <=1e-12 uses scale 1; other dimensions use max(std,1e-4). This keeps constant training-length features at native scale externally.',
        'prediction': 'shared_head(alt_features) - shared_head(ref_features)',
        'interpretation': 'Readout accessibility for this probe; failure does not prove information absent.'})
    return result
