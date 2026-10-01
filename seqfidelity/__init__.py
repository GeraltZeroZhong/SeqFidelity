"""Reusable paired sequence models and locus-aware evaluation."""
from .model import ModelConfig, PairedSequenceModel, CompactCNN, RouteState, build_model
from .encoding import encode_sequence, PairDataset, collate_pairs
from .metrics import EvaluationThresholds, fit_thresholds, evaluate_predictions

__all__ = [
    'ModelConfig', 'PairedSequenceModel', 'CompactCNN', 'RouteState', 'build_model',
    'encode_sequence', 'PairDataset', 'collate_pairs',
    'EvaluationThresholds', 'fit_thresholds', 'evaluate_predictions',
]
