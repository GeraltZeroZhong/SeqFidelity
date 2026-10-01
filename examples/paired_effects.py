"""Run an independent synthetic example through the paired prediction API."""
import argparse
from pathlib import Path

import torch

from seqfidelity.model import ModelConfig, build_model
from seqfidelity.synthetic import toy_pairs
from seqfidelity.training import TrainConfig, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    frame = toy_pairs(n_pairs=8, length=24, seed=0)
    model = build_model(ModelConfig(mode="aggregate", dim=16, depth=1, heads=2,
                                   compression=4, encoder_depth=1, decoder_depth=1))
    predictions = predict(model, frame, TrainConfig(device="cpu", batch_size=4, amp=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(predictions)} synthetic predictions to {args.output}")


if __name__ == "__main__":
    main()
