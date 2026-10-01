"""Command-line tools for paired sequence prediction and synthetic examples."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("Input tables must use .csv or .tsv filenames")
    return pd.read_csv(path, sep="," if path.suffix.lower()==".csv" else "\t")


def save_table(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("Output tables must use .csv or .tsv filenames")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="," if path.suffix.lower()==".csv" else "\t", index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="seqfidelity", description="Paired DNA effect prediction and compression analysis.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a small CPU example using independent synthetic pairs.")
    demo.add_argument("--output", type=Path, required=True, help="Directory for toy inputs and example predictions.")
    demo.add_argument("--pairs", type=int, default=16)
    demo.add_argument("--length", type=int, default=32)
    demo.add_argument("--seed", type=int, default=0)
    generate = commands.add_parser("generate", help="Create toy pairs with analytically defined activity labels.")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--pairs", type=int, default=32)
    generate.add_argument("--length", type=int, default=32)
    generate.add_argument("--seed", type=int, default=0)
    train = commands.add_parser("train", help="Fit a model to user-provided training and validation tables.")
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--mode", choices=["none", "fixed", "aggregate", "learned", "shared", "coupled", "no_backbone", "cnn"], default="aggregate")
    train.add_argument("--dim", type=int, default=32)
    train.add_argument("--depth", type=int, default=2)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--compression", type=int, default=4)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--patience", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--objective", choices=["activity", "delta_activity"], default="delta_activity")
    train.add_argument("--absolute-weight", type=float, default=0.1, help="Activity-loss weight for the combined objective; zero uses only paired effects.")
    train.add_argument("--without-skip", action="store_true", help="Remove the full-resolution encoder-to-decoder connection.")
    train.add_argument("--fusion", choices=["sum", "rms_balanced"], default="sum")
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=0)
    predict = commands.add_parser("predict", help="Predict paired effects from a local checkpoint.")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--batch-size", type=int, default=32)
    evaluate = commands.add_parser("evaluate", help="Evaluate predictions using explicit or validation-fitted thresholds.")
    evaluate.add_argument("--input", type=Path, required=True)
    threshold = evaluate.add_mutually_exclusive_group(required=True)
    threshold.add_argument("--thresholds", type=Path, help="JSON containing an EvaluationThresholds configuration.")
    threshold.add_argument("--validation-predictions", type=Path, help="Separate labelled validation predictions for fitting thresholds.")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=0, help="Component bootstrap draws; zero returns point estimates.")
    evaluate.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.command == "generate":
        from .synthetic import toy_pairs
        save_table(toy_pairs(args.pairs, args.length, args.seed), args.output)
        print(f"Wrote {args.pairs} synthetic pairs to {args.output}")
        return

    if args.command == "evaluate":
        from .metrics import EvaluationThresholds, evaluate_predictions, fit_thresholds
        from .training import write_json
        thresholds = (EvaluationThresholds.from_dict(json.loads(args.thresholds.read_text()))
                      if args.thresholds else fit_thresholds(read_table(args.validation_predictions)))
        result = evaluate_predictions(read_table(args.input), thresholds, n_bootstrap=args.bootstrap, seed=args.seed)
        write_json(args.output, result)
        print(f"Wrote evaluation to {args.output}")
        return

    from .model import ModelConfig, build_model
    from .training import TrainConfig, load_model, predict as predict_pairs, train_model, write_json
    if args.command == "demo":
        import torch
        from .synthetic import toy_pairs
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        pairs = toy_pairs(args.pairs, args.length, args.seed)
        config = ModelConfig(mode="aggregate", dim=16, depth=1, heads=2, compression=4,
                             encoder_depth=1, decoder_depth=1, kernel_size=3)
        model = build_model(config)
        predictions = predict_pairs(model, pairs, TrainConfig(device="cpu", batch_size=8, amp=False))
        save_table(pairs, args.output/"pairs.tsv")
        save_table(predictions, args.output/"predictions.tsv")
        write_json(args.output/"example.json", {"provenance":"Independent synthetic DNA with analytic GC-balance labels",
                   "model":"Randomly initialized demonstration model", "seed":args.seed,
                   "pairs":len(pairs), "model_config":asdict(config)})
        print(f"Wrote the synthetic example to {args.output}")
    elif args.command == "train":
        config = ModelConfig(mode=args.mode, dim=args.dim, depth=args.depth, heads=args.heads,
                             compression=args.compression, external_skip=not args.without_skip, branch_fusion=args.fusion)
        training = TrainConfig(device=args.device, seed=args.seed, epochs=args.epochs,
                               min_epochs=min(5, args.epochs), patience=args.patience,
                               batch_size=args.batch_size, learning_rate=args.learning_rate,
                               objective=args.objective, absolute_weight=args.absolute_weight, amp=args.device.startswith("cuda"))
        train_model(config, training, read_table(args.train), read_table(args.validation), args.output)
        print(f"Wrote the trained model and validation outputs to {args.output}")
    elif args.command == "predict":
        model, config = load_model(args.checkpoint, device=args.device)
        config.batch_size = args.batch_size
        config.amp = args.device.startswith("cuda")
        save_table(predict_pairs(model, read_table(args.input), config), args.output)
        print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
