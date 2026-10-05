# SeqFidelity

SeqFidelity provides PyTorch sequence models and tools for paired DNA effect
prediction. It includes configurable compression, explicit routing state,
full-resolution skip connections, component-aware evaluation, and frozen
representation readouts.

The package accepts equal-length reference and alternate sequences within each
pair. Different pairs can have different lengths. A shared scalar head scores
each sequence, and the predicted effect is the alternate score minus the
reference score. Sequence length masks separate valid nucleotides from padding.

## Installation

Use Python 3.10 or newer and a compatible PyTorch installation. Install the
tagged release from GitHub:

```bash
python -m pip install 'seqfidelity[io] @ git+https://github.com/GeraltZeroZhong/SeqFidelity.git@v0.1.0'
```

For development, clone the repository and install from its root:

```bash
git clone https://github.com/GeraltZeroZhong/SeqFidelity.git
cd SeqFidelity
python -m pip install -e '.[test,io]'
```

Core dependencies are PyTorch, NumPy, pandas, and SciPy. The `io` extra adds
PyArrow for Parquet-backed feature and readout workflows. CPU execution is the
default. Select an available device explicitly when using another backend.

## Synthetic quick start

```bash
python -m seqfidelity demo --output output/demo
```

This creates independently generated DNA pairs, analytic toy activity labels,
and predictions from a small randomly initialized model. It writes `pairs.tsv`,
`predictions.tsv`, and `example.json` to the selected directory. These outputs
illustrate the interface; numerical values describe this toy example.

A train-and-predict example uses separate synthetic inputs:

```bash
seqfidelity generate --output output/train.tsv --pairs 32 --length 32 --seed 0
seqfidelity generate --output output/validation.tsv --pairs 16 --length 32 --seed 1
seqfidelity train --train output/train.tsv --validation output/validation.tsv --output output/model --epochs 1 --dim 16 --depth 1 --heads 2 --batch-size 8
seqfidelity predict --checkpoint output/model/checkpoints/best.pt --input output/validation.tsv --output output/predictions.tsv
```

Training saves checkpoints, configuration, training history, and validation
predictions beneath the chosen output directory. Treat checkpoints as trusted
local files when loading them with PyTorch.

## Input and output tables

CSV and TSV tables use the following fields:

| Field | Meaning | Required for |
| --- | --- | --- |
| `ref_seq`, `alt_seq` | Nonempty DNA strings using A, C, G, T, N; letter case is normalized | Prediction and training |
| `pair_id` | A unique pair identifier | Matched comparisons and traceable evaluation |
| `locus_id` | A grouping identifier for component-weighted metrics | Training selection and evaluation |
| `delta` | Alternate-minus-reference effect label in user-defined units | Training and validation |
| `y_ref`, `y_alt` | Individual sequence activity labels in consistent units | Activity supervision |

Prediction retains supplied metadata and writes `delta_pred`, `y_ref_pred`,
`y_alt_pred`, and `delta_true` when labels are supplied. Routed models also write
`k_ref` and `k_alt`, the retained token counts. A missing activity label can be
represented by a missing value for the combined objective. Activity-only fitting
requires available activity labels.

Evaluation reads `delta_true`, `delta_pred`, and `locus_id`. Matched model
comparisons also require `pair_id`.
Optional quality, standard-error, and significance fields support reliable-effect
summaries. See [input schemas](docs/input_schema.md) for the exact interpretation.

## Model and analysis APIs

```python
from seqfidelity.model import ModelConfig, build_model
from seqfidelity.synthetic import toy_pairs
from seqfidelity.training import TrainConfig, predict

pairs = toy_pairs(n_pairs=8, length=24, seed=0)
model = build_model(ModelConfig(mode="aggregate", dim=16, depth=1, heads=2))
predictions = predict(model, pairs, TrainConfig(device="cpu", amp=False))
```

Compression modes include full resolution, fixed sampling, fixed interval
aggregation, and learned routing. The configuration also exposes shared-route,
coupled-route, backbone-bypass, and convolutional controls. The retained token
budget follows each mode's documented rule. `external_skip` controls the
encoder-to-decoder connection; internal residual operations retain their defined
behavior. See the [API guide](docs/api.md) for architecture and analysis details.

Evaluation thresholds are supplied explicitly or fitted on a separate validation
prediction table:

```bash
seqfidelity evaluate --input output/predictions.tsv --validation-predictions output/model/validation_predictions.tsv --output output/metrics.json --bootstrap 0
```

Use validation and evaluation tables with distinct evaluation roles in an actual
application. The command above demonstrates file interfaces on synthetic data.
Component bootstrap intervals describe uncertainty over the supplied components
conditional on fitted models. Assess training-seed variation through repeated
training runs.

## Documentation and releases

- [API guide](docs/api.md): models, training, evaluation, and representation tools.
- [Input schemas](docs/input_schema.md): sequence, label, and prediction fields.
- [Test guide](docs/testing.md): synthetic fixture provenance and checks.
- [Changelog](CHANGELOG.md): versioned software changes.
- [Releases](https://github.com/GeraltZeroZhong/SeqFidelity/releases): tagged source
  and installable distributions.

The distribution contains reusable software and independently generated examples.
Model fitting and evaluation use inputs and output locations supplied by the user.

## Tests and examples

```bash
python -m pytest
python examples/paired_effects.py --output output/api-example.tsv
```

Tests use independently constructed synthetic tensors, sequences, and tables.
They cover routing and padding invariants, paired effects, losses, weighted
metrics, matched comparisons, and representation diagnostics. Synthetic fixtures
have explicit provenance in [the test guide](docs/testing.md).

## Attribution

The routing implementation is inspired by the
[H-Net architecture](https://github.com/goombalab/hnet). It uses a paired,
bidirectional implementation with configurable token selection and decompression.
The LegNet baseline includes code adapted from `human_legnet`;
its original MIT license text is preserved in `licenses/human_legnet-MIT.txt`,
with attribution in `NOTICE`.

See [contribution guidance](CONTRIBUTING.md) for development conventions.

## License

MIT License, copyright (c) 2026 Shuyu Zhong. See [LICENSE](LICENSE).
Third-party attribution and retained notices are listed in [NOTICE](NOTICE).
