# API guide

## Models

`seqfidelity.model.ModelConfig` defines hidden width, backbone depth, attention
heads, convolutional kernel size, compression, encoder/decoder depth, dropout,
architecture, routing mode, and branch fusion. Hidden width must be divisible by
the number of heads. Convolutional kernel sizes are odd.

`build_model(config)` creates a `PairedSequenceModel` or the attributed LegNet
baseline. The routed model supports a convolutional control via `mode="cnn"`;
`CompactCNN` is also available as a standalone class. Use `architecture="routed"` for the
routing family and `architecture="legnet"` for the convolutional baseline.

The routing modes are:

| Mode | Behavior |
| --- | --- |
| `none` | Full-resolution tokens |
| `fixed` | Regularly spaced token selection |
| `aggregate` | Fixed interval aggregation at the compression budget |
| `learned` | Context-dependent boundary selection |
| `shared` | Shared route from mean reference/alternate boundary probabilities |
| `coupled` | Per-allele routes from 75% own and 25% partner probabilities |
| `no_backbone` | Encoder and decoder path with the deep backbone bypassed |
| `cnn` | Compact convolutional control |

`RouteState` exposes selected indices, masks, probabilities, coordinates,
sequence lengths, and the decompression mapping. Positions use zero-based input
coordinates. Rebuild derived state with `route_from_probs` when changing selection
indices or boundary probabilities. Mask padded positions in downstream analyses.

`forward_pair(ref, alt, lengths)` returns paired and per-allele predictions. With
feature capture enabled it also exposes intermediate representations for
analysis. `external_skip=False` removes the full-resolution external connection.
Fusion choices are `sum` and `rms_balanced`.

## Training and prediction

`seqfidelity.training.TrainConfig` configures the optimizer, supervision, early
stopping, seed, batch size, augmentation, device, and mixed precision.
`train_model(model_config, config, train_frame, validation_frame, output_dir)`
selects a model using validation component-macro effect MSE and writes local
checkpoints beneath `output_dir/checkpoints`.

`load_model(checkpoint, device="cpu")` returns a model and its saved training
configuration. `predict(model, frame, config)` returns a prediction DataFrame.
`loss_for` combines standardized paired and optional activity errors, or uses the
activity-only objective. Set the activity weight to zero for effect-only loss.

## Evaluation and comparison

`seqfidelity.metrics` provides `EvaluationThresholds`, `fit_thresholds`,
`evaluate_predictions`, paired model comparisons, and error decomposition.
`seqfidelity.comparisons` supplies complete-pair alignment and component-weighted
contrast utilities. Preserve pair identity, labels, and the grouping unit across
model comparisons. Bootstrap resamples components; report its conditioning on
trained models separately from variation across training runs.

## Representation and intervention tools

The package separates reusable analysis by function:

- `diagnostics`: representation capture, local perturbations, and routing diagnostics.
- `precision`: execution-policy comparisons and component-aware summaries.
- `representations`: pooling and target-specific representation heads.
- `states`: masked state comparisons and decompression helpers.
- `interventions`: routing, branch, and gradient interventions.
- `attribution`: fixed-model path-response analysis.
- `probes`: pooled frozen-representation readouts.
- `token_probes`: masked token readouts and positional features.
- `features`: reading and matching local feature caches.

Parquet-backed workflows use the `io` optional dependency. Each workflow accepts
explicit inputs and output locations. Feature coordinates, masks, supervision
scales, and pooling options form part of the readout configuration; preserve them
when comparing representations.

## Command-line interface

Run `seqfidelity --help` or `python -m seqfidelity --help` to list commands.
`demo` and `generate` construct independent synthetic examples. `train`, `predict`,
and `evaluate` accept explicit input and output paths. All commands use CPU by
default. The API exposes the detailed configuration surface used by the models
and diagnostics.

## Pooled feature scales

`diagnostics.pooled_branch_features(features, length_scale=1.0)` returns a
dictionary of pooled tensors for the full, encoder, and backbone representations.
Each row contains `3 * hidden_width + 2` features: masked mean, masked maximum, coordinate-weighted
mean, retained-length fraction, and sequence length divided by `length_scale`.
Coordinate weights use input position divided by `sequence_length - 1`, with a
safe denominator for single-position inputs. Supply a positive scale and keep it
fixed across splits, models, and reused readouts. The same parameter is accepted
by `diagnostics.extract_pair_features` and `probes.cache_representations`; the
cache manifest records it.

Treat each representation cache as a fixed snapshot of its input data and model.
Reuse checks compare checkpoint path and modification time, extraction settings,
split sizes, and ordered pair identifiers. Sequence and label contents are outside
these checks. Choose a fresh cache directory whenever sequences, labels, or model
weights change, including changes that retain the same pair identifiers.

The LegNet architecture requires sequences of at least 16 nucleotides. Its
training batches require equal sequence lengths. Prediction groups sequences by
length before passing them to this architecture.
