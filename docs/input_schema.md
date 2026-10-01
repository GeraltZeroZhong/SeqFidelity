# Input schemas and metric conventions

## Sequence pairs

`ref_seq` and `alt_seq` contain nonempty ACGTN strings. Both alleles of one pair
have equal length; batches may contain pairs of different lengths. Encoding maps
A/C/G/T/N to 0/1/2/3/4, and right padding also uses 4. Length masks distinguish
padding from an N inside a valid sequence. Unsupported symbols raise an error.

`delta` is the supervised paired effect. Supply it directly when an estimator
provides a paired effect: it need not equal separately estimated `y_alt - y_ref`.
The model output uses alternate-minus-reference direction. Activity-only and
combined supervision operate on the optional `y_ref` and `y_alt` values. Missing
activity values are masked.

Use `pair_id` to identify matching observations and `locus_id` to define the unit
that should receive equal weight. Identifiers are user-provided opaque strings.
For inference, individual sequence labels may be omitted. Training and validation
require finite `delta` labels and nonmissing `locus_id` values.

## Prediction tables

`predict` retains input metadata, removes the sequence strings, and adds:

| Column | Definition |
| --- | --- |
| `delta_true` | Supplied `delta`, when present |
| `delta_pred` | Predicted alternate score minus reference score |
| `y_ref_pred`, `y_alt_pred` | Predicted scalar activity for each allele |
| `k_ref`, `k_alt` | Selected token counts when the architecture exposes routes |

Evaluation requires `delta_true`, `delta_pred`, and `locus_id`. Matched model
comparisons also require `pair_id`. All
models in a matched comparison must use the same complete pair set and compatible
labels and grouping metadata.

## Quality and reliability

Optional `quality_pass` describes an upstream quality decision. Missing quality
information remains unknown. `q_value`, `p_value`, and `se` may describe a formal
paired estimate and its uncertainty. When present, reliability first uses a
finite q-value, then a finite p-value, and finally an effect-versus-standard-error
criterion. The `EvaluationThresholds` configuration controls these cutoffs and the
strong-effect definition. Threshold fitting uses a separately supplied validation
table; preserve the fitted configuration when comparing models.

`macro_delta_mse` first averages squared errors within each `locus_id`, then gives
each locus equal weight. Row-level summaries instead weight each observed pair.
Conditional metrics use their defined eligible target sets; undefined correlations
or empty strata retain their missing-value status. Specify the desired number of
component bootstrap draws explicitly for interval estimation.

Reliable strong-effect endpoints require all three conditions: an effect meeting
the magnitude threshold, a true `quality_pass` flag, and supporting q-value,
p-value, or standard-error evidence. Missing quality flags remain unknown and
therefore do not qualify for these conditional endpoints.
