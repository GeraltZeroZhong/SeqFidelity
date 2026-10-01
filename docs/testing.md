# Synthetic test guide

The public tests construct inputs independently from short DNA strings, seeded
random tensors, and hand-calculated tables. `seqfidelity.synthetic.toy_pairs`
generates random A/C/G/T sequences, changes one nucleotide per pair, and defines
activity as `(G count - C count) / sequence length`. The alternate-minus-reference
activity difference supplies its effect label.

Run the suite on CPU with:

```bash
python -m pytest
```

The suite covers:

- DNA encoding, reverse complements, input validation, and padded batches.
- Routing budgets, mandatory initial boundaries, monotone coordinates, masking,
  aggregation, skip removal, branch fusion, and gradients.
- Scalar-head paired effects, analytic losses, and normalization behavior.
- Component-weighted metrics, quality eligibility, fixed-budget selection,
  matched comparisons, uncertainty summaries, and risk identities.
- Intermediate-state comparisons, frozen readouts, token pooling, branch
  interventions, and paired antisymmetry.

Expected values come from the constructed inputs or algebraic invariants. Tiny
training checks exercise optimization and checkpoint loading. They establish
software behavior; empirical accuracy requires independently specified datasets,
partitions, trained models, and evaluation protocols.

`examples/paired_effects.py` writes a prediction table from an untrained model.
The `demo` command also records the synthetic generator and model configuration.
Both accept an explicit output destination and use CPU execution by default.
