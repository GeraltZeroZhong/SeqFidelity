"""Independent toy DNA pairs with analytically defined activity labels."""
from __future__ import annotations

import numpy as np
import pandas as pd


def toy_pairs(n_pairs: int = 32, length: int = 32, seed: int = 0) -> pd.DataFrame:
    """Generate single-base substitutions and a GC-balance activity difference.

    Activity is (number of G bases - number of C bases) / sequence length.
    All sequences and identifiers are generated here, independently of observations.
    """
    if n_pairs < 1 or length < 1:
        raise ValueError("n_pairs and length must be positive")
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"))
    rows = []
    for i in range(n_pairs):
        bases = rng.integers(0, 4, size=length)
        position = int(rng.integers(length))
        alternative = bases.copy()
        alternative[position] = (alternative[position] + int(rng.integers(1, 4))) % 4
        ref_seq = "".join(alphabet[bases])
        alt_seq = "".join(alphabet[alternative])
        y_ref = (ref_seq.count("G") - ref_seq.count("C")) / length
        y_alt = (alt_seq.count("G") - alt_seq.count("C")) / length
        rows.append({"pair_id": f"toy-{i:04d}", "locus_id": f"toy-group-{i // 4:03d}",
                     "ref_seq": ref_seq, "alt_seq": alt_seq, "delta": y_alt-y_ref,
                     "y_ref": y_ref, "y_alt": y_alt})
    return pd.DataFrame(rows)
