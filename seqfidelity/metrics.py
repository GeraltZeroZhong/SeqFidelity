"""Variant-effect endpoints with the locus, not the row, as sampling unit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class EvaluationThresholds:
    strong_abs_delta: float
    screening_fraction: float = 0.1
    tail_quantile: float = 0.95
    strong_quantile: float = 0.8
    fitted_on: str = "validation"
    validation_rows: int = 0
    quality_definition: str = "caller-supplied quality_pass; unknown remains unknown"
    significance_max: float = 0.05
    se_z: float = 1.96

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "EvaluationThresholds":
        return cls(**values)


Thresholds = EvaluationThresholds


def fit_thresholds(
    validation: pd.DataFrame,
    strong_quantile: float = 0.8,
    screening_fraction: float = 0.1,
    tail_quantile: float = 0.95,
    quality_definition: str = "caller-supplied quality_pass; unknown remains unknown",
) -> EvaluationThresholds:
    """Lock effect magnitude and screening rules using validation labels only.

    A large magnitude is not itself evidence of measurement reliability. The
    caller supplies source-grounded quality_pass flags; P values are never
    converted to uncertainty estimates or quality flags by this module.
    """
    if "split" in validation and not validation["split"].isin(["validation", "val"]).all():
        raise ValueError("Threshold fitting accepts validation rows only")
    if not 0 < screening_fraction <= 1 or not 0 < tail_quantile < 1:
        raise ValueError("Invalid screening fraction or tail quantile")
    column = "delta_true" if "delta_true" in validation else "delta"
    truth = pd.to_numeric(validation[column], errors="coerce").to_numpy(float)
    truth = truth[np.isfinite(truth)]
    if not len(truth):
        raise ValueError("No finite validation labels for threshold selection")
    return EvaluationThresholds(
        strong_abs_delta=float(np.quantile(np.abs(truth), strong_quantile)),
        screening_fraction=screening_fraction,
        tail_quantile=tail_quantile,
        strong_quantile=strong_quantile,
        validation_rows=len(truth),
        quality_definition=quality_definition,
    )


def _correlation(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(truth) < 2 or np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        return None
    return float(spearmanr(truth, prediction).statistic)


def _quality_mask(frame: pd.DataFrame) -> pd.Series:
    # Explicit comparisons avoid treating NaN or the string "False" as true.
    if "quality_pass" not in frame:
        return pd.Series(False, index=frame.index)
    return frame["quality_pass"].eq(True) | frame["quality_pass"].astype(str).str.lower().isin(["true", "1"])


def _reliability_mask(frame: pd.DataFrame, thresholds: EvaluationThresholds) -> tuple[np.ndarray, bool]:
    supported = np.zeros(len(frame), dtype=bool)
    known = np.zeros(len(frame), dtype=bool)
    for name in ["q_value", "p_value"]:
        if name in frame:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            available = np.isfinite(values) & (values >= 0) & (values <= 1) & ~known
            supported[available] = values[available] <= thresholds.significance_max
            known |= available
    if "se" in frame:
        se = pd.to_numeric(frame.se, errors="coerce").to_numpy(float)
        available = np.isfinite(se) & (se >= 0) & ~known
        supported[available] = np.abs(frame.delta_true.to_numpy(float)[available]) > thresholds.se_z * se[available]
        known |= available
    quality_known = "quality_pass" in frame and frame.quality_pass.notna().any()
    return supported & _quality_mask(frame).to_numpy(), bool(known.any() and quality_known)


def _finite_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for name in ["delta_true", "delta_pred"]:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    finite_truth = np.isfinite(result.delta_true)
    if (finite_truth & ~np.isfinite(result.delta_pred)).any():
        raise ValueError("Nonfinite model predictions for eligible measured effects invalidate evaluation")
    result = result.loc[finite_truth].copy()
    if result.locus_id.isna().any():
        raise ValueError("Locus identifiers are required for every evaluated pair")
    result["locus_id"] = result.locus_id.astype(str)
    return result.reset_index(drop=True)


def _large_effect(
    frame: pd.DataFrame, thresholds: EvaluationThresholds, selected: np.ndarray,
    reliable: bool = False, high_quality: bool = False,
) -> dict[str, Any]:
    truth = frame.delta_true.to_numpy(float)
    prediction = frame.delta_pred.to_numpy(float)
    large = (np.abs(truth) >= thresholds.strong_abs_delta) & (truth != 0)
    if reliable:
        large &= _reliability_mask(frame, thresholds)[0]
    if high_quality:
        large &= _quality_mask(frame).to_numpy()
    if not large.any():
        return {"n": 0, "n_loci": 0, "sign_accuracy": None, "macro_sign_accuracy": None,
                "recall_at_budget": None, "macro_recall_at_budget": None, "delta_mse": None,
                "delta_mae": None, "absolute_error_p95": None}
    correct = np.sign(truth) == np.sign(prediction)
    local = pd.DataFrame({"locus": frame.locus_id[large].to_numpy(),
                          "correct": correct[large], "selected": selected[large]})
    per_locus = local.groupby("locus", sort=False)[["correct", "selected"]].mean()
    errors = prediction[large] - truth[large]
    return {"n": int(large.sum()), "n_loci": len(per_locus),
            "sign_accuracy": float(correct[large].mean()),
            "macro_sign_accuracy": float(per_locus.correct.mean()),
            "recall_at_budget": float(selected[large].mean()),
            "macro_recall_at_budget": float(per_locus.selected.mean()),
            "delta_mse": float(np.mean(errors ** 2)),
            "delta_mae": float(np.mean(np.abs(errors))),
            "absolute_error_p95": float(np.quantile(np.abs(errors), .95))}


def _locus_table(frame: pd.DataFrame, thresholds: EvaluationThresholds) -> pd.DataFrame:
    rows = []
    for locus, group in frame.groupby("locus_id", sort=False):
        true, pred = group.delta_true.to_numpy(float), group.delta_pred.to_numpy(float)
        ae = np.abs(pred - true)
        q = float(np.quantile(ae, thresholds.tail_quantile))
        rows.append({"locus_id": locus, "n": len(group), "delta_mse": np.mean(ae ** 2),
                     "delta_mae": np.mean(ae), "spearman": _correlation(true, pred),
                     "absolute_error_p95": np.quantile(ae, .95),
                     "tail_mean_absolute_error": np.mean(ae[ae >= q])})
    return pd.DataFrame(rows)


def _bootstrap_means(values: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, Any]:
    if len(values) < 2 or n_bootstrap <= 0:
        return {"low": None, "high": None, "n_resamples": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    # One vector per draw keeps memory bounded for large independent-locus sets.
    for iteration in range(n_bootstrap):
        means[iteration] = values[rng.integers(0, len(values), len(values))].mean()
    return {"low": float(np.quantile(means, .025)), "high": float(np.quantile(means, .975)),
            "n_resamples": n_bootstrap}


def _evaluate_subset(frame: pd.DataFrame, thresholds: EvaluationThresholds,
                     n_bootstrap: int, seed: int) -> dict[str, Any]:
    if frame.empty:
        return {"n": 0, "n_loci": 0, "status": "no_eligible_rows"}
    truth, prediction = frame.delta_true.to_numpy(float), frame.delta_pred.to_numpy(float)
    error = prediction - truth
    absolute_error = np.abs(error)
    budget = max(1, int(np.ceil(thresholds.screening_fraction * len(frame))))
    # Stable order makes tied model scores deterministic without consulting labels.
    order = np.argsort(-np.abs(prediction), kind="stable")
    selected = np.zeros(len(frame), dtype=bool)
    selected[order[:budget]] = True
    locus_table = _locus_table(frame, thresholds)
    correlations = locus_table.spearman.dropna()
    tail_cutoff = float(np.quantile(absolute_error, thresholds.tail_quantile))
    cis = {}
    for name in ["delta_mse", "delta_mae"]:
        cis[f"macro_{name}"] = _bootstrap_means(locus_table[name].to_numpy(), n_bootstrap, seed)
    _, reliability_known = _reliability_mask(frame, thresholds)
    cis.update({"unit": "locus", "level": .95, "method": "percentile cluster bootstrap",
                "n_independent_units": len(locus_table)})
    quality_known = "quality_pass" in frame and frame.quality_pass.notna().any()
    return {"n": len(frame), "n_loci": len(locus_table),
            "macro_delta_mse": float(locus_table.delta_mse.mean()),
            "macro_delta_mae": float(locus_table.delta_mae.mean()),
            "micro_delta_mse": float(np.mean(error ** 2)), "micro_delta_mae": float(absolute_error.mean()),
            "spearman": _correlation(truth, prediction),
            "macro_spearman": float(correlations.mean()) if len(correlations) else None,
            "n_loci_with_defined_spearman": len(correlations),
            "absolute_error_p95": float(np.quantile(absolute_error, .95)),
            "absolute_error_max": float(absolute_error.max()),
            "tail_mean_absolute_error": float(absolute_error[absolute_error >= tail_cutoff].mean()),
            "macro_tail_mean_absolute_error": float(locus_table.tail_mean_absolute_error.mean()),
            "screening_budget": budget, "screening_fraction": thresholds.screening_fraction,
            "large_effect": _large_effect(frame, thresholds, selected),
            "high_quality_large": _large_effect(frame, thresholds, selected, high_quality=True) if quality_known else None,
            "reliable_large": _large_effect(frame, thresholds, selected, reliable=True) if reliability_known else None,
            "quality_known_rows": int(frame.quality_pass.notna().sum()) if quality_known else 0,
            "confidence_intervals": cis}


def evaluate_predictions(frame: pd.DataFrame, thresholds: EvaluationThresholds | dict,
                         n_bootstrap: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Evaluate all rows, source-defined quality strata, and individual elements.

    Confidence intervals resample independent locus means. Multiple variants
    assigned to one locus remain one sampling unit.
    """
    if isinstance(thresholds, dict):
        thresholds = EvaluationThresholds.from_dict(thresholds)
    eligible = _finite_frame(frame)
    overall = _evaluate_subset(eligible, thresholds, n_bootstrap, seed)
    high_quality = eligible.loc[_quality_mask(eligible)].reset_index(drop=True)
    elements = {}
    if "element" in eligible:
        for element, group in eligible.dropna(subset=["element"]).groupby("element", sort=True):
            elements[str(element)] = _evaluate_subset(group.reset_index(drop=True), thresholds, n_bootstrap, seed)
    result = {"all": overall,
              "high_quality": _evaluate_subset(high_quality, thresholds, n_bootstrap, seed),
              "per_element": elements, "thresholds": thresholds.to_dict(),
              "excluded_nonfinite_rows": len(frame) - len(eligible),
              "uncertainty_note": "Bootstrap resamples complete independent loci; variants within a locus share a sampling unit.",
              "quality_note": "Reliable_large requires quality_pass, the locked magnitude threshold, and supplied q<=alpha (preferred), p<=alpha, or a nonzero normal-approximation interval from an actual SE if neither q nor p exists. No SE is inferred from P values."}
    for name in ["macro_delta_mse", "macro_delta_mae", "micro_delta_mse", "micro_delta_mae", "spearman"]:
        result[name] = overall.get(name)
    return result


def per_locus_metrics(frame: pd.DataFrame, thresholds: EvaluationThresholds) -> pd.DataFrame:
    return _locus_table(_finite_frame(frame), thresholds)


def paired_comparison(frame: pd.DataFrame, candidate: str, reference: str = "delta_pred",
                      truth: str = "delta_true", n_bootstrap: int = 1000,
                      seed: int = 0) -> dict[str, Any]:
    """Paired locus CI; a negative difference favors the candidate intervention."""
    work = frame[["locus_id", truth, candidate, reference]].dropna().copy()
    finite = np.isfinite(work[[truth, candidate, reference]].to_numpy(float)).all(axis=1)
    work = work.loc[finite]
    result: dict[str, Any] = {"n": len(work), "n_loci": work.locus_id.nunique(),
                              "candidate": candidate, "reference": reference,
                              "direction": "negative favors candidate", "bootstrap_unit": "locus"}
    for metric, power in [("delta_mse", 2), ("delta_mae", 1)]:
        diff = np.abs(work[candidate] - work[truth]) ** power - np.abs(work[reference] - work[truth]) ** power
        grouped = diff.groupby(work.locus_id, sort=False).mean().to_numpy()
        result[f"macro_{metric}_difference"] = float(grouped.mean()) if len(grouped) else None
        result[f"macro_{metric}_difference_ci95"] = _bootstrap_means(grouped, n_bootstrap, seed)
    return result


def absolute_error_decomposition(y_ref, y_alt, pred_ref, pred_alt) -> dict[str, float]:
    """Population-moment identity for paired errors, retaining covariance."""
    er = np.asarray(pred_ref, float) - np.asarray(y_ref, float)
    ea = np.asarray(pred_alt, float) - np.asarray(y_alt, float)
    valid = np.isfinite(er) & np.isfinite(ea)
    er, ea = er[valid], ea[valid]
    if not len(er):
        return {"n": 0}
    covariance = np.mean((er - er.mean()) * (ea - ea.mean()))
    return {"n": len(er), "target_definition": "Difference of the provided absolute activity labels; may differ from the separately supplied paired-contrast effect endpoint.", "var_error_ref": float(er.var()), "var_error_alt": float(ea.var()),
            "cov_error_ref_alt": float(covariance), "squared_bias_difference": float((ea.mean() - er.mean()) ** 2),
            "delta_mse": float(np.mean((ea - er) ** 2)),
            "decomposed_delta_mse": float(er.var() + ea.var() - 2 * covariance + (ea.mean() - er.mean()) ** 2)}
