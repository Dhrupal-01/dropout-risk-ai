"""
Generalized Algorithmic Fairness and Bias Audit Engine
Context: Empirical benchmarking (UCI ID 697, OULAD UCI ID 349) and simulated cohort validation.

Audits out-of-fold or holdout predictions against an isolated audit frame of protected attributes
that were strictly excluded from model feature matrices.

For each demographic group:
- n_samples (count)
- base_rate (observed positive prevalence)
- selection_rate at top-20% predicted risk threshold
- True Positive Rate (TPR / Recall)
- False Negative Rate (FNR / Miss Rate)
- False Positive Rate (FPR / Alarm Rate)
- Precision
- Within-group Expected Calibration Error (ECE across 10 equal bins)
- Disparity gaps vs reference group with 95% bootstrap confidence intervals (1,000 resamples)
- Small-sample safeguard: groups with n < 50 are flagged 'insufficient_sample' with gap suppressed
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

MINIMUM_AUDIT_SAMPLE_SIZE = 50


def compute_within_group_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) within a demographic subgroup
    partitioned across n_bins equal intervals in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low_b, high_b = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low_b) & (y_prob <= high_b)
        else:
            mask = (y_prob >= low_b) & (y_prob < high_b)

        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


def compute_group_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes core fairness and performance metrics for a specific group slice.
    """
    n_samples = int(len(y_true))
    if n_samples == 0:
        return {
            "n_samples": 0,
            "insufficient_sample": True,
            "status": "insufficient_sample",
        }

    actual_positives = int(np.sum(y_true == 1))
    actual_negatives = int(np.sum(y_true == 0))
    predicted_positives = int(np.sum(y_pred == 1))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    tpr = float(tp / actual_positives) if actual_positives > 0 else 0.0
    fnr = float(fn / actual_positives) if actual_positives > 0 else 0.0
    fpr = float(fp / actual_negatives) if actual_negatives > 0 else 0.0
    prec = float(tp / predicted_positives) if predicted_positives > 0 else 0.0
    base_rate = float(actual_positives / n_samples)
    selection_rate = float(predicted_positives / n_samples)
    ece = compute_within_group_ece(y_true, y_prob)

    is_small = n_samples < MINIMUM_AUDIT_SAMPLE_SIZE

    return {
        "n_samples": n_samples,
        "insufficient_sample": is_small,
        "status": "insufficient_sample" if is_small else "ok",
        "base_rate": round(base_rate, 4),
        "selection_rate": round(selection_rate, 4),
        "true_positives": tp,
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positives": fp,
        "tpr_recall": round(tpr, 4),
        "fnr_miss_rate": round(fnr, 4),
        "fpr_alarm_rate": round(fpr, 4),
        "precision": round(prec, 4),
        "within_group_ece": round(ece, 4),
    }


def audit_single_attribute(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    attr_series: pd.Series,
    attribute_name: str,
    reference_group: Optional[str] = None,
    top_k_pct: float = 0.20,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Audits an individual sensitive attribute across all demographic categories.
    Computes top-20% threshold, group metrics, gaps vs reference group, and 95% bootstrap CIs.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_total = len(y_true)

    # Determine selection threshold at top_k_pct (default top 20% highest risk)
    threshold = float(np.quantile(y_prob, 1.0 - top_k_pct))
    y_pred = (y_prob >= threshold).astype(int)

    # Clean attribute series (convert to string and drop NaN)
    valid_mask = attr_series.notna().values
    y_true_v = y_true[valid_mask]
    y_prob_v = y_prob[valid_mask]
    y_pred_v = y_pred[valid_mask]
    attr_v = attr_series[valid_mask].astype(str).values

    unique_groups = sorted(list(set(attr_v)))
    if not unique_groups:
        return {"attribute": attribute_name, "error": "No valid group values found"}

    # Assign reference group (default to largest category if not specified)
    if reference_group is None or reference_group not in unique_groups:
        group_counts = {g: int(np.sum(attr_v == g)) for g in unique_groups}
        reference_group = max(group_counts, key=group_counts.get)

    ref_mask = attr_v == reference_group
    ref_metrics = compute_group_metrics(y_true_v[ref_mask], y_prob_v[ref_mask], y_pred_v[ref_mask])
    ref_insufficient = ref_metrics.get("insufficient_sample", True)

    groups_summary: Dict[str, Any] = {}
    gaps_summary: Dict[str, Any] = {}

    for g in unique_groups:
        g_mask = attr_v == g
        m = compute_group_metrics(y_true_v[g_mask], y_prob_v[g_mask], y_pred_v[g_mask])
        groups_summary[g] = m

        if g == reference_group:
            gaps_summary[g] = {
                "is_reference": True,
                "status": "reference_group",
                "fnr_gap": 0.0,
                "tpr_gap": 0.0,
                "selection_rate_gap": 0.0,
                "disparate_impact_ratio": 1.0,
            }
            continue

        if m["insufficient_sample"] or ref_insufficient:
            gaps_summary[g] = {
                "is_reference": False,
                "status": "insufficient_sample",
                "note": "insufficient_sample (n < 50)",
                "reason": f"Sample size (group={m['n_samples']}, ref={ref_metrics['n_samples']}) < {MINIMUM_AUDIT_SAMPLE_SIZE}",
                "fnr_gap": None,
                "fnr_gap_95ci": None,
                "tpr_gap": None,
                "selection_rate_gap": None,
                "disparate_impact_ratio": None,
            }
        else:
            fnr_gap = round(float(m["fnr_miss_rate"] - ref_metrics["fnr_miss_rate"]), 4)
            tpr_gap = round(float(m["tpr_recall"] - ref_metrics["tpr_recall"]), 4)
            sel_gap = round(float(m["selection_rate"] - ref_metrics["selection_rate"]), 4)
            disp_impact = round(float(m["selection_rate"] / max(ref_metrics["selection_rate"], 1e-5)), 4)
            gaps_summary[g] = {
                "is_reference": False,
                "status": "ok",
                "fnr_gap": fnr_gap,
                "fnr_gap_95ci": None,
                "tpr_gap": tpr_gap,
                "selection_rate_gap": sel_gap,
                "disparate_impact_ratio": disp_impact,
            }

    # Deterministic Bootstrap Confidence Intervals for groups and gaps (n >= 50)
    boot_ci_results = compute_bootstrap_fairness_cis(
        y_true_v=y_true_v,
        y_prob_v=y_prob_v,
        attr_v=attr_v,
        threshold=threshold,
        unique_groups=unique_groups,
        reference_group=reference_group,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # Attach CIs to groups and gaps
    for g, g_cis in boot_ci_results["groups"].items():
        if g in groups_summary:
            groups_summary[g]["confidence_intervals_95"] = g_cis

    for g, gap_cis in boot_ci_results["gaps"].items():
        if g in gaps_summary and gaps_summary[g].get("status") == "ok":
            gaps_summary[g]["confidence_intervals_95"] = gap_cis
            gaps_summary[g]["fnr_gap_95ci"] = gap_cis.get("fnr_gap")

    return {
        "attribute_name": attribute_name,
        "reference_group": reference_group,
        "n_total": n_total,
        "n_audited": len(y_true_v),
        "top_k_pct": top_k_pct,
        "selection_threshold": round(threshold, 4),
        "groups": groups_summary,
        "gaps_vs_reference": gaps_summary,
    }


def compute_bootstrap_fairness_cis(
    y_true_v: np.ndarray,
    y_prob_v: np.ndarray,
    attr_v: np.ndarray,
    threshold: float,
    unique_groups: List[str],
    reference_group: str,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes 95% bootstrap confidence intervals across n_bootstraps resamples with replacement.
    Fully deterministic given seed.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true_v)

    # Determine which groups qualify for CIs (n >= 50)
    qualifying_groups = {g for g in unique_groups if np.sum(attr_v == g) >= MINIMUM_AUDIT_SAMPLE_SIZE}
    ref_qualifies = reference_group in qualifying_groups

    boot_group_metrics: Dict[str, Dict[str, List[float]]] = {
        g: {"tpr": [], "fnr": [], "fpr": [], "selection_rate": [], "ece": []}
        for g in qualifying_groups
    }
    boot_gap_metrics: Dict[str, Dict[str, List[float]]] = {
        g: {"fnr_gap": [], "tpr_gap": [], "selection_rate_gap": [], "disparate_impact": []}
        for g in qualifying_groups if g != reference_group and ref_qualifies
    }

    attempts = 0
    max_attempts = n_bootstraps * 3
    valid_boots = 0

    while valid_boots < n_bootstraps and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        y_b = y_true_v[idx]
        p_b = y_prob_v[idx]
        pred_b = (p_b >= threshold).astype(int)
        a_b = attr_v[idx]

        # Ensure both classes exist
        if len(np.unique(y_b)) < 2:
            continue

        ref_fnr, ref_tpr, ref_sel = None, None, None
        if ref_qualifies:
            ref_idx = a_b == reference_group
            if np.sum(ref_idx) > 5 and np.sum(y_b[ref_idx] == 1) > 0:
                ref_m = compute_group_metrics(y_b[ref_idx], p_b[ref_idx], pred_b[ref_idx])
                ref_fnr = ref_m["fnr_miss_rate"]
                ref_tpr = ref_m["tpr_recall"]
                ref_sel = ref_m["selection_rate"]

        for g in qualifying_groups:
            g_idx = a_b == g
            if np.sum(g_idx) < 5 or np.sum(y_b[g_idx] == 1) == 0:
                continue
            gm = compute_group_metrics(y_b[g_idx], p_b[g_idx], pred_b[g_idx])
            boot_group_metrics[g]["tpr"].append(gm["tpr_recall"])
            boot_group_metrics[g]["fnr"].append(gm["fnr_miss_rate"])
            boot_group_metrics[g]["fpr"].append(gm["fpr_alarm_rate"])
            boot_group_metrics[g]["selection_rate"].append(gm["selection_rate"])
            boot_group_metrics[g]["ece"].append(gm["within_group_ece"])

            if g != reference_group and ref_fnr is not None:
                boot_gap_metrics[g]["fnr_gap"].append(gm["fnr_miss_rate"] - ref_fnr)
                boot_gap_metrics[g]["tpr_gap"].append(gm["tpr_recall"] - ref_tpr)
                boot_gap_metrics[g]["selection_rate_gap"].append(gm["selection_rate"] - ref_sel)
                boot_gap_metrics[g]["disparate_impact"].append(gm["selection_rate"] / max(ref_sel, 1e-5))

        valid_boots += 1

    # Summarize CIs (2.5th and 97.5th percentiles)
    ci_groups: Dict[str, Dict[str, List[float]]] = {}
    for g, m_dict in boot_group_metrics.items():
        ci_groups[g] = {}
        for m_name, vals in m_dict.items():
            if len(vals) >= 100:
                lo = float(np.percentile(vals, 2.5))
                hi = float(np.percentile(vals, 97.5))
                ci_groups[g][m_name] = [round(lo, 4), round(hi, 4)]
            else:
                ci_groups[g][m_name] = [0.0, 0.0]

    ci_gaps: Dict[str, Dict[str, List[float]]] = {}
    for g, m_dict in boot_gap_metrics.items():
        ci_gaps[g] = {}
        for m_name, vals in m_dict.items():
            if len(vals) >= 100:
                lo = float(np.percentile(vals, 2.5))
                hi = float(np.percentile(vals, 97.5))
                ci_gaps[g][m_name] = [round(lo, 4), round(hi, 4)]
            else:
                ci_gaps[g][m_name] = [0.0, 0.0]

    return {"groups": ci_groups, "gaps": ci_gaps}


def audit_model_fairness(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    audit_df: pd.DataFrame,
    attribute_configs: Optional[Dict[str, Optional[str]]] = None,
    top_k_pct: float = 0.20,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes a comprehensive fairness audit across multiple protected attributes.

    Parameters:
    -----------
    y_true: Ground truth binary target array (0 or 1)
    y_prob: Predicted continuous probabilities
    audit_df: Isolated dataframe of protected/sensitive attributes
    attribute_configs: Optional mapping {attribute_col: reference_group}
    top_k_pct: Operational selection percentile (default 0.20 for top 20% highest risk)
    n_bootstraps: Number of bootstrap iterations (default 1,000)
    seed: Random seed for deterministic reproducibility
    """
    if len(y_true) != len(audit_df):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} rows, audit_df has {len(audit_df)} rows")

    attrs_to_audit = attribute_configs or {col: None for col in audit_df.columns}
    results: Dict[str, Any] = {}

    for attr_col, ref_group in attrs_to_audit.items():
        if attr_col not in audit_df.columns:
            logger.warning("Audit attribute '%s' not present in audit_df; skipping.", attr_col)
            continue

        attr_series = audit_df[attr_col]
        audit_res = audit_single_attribute(
            y_true=y_true,
            y_prob=y_prob,
            attr_series=attr_series,
            attribute_name=attr_col,
            reference_group=ref_group,
            top_k_pct=top_k_pct,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
        results[attr_col] = audit_res

    overall_base_rate = float(np.mean(np.asarray(y_true) == 1))
    return {
        "evaluation_type": "fairness_and_bias_audit",
        "top_k_pct": top_k_pct,
        "n_samples": len(y_true),
        "base_rate": round(overall_base_rate, 4),
        "top_k_selection_rate": top_k_pct,
        "n_bootstraps": n_bootstraps,
        "attributes": results,
    }
