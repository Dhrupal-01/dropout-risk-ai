"""
Fairness and Bias Audit Pipeline for DropoutGuard
Context: SIH 2026 PSID 7-L / SDG 4 (Quality Education)

Computes quantitative demographic parity, False Negative Rate (FNR) parity,
and Equal Opportunity metrics across:
1. Gender (Male vs. Female)
2. Economic Proxy (Family Income: <5 LPA vs. >=5 LPA)
3. First-Generation Learner Status (First-Gen vs. Non-First-Gen)
4. Age band (protected, audit frame only): at or below vs above the cohort median age

Outputs real, computed results directly to docs/ethics_and_fairness.md.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ml.config import (
    PROCESSED_DATA_PATH,
    MODEL_ARTIFACT_PATH,
    FEATURE_NAMES_PATH,
    FAIRNESS_METRICS_PATH
)
from ml.fairness.audit import MINIMUM_AUDIT_SAMPLE_SIZE
from ml.models.calibrate import predict_student_risk
from ml.provenance import add_allow_dirty_argument, build_provenance, require_clean_tree, simulated_inputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_group_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Computes confusion matrix and fairness rates for a specific demographic slice.
    """
    y_t = y_true[group_mask]
    y_p = y_pred[group_mask]

    n_samples = int(len(y_t))
    if n_samples == 0:
        return {}

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    actual_positives = tp + fn
    actual_negatives = tn + fp
    predicted_positives = tp + fp

    # True Positive Rate (Recall/Sensitivity) & False Negative Rate (FNR = 1 - TPR)
    tpr = float(tp / actual_positives) if actual_positives > 0 else 0.0
    fnr = float(fn / actual_positives) if actual_positives > 0 else 0.0

    # False Positive Rate (FPR = FP / (FP + TN))
    fpr = float(fp / actual_negatives) if actual_negatives > 0 else 0.0

    # Selection Rate (Demographic Parity metric)
    selection_rate = float(predicted_positives / n_samples)
    base_rate = float(actual_positives / n_samples)

    return {
        "n_samples": n_samples,
        "base_rate": round(base_rate, 4),
        "selection_rate": round(selection_rate, 4),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "tpr_recall": round(tpr, 4),
        "fnr_miss_rate": round(fnr, 4),
        "fpr_alarm_rate": round(fpr, 4)
    }


def compute_age_band_audit(
    audit_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, cut: float
) -> Dict[str, Any]:
    """
    FNR parity by age band. age is protected (never a model feature); it is read from the audit
    frame only. `cut` is the median age of the full simulated cohort, computed by the caller.
    """
    low = compute_group_fairness_metrics(y_true, y_pred, (audit_df["age"] <= cut).values)
    high = compute_group_fairness_metrics(y_true, y_pred, (audit_df["age"] > cut).values)
    sufficient = min(low.get("n_samples", 0), high.get("n_samples", 0)) >= MINIMUM_AUDIT_SAMPLE_SIZE
    return {
        "cut_median_age": round(float(cut), 2),
        "at_or_below_median": low,
        "above_median": high,
        "status": "ok" if sufficient else "insufficient_sample",
        "fnr_disparity": round(abs(low["fnr_miss_rate"] - high["fnr_miss_rate"]), 4) if sufficient else None,
    }


def compute_cross_validated_fairness_audit(n_splits: int = 5) -> Dict[str, Any]:
    """
    Executes 5-fold Stratified Cross-Validation fairness audit across the full cohort (N = 2,000).
    Aggregates out-of-fold predictions to evaluate fairness on a large effective sample (100+ False Negatives),
    verifying whether single-split disparity gaps hold up, shrink, or grow with greater statistical power.
    """
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from ml.config import RANDOM_SEED

    full_df = pd.read_csv(PROCESSED_DATA_PATH)
    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)

    X = full_df[feature_names]
    y = full_df["is_dropout"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof_probs = np.zeros(len(full_df))

    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val = X.iloc[val_idx]

        base_clf = XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.045,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.25,
            reg_alpha=0.15,
            reg_lambda=1.20,
            scale_pos_weight=1.82,
            random_state=RANDOM_SEED,
            eval_metric="logloss"
        )
        calibrated = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
        calibrated.fit(X_tr, y_tr)
        oof_probs[val_idx] = calibrated.predict_proba(X_val)[:, 1]

    oof_preds = (oof_probs >= 0.50).astype(int)

    # 1. Gender Parity
    male_mask = (full_df["gender"] == "Male").values
    female_mask = (full_df["gender"] == "Female").values
    g_m = compute_group_fairness_metrics(y, oof_preds, male_mask)
    g_f = compute_group_fairness_metrics(y, oof_preds, female_mask)
    g_fnr_diff = abs(g_m["fnr_miss_rate"] - g_f["fnr_miss_rate"])
    g_disp_impact = round(g_f["selection_rate"] / max(g_m["selection_rate"], 1e-5), 3)

    # 2. Income Proxy Parity
    low_inc_mask = (full_df["income_slab_idx"] <= 1).values
    high_inc_mask = (full_df["income_slab_idx"] >= 2).values
    inc_l = compute_group_fairness_metrics(y, oof_preds, low_inc_mask)
    inc_h = compute_group_fairness_metrics(y, oof_preds, high_inc_mask)
    inc_fnr_diff = abs(inc_l["fnr_miss_rate"] - inc_h["fnr_miss_rate"])

    # 3. First-Generation Parity
    fg_mask = (full_df["is_first_generation"] == 1).values
    nfg_mask = (full_df["is_first_generation"] == 0).values
    fg_y = compute_group_fairness_metrics(y, oof_preds, fg_mask)
    fg_n = compute_group_fairness_metrics(y, oof_preds, nfg_mask)
    fg_fnr_diff = abs(fg_y["fnr_miss_rate"] - fg_n["fnr_miss_rate"])

    # 4. Age band (median split of the full cohort)
    age_band = compute_age_band_audit(full_df, y, oof_preds, float(full_df["age"].median()))

    return {
        "evaluation_scope": f"5-Fold Stratified Cross-Validation (N = {len(full_df)} out-of-fold samples)",
        "total_false_negatives": int(np.sum((y == 1) & (oof_preds == 0))),
        "gender": {
            "male": g_m,
            "female": g_f,
            "fnr_disparity": round(g_fnr_diff, 4),
            "disparate_impact_ratio": g_disp_impact
        },
        "economic_proxy": {
            "lower_income_under_5lpa": inc_l,
            "higher_income_above_5lpa": inc_h,
            "fnr_disparity": round(inc_fnr_diff, 4)
        },
        "first_generation": {
            "first_gen": fg_y,
            "non_first_gen": fg_n,
            "fnr_disparity": round(fg_fnr_diff, 4)
        },
        "age_band": age_band,
    }


def run_comprehensive_fairness_audit(
    fairness_dir: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    allow_dirty: bool = False,
) -> Dict[str, Any]:
    """
    Executes comprehensive fairness audit on both the held-out test split (N=300)
    and full 5-fold cross-validation (N=2,000). Writes generator_sanity_check.json and
    fairness_metrics.json (with provenance). Output paths default to the repository locations.
    docs/ethics_and_fairness.md is not rendered here (scripts/render_fairness_report.py, run by
    `python -m ml.pipeline run-all` after every step has succeeded).
    Refuses to run on a dirty tree unless allow_dirty (recorded in the JSON provenance).
    """
    require_clean_tree(allow_dirty)
    metrics_path = Path(metrics_path) if metrics_path else FAIRNESS_METRICS_PATH
    fairness_dir = Path(fairness_dir) if fairness_dir else FAIRNESS_METRICS_PATH.parent / "fairness"
    from sklearn.model_selection import train_test_split
    from ml.config import RANDOM_SEED

    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Calibrated model not found at {MODEL_ARTIFACT_PATH}. Run calibration first.")

    model = joblib.load(MODEL_ARTIFACT_PATH)
    full_df = pd.read_csv(PROCESSED_DATA_PATH)
    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)

    # 1. Single Held-Out Test Split (N = 300)
    _, test_df = train_test_split(
        full_df, test_size=0.15, random_state=RANDOM_SEED, stratify=full_df["is_dropout"]
    )
    X_test = test_df[feature_names]
    y_test = test_df["is_dropout"].values

    test_probs, _ = predict_student_risk(model, X_test)
    test_preds = (test_probs >= 0.50).astype(int)

    # Gender Audit (Test Split)
    male_mask_t = (test_df["gender"] == "Male").values
    female_mask_t = (test_df["gender"] == "Female").values
    male_m_t = compute_group_fairness_metrics(y_test, test_preds, male_mask_t)
    fem_m_t = compute_group_fairness_metrics(y_test, test_preds, female_mask_t)
    g_fnr_diff_t = abs(male_m_t["fnr_miss_rate"] - fem_m_t["fnr_miss_rate"])
    g_disp_t = round(fem_m_t["selection_rate"] / max(male_m_t["selection_rate"], 1e-5), 3)

    # Economic Proxy Audit (Test Split)
    low_inc_mask_t = (test_df["income_slab_idx"] <= 1).values
    high_inc_mask_t = (test_df["income_slab_idx"] >= 2).values
    low_inc_m_t = compute_group_fairness_metrics(y_test, test_preds, low_inc_mask_t)
    high_inc_m_t = compute_group_fairness_metrics(y_test, test_preds, high_inc_mask_t)
    inc_fnr_diff_t = abs(low_inc_m_t["fnr_miss_rate"] - high_inc_m_t["fnr_miss_rate"])

    # First-Gen Audit (Test Split)
    fg_mask_t = (test_df["is_first_generation"] == 1).values
    nfg_mask_t = (test_df["is_first_generation"] == 0).values
    fg_m_t = compute_group_fairness_metrics(y_test, test_preds, fg_mask_t)
    nfg_m_t = compute_group_fairness_metrics(y_test, test_preds, nfg_mask_t)
    fg_fnr_diff_t = abs(fg_m_t["fnr_miss_rate"] - nfg_m_t["fnr_miss_rate"])

    # Age Band Audit (Test Split; cut = median age of the full cohort)
    age_band_t = compute_age_band_audit(test_df, y_test, test_preds, float(full_df["age"].median()))

    # 2. 5-Fold Cross-Validation Audit (N = 2,000 full cohort)
    cv_audit = compute_cross_validated_fairness_audit(n_splits=5)

    test_split_summary = {
        "evaluation_scope": "Held-Out Test Set (N = 300 students, unseen 15% split)",
        "total_false_negatives": int(np.sum((y_test == 1) & (test_preds == 0))),
        "gender": {
            "male": male_m_t,
            "female": fem_m_t,
            "fnr_disparity": round(g_fnr_diff_t, 4),
            "disparate_impact_ratio": g_disp_t
        },
        "economic_proxy": {
            "lower_income_under_5lpa": low_inc_m_t,
            "higher_income_above_5lpa": high_inc_m_t,
            "fnr_disparity": round(inc_fnr_diff_t, 4)
        },
        "first_generation": {
            "first_gen": fg_m_t,
            "non_first_gen": nfg_m_t,
            "fnr_disparity": round(fg_fnr_diff_t, 4)
        },
        "age_band": age_band_t,
    }

    # -------------------------------------------------------------
    # Write Generator Sanity Check & Synchronize Report
    # -------------------------------------------------------------
    provenance = build_provenance(simulated_inputs(), allow_dirty=allow_dirty)
    gen_check_dict = {
        "benchmark": "generator_sanity_check_simulated_cohort",
        "description": "Verification of simulated Indian cohort generator. Note: metrics reflect generator design parameters and are NOT evidence of real-world predictive validity.",
        "test_split_n300": test_split_summary,
        "cross_validation_n2000": cv_audit,
        "provenance": provenance,
    }

    fairness_dir.mkdir(parents=True, exist_ok=True)
    with open(fairness_dir / "generator_sanity_check.json", "w") as f:
        json.dump(gen_check_dict, f, indent=2)

    # Synchronize unified metrics file
    existing_metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                existing_metrics = json.load(f)
        except Exception:
            existing_metrics = {}

    summary_dict = {
        "gender": test_split_summary["gender"],
        "economic_proxy": test_split_summary["economic_proxy"],
        "first_generation": test_split_summary["first_generation"],
        "age_band": test_split_summary["age_band"],
        "test_split": test_split_summary,
        "cross_validation": cv_audit,
    }

    existing_metrics["gender"] = summary_dict["gender"]
    existing_metrics["economic_proxy"] = summary_dict["economic_proxy"]
    existing_metrics["first_generation"] = summary_dict["first_generation"]
    existing_metrics["age_band"] = summary_dict["age_band"]
    existing_metrics["generator_sanity_check"] = {
        "description": gen_check_dict["description"],
        "test_split_fnr_gaps": {
            "gender": test_split_summary["gender"]["fnr_disparity"],
            "economic_proxy": test_split_summary["economic_proxy"]["fnr_disparity"],
            "first_generation": test_split_summary["first_generation"]["fnr_disparity"],
            "age_band": test_split_summary["age_band"]["fnr_disparity"],
        },
        "cross_validation_fnr_gaps": {
            "gender": cv_audit["gender"]["fnr_disparity"],
            "economic_proxy": cv_audit["economic_proxy"]["fnr_disparity"],
            "first_generation": cv_audit["first_generation"]["fnr_disparity"],
            "age_band": cv_audit["age_band"]["fnr_disparity"],
        },
    }

    existing_metrics["generator_sanity_check"]["provenance"] = provenance

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(existing_metrics, f, indent=2)
    logger.info("Saved fairness metrics JSON to %s", metrics_path)

    return summary_dict


def build_arg_parser():
    import argparse
    return add_allow_dirty_argument(argparse.ArgumentParser(description="Simulated-cohort fairness audit"))


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    summary = run_comprehensive_fairness_audit(allow_dirty=args.allow_dirty)
    print("Fairness Audit Completed Successfully!")
    print(json.dumps(summary, indent=2))
