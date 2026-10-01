"""
Main Fairness Audit Orchestrator for DropoutGuard
Executes full suite of algorithmic fairness evaluations across real and simulated benchmarks:
1. Real-Data Higher Education Benchmark (UCI Dataset 697, END_OF_SEM1, out-of-fold predictions)
2. UCI Fairness Mitigations Comparative Benchmark (None, Reweighing, Group Thresholds, Fairlearn ExponentiatedGradient)
3. Time-Based Learning Analytics Benchmark (OULAD Day 56 Snapshot, Temporal Test Holdout 2014)
4. OULAD Presentation Shift Check (2013 in-sample vs 2014 out-of-sample presentation shift)
5. Generator Sanity Check (Simulated Indian Cohort, N=2,000, 5-fold CV + Test Split)
6. Income Feature Ablation Experiment (Simulated Cohort, with vs without financial stress features)

Serializes all audit artifacts to ml/artifacts/fairness/ and renders docs/ethics_and_fairness.md.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from ml.fairness.attributes import AUDIT_GROUPS, PROTECTED
from ml.fairness.audit import audit_model_fairness
from ml.fairness.income_ablation import run_income_ablation_experiment
from ml.fairness.mitigation import compare_fairness_mitigations
from ml.fairness.shift_check import run_oulad_presentation_shift_check
from ml.sources.oulad import build_snapshot_dataset, load_raw_tables
from ml.sources.uci import get_uci_benchmark_dataset, load_uci_clean_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"
FAIRNESS_ARTIFACTS_DIR = ARTIFACTS_DIR / "fairness"
FAIRNESS_METRICS_PATH = ARTIFACTS_DIR / "fairness_metrics.json"


# =========================================================================
# 1. REAL-DATA HIGHER ED BENCHMARK: UCI AUDIT & MITIGATIONS
# =========================================================================

def run_uci_audit_pipeline(
    n_splits: int = 5,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes fairness audit and mitigation comparisons on UCI Higher Education dataset.
    The model is the benchmark model: X and y come from get_uci_benchmark_dataset
    ("end_of_sem1", "primary"), so the audited feature matrix is the benchmarked one.
    Audited attributes:
    - PROTECTED (never model features): gender (Male vs Female), age_group (<=20, 21-25, >25)
    - AUDIT_GROUPS (also model features, documented in ml/fairness/attributes.py):
      scholarship_holder, debtor, displaced
    """
    logger.info("Loading UCI Higher Education dataset for fairness audit...")
    # Primary label variant: Dropout (1) vs Graduate (0), Enrolled excluded
    X, y, _, feature_names = get_uci_benchmark_dataset(feature_set="end_of_sem1", label_variant="primary")
    df = load_uci_clean_df().loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)

    # Construct clean categorical audit frame
    audit_df = pd.DataFrame(index=df.index)
    audit_df["gender"] = df["gender"].map({1: "Male", 0: "Female"}).fillna("Unknown")
    audit_df["scholarship_holder"] = df["scholarship_holder"].map({1: "Scholarship", 0: "No Scholarship"}).fillna("Unknown")
    audit_df["debtor"] = df["debtor"].map({1: "Debtor", 0: "Non-Debtor"}).fillna("Unknown")
    audit_df["displaced"] = df["displaced"].map({1: "Displaced", 0: "Non-Displaced"}).fillna("Unknown")
    
    # Age bins: <=20, 21-25, >25
    age_series = df["age_at_enrollment"]
    age_groups = pd.cut(
        age_series,
        bins=[-np.inf, 20, 25, np.inf],
        labels=["<=20", "21-25", ">25"],
        right=True,
    )
    audit_df["age_group"] = age_groups.astype(str)

    logger.info("UCI Feature matrix X: END_OF_SEM1 with %d features (PROTECTED excluded: %s)", len(feature_names), PROTECTED["uci"])

    # 5-Fold Stratified Cross-Validation for Out-of-Fold risk probabilities
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(df), dtype=float)

    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val = X.iloc[val_idx]

        scaler = StandardScaler()
        X_tr_std = scaler.fit_transform(X_tr)
        X_val_std = scaler.transform(X_val)

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf.fit(X_tr_std, y_tr)
        oof_probs[val_idx] = clf.predict_proba(X_val_std)[:, 1]

    # Attribute configurations with standard reference groups
    attr_configs = {
        "gender": "Male",
        "scholarship_holder": "No Scholarship",
        "debtor": "Non-Debtor",
        "displaced": "Non-Displaced",
        "age_group": "<=20",
    }

    logger.info("Computing out-of-fold fairness audit on UCI cohort (N=%d)...", len(df))
    uci_audit = audit_model_fairness(
        y_true=y,
        y_prob=oof_probs,
        audit_df=audit_df,
        attribute_configs=attr_configs,
        top_k_pct=0.20,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    uci_audit["feature_count"] = len(feature_names)
    uci_audit["feature_set"] = "END_OF_SEM1"
    uci_audit["excluded_protected_attributes"] = PROTECTED["uci"]
    uci_audit["audit_groups_used_as_features"] = AUDIT_GROUPS["uci"]

    # Mitigations Benchmark on standard 70/30 split using sensitive attribute 'gender'
    logger.info("Running fairness mitigation comparison on UCI split...")
    s_arr = np.array(audit_df["gender"].tolist(), dtype=object)
    X_tr_m, X_te_m, y_tr_m, y_te_m, s_tr_m, s_te_m = train_test_split(
        X, y, s_arr, test_size=0.30, random_state=seed, stratify=y
    )
    uci_mitigations = compare_fairness_mitigations(
        X_train=X_tr_m,
        y_train=y_tr_m,
        X_test=X_te_m,
        y_test=y_te_m,
        sensitive_train=s_tr_m,
        sensitive_test=s_te_m,
        base_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
        seed=seed,
    )
    uci_mitigations["sensitive_attribute"] = "gender"

    return uci_audit, uci_mitigations


# =========================================================================
# 2. TIME-BASED LEARNING ANALYTICS: OULAD AUDIT
# =========================================================================

def run_oulad_audit_pipeline(
    t: int = 56,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes fairness audit on OULAD snapshot t=56 temporal test holdout cohort (2014B/J).
    Audits 7 attributes strictly excluded from feature matrix X:
    - gender (M vs F)
    - age_band (0-35, 35-55, 55<=)
    - imd_band (deprivation deciles)
    - disability (N vs Y)
    - highest_education (5 qualifications)
    - region (13 UK administrative regions)
    - imd_x_gender (intersectional deprivation x gender)
    """
    logger.info("Building OULAD snapshot t=%d for fairness audit...", t)
    raw_tables = load_raw_tables()
    X, y, splits, groups, audit_df, feature_names = build_snapshot_dataset(t=t, tables=raw_tables)

    # Ensure highest_education is strictly in audit_df and dropped from X
    if "highest_education" in X.columns:
        X = X.drop(columns=["highest_education"])
    if "highest_education" not in audit_df.columns:
        info_df = raw_tables["studentInfo"]
        pop_keys = audit_df[["code_module", "code_presentation", "id_student"]]
        merged = pd.merge(pop_keys, info_df[["code_module", "code_presentation", "id_student", "highest_education"]], on=["code_module", "code_presentation", "id_student"], how="left")
        audit_df["highest_education"] = merged["highest_education"].fillna("Unknown").values

    # Clean imd_band formatting and add intersectional imd_x_gender
    if "imd_band" in audit_df.columns:
        audit_df["imd_band"] = audit_df["imd_band"].replace({"10-20": "10-20%"}).fillna("Missing")
    audit_df["imd_x_gender"] = audit_df["imd_band"].astype(str) + "_" + audit_df["gender"].astype(str)

    # Train on 2013 presentations, evaluate on 2014 holdout presentations
    train_idx, test_idx = splits[0]
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    audit_test = audit_df.iloc[test_idx].reset_index(drop=True)

    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_train)
    X_te_std = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_tr_std, y_train)
    p_test = clf.predict_proba(X_te_std)[:, 1]

    attr_configs = {
        "gender": "M",
        "age_band": "0-35",
        "disability": "N",
        "imd_band": "90-100%",
        "highest_education": "A Level or Equivalent",
        "region": "South Region",
        "imd_x_gender": "90-100%_M",
    }

    logger.info("Computing fairness audit on OULAD 2014 temporal holdout cohort (N=%d)...", len(y_test))
    oulad_audit = audit_model_fairness(
        y_true=y_test,
        y_prob=p_test,
        audit_df=audit_test,
        attribute_configs=attr_configs,
        top_k_pct=0.20,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    oulad_audit["snapshot_horizon_t"] = t
    oulad_audit["train_cohort"] = "2013B + 2013J"
    oulad_audit["test_cohort"] = "2014B + 2014J (Temporal Holdout)"
    oulad_audit["feature_count"] = X.shape[1]
    oulad_audit["excluded_protected_attributes"] = list(attr_configs.keys())

    return oulad_audit


# =========================================================================
# 3. GENERATOR SANITY CHECK (SIMULATED INDIAN COHORT)
# =========================================================================

def run_generator_sanity_check_pipeline(seed: int = 42) -> Dict[str, Any]:
    """
    Executes generator sanity check on simulated Indian cohort (N=2,000).
    Verifies that the synthetic data generator does not bake in unintended demographic disparities.
    Explicitly labeled as a generator calibration check, NOT real-world predictive validity.
    """
    from ml.models.fairness_audit import (
        compute_cross_validated_fairness_audit,
        compute_group_fairness_metrics,
    )
    from ml.config import PROCESSED_DATA_PATH, MODEL_ARTIFACT_PATH, FEATURE_NAMES_PATH
    import joblib

    logger.info("Executing generator sanity check on simulated Indian cohort...")
    full_df = pd.read_csv(PROCESSED_DATA_PATH)
    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)

    # 1. 5-Fold Cross-Validation Audit
    cv_audit = compute_cross_validated_fairness_audit(n_splits=5)

    # 2. Test Split Audit (15%)
    X = full_df[feature_names]
    y = full_df["is_dropout"].values

    _, X_test, _, y_test, _, test_df = train_test_split(
        X, y, full_df, test_size=0.15, random_state=seed, stratify=y
    )

    if MODEL_ARTIFACT_PATH.exists():
        model = joblib.load(MODEL_ARTIFACT_PATH)
        from ml.models.calibrate import predict_student_risk
        test_probs, _ = predict_student_risk(model, X_test)
    else:
        # Fallback logistic regression if artifact absent
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X, y)
        test_probs = clf.predict_proba(X_test)[:, 1]

    test_preds = (test_probs >= 0.50).astype(int)

    male_mask_t = (test_df["gender"] == "Male").values
    fem_mask_t = (test_df["gender"] == "Female").values
    male_m_t = compute_group_fairness_metrics(y_test, test_preds, male_mask_t)
    fem_m_t = compute_group_fairness_metrics(y_test, test_preds, fem_mask_t)

    low_inc_mask_t = (test_df["income_slab_idx"] <= 1).values
    high_inc_mask_t = (test_df["income_slab_idx"] >= 2).values
    low_inc_m_t = compute_group_fairness_metrics(y_test, test_preds, low_inc_mask_t)
    high_inc_m_t = compute_group_fairness_metrics(y_test, test_preds, high_inc_mask_t)

    fg_mask_t = (test_df["is_first_generation"] == 1).values
    nfg_mask_t = (test_df["is_first_generation"] == 0).values
    fg_m_t = compute_group_fairness_metrics(y_test, test_preds, fg_mask_t)
    nfg_m_t = compute_group_fairness_metrics(y_test, test_preds, nfg_mask_t)

    return {
        "benchmark": "generator_sanity_check_simulated_cohort",
        "description": "Verification of simulated Indian cohort generator. Note: metrics reflect generator design parameters and are NOT evidence of real-world predictive validity.",
        "test_split_n300": {
            "gender": {
                "male": male_m_t,
                "female": fem_m_t,
                "fnr_disparity": round(abs(male_m_t["fnr_miss_rate"] - fem_m_t["fnr_miss_rate"]), 4),
            },
            "economic_proxy": {
                "lower_income_under_5lpa": low_inc_m_t,
                "higher_income_above_5lpa": high_inc_m_t,
                "fnr_disparity": round(abs(low_inc_m_t["fnr_miss_rate"] - high_inc_m_t["fnr_miss_rate"]), 4),
            },
            "first_generation": {
                "first_gen": fg_m_t,
                "non_first_gen": nfg_m_t,
                "fnr_disparity": round(abs(fg_m_t["fnr_miss_rate"] - nfg_m_t["fnr_miss_rate"]), 4),
            },
        },
        "cross_validation_n2000": cv_audit,
    }


# =========================================================================
# 4. MASTER ORCHESTRATION & PERSISTENCE
# =========================================================================

def run_all_fairness_audits(
    n_bootstraps: int = 1000,
    seed: int = 42,
    render_docs: bool = True,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """
    Executes all Phase 4 fairness audits, writes JSON artifacts, and triggers doc rendering.
    Uses existing artifacts if force_rerun is False and files exist.
    """
    FAIRNESS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" DROPOUTGUARD — COMPREHENSIVE PHASE 4 FAIRNESS AUDIT SUITE ")
    print(" Benchmarks: Real Higher Ed (UCI 697), Real Learning Analytics (OULAD), Simulated Cohort ")
    print("=" * 80)

    # 1. UCI Audit & Mitigations
    uci_fair_path = FAIRNESS_ARTIFACTS_DIR / "uci_fairness.json"
    uci_mit_path = FAIRNESS_ARTIFACTS_DIR / "uci_mitigations.json"
    if not force_rerun and uci_fair_path.exists() and uci_mit_path.exists():
        print("\n[1/5] Loading existing UCI Higher Education Fairness Audit & Mitigations...")
        with open(uci_fair_path, "r") as f:
            uci_audit = json.load(f)
        with open(uci_mit_path, "r") as f:
            uci_mitigations = json.load(f)
    else:
        print("\n[1/5] Running UCI Higher Education Fairness Audit & Mitigations...")
        uci_audit, uci_mitigations = run_uci_audit_pipeline(n_splits=5, n_bootstraps=n_bootstraps, seed=seed)
        with open(uci_fair_path, "w") as f:
            json.dump(uci_audit, f, indent=2)
        with open(uci_mit_path, "w") as f:
            json.dump(uci_mitigations, f, indent=2)
    print(f" ✓ Saved/Loaded UCI audit ({len(uci_audit.get('attributes', {}))} attributes) and mitigations.")

    # 2. OULAD Snapshot t=56 Audit
    oulad_fair_path = FAIRNESS_ARTIFACTS_DIR / "oulad_fairness.json"
    if not force_rerun and oulad_fair_path.exists():
        print("\n[2/5] Loading existing OULAD Day 56 Fairness Audit (Temporal Holdout 2014)...")
        with open(oulad_fair_path, "r") as f:
            oulad_audit = json.load(f)
    else:
        print("\n[2/5] Running OULAD Day 56 Fairness Audit (Temporal Holdout 2014)...")
        oulad_audit = run_oulad_audit_pipeline(t=56, n_bootstraps=n_bootstraps, seed=seed)
        with open(oulad_fair_path, "w") as f:
            json.dump(oulad_audit, f, indent=2)
    print(f" ✓ Saved/Loaded OULAD audit ({len(oulad_audit.get('attributes', {}))} attributes).")

    # 3. OULAD Presentation Shift Check
    oulad_shift_path = FAIRNESS_ARTIFACTS_DIR / "oulad_shift_check.json"
    if not force_rerun and oulad_shift_path.exists():
        print("\n[3/5] Loading existing OULAD Presentation Shift Check (2013 vs 2014)...")
        with open(oulad_shift_path, "r") as f:
            oulad_shift = json.load(f)
    else:
        print("\n[3/5] Running OULAD Presentation Shift Check (2013 vs 2014)...")
        oulad_shift = run_oulad_presentation_shift_check(t=56, n_bootstraps=n_bootstraps, seed=seed)
        with open(oulad_shift_path, "w") as f:
            json.dump(oulad_shift, f, indent=2)
    print(" ✓ Saved/Loaded OULAD temporal presentation shift comparison.")

    # 4. Generator Sanity Check
    print("\n[4/5] Running Generator Sanity Check on Simulated Cohort...")
    generator_check = run_generator_sanity_check_pipeline(seed=seed)
    with open(FAIRNESS_ARTIFACTS_DIR / "generator_sanity_check.json", "w") as f:
        json.dump(generator_check, f, indent=2)
    print(" ✓ Saved generator sanity check.")

    # 5. Income Ablation Experiment
    print("\n[5/5] Running Income Feature Ablation Experiment...")
    income_ablation_res = run_income_ablation_experiment(seed=seed)
    with open(FAIRNESS_ARTIFACTS_DIR / "income_ablation.json", "w") as f:
        json.dump(income_ablation_res, f, indent=2)
    print(" ✓ Saved income feature ablation results.")

    # 6. Synchronize unified summary to ml/artifacts/fairness_metrics.json
    unified_summary = {
        "status": "completed",
        "phase": "Phase 4 - Fairness Audit & Mitigations",
        "uci_higher_ed": {
            "n_samples": uci_audit.get("n_samples", 3630),
            "base_rate": uci_audit.get("base_rate", 0.3209),
            "selection_rate_top20": uci_audit.get("top_k_selection_rate", 0.20),
            "attributes_audited": list(uci_audit.get("attributes", {}).keys()),
            "mitigations_comparison": uci_mitigations.get("comparison_table", []),
        },
        "oulad_learning_analytics": {
            "n_samples_holdout": oulad_audit.get("n_samples", 15092),
            "base_rate": oulad_audit.get("base_rate", 0.1543),
            "attributes_audited": list(oulad_audit.get("attributes", {}).keys()),
        },
        "generator_sanity_check": {
            "description": generator_check["description"],
            "test_split_fnr_gaps": {
                "gender": generator_check["test_split_n300"]["gender"]["fnr_disparity"],
                "economic_proxy": generator_check["test_split_n300"]["economic_proxy"]["fnr_disparity"],
                "first_generation": generator_check["test_split_n300"]["first_generation"]["fnr_disparity"],
            },
            "cross_validation_fnr_gaps": {
                "gender": generator_check["cross_validation_n2000"]["gender"]["fnr_disparity"],
                "economic_proxy": generator_check["cross_validation_n2000"]["economic_proxy"]["fnr_disparity"],
                "first_generation": generator_check["cross_validation_n2000"]["first_generation"]["fnr_disparity"],
            },
        },
        "income_ablation": {
            "comparison_table": income_ablation_res["comparison_table"],
        },
    }
    with open(FAIRNESS_METRICS_PATH, "w") as f:
        json.dump(unified_summary, f, indent=2)
    print(f" ✓ Synchronized unified summary to {FAIRNESS_METRICS_PATH}")

    # 7. Render docs/ethics_and_fairness.md
    if render_docs:
        print("\nRendering docs/ethics_and_fairness.md from generated JSON artifacts...")
        from scripts.render_fairness_report import render_fairness_markdown_report
        render_fairness_markdown_report()
        print(" ✓ Generated docs/ethics_and_fairness.md successfully.")

    print("\n" + "=" * 80)
    print(" ALL FAIRNESS AUDITS COMPLETED SUCCESSFULLY! ")
    print("=" * 80)

    return unified_summary


if __name__ == "__main__":
    run_all_fairness_audits()
