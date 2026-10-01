"""
Unit and Integration Tests for Algorithmic Fairness Audit & Mitigations
Context: DropoutGuard Phase 4 Fairness Engineering

Validates:
1. Small-group flag works (subgroups with N < 50 flagged 'insufficient_sample' with gaps suppressed).
2. Bootstrap confidence intervals are deterministic with fixed random seed.
3. PROTECTED attributes (ml/fairness/attributes.py) never appear in any actual model feature matrix (UCI, OULAD, Simulated).
4. Mitigation benchmarking produces structured comparative metrics across all 4 strategies.
5. Within-group ECE (Expected Calibration Error) calculates valid calibration errors.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from backend.app.services.ml_service import RAW_FEATURE_COLUMNS
from ml.data_pipeline.feature_engineering import generate_processed_feature_dataset
from ml.fairness.attributes import PROTECTED
from ml.fairness.audit import (
    audit_model_fairness,
    compute_within_group_ece,
    compute_group_metrics,
)
from ml.fairness.mitigation import compare_fairness_mitigations
from ml.models.train import prepare_training_data
from ml.sources.oulad import SNAPSHOT_DAYS, build_snapshot_dataset, load_raw_tables
from ml.sources.uci import FEATURE_SETS, get_uci_benchmark_dataset

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"


def test_small_group_flag_works():
    """
    Subgroups with sample size n < 50 must be flagged with 'insufficient_sample' = True,
    and their disparity gaps vs the reference group must be suppressed (None).
    """
    # Create synthetic cohort: Large Group (N=200), Small Group (N=35)
    rng = np.random.default_rng(42)
    n_large = 200
    n_small = 35
    total_n = n_large + n_small

    y_true = rng.integers(0, 2, size=total_n)
    y_prob = rng.uniform(0.1, 0.9, size=total_n)

    group_labels = ["LargeGroup"] * n_large + ["SmallGroup"] * n_small
    audit_df = pd.DataFrame({"subgroup": group_labels})

    attr_configs = {"subgroup": "LargeGroup"}

    results = audit_model_fairness(
        y_true=y_true,
        y_prob=y_prob,
        audit_df=audit_df,
        attribute_configs=attr_configs,
        top_k_pct=0.20,
        n_bootstraps=100,
        seed=42,
    )

    group_res = results["attributes"]["subgroup"]["groups"]
    gap_res = results["attributes"]["subgroup"]["gaps_vs_reference"]

    # Verify sample counts
    assert group_res["LargeGroup"]["n_samples"] == 200
    assert group_res["SmallGroup"]["n_samples"] == 35

    # Verify small group flagging
    assert group_res["LargeGroup"]["insufficient_sample"] is False
    assert group_res["SmallGroup"]["insufficient_sample"] is True

    # Verify gap suppression
    assert gap_res["SmallGroup"]["fnr_gap"] is None
    assert gap_res["SmallGroup"]["fnr_gap_95ci"] is None
    assert gap_res["SmallGroup"]["note"] == "insufficient_sample (n < 50)"


def test_bootstrap_cis_deterministic_with_seed():
    """
    Bootstrap confidence interval computations must be completely reproducible
    and deterministic when providing the same random seed.
    """
    rng = np.random.default_rng(101)
    n_samples = 300

    y_true = rng.integers(0, 2, size=n_samples)
    y_prob = rng.uniform(0.05, 0.95, size=n_samples)

    groups = rng.choice(["GroupA", "GroupB"], size=n_samples, p=[0.6, 0.4])
    audit_df = pd.DataFrame({"cohort": groups})

    attr_configs = {"cohort": "GroupA"}

    # Run audit 1
    res1 = audit_model_fairness(
        y_true=y_true,
        y_prob=y_prob,
        audit_df=audit_df,
        attribute_configs=attr_configs,
        top_k_pct=0.20,
        n_bootstraps=250,
        seed=999,
    )

    # Run audit 2 with identical seed
    res2 = audit_model_fairness(
        y_true=y_true,
        y_prob=y_prob,
        audit_df=audit_df,
        attribute_configs=attr_configs,
        top_k_pct=0.20,
        n_bootstraps=250,
        seed=999,
    )

    ci1 = res1["attributes"]["cohort"]["gaps_vs_reference"]["GroupB"]["fnr_gap_95ci"]
    ci2 = res2["attributes"]["cohort"]["gaps_vs_reference"]["GroupB"]["fnr_gap_95ci"]

    assert ci1 is not None and ci2 is not None
    assert ci1 == ci2, f"Bootstrap CIs were non-deterministic: {ci1} vs {ci2}"


@pytest.mark.data
def test_uci_feature_matrices_exclude_protected():
    """
    Every UCI feature matrix the benchmark and fairness audit train on (all feature sets x both
    label variants) shares no column with PROTECTED["uci"]. Columns are taken as built.
    """
    for feature_set in FEATURE_SETS:
        for label_variant in ("primary", "sensitivity"):
            X, _, _, _ = get_uci_benchmark_dataset(feature_set=feature_set, label_variant=label_variant)
            leaked = set(X.columns) & set(PROTECTED["uci"])
            assert leaked == set(), f"UCI {feature_set}/{label_variant} uses protected attributes: {sorted(leaked)}"


@pytest.mark.data
def test_oulad_snapshot_matrices_exclude_protected():
    """
    Every OULAD snapshot feature matrix (all t in SNAPSHOT_DAYS) shares no column with
    PROTECTED["oulad"]. Columns are taken as built.
    """
    tables = load_raw_tables()
    for t in SNAPSHOT_DAYS:
        X, _, _, _, _, _ = build_snapshot_dataset(t=t, tables=tables)
        leaked = set(X.columns) & set(PROTECTED["oulad"])
        assert leaked == set(), f"OULAD snapshot t={t} uses protected attributes: {sorted(leaked)}"


def test_simulated_feature_matrices_exclude_protected(tmp_path):
    """
    The simulated-cohort training matrix (prepare_training_data on a generated cohort) and the
    serving contract (feature_names.json, RAW_FEATURE_COLUMNS) share no column with
    PROTECTED["simulated"]. Columns are taken as built.
    """
    features_csv = tmp_path / "features.csv"
    generate_processed_feature_dataset(n_students=300, seed=42, output_path=features_csv)
    X, _, _, _ = prepare_training_data(data_path=features_csv)
    leaked = set(X.columns) & set(PROTECTED["simulated"])
    assert leaked == set(), f"Simulated training matrix uses protected attributes: {sorted(leaked)}"

    with open(FEATURE_NAMES_PATH, "r") as f:
        serving_features = json.load(f)
    leaked = set(serving_features) & set(PROTECTED["simulated"])
    assert leaked == set(), f"feature_names.json uses protected attributes: {sorted(leaked)}"

    leaked = set(RAW_FEATURE_COLUMNS) & set(PROTECTED["simulated"])
    assert leaked == set(), f"RAW_FEATURE_COLUMNS uses protected attributes: {sorted(leaked)}"


def test_mitigations_comparison_structure():
    """
    Validates that compare_fairness_mitigations evaluates all 4 strategies:
    None, Sample Reweighing, Group-Specific Thresholds, and Fairlearn ExponentiatedGradient.
    """
    rng = np.random.default_rng(42)
    n_tr, n_te = 150, 60
    n_feats = 4

    X_tr = pd.DataFrame(rng.normal(size=(n_tr, n_feats)), columns=[f"f_{i}" for i in range(n_feats)])
    y_tr = rng.integers(0, 2, size=n_tr)
    s_tr = rng.choice(["GroupA", "GroupB"], size=n_tr)

    X_te = pd.DataFrame(rng.normal(size=(n_te, n_feats)), columns=[f"f_{i}" for i in range(n_feats)])
    y_te = rng.integers(0, 2, size=n_te)
    s_te = rng.choice(["GroupA", "GroupB"], size=n_te)

    res = compare_fairness_mitigations(
        X_train=X_tr,
        y_train=y_tr,
        sensitive_train=s_tr,
        X_test=X_te,
        y_test=y_te,
        sensitive_test=s_te,
        base_estimator=LogisticRegression(C=1.0, max_iter=200, random_state=42),
        seed=42,
    )

    assert "comparison_table" in res
    strategies = [row["strategy"] for row in res["comparison_table"]]

    expected_strategies = [
        "None (Unmitigated)",
        "Sample Reweighing",
        "Group-Specific Thresholds (FNR Parity)",
        "Fairlearn ExponentiatedGradient",
    ]
    assert strategies == expected_strategies

    for row in res["comparison_table"]:
        assert "roc_auc" in row
        assert "pr_auc" in row
        assert "recall" in row
        assert "precision" in row
        assert "brier_score" in row
        assert "max_fnr_gap" in row


def test_within_group_ece_calculation():
    """
    Tests Expected Calibration Error calculation.
    Perfect predictions should yield an ECE close to 0.0.
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.02, 0.04, 0.01, 0.03, 0.98, 0.99, 0.96, 0.97])

    ece = compute_within_group_ece(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 0.05, f"Expected near-zero ECE for well-calibrated probabilities, got {ece}"
