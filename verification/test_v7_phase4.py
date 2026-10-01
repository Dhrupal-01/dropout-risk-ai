"""
Verification tests for V7 (Phase 4 — fairness).
"""

import json
import re
from pathlib import Path
import pytest
import numpy as np

from verification.conftest import PROJECT_ROOT


class TestV7Phase4Fairness:
    @pytest.mark.data
    def test_v7_1_audit_attributes_strictly_excluded(self, tmp_path):
        """V7.1: Actual feature-matrix columns of UCI, OULAD and the simulated cohort never intersect PROTECTED."""
        from backend.app.services.ml_service import RAW_FEATURE_COLUMNS
        from ml.data_pipeline.feature_engineering import generate_processed_feature_dataset
        from ml.fairness.attributes import PROTECTED
        from ml.models.train import prepare_training_data
        from ml.sources.oulad import SNAPSHOT_DAYS, build_snapshot_dataset, load_raw_tables
        from ml.sources.uci import FEATURE_SETS, get_uci_benchmark_dataset

        # UCI: every feature set x label variant, columns as built
        for feature_set in FEATURE_SETS:
            for label_variant in ("primary", "sensitivity"):
                X, _, _, _ = get_uci_benchmark_dataset(feature_set=feature_set, label_variant=label_variant)
                leaked = set(X.columns) & set(PROTECTED["uci"])
                assert leaked == set(), f"UCI {feature_set}/{label_variant}: {sorted(leaked)}"

        # OULAD: every snapshot day, columns as built
        tables = load_raw_tables()
        for t in SNAPSHOT_DAYS:
            X, _, _, _, _, _ = build_snapshot_dataset(t=t, tables=tables)
            leaked = set(X.columns) & set(PROTECTED["oulad"])
            assert leaked == set(), f"OULAD t={t}: {sorted(leaked)}"

        # Simulated cohort: training matrix from a generated cohort, plus the serving contract
        features_csv = tmp_path / "features.csv"
        generate_processed_feature_dataset(n_students=300, seed=42, output_path=features_csv)
        X, _, _, _ = prepare_training_data(data_path=features_csv)
        leaked = set(X.columns) & set(PROTECTED["simulated"])
        assert leaked == set(), f"Simulated training matrix: {sorted(leaked)}"

        serving_features = json.loads((PROJECT_ROOT / "ml" / "artifacts" / "feature_names.json").read_text())
        leaked = set(serving_features) & set(PROTECTED["simulated"])
        assert leaked == set(), f"feature_names.json: {sorted(leaked)}"
        leaked = set(RAW_FEATURE_COLUMNS) & set(PROTECTED["simulated"])
        assert leaked == set(), f"RAW_FEATURE_COLUMNS: {sorted(leaked)}"

    def test_v7_2_small_groups_flagged(self):
        """V7.2: Groups with n < 50 are flagged with insufficient sample."""
        from ml.fairness.audit import compute_group_metrics

        y_true = np.array([0, 1] * 20)  # n = 40 (< 50)
        y_prob = np.array([0.2, 0.8] * 20)
        y_pred = np.array([0, 1] * 20)
        res = compute_group_metrics(y_true, y_prob, y_pred)
        assert res.get("insufficient_sample") is True
        assert res.get("status") == "insufficient_sample"

    def test_v7_3_bootstrap_cis_deterministic(self):
        """V7.3: Bootstrap CIs are deterministic with a fixed seed."""
        from ml.fairness.audit import compute_bootstrap_fairness_cis

        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 10)
        y_prob = np.array([0.8, 0.2, 0.7, 0.9, 0.1, 0.3, 0.8, 0.2] * 10)
        attr = np.array(["A", "B", "A", "B", "A", "B", "A", "B"] * 10)

        ci1 = compute_bootstrap_fairness_cis(
            y_true_v=y_true,
            y_prob_v=y_prob,
            attr_v=attr,
            threshold=0.5,
            unique_groups=["A", "B"],
            reference_group="A",
            n_bootstraps=50,
            seed=42,
        )
        ci2 = compute_bootstrap_fairness_cis(
            y_true_v=y_true,
            y_prob_v=y_prob,
            attr_v=attr,
            threshold=0.5,
            unique_groups=["A", "B"],
            reference_group="A",
            n_bootstraps=50,
            seed=42,
        )
        assert ci1 == ci2, "Bootstrap CI was non-deterministic with same seed"

    def test_v7_4_all_four_mitigations_comparison(self):
        """V7.4: All four mitigation conditions ran on identical splits."""
        fairness_json = PROJECT_ROOT / "ml" / "artifacts" / "fairness_metrics.json"
        assert fairness_json.exists()
        with open(fairness_json, "r") as f:
            data = json.load(f)
        mitigations = data.get("uci_higher_ed", {}).get("mitigations_comparison", [])
        assert len(mitigations) >= 4, f"Expected 4 mitigation conditions, found {len(mitigations)}"

        uci_mit_file = PROJECT_ROOT / "ml" / "artifacts" / "fairness" / "uci_mitigations.json"
        assert uci_mit_file.exists()

    def test_v7_5_oulad_2013_vs_2014_shift_reported(self):
        """V7.5: OULAD 2013 vs 2014 gaps and income ablation are reported."""
        fairness_json = PROJECT_ROOT / "ml" / "artifacts" / "fairness_metrics.json"
        assert fairness_json.exists()
        with open(fairness_json, "r") as f:
            data = json.load(f)
        assert "income_ablation" in data

        shift_file = PROJECT_ROOT / "ml" / "artifacts" / "fairness" / "oulad_shift_check.json"
        assert shift_file.exists()

    def test_v7_6_forbidden_claims_in_ethics_doc(self):
        """V7.6: docs/ethics_and_fairness.md contains no forbidden claims outside static section."""
        doc_path = PROJECT_ROOT / "docs" / "ethics_and_fairness.md"
        assert doc_path.exists()
        content = doc_path.read_text(encoding="utf-8")

        # Split at Responsible Use section
        parts = content.split("## Responsible Use")
        audit_part = parts[0]

        forbidden_regex = re.compile(r"\b(is fair|unbiased|no (significant )?bias|free of bias|guarantee)\b", re.IGNORECASE)
        matches = forbidden_regex.findall(audit_part)
        assert not matches, f"Forbidden claims found in ethics doc: {matches}"
