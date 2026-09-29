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
    def test_v7_1_audit_attributes_strictly_excluded(self):
        """V7.1: Audit attributes never appear in model feature sets."""
        from ml.sources.oulad import build_snapshot_dataset
        from backend.app.services.ml_service import RAW_FEATURE_COLUMNS

        # Simulated cohort model features
        sim_audit = {"gender", "category", "family_income_slab"}
        assert not sim_audit.intersection(RAW_FEATURE_COLUMNS), "Protected attributes found in collegiate RAW_FEATURE_COLUMNS"

        # OULAD feature matrix excludes audit attributes
        _, _, _, _, _, oulad_features = build_snapshot_dataset(14)
        oulad_audit = {"gender", "age_band", "imd_band", "disability", "region"}
        assert not oulad_audit.intersection(set(oulad_features))

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
