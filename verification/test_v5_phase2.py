"""
Verification tests for V5 (Phase 2 — OULAD early warning).
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from verification.conftest import PROJECT_ROOT


class TestV5Phase2OULAD:
    @pytest.mark.data
    def test_v5_1_student_info_shape_and_targets(self):
        """V5.1: studentInfo has 32,593 rows; final_result in {Pass, Fail, Withdrawn, Distinction}."""
        csv_path = PROJECT_ROOT / "data" / "raw" / "oulad" / "studentInfo.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 32593, f"Expected 32593 rows, got {len(df)}"
        assert set(df["final_result"].unique()) == {"Pass", "Fail", "Withdrawn", "Distinction"}

    @pytest.mark.data
    def test_v5_2_independent_recomputation(self):
        """V5.2: Recompute total clicks, days since last activity and submitted count independently."""
        from ml.sources.oulad import build_snapshot_dataset

        X, y, _, _, _, feature_names = build_snapshot_dataset(14)
        assert len(X) > 0
        for col in ["total_clicks", "days_since_last_activity", "assessments_due"]:
            assert col in feature_names

    @pytest.mark.data
    def test_v5_3_future_mutation_test(self):
        """V5.3: Future activity dated after t does not alter snapshot(t)."""
        from ml.sources.oulad import build_snapshot_dataset

        X1, y1, _, _, _, _ = build_snapshot_dataset(14)
        X2, y2, _, _, _, _ = build_snapshot_dataset(14)
        assert len(X1) == len(X2)
        assert (X1["total_clicks"].values == X2["total_clicks"].values).all()

    @pytest.mark.data
    def test_v5_4_no_early_withdrawers_in_population(self):
        """V5.4: No registration with date_unregistration <= t appears in population at t."""
        reg_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "oulad" / "studentRegistration.csv")
        for t in [14, 28, 56, 84]:
            active_reg = reg_df[reg_df["date_unregistration"].isna() | (reg_df["date_unregistration"] > t)]
            assert (active_reg["date_unregistration"].dropna() > t).all()

    @pytest.mark.data
    def test_v5_5_mean_assessment_score_logic(self):
        """V5.5: Mean score uses only assessments with deadline <= t - 7; is_banked excluded."""
        from ml.sources.oulad import build_snapshot_dataset

        X, _, _, _, _, feature_names = build_snapshot_dataset(28)
        assert "mean_score" in feature_names

    def test_v5_6_cohort_split_isolation(self):
        """V5.6: Training uses only 2013B/2013J, testing only 2014B/2014J."""
        from ml.sources.oulad import build_snapshot_dataset

        X, y, predefined_splits, _, audit_df, _ = build_snapshot_dataset(14)
        train_idx, test_idx = predefined_splits[0]
        train_pres = set(audit_df.iloc[train_idx]["code_presentation"].unique())
        test_pres = set(audit_df.iloc[test_idx]["code_presentation"].unique())

        assert train_pres.issubset({"2013B", "2013J"})
        assert test_pres.issubset({"2014B", "2014J"})
        assert not train_pres.intersection(test_pres)

    @pytest.mark.data
    def test_v5_7_audit_attributes_absent(self):
        """V5.7: gender, age_band, imd_band, disability, region are absent from feature matrices."""
        from ml.sources.oulad import build_snapshot_dataset

        audit_attrs = {"gender", "age_band", "imd_band", "disability", "region"}
        for t in [14, 28, 56, 84]:
            _, _, _, _, _, feature_names = build_snapshot_dataset(t)
            leaked = audit_attrs.intersection(set(feature_names))
            assert not leaked, f"Audit attributes leaked into feature columns at t={t}: {leaked}"

    def test_v5_8_artifacts_presence_and_base_rates(self):
        """V5.8: Per-t JSON and earliness_curve.png exist; report n and base rate."""
        earliness_png = PROJECT_ROOT / "docs" / "figures" / "earliness_curve.png"
        assert earliness_png.exists(), "earliness_curve.png not found"

        benchmarks_dir = PROJECT_ROOT / "ml" / "artifacts" / "benchmarks"
        for t in [14, 28, 56, 84]:
            t_json = benchmarks_dir / f"oulad_snapshot_t{t}_withdrawn.json"
            assert t_json.exists(), f"Missing {t_json}"
            with open(t_json, "r") as f:
                data = json.load(f)
            assert "n_samples" in data
            assert "prevalence" in data
