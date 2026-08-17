"""
Unit & Integration Tests for DropoutGuard Data Pipeline
Tests:
- Schema correctness across all 4 pillars
- No unexpected nulls in required columns
- Statistical validity and ground-truth correlation directions and magnitudes
- Feature engineering transformations and interaction terms
- UCI & OULAD loaders
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.data_pipeline.load_uci import load_clean_uci_data
from ml.data_pipeline.load_oulad import load_clean_oulad_data
from ml.data_pipeline.generate_synthetic_indian import generate_indian_student_cohort
from ml.data_pipeline.feature_engineering import (
    build_engineered_features,
    generate_processed_feature_dataset,
    PILLAR_COLUMNS
)


class TestDataPipelineLoaders:
    """Tests for external and raw dataset loaders."""

    def test_uci_loader_schema_and_target(self):
        """Test that UCI dataset loader returns expected columns and binary target."""
        df = load_clean_uci_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "is_dropout" in df.columns
        assert set(df["is_dropout"].unique()).issubset({0, 1})
        # Check presence of key academic & socio-economic signals
        assert "admission_grade" in df.columns or "previous_qualification_grade" in df.columns
        assert "tuition_fees_up_to_date" in df.columns or "is_debtor" in df.columns

    def test_oulad_loader_schema_and_target(self):
        """Test that OULAD dataset loader returns behavioral features and binary target."""
        df = load_clean_oulad_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "is_dropout" in df.columns
        assert set(df["is_dropout"].unique()).issubset({0, 1})
        # Check presence of key behavioral signals
        assert "sum_click" in df.columns
        assert "avg_submission_lag_days" in df.columns or "late_submission_rate" in df.columns


class TestSyntheticIndianCohort:
    """Tests for the synthetic Indian college student cohort generation."""

    @pytest.fixture
    def cohort_df(self):
        return generate_indian_student_cohort(n_students=500, seed=42, output_path=None)

    def test_cohort_row_count_and_uniqueness(self, cohort_df):
        """Verify row count and unique student IDs."""
        assert len(cohort_df) == 500
        assert cohort_df["student_id"].nunique() == 500

    def test_cohort_value_ranges(self, cohort_df):
        """Verify numerical ranges are realistic and bounded."""
        assert cohort_df["attendance_percentage"].between(0.0, 100.0).all()
        assert cohort_df["current_cgpa"].between(0.0, 10.0).all()
        assert cohort_df["prev_sem_cgpa"].between(0.0, 10.0).all()
        assert cohort_df["backlog_count"].ge(0).all()
        assert cohort_df["fee_payment_delay_days"].ge(0).all()
        assert cohort_df["lms_logins_per_week"].ge(0).all()
        assert cohort_df["ground_truth_risk_prob"].between(0.0, 1.0).all()
        assert set(cohort_df["is_dropout"].unique()).issubset({0, 1})

    def test_mandatory_75_percent_attendance_flag(self, cohort_df):
        """Verify that attendance_risk_flag adheres strictly to the 75% rule."""
        expected_flag = (cohort_df["attendance_percentage"] < 75.0).astype(int)
        pd.testing.assert_series_equal(
            cohort_df["attendance_risk_flag"],
            expected_flag,
            check_names=False
        )

    def test_correlation_direction_and_magnitude(self, cohort_df):
        """
        Verify that the synthetic dropout label correlates logically with ground truth risk drivers:
        - Attendance %: Negative correlation (higher attendance -> lower dropout)
        - CGPA: Negative correlation (higher CGPA -> lower dropout)
        - Backlogs: Positive correlation (more backlogs -> higher dropout)
        - Fee Payment Delay: Positive correlation (longer delay -> higher dropout)
        - LMS Inactivity: Positive correlation (higher inactivity -> higher dropout)
        """
        corr = cohort_df.select_dtypes(include=[np.number]).corr()["is_dropout"]

        # 1. Attendance must negatively correlate with dropout (r < -0.30)
        assert corr["attendance_percentage"] < -0.30, f"Expected strong negative correlation for attendance, got {corr['attendance_percentage']}"

        # 2. CGPA must negatively correlate with dropout (r < -0.35)
        assert corr["current_cgpa"] < -0.35, f"Expected strong negative correlation for CGPA, got {corr['current_cgpa']}"

        # 3. Backlog count must positively correlate with dropout (r > +0.35)
        assert corr["backlog_count"] > 0.35, f"Expected strong positive correlation for backlogs, got {corr['backlog_count']}"

        # 4. Fee payment delay must positively correlate with dropout (r > +0.18)
        assert corr["fee_payment_delay_days"] > 0.18, f"Expected positive correlation for fee delay, got {corr['fee_payment_delay_days']}"

        # 5. LMS inactivity recency must positively correlate with dropout (r > +0.20)
        assert corr["days_since_last_lms_activity"] > 0.20, f"Expected positive correlation for LMS inactivity, got {corr['days_since_last_lms_activity']}"


class TestFeatureEngineering:
    """Tests for 4-pillar feature transformations and final processed dataset."""

    @pytest.fixture
    def processed_df(self, tmp_path):
        test_path = tmp_path / "test_features.csv"
        return generate_processed_feature_dataset(n_students=500, seed=42, output_path=test_path)

    def test_no_null_values_in_required_features(self, processed_df):
        """Ensure no missing or NaN values in processed feature set."""
        assert processed_df.isna().sum().sum() == 0, f"Found nulls: {processed_df.isna().sum()[processed_df.isna().sum() > 0]}"

    def test_pillar_columns_present(self, processed_df):
        """Ensure all required columns across the 4 pillars exist in the dataset."""
        for pillar_name, cols in PILLAR_COLUMNS.items():
            for col in cols:
                assert col in processed_df.columns, f"Missing column '{col}' for pillar '{pillar_name}'"

    def test_composite_indices(self, processed_df):
        """Verify behavior and financial stress indices are bounded between [0, 1]."""
        assert processed_df["behavioral_disengagement_index"].between(0.0, 1.0).all()
        assert processed_df["financial_stress_index"].between(0.0, 1.0).all()

    def test_interaction_features_non_negative(self, processed_df):
        """Verify interaction terms are computed properly and are non-negative."""
        assert (processed_df["interaction_att_x_fee"] >= 0.0).all()
        assert (processed_df["interaction_cgpa_x_backlog"] >= 0.0).all()
        assert (processed_df["interaction_firstgen_x_inactivity"] >= 0.0).all()
        assert (processed_df["interaction_att_x_cgpa_drop"] >= 0.0).all()

    def test_academic_crisis_flag_logic(self, processed_df):
        """Verify academic_crisis_flag = (CGPA < 5.0 | backlog >= 2)."""
        expected = ((processed_df["current_cgpa"] < 5.0) | (processed_df["backlog_count"] >= 2)).astype(int)
        pd.testing.assert_series_equal(
            processed_df["academic_crisis_flag"],
            expected,
            check_names=False
        )
