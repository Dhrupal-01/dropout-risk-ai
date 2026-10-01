"""
Unit Tests for Phase 2: OULAD Time-Based Early Warning Benchmark
Tests:
1. Temporal leakage invariance: inserting a fabricated activity row with date > t does not alter features
2. Population exclusion: early withdrawals (date_unregistration <= t) are strictly excluded from population
3. Audit attribute separation: gender, age_band, imd_band, disability, region are strictly excluded from X
4. Schema & target assertions: row count (32,593) and expected target categories
5. Missing raw files instructions: raises FileNotFoundError with explicit download instructions
6. PyTorch GRU estimator: fits on weekly sequence + static features and outputs well-calibrated probabilities
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.sources.oulad import (
    EXPECTED_RESULTS,
    EXPECTED_ROWS,
    REQUIRED_TABLES,
    build_snapshot_dataset,
    check_and_get_oulad_dir,
    load_raw_tables,
)
from ml.models.gru import HAS_TORCH, PyTorchGRUEstimator


@pytest.fixture
def mock_oulad_tables():
    """
    Creates a minimal, schema-conforming set of OULAD tables for unit testing.
    """
    # 4 students:
    # 101: withdraws at day 10 (early withdrawal for t=14)
    # 102: withdraws at day 20 (active at t=14, withdrew at t=28)
    # 103: withdraws at day 50 (active at t=14, 28)
    # 104: never withdraws (active at all t)
    info_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2014J", "2014J"],
        "id_student": [101, 102, 103, 104],
        "gender": ["M", "F", "M", "F"],
        "region": ["East", "West", "North", "South"],
        "highest_education": ["A Level or Equivalent", "HE Qualification", "Lower Than A Level", "No Formal quals"],
        "imd_band": ["10-20%", "20-30%", "30-40%", "40-50%"],
        "age_band": ["0-35", "35-55", "0-35", "55<="],
        "num_of_prev_attempts": [0, 1, 0, 2],
        "studied_credits": [60, 120, 60, 90],
        "disability": ["N", "N", "Y", "N"],
        "final_result": ["Withdrawn", "Withdrawn", "Withdrawn", "Pass"],
    })

    reg_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2014J", "2014J"],
        "id_student": [101, 102, 103, 104],
        "date_registration": [-30, -25, -20, -15],
        "date_unregistration": [10.0, 20.0, 50.0, np.nan],
    })

    vle_df = pd.DataFrame({
        "id_site": [1, 2, 3],
        "code_module": ["AAA", "AAA", "BBB"],
        "code_presentation": ["2013J", "2013J", "2014J"],
        "activity_type": ["forumng", "oucontent", "resource"],
        "week_from": [np.nan, np.nan, np.nan],
        "week_to": [np.nan, np.nan, np.nan],
    })

    # Clicks at various dates
    svle_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2014J", "2014J", "2014J"],
        "id_student": [102, 102, 103, 104, 104],
        "id_site": [1, 2, 3, 3, 3],
        "date": [5, 12, 10, 8, 25],  # 25 is after t=14
        "sum_click": [10, 15, 20, 5, 50],
    })

    assess_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2014J", "2014J"],
        "id_assessment": [1001, 1002, 2001, 2002],
        "assessment_type": ["TMA", "Exam", "TMA", "TMA"],
        "date": [10.0, np.nan, 12.0, 40.0],  # 1002 is exam with NaN date
        "weight": [20.0, 80.0, 30.0, 30.0],
    })

    student_assess_df = pd.DataFrame({
        "id_assessment": [1001, 2001, 2001],
        "id_student": [102, 103, 104],
        "date_submitted": [9.0, 14.0, 11.0],
        "is_banked": [0, 0, 0],
        "score": [75.0, 60.0, 85.0],
    })

    courses_df = pd.DataFrame({
        "code_module": ["AAA", "BBB"],
        "code_presentation": ["2013J", "2014J"],
        "module_presentation_length": [268, 269],
    })

    return {
        "studentInfo": info_df,
        "studentRegistration": reg_df,
        "vle": vle_df,
        "studentVle": svle_df,
        "assessments": assess_df,
        "studentAssessment": student_assess_df,
        "courses": courses_df,
    }


def test_early_withdrawals_are_excluded(mock_oulad_tables):
    """
    Verifies that students who already withdrew by day t (date_unregistration <= t)
    are strictly excluded from the population at snapshot t.
    """
    # At t=14: Student 101 unregistered on day 10 <= 14 -> must be EXCLUDED
    # Students 102 (unreg 20), 103 (unreg 50), 104 (never) -> must be INCLUDED
    X, y, splits, groups, audit_df, feature_names = build_snapshot_dataset(t=14, tables=mock_oulad_tables)

    assert 101 not in audit_df["id_student"].values
    assert 102 in audit_df["id_student"].values
    assert 103 in audit_df["id_student"].values
    assert 104 in audit_df["id_student"].values
    assert len(X) == 3

    # At t=28: Student 101 (day 10) and 102 (day 20) must be EXCLUDED
    X_28, y_28, _, _, audit_28, _ = build_snapshot_dataset(t=28, tables=mock_oulad_tables)
    assert 101 not in audit_28["id_student"].values
    assert 102 not in audit_28["id_student"].values
    assert 103 in audit_28["id_student"].values
    assert 104 in audit_28["id_student"].values
    assert len(X_28) == 2


def test_temporal_leakage_invariance_on_future_activity(mock_oulad_tables):
    """
    CRITICAL TEMPORAL TEST:
    Inserting a fabricated activity row with date > t does not change any feature for snapshot t.
    """
    t = 14
    X_baseline, _, _, _, _, _ = build_snapshot_dataset(t=t, tables=mock_oulad_tables)

    # Clone tables and insert a massive fabricated interaction at date = t + 5
    tables_modified = {k: v.copy() for k, v in mock_oulad_tables.items()}
    future_row = pd.DataFrame([{
        "code_module": "AAA",
        "code_presentation": "2013J",
        "id_student": 102,
        "id_site": 1,
        "date": t + 5,
        "sum_click": 9999,
    }])
    tables_modified["studentVle"] = pd.concat([tables_modified["studentVle"], future_row], ignore_index=True)

    # Insert a future assessment submission after t
    future_subm = pd.DataFrame([{
        "id_assessment": 1001,
        "id_student": 102,
        "date_submitted": t + 10,
        "is_banked": 0,
        "score": 100.0,
    }])
    tables_modified["studentAssessment"] = pd.concat([tables_modified["studentAssessment"], future_subm], ignore_index=True)

    X_after, _, _, _, _, _ = build_snapshot_dataset(t=t, tables=tables_modified)

    # Assert exactly identical feature matrices
    pd.testing.assert_frame_equal(X_baseline, X_after)


def test_audit_attributes_strictly_separated(mock_oulad_tables):
    """
    Verifies that no demographic/protected attributes (gender, age_band, imd_band, disability, region)
    exist in model feature matrix X, and that they exist in audit_df.
    """
    X, y, splits, groups, audit_df, feature_names = build_snapshot_dataset(t=14, tables=mock_oulad_tables)

    forbidden_attributes = ["gender", "age_band", "imd_band", "disability", "region"]
    for attr in forbidden_attributes:
        assert attr not in X.columns, f"Forbidden audit attribute '{attr}' leaked into feature matrix X!"
        assert attr in audit_df.columns, f"Audit attribute '{attr}' missing from audit_df!"


def test_shared_builder_audit_frame_carries_audit_groups(mock_oulad_tables):
    """
    The single OULAD builder puts highest_education (raw labels), normalised imd_band and
    imd_x_gender in the audit frame, so fairness code needs no post-processing of X.
    highest_education stays in X as an encoded AUDIT_GROUP feature.
    """
    X, _, _, _, audit_df, feature_names = build_snapshot_dataset(t=14, tables=mock_oulad_tables)

    assert "highest_education" in X.columns
    assert list(X.columns) == feature_names
    assert list(audit_df["highest_education"]) == list(
        mock_oulad_tables["studentInfo"].set_index("id_student").loc[audit_df["id_student"], "highest_education"]
    )
    assert list(audit_df["imd_x_gender"]) == [f"{imd}_{g}" for imd, g in zip(audit_df["imd_band"], audit_df["gender"])]
    assert not audit_df["imd_band"].isna().any()


def test_schema_and_target_assertions(tmp_path):
    """
    Verifies that load_raw_tables raises AssertionError if studentInfo row count
    differs from 32,593 or final_result domain is invalid.
    """
    csv_dir = tmp_path / "oulad"
    csv_dir.mkdir(parents=True)

    # Create dummy tables with invalid row count
    dummy_info = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Withdrawn"],
    })
    dummy_info.to_csv(csv_dir / "studentInfo.csv", index=False)

    for req in REQUIRED_TABLES:
        if req != "studentInfo.csv":
            pd.DataFrame({"dummy": [1]}).to_csv(csv_dir / req, index=False)

    with pytest.raises(AssertionError, match=f"Expected {EXPECTED_ROWS} rows"):
        load_raw_tables(data_dir=csv_dir)


def test_missing_raw_files_instructions(tmp_path):
    """
    Verifies that missing raw directory raises FileNotFoundError with explicit download instructions.
    """
    empty_dir = tmp_path / "non_existent_oulad"
    with pytest.raises(FileNotFoundError, match="Manual download instructions"):
        check_and_get_oulad_dir(empty_dir)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for GRU test")
def test_pytorch_gru_estimator():
    """
    Tests that PyTorchGRUEstimator fits over weekly click vectors + static features
    and produces well-formed probabilities in [0, 1] summing to 1.
    """
    rng = np.random.default_rng(42)
    n = 100
    X_df = pd.DataFrame({
        "total_clicks": rng.integers(10, 500, size=n).astype(float),
        "clicks_week_1": rng.integers(0, 50, size=n).astype(float),
        "clicks_week_2": rng.integers(0, 50, size=n).astype(float),
        "clicks_week_3": rng.integers(0, 50, size=n).astype(float),
        "clicks_week_4": rng.integers(0, 50, size=n).astype(float),
        "days_since_last_activity": rng.integers(0, 20, size=n).astype(float),
        "active_days": rng.integers(1, 15, size=n).astype(float),
        "num_of_prev_attempts": rng.integers(0, 3, size=n).astype(float),
        "studied_credits": rng.choice([60, 120], size=n).astype(float),
        "highest_education": rng.integers(0, 4, size=n).astype(float),
    })
    y = rng.choice([0, 1], size=n, p=[0.7, 0.3])

    gru = PyTorchGRUEstimator(
        hidden_dim=16,
        lr=0.01,
        batch_size=32,
        max_epochs=5,
        patience=2,
        random_state=42,
    )
    gru.fit(X_df, y)

    probs = gru.predict_proba(X_df)
    assert probs.shape == (n, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(n), rtol=1e-5)

    preds = gru.predict(X_df)
    assert len(preds) == n
    assert set(np.unique(preds)).issubset({0, 1})
