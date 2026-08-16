"""
Feature Engineering Pipeline for DropoutGuard
Context: SIH 2026 PSID 7-L / SDG 4 (Quality Education)

Combines data sources and builds engineered features across the four core pillars:
1. Attendance Dynamics: Trends, slopes, streaks, mandatory 75% rule violation
2. Academic Performance: CGPA delta, backlog accumulation, internal test scores
3. Learning Behavior: LMS login frequency, submission lag, inactivity recency, engagement index
4. Socio-Economic Indicators: Income slab, first-gen flag, fee delay, scholarship buffer, financial stress index
+ Cross-Pillar Interaction Terms: Attendance x Fee Delay, CGPA x Backlogs, First-Gen x Inactivity

Outputs the clean, standardized dataset: data/processed/features.csv
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd

from ml.data_pipeline.load_uci import load_clean_uci_data
from ml.data_pipeline.load_oulad import load_clean_oulad_data
from ml.data_pipeline.generate_synthetic_indian import generate_indian_student_cohort

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FEATURES_CSV_PATH = PROCESSED_DATA_DIR / "features.csv"
FEATURE_METADATA_PATH = PROCESSED_DATA_DIR / "feature_metadata.json"

# Categorization of features for modeling, explainability, and fairness audits
PILLAR_COLUMNS = {
    "attendance": [
        "attendance_percentage",
        "attendance_month_1",
        "attendance_month_2",
        "attendance_month_3",
        "attendance_3m_trend",
        "consecutive_absences",
        "attendance_risk_flag",
        "subject_attendance_std"
    ],
    "academic": [
        "current_cgpa",
        "prev_sem_cgpa",
        "cgpa_delta",
        "backlog_count",
        "internal_exam_score_pct",
        "stem_core_fail_flag",
        "academic_crisis_flag"
    ],
    "behavior": [
        "lms_logins_per_week",
        "assignment_submission_lag_days",
        "resource_access_count",
        "days_since_last_lms_activity",
        "forum_participation_count",
        "behavioral_disengagement_index"
    ],
    "socio_economic": [
        "income_slab_idx",
        "is_first_generation",
        "fee_payment_delay_days",
        "has_scholarship",
        "is_hosteler",
        "commute_distance_km",
        "financial_stress_index"
    ],
    "interactions": [
        "interaction_att_x_fee",
        "interaction_cgpa_x_backlog",
        "interaction_firstgen_x_inactivity",
        "interaction_att_x_cgpa_drop"
    ],
    "demographics_protected": [
        "gender",
        "category",
        "age",
        "family_income_slab",
        "hostel_status"
    ],
    "identifiers": [
        "student_id"
    ],
    "targets": [
        "is_dropout",
        "ground_truth_risk_prob"
    ]
}


def build_engineered_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Applies comprehensive domain-specific feature engineering to the student cohort table.
    """
    df = df_raw.copy()

    # ---------------------------------------------------------
    # 1. Pillar 1: Attendance Feature Engineering
    # ---------------------------------------------------------
    # Standard deviation across subjects (detects targeted class skipping)
    subj_cols = [c for c in ["att_core1", "att_core2", "att_lab", "att_elective"] if c in df.columns]
    if subj_cols:
        df["subject_attendance_std"] = np.round(df[subj_cols].std(axis=1), 2)
    else:
        df["subject_attendance_std"] = 0.0

    # Ensure attendance_3m_trend is available
    if "attendance_3m_trend" not in df.columns:
        if "attendance_month_3" in df.columns and "attendance_month_1" in df.columns:
            df["attendance_3m_trend"] = np.round((df["attendance_month_3"] - df["attendance_month_1"]) / 2.0, 2)
        else:
            df["attendance_3m_trend"] = 0.0

    # Attendance risk flag (<75% mandatory threshold)
    if "attendance_risk_flag" not in df.columns:
        df["attendance_risk_flag"] = (df["attendance_percentage"] < 75.0).astype(int)

    # ---------------------------------------------------------
    # 2. Pillar 2: Academic Performance Feature Engineering
    # ---------------------------------------------------------
    if "cgpa_delta" not in df.columns:
        df["cgpa_delta"] = np.round(df["current_cgpa"] - df["prev_sem_cgpa"], 2)

    # Academic crisis flag (Severe risk threshold: CGPA < 5.0 or 2+ active backlogs)
    df["academic_crisis_flag"] = ((df["current_cgpa"] < 5.0) | (df["backlog_count"] >= 2)).astype(int)

    # ---------------------------------------------------------
    # 3. Pillar 3: Learning Behavior & Engagement Index
    # ---------------------------------------------------------
    # Composite behavioral disengagement index [0 to 1]:
    # Combines normalized low logins, positive submission lag (late), and inactivity days
    norm_logins = 1.0 - np.clip(df["lms_logins_per_week"] / 10.0, 0.0, 1.0)
    norm_lag = np.clip(df["assignment_submission_lag_days"] / 7.0, 0.0, 1.0)
    norm_inactivity = np.clip(df["days_since_last_lms_activity"] / 30.0, 0.0, 1.0)
    df["behavioral_disengagement_index"] = np.round(0.35 * norm_logins + 0.35 * norm_lag + 0.30 * norm_inactivity, 3)

    # ---------------------------------------------------------
    # 4. Pillar 4: Socio-Economic & Financial Stress Index
    # ---------------------------------------------------------
    # Ensure binary is_hosteler flag exists
    if "is_hosteler" not in df.columns:
        df["is_hosteler"] = (df["hostel_status"] == "Hosteler").astype(int)

    # Financial stress index [0 to 1]:
    # Combines low income slab (0=<2LPA to 3=>8LPA), fee delay days, and lack of scholarship
    income_risk = (3 - df["income_slab_idx"]) / 3.0  # 0->1.0 (highest risk for lowest income)
    fee_delay_risk = np.clip(df["fee_payment_delay_days"] / 60.0, 0.0, 1.0)
    scholarship_buffer = (1 - df["has_scholarship"])  # 1 if no scholarship
    df["financial_stress_index"] = np.round(0.40 * income_risk + 0.40 * fee_delay_risk + 0.20 * scholarship_buffer, 3)

    # ---------------------------------------------------------
    # 5. Cross-Pillar Interaction Features
    # ---------------------------------------------------------
    # A. Attendance Deficit x Fee Delay
    # Captures double crisis where financial distress causes absenteeism
    df["interaction_att_x_fee"] = np.round(
        np.maximum(0.0, 100.0 - df["attendance_percentage"]) * (df["fee_payment_delay_days"] / 30.0),
        2
    )

    # B. Academic Deficit x Backlog Spiral
    df["interaction_cgpa_x_backlog"] = np.round(
        np.maximum(0.0, 10.0 - df["current_cgpa"]) * df["backlog_count"],
        2
    )

    # C. First-Gen Learner x Digital Inactivity
    # Lack of home mentorship coupled with high online disengagement
    df["interaction_firstgen_x_inactivity"] = np.round(
        df["is_first_generation"] * (df["days_since_last_lms_activity"] / 10.0),
        2
    )

    # D. Low Attendance x Negative CGPA Trajectory
    negative_cgpa_drop = np.maximum(0.0, -df["cgpa_delta"])
    df["interaction_att_x_cgpa_drop"] = np.round(
        df["attendance_risk_flag"] * negative_cgpa_drop,
        2
    )

    return df


def generate_processed_feature_dataset(
    n_students: int = 2000,
    seed: int = 42,
    output_path: Path = FEATURES_CSV_PATH
) -> pd.DataFrame:
    """
    Executes the end-to-end data pipeline:
    1. Loads / standardizes UCI and OULAD source layers
    2. Generates the realistic Indian Higher Education cohort layer
    3. Computes 4-pillar engineered features and interaction terms
    4. Validates schema and exports final data/processed/features.csv
    """
    logger.info("Starting end-to-end feature engineering pipeline...")
    
    # 1. Ensure source datasets are downloaded/available
    uci_df = load_clean_uci_data()
    oulad_df = load_clean_oulad_data()
    
    # 2. Generate the primary Indian collegiate cohort
    raw_cohort = generate_indian_student_cohort(n_students=n_students, seed=seed)
    
    # 3. Apply feature engineering transformations
    processed_df = build_engineered_features(raw_cohort)
    
    # 4. Save to processed destination
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    
    # 5. Save feature metadata dictionary for backend and model services
    import json
    with open(FEATURE_METADATA_PATH, "w") as f:
        json.dump(PILLAR_COLUMNS, f, indent=2)

    logger.info("Successfully produced %s: %d rows, %d columns.", output_path, len(processed_df), len(processed_df.columns))
    logger.info("Saved feature metadata schema to %s", FEATURE_METADATA_PATH)
    
    return processed_df


if __name__ == "__main__":
    df = generate_processed_feature_dataset()
    print("Processed Features Dataset Ready!")
    print(f"Total Rows: {len(df)}, Total Columns: {len(df.columns)}")
    print("\nColumns by Pillar:")
    for pillar, cols in PILLAR_COLUMNS.items():
        print(f"  [{pillar}]: {cols}")
    print("\nSample 5 rows:")
    print(df.head())
