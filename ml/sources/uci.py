"""
UCI Dataset Source for Higher Education Benchmark (ID 697)
Dataset: "Predict Students' Dropout and Academic Success" (Portuguese Higher Education)

Provides clean snake_case DataFrame, strict row/target assertions,
three temporal feature sets (ENROLMENT_TIME, END_OF_SEM1, FULL),
and primary vs sensitivity label variants.
"""

import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from ml.sources.integrity import refuse_synthetic, verify_checksum

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
UCI_CSV_PATH = RAW_DATA_DIR / "uci_dropout.csv"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"

EXPECTED_ROWS = 4424
EXPECTED_TARGETS = {"Dropout", "Graduate", "Enrolled"}

# Standard clean snake_case column mapping from raw UCI headers
UCI_COLUMN_MAPPING: Dict[str, str] = {
    "Marital Status": "marital_status",
    "Application mode": "application_mode",
    "Application order": "application_order",
    "Course": "course",
    "Daytime/evening attendance\t": "daytime_evening_attendance",
    "Daytime/evening attendance": "daytime_evening_attendance",
    "Previous qualification": "previous_qualification",
    "Previous qualification (grade)": "previous_qualification_grade",
    "Nacionality": "nationality",
    "Mother's qualification": "mothers_qualification",
    "Father's qualification": "fathers_qualification",
    "Mother's occupation": "mothers_occupation",
    "Father's occupation": "fathers_occupation",
    "Admission grade": "admission_grade",
    "Displaced": "displaced",
    "Educational special needs": "educational_special_needs",
    "Debtor": "debtor",
    "Tuition fees up to date": "tuition_fees_up_to_date",
    "Gender": "gender",
    "Scholarship holder": "scholarship_holder",
    "Age at enrollment": "age_at_enrollment",
    "International": "international",
    "Curricular units 1st sem (credited)": "cu_1st_sem_credited",
    "Curricular units 1st sem (enrolled)": "cu_1st_sem_enrolled",
    "Curricular units 1st sem (evaluations)": "cu_1st_sem_evaluations",
    "Curricular units 1st sem (approved)": "cu_1st_sem_approved",
    "Curricular units 1st sem (grade)": "cu_1st_sem_grade",
    "Curricular units 1st sem (without evaluations)": "cu_1st_sem_without_evaluations",
    "Curricular units 2nd sem (credited)": "cu_2nd_sem_credited",
    "Curricular units 2nd sem (enrolled)": "cu_2nd_sem_enrolled",
    "Curricular units 2nd sem (evaluations)": "cu_2nd_sem_evaluations",
    "Curricular units 2nd sem (approved)": "cu_2nd_sem_approved",
    "Curricular units 2nd sem (grade)": "cu_2nd_sem_grade",
    "Curricular units 2nd sem (without evaluations)": "cu_2nd_sem_without_evaluations",
    "Unemployment rate": "unemployment_rate",
    "Inflation rate": "inflation_rate",
    "GDP": "gdp",
    "Target": "target",
}

# 1. ENROLMENT_TIME: Only features available at admission.
# PROTECTED attributes (gender, age_at_enrollment) are never model features; see
# ml/fairness/attributes.py. AUDIT_GROUPS (scholarship_holder, debtor, displaced) stay as
# documented features and are also audited.
ENROLMENT_TIME: List[str] = [
    "marital_status",
    "application_mode",
    "application_order",
    "course",
    "daytime_evening_attendance",
    "previous_qualification",
    "previous_qualification_grade",
    "nationality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
    "admission_grade",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "scholarship_holder",
    "international",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
]

# 2. END_OF_SEM1: ENROLMENT_TIME + 1st semester curricular units (strictly no 2nd-sem columns)
END_OF_SEM1: List[str] = ENROLMENT_TIME + [
    "cu_1st_sem_credited",
    "cu_1st_sem_enrolled",
    "cu_1st_sem_evaluations",
    "cu_1st_sem_approved",
    "cu_1st_sem_grade",
    "cu_1st_sem_without_evaluations",
]

# 3. FULL: Everything (marked as "not early warning")
FULL: List[str] = END_OF_SEM1 + [
    "cu_2nd_sem_credited",
    "cu_2nd_sem_enrolled",
    "cu_2nd_sem_evaluations",
    "cu_2nd_sem_approved",
    "cu_2nd_sem_grade",
    "cu_2nd_sem_without_evaluations",
]

FEATURE_SETS: Dict[str, List[str]] = {
    "enrolment_time": ENROLMENT_TIME,
    "end_of_sem1": END_OF_SEM1,
    "full": FULL,
}


def _download_instructions(target_path: Path) -> str:
    return (
        f"Manual download instructions:\n"
        f"1. Download the zip archive from {UCI_ZIP_URL}\n"
        f"2. Extract 'data.csv' unchanged (do not re-save it) to {target_path}\n"
        f"No synthetic fallback is permitted in this phase."
    )


def download_uci_dataset(target_path: Path = UCI_CSV_PATH) -> pd.DataFrame:
    """
    Downloads official UCI Dataset 697 from archive zip URL and caches the zip's CSV member
    byte-for-byte at target_path (so the cached file matches the recorded checksum).
    Raises RuntimeError on failure with manual instructions.
    No synthetic fallback.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Attempting download of UCI dataset from %s", UCI_ZIP_URL)

    try:
        resp = requests.get(UCI_ZIP_URL, timeout=30)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_names = [name for name in z.namelist() if name.endswith(".csv")]
                if csv_names:
                    target_path.write_bytes(z.read(csv_names[0]))
                    logger.info("Successfully downloaded and cached UCI dataset at %s", target_path)
                    return pd.read_csv(target_path, sep=";")
    except Exception as exc:
        logger.error("Download failed from %s: %s", UCI_ZIP_URL, exc)

    raise RuntimeError(
        f"Failed to download UCI dataset (id=697) from {UCI_ZIP_URL}. "
        + _download_instructions(target_path)
    )


def load_uci_clean_df(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads UCI dataset from cache (or downloads if missing), cleans column names to snake_case,
    and asserts expected row count and target values.
    Refuses files with an `is_synthetic` column and files that do not match the official checksum.
    """
    path = csv_path or UCI_CSV_PATH

    if not path.exists():
        download_uci_dataset(path)

    # Load from cached CSV
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) <= 1:
            df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")

    refuse_synthetic(df.columns, path)

    # Rename columns to standardized snake_case
    raw_to_clean = {}
    for col in df.columns:
        clean_key = col.strip()
        if clean_key in UCI_COLUMN_MAPPING:
            raw_to_clean[col] = UCI_COLUMN_MAPPING[clean_key]
        else:
            # Fallback cleaning
            raw_to_clean[col] = clean_key.lower().replace(" ", "_").replace("'", "").replace("(", "").replace(")", "")
    df = df.rename(columns=raw_to_clean)

    # Clean target string values
    df["target"] = df["target"].astype(str).str.strip()

    # Assert integrity: 4,424 rows and exact target classes
    assert len(df) == EXPECTED_ROWS, (
        f"Integrity error: Expected {EXPECTED_ROWS} rows in UCI dataset, got {len(df)}."
    )
    unique_targets = set(df["target"].unique())
    assert unique_targets == EXPECTED_TARGETS, (
        f"Integrity error: Expected target classes {EXPECTED_TARGETS}, got {unique_targets}."
    )

    verify_checksum("uci_697", UCI_CSV_PATH.name, path, _download_instructions(path))

    return df


def get_uci_benchmark_dataset(
    feature_set: str = "enrolment_time",
    label_variant: str = "primary",
    csv_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Prepares feature matrix X, target array y, groups (course ID), and feature names list.

    Parameters:
    -----------
    feature_set: 'enrolment_time', 'end_of_sem1', or 'full'
    label_variant:
        'primary': Dropout = 1 vs Graduate = 0 (Enrolled excluded, N = 3,630)
        'sensitivity': Dropout = 1 vs Graduate/Enrolled = 0 (N = 4,424)

    Returns:
    --------
    (X, y, groups, feature_names)
    """
    df = load_uci_clean_df(csv_path)

    fs_key = feature_set.lower().strip()
    if fs_key not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Must be one of: {list(FEATURE_SETS.keys())}")

    lv_key = label_variant.lower().strip()
    if lv_key == "primary":
        # Exclude Enrolled
        df = df[df["target"].isin(["Dropout", "Graduate"])].copy()
        y = (df["target"] == "Dropout").astype(int).values
    elif lv_key == "sensitivity":
        # Treat Enrolled as 0 (not dropped out)
        y = (df["target"] == "Dropout").astype(int).values
    else:
        raise ValueError(f"Unknown label variant '{label_variant}'. Must be 'primary' or 'sensitivity'.")

    feature_cols = FEATURE_SETS[fs_key]
    X = df[feature_cols].copy()
    groups = df["course"].values

    return X, y, groups, feature_cols
