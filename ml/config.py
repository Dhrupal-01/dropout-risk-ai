"""
DropoutGuard Central Configuration
Reads configurable parameters from environment variables with sensible defaults.
Zero hardcoded secrets, central risk thresholds.
"""

import os
from pathlib import Path
from typing import Tuple, List

# Base Directories
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "features.csv"
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"
DOCS_DIR = BASE_DIR / "docs"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Risk Thresholds (Configurable via environment)
# Low: < RISK_THRESHOLD_LOW (e.g. 0.33)
# Medium: [RISK_THRESHOLD_LOW, RISK_THRESHOLD_HIGH] (e.g. [0.33, 0.66])
# High: > RISK_THRESHOLD_HIGH (e.g. 0.66)
RISK_THRESHOLD_LOW = float(os.getenv("RISK_THRESHOLD_LOW", "0.33"))
RISK_THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", "0.66"))

# Artifact File Paths
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "calibrated_model.joblib"
BASE_MODEL_PATH = ARTIFACTS_DIR / "base_xgboost_model.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"
METRICS_REPORT_PATH = ARTIFACTS_DIR / "model_metrics.json"
SHAP_EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.joblib"
FAIRNESS_REPORT_PATH = DOCS_DIR / "ethics_and_fairness.md"

# Random Seed for Reproducibility
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Feature Exclusions for Model Training (IDs, targets, protected raw strings)
EXCLUDED_FEATURES = [
    "student_id",
    "gender",
    "category",
    "family_income_slab",
    "hostel_status",
    "ground_truth_risk_prob",
    "is_dropout"
]


def get_risk_tier(probability: float) -> str:
    """
    Maps a calibrated dropout probability to a discrete risk tier using configurable thresholds.
    """
    if probability < RISK_THRESHOLD_LOW:
        return "Low"
    elif probability <= RISK_THRESHOLD_HIGH:
        return "Medium"
    else:
        return "High"
