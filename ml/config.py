"""
DropoutGuard Central Configuration
Reads configurable parameters from environment variables with sensible defaults.
Zero hardcoded secrets, central risk thresholds.
"""

import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
# Overridable so the test suite can build generated artifacts in a temp dir instead of the repo
PROCESSED_DATA_PATH = Path(os.getenv("DROPOUTGUARD_PROCESSED_DATA_PATH", str(DATA_DIR / "processed" / "features.csv")))
ARTIFACTS_DIR = Path(os.getenv("DROPOUTGUARD_ARTIFACTS_DIR", str(BASE_DIR / "ml" / "artifacts")))
DOCS_DIR = BASE_DIR / "docs"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Runtime settings shared by the API, the tests and the pipeline. One source: the process
# environment first, then backend/.env, then .env (the same precedence pydantic-settings uses for
# backend Settings), then the defaults below. os.environ is never modified.
ENV_FILES = (BASE_DIR / "backend" / ".env", BASE_DIR / ".env")  # earlier file wins
DEFAULT_RISK_THRESHOLD_LOW = 0.33
DEFAULT_RISK_THRESHOLD_HIGH = 0.66
DEFAULT_RANDOM_SEED = 42


def _setting(name: str, default):
    if name in os.environ:
        return os.environ[name]
    from dotenv import dotenv_values

    for env_file in ENV_FILES:
        if env_file.exists():
            value = dotenv_values(env_file).get(name)
            if value is not None:
                return value
    return default


def validate_risk_thresholds(low: float, high: float) -> None:
    """Raises ValueError unless 0 < low < high < 1."""
    if not 0.0 < low < high < 1.0:
        raise ValueError(
            f"Invalid risk thresholds: need 0 < RISK_THRESHOLD_LOW < RISK_THRESHOLD_HIGH < 1, "
            f"got low={low}, high={high}"
        )


# Risk Thresholds
# Low: < RISK_THRESHOLD_LOW
# Medium: [RISK_THRESHOLD_LOW, RISK_THRESHOLD_HIGH]
# High: > RISK_THRESHOLD_HIGH
RISK_THRESHOLD_LOW = float(_setting("RISK_THRESHOLD_LOW", DEFAULT_RISK_THRESHOLD_LOW))
RISK_THRESHOLD_HIGH = float(_setting("RISK_THRESHOLD_HIGH", DEFAULT_RISK_THRESHOLD_HIGH))
validate_risk_thresholds(RISK_THRESHOLD_LOW, RISK_THRESHOLD_HIGH)

# Artifact File Paths
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "calibrated_model.joblib"
BASE_MODEL_PATH = ARTIFACTS_DIR / "base_xgboost_model.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"
METRICS_REPORT_PATH = ARTIFACTS_DIR / "model_metrics.json"
SHAP_EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.joblib"
FAIRNESS_REPORT_PATH = DOCS_DIR / "ethics_and_fairness.md"
FAIRNESS_METRICS_PATH = ARTIFACTS_DIR / "fairness_metrics.json"

# Random Seed for Reproducibility
RANDOM_SEED = int(_setting("RANDOM_SEED", DEFAULT_RANDOM_SEED))

ASSUMPTIONS_PATH = BASE_DIR / "ml" / "simulation" / "assumptions.yaml"


def _attendance_threshold() -> float:
    """Statutory attendance requirement (%), the one value shared by the generator, the
    attendance_risk_flag feature and the rule-based alerts. A missing key raises."""
    import yaml

    with open(ASSUMPTIONS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return float(data["regulations_and_thresholds"]["mandatory_attendance_threshold"]["value"])


ATTENDANCE_THRESHOLD = _attendance_threshold()

# Feature Exclusions for Model Training (IDs, targets, protected raw strings).
# NOTE: Income IS used as a model input via `income_slab_idx` (serving as an objective
# need signal for routing institutional financial support and feeding `financial_stress_index`).
# Only its duplicate raw string label `family_income_slab` is excluded here to avoid categorical redundancy.
# Protected attributes (gender, category, age) stay in the cohort for fairness audits only (ml/fairness/attributes.py).
EXCLUDED_FEATURES = [
    "student_id",
    "gender",
    "category",
    "age",
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
