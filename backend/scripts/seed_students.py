"""
Import the reference cohort from data/processed/features.csv into PostgreSQL.

Usage:
    python -m backend.scripts.seed_students                 # all rows
    python -m backend.scripts.seed_students --limit 200     # subset
    python -m backend.scripts.seed_students --predict       # also score each student

SAFETY RULES ENFORCED HERE
  * `is_dropout` and `ground_truth_risk_prob` are LABELS. They are dropped before anything
    is stored and can never reach predict_proba.
  * `gender`, `category` and `family_income_slab` are protected/redundant attributes
    excluded from the model by ml.config.EXCLUDED_FEATURES. They are not persisted.
    (`income_slab_idx`, the ordinal encoding, IS a model feature and is kept.)
  * The 9 engineered columns present in the CSV are NOT stored: they are recomputed
    deterministically at predict time from the raw inputs.
  * `name` and `department` do not exist in the dataset. They are synthesised
    deterministically from student_id purely for dashboard display, and are kept out of
    the `features` payload.
"""

import argparse
import hashlib
import logging
import sys
from typing import Any, Dict, List

import pandas as pd

from backend.app.db.session import SessionLocal
from backend.app.services import student_service
from backend.app.services.ml_service import (
    LABEL_COLUMNS,
    RAW_FEATURE_COLUMNS,
    ml_service,
    to_native,
)
from ml.config import PROCESSED_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Columns copied verbatim into the stored `features` payload.
STORED_FEATURE_COLUMNS = list(RAW_FEATURE_COLUMNS) + ["hostel_status"]

# Deterministic display metadata. The ML dataset carries neither, and inventing them at
# random would make the seed non-reproducible across runs.
DEPARTMENTS = [
    "Computer Science & Engineering",
    "Information Technology",
    "Electronics & Communication",
    "Mechanical Engineering",
    "Civil Engineering",
    "Electrical Engineering",
]
FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Diya", "Ananya", "Ishaan", "Saanvi", "Kabir",
    "Meera", "Rohan", "Priya", "Arjun", "Neha", "Kiran", "Riya", "Aryan",
]
LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Reddy", "Iyer", "Nair", "Singh", "Gupta",
    "Desai", "Menon", "Joshi", "Kulkarni",
]
MENTORS = [f"FAC_{i:03d}" for i in range(1, 13)]


def _stable_index(student_id: str, salt: str, modulo: int) -> int:
    """Deterministic hash -> index. Stable across runs, processes and machines."""
    digest = hashlib.sha256(f"{salt}:{student_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def make_display_metadata(student_id: str) -> Dict[str, str]:
    """Fictitious but stable display fields. Never used as model input."""
    first = FIRST_NAMES[_stable_index(student_id, "first", len(FIRST_NAMES))]
    last = LAST_NAMES[_stable_index(student_id, "last", len(LAST_NAMES))]
    return {
        "name": f"{first} {last}",
        "department": DEPARTMENTS[_stable_index(student_id, "dept", len(DEPARTMENTS))],
        "assigned_mentor_id": MENTORS[_stable_index(student_id, "mentor", len(MENTORS))],
    }


def build_feature_payload(row: pd.Series) -> Dict[str, Any]:
    """Extract the canonical raw inference payload from one CSV row."""
    payload = {col: to_native(row[col]) for col in STORED_FEATURE_COLUMNS if col in row.index}

    missing = [c for c in RAW_FEATURE_COLUMNS if c not in payload]
    if missing:
        raise ValueError(f"features.csv is missing required raw column(s): {missing}")

    leaked = LABEL_COLUMNS.intersection(payload)
    if leaked:  # defensive; STORED_FEATURE_COLUMNS cannot contain labels
        raise ValueError(f"Label leak into feature payload: {sorted(leaked)}")

    return payload


def seed(
    limit: int | None = None,
    with_predictions: bool = False,
    batch_size: int = 200,
    session_factory=None,
) -> int:
    """
    Load the cohort. Idempotent: re-running upserts by student_id.

    `session_factory` defaults to the application's SessionLocal; tests inject a factory
    bound to the test database so seeding never touches the app database.
    """
    session_factory = session_factory or SessionLocal
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_DATA_PATH} not found. Generate it first:\n"
            "    python -m ml.data_pipeline.feature_engineering"
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)
    logger.info("Read %s: %d rows, %d columns", PROCESSED_DATA_PATH, len(df), len(df.columns))

    present_labels = LABEL_COLUMNS.intersection(df.columns)
    if present_labels:
        logger.info("Dropping label column(s) — never stored, never scored: %s", sorted(present_labels))

    if limit:
        df = df.head(limit)

    if with_predictions and not ml_service.is_loaded:
        ml_service.load()
        if not ml_service.is_loaded:
            raise RuntimeError(f"--predict requires ML artifacts: {ml_service.load_error}")

    inserted = 0
    with session_factory() as db:
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start : start + batch_size]
            for _, row in chunk.iterrows():
                student_id = str(row["student_id"])
                features = build_feature_payload(row)
                meta = make_display_metadata(student_id)

                student = student_service.upsert_student(
                    db, student_id=student_id, features=features, **meta
                )

                if with_predictions:
                    result = ml_service.predict(features)
                    student_service.record_prediction(
                        db,
                        student=student,
                        calibrated_risk_probability=result["calibrated_risk_probability"],
                        risk_tier=result["risk_tier"],
                        risk_score_percentage=result["risk_score_percentage"],
                        input_features=result["input_features"],
                        model_version=result["model_version"],
                    )
                inserted += 1

            db.commit()
            logger.info("Committed %d / %d students", min(start + batch_size, len(df)), len(df))

    logger.info("Seed complete: %d students%s", inserted, " (with predictions)" if with_predictions else "")
    return inserted


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the DropoutGuard reference cohort.")
    parser.add_argument("--limit", type=int, default=None, help="Only import the first N rows.")
    parser.add_argument(
        "--predict", action="store_true", help="Also score each student and store a prediction row."
    )
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        seed(limit=args.limit, with_predictions=args.predict, batch_size=args.batch_size)
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        logger.error("Seed failed: %s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
