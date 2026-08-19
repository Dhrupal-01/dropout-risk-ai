"""
Prediction endpoints.

    POST /api/v1/predict             score one student
    POST /api/v1/predict/batch       score a JSON list in one vectorised call
    POST /api/v1/predict/batch/csv   score a multipart CSV upload

Risk tiers always come from `ml.models.calibrate.predict_student_risk`, which delegates
to `ml.config.get_risk_tier`. The API never derives Low/Medium/High itself.
"""

import io
import logging
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.core.errors import FeatureContractError
from backend.app.db.session import get_db
from backend.app.schemas.prediction import (
    MAX_BATCH_SIZE,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.schemas.student import StudentFeatureInput
from backend.app.services import prediction_service
from backend.app.services.ml_service import RAW_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict")


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Score one student and append to their prediction history",
)
def predict(payload: PredictionRequest, db: Session = Depends(get_db)) -> PredictionResponse:
    """
    Validate -> engineer features via the ML pipeline -> align to feature_names.json ->
    `predict_student_risk` -> upsert student -> append a NEW prediction row.

    Top SHAP drivers are computed and stored with the prediction (~43ms) so the exact
    explanation behind this score stays reproducible. Set `include_explanation` to also
    receive them in the response.
    """
    results = prediction_service.score_and_persist(
        db, [payload], include_explanations=payload.include_explanation
    )
    return results[0]


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Score a JSON list of students in a single inference call",
)
def predict_batch(
    payload: BatchPredictionRequest, db: Session = Depends(get_db)
) -> BatchPredictionResponse:
    """
    All students are scored with ONE aligned DataFrame and one `predict_student_risk`
    call — roughly 0.5ms per student versus 110ms when looping.

    Explanations are opt-in here: SHAP is per-row and would dominate the vectorised call.
    """
    duplicates = _duplicate_ids([item.student_id for item in payload.students])
    if duplicates:
        raise FeatureContractError(
            f"Duplicate student_id(s) in batch: {duplicates}. Each student may appear once."
        )

    results = prediction_service.score_and_persist(
        db,
        payload.students,
        include_explanations=payload.include_explanations,
        persist_top_drivers=payload.include_explanations,
    )
    return _build_batch_response(results, sort_by_risk_desc=payload.sort_by_risk_desc)


@router.post(
    "/batch/csv",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Score a CSV upload using the project's own features.csv column names",
)
def predict_batch_csv(
    file: UploadFile = File(..., description="CSV with student_id plus the 28 raw feature columns"),
    sort_by_risk_desc: bool = Query(default=False),
    include_explanations: bool = Query(default=False),
    db: Session = Depends(get_db),
) -> BatchPredictionResponse:
    """
    Accepts the same column names as data/processed/features.csv, so an export of the
    reference cohort can be scored directly.

    Extra columns (engineered features, labels, protected attributes) are ignored rather
    than rejected — a real features.csv contains all of them — but labels are dropped
    before anything reaches the model.
    """
    if not (file.filename or "").lower().endswith(".csv"):
        raise FeatureContractError("Upload must be a .csv file")

    try:
        frame = pd.read_csv(io.BytesIO(file.file.read()))
    except Exception as exc:  # noqa: BLE001 — malformed upload is a client error
        raise FeatureContractError(f"Could not parse CSV: {exc}") from exc
    finally:
        file.file.close()

    if frame.empty:
        raise FeatureContractError("CSV contains no data rows")
    if len(frame) > MAX_BATCH_SIZE:
        raise FeatureContractError(
            f"CSV has {len(frame)} rows; the maximum batch size is {MAX_BATCH_SIZE}."
        )
    if "student_id" not in frame.columns:
        raise FeatureContractError("CSV must include a 'student_id' column")

    missing = [column for column in RAW_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise FeatureContractError(f"CSV is missing required feature column(s): {missing}")
    if "hostel_status" not in frame.columns and "is_hosteler" not in frame.columns:
        raise FeatureContractError("CSV must include either 'hostel_status' or 'is_hosteler'")

    accepted = set(StudentFeatureInput.model_fields)
    requests: List[PredictionRequest] = []
    for position, row in enumerate(frame.to_dict(orient="records")):
        # Keep only recognised feature fields: silently drops labels, engineered columns
        # and protected attributes that a real features.csv export carries.
        features = {
            key: value
            for key, value in row.items()
            if key in accepted and value is not None and not pd.isna(value)
        }
        try:
            requests.append(
                PredictionRequest(
                    student_id=str(row["student_id"]),
                    features=StudentFeatureInput(**features),
                )
            )
        except Exception as exc:  # noqa: BLE001 — report the offending row to the caller
            raise FeatureContractError(f"Row {position} (student_id={row['student_id']}): {exc}") from exc

    duplicates = _duplicate_ids([item.student_id for item in requests])
    if duplicates:
        raise FeatureContractError(f"Duplicate student_id(s) in CSV: {duplicates}")

    results = prediction_service.score_and_persist(
        db,
        requests,
        include_explanations=include_explanations,
        persist_top_drivers=include_explanations,
    )
    return _build_batch_response(results, sort_by_risk_desc=sort_by_risk_desc)


def _duplicate_ids(student_ids: List[str]) -> List[str]:
    """
    Duplicates within one batch would make two rows fight over the same student record,
    leaving a non-deterministic winner. Rejected up front.
    """
    seen, duplicates = set(), []
    for student_id in student_ids:
        if student_id in seen and student_id not in duplicates:
            duplicates.append(student_id)
        seen.add(student_id)
    return duplicates


def _build_batch_response(
    results: List[PredictionResponse], *, sort_by_risk_desc: bool
) -> BatchPredictionResponse:
    counts = prediction_service.tier_counts(results)
    ordered = (
        sorted(results, key=lambda r: r.calibrated_risk_probability, reverse=True)
        if sort_by_risk_desc
        else results
    )
    return BatchPredictionResponse(
        total_students_evaluated=len(results),
        high_risk_count=counts["High"],
        medium_risk_count=counts["Medium"],
        low_risk_count=counts["Low"],
        model_version=results[0].model_version if results else None,
        results=ordered,
    )
