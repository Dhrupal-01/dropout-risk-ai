"""
Prediction orchestration shared by the single, JSON-batch and CSV endpoints.

One code path scores, persists and shapes results, so the three entry points can never
drift apart. Scoring always goes through `MLService.predict_batch`, which issues a single
aligned `predict_student_risk` call regardless of batch size.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from backend.app.schemas.prediction import PredictionRequest, PredictionResponse
from backend.app.services import student_service
from backend.app.services.ml_service import RAW_FEATURE_COLUMNS, ml_service

logger = logging.getLogger(__name__)

# SHAP costs ~43ms/student against ~110ms for a single prediction, so drivers are stored
# by default on the single-student path: it makes the exact explanation behind a demo
# score permanently reproducible. Batch leaves it opt-in, where the per-row cost would
# dominate an otherwise vectorised call.
SINGLE_PREDICTION_TOP_K = 5


def canonical_stored_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    The representation persisted on `students.features`: the 28 raw model inputs plus the
    residency source column.

    `is_hosteler` is an ENGINEERED feature, so it is not stored when `hostel_status` is
    available to derive it from — that would create a second source of truth. It is kept
    only when it is the sole residency signal the caller provided.
    """
    stored = {column: raw[column] for column in RAW_FEATURE_COLUMNS if column in raw}
    if raw.get("hostel_status") is not None:
        stored["hostel_status"] = raw["hostel_status"]
    elif raw.get("is_hosteler") is not None:
        stored["is_hosteler"] = raw["is_hosteler"]
    return stored


def score_and_persist(
    db: Session,
    requests: Sequence[PredictionRequest],
    *,
    include_explanations: bool = False,
    persist_top_drivers: bool = True,
    top_k: int = SINGLE_PREDICTION_TOP_K,
) -> List[PredictionResponse]:
    """
    Score every request, upsert each student, and append one new prediction row each.

    Existing prediction rows are never modified — history is append-only.
    The caller owns the transaction boundary only in the sense that this commits once at
    the end; a failure anywhere rolls the whole batch back.
    """
    if not requests:
        return []

    raw_rows = [request.features.to_model_input() for request in requests]

    # Single vectorised inference call for the whole batch.
    scored = ml_service.predict_batch(raw_rows)

    responses: List[PredictionResponse] = []
    for request, raw, result in zip(requests, raw_rows, scored):
        drivers: Optional[List[Dict[str, Any]]] = None
        if include_explanations or persist_top_drivers:
            drivers = ml_service.explain_from_snapshot(result["input_features"], top_k=top_k)

        student = student_service.upsert_student(
            db,
            student_id=request.student_id,
            features=canonical_stored_features(raw),
            name=request.name,
            department=request.department,
            assigned_mentor_id=request.assigned_mentor_id,
        )

        prediction = student_service.record_prediction(
            db,
            student=student,
            calibrated_risk_probability=result["calibrated_risk_probability"],
            risk_tier=result["risk_tier"],
            risk_score_percentage=result["risk_score_percentage"],
            input_features=result["input_features"],
            model_version=result["model_version"],
            top_drivers=drivers if persist_top_drivers else None,
        )

        responses.append(
            PredictionResponse(
                student_id=request.student_id,
                calibrated_risk_probability=result["calibrated_risk_probability"],
                risk_tier=result["risk_tier"],
                risk_score_percentage=result["risk_score_percentage"],
                evaluation_timestamp=prediction.evaluated_at,
                model_version=result["model_version"],
                prediction_id=prediction.id,
                top_drivers=drivers if include_explanations else None,
            )
        )

    db.commit()
    logger.info("Scored and persisted %d student(s)", len(responses))
    return responses


def tier_counts(responses: Sequence[PredictionResponse]) -> Dict[str, int]:
    """Roll up tiers as assigned by the model — never recomputed from probabilities."""
    counts = {"Low": 0, "Medium": 0, "High": 0}
    for response in responses:
        counts[response.risk_tier] += 1
    return counts
