"""
Explanation endpoint.

    GET /api/v1/students/{student_id}/explanation

Explains the student's LATEST PERSISTED PREDICTION, using the exact 37-feature snapshot
that produced it — never the current mutable `students.features` record. If the student's
features have been updated since they were scored, explaining the live record would
attribute drivers to a probability that was never computed from them, and the SHAP values
would not reconcile with the displayed risk score.

SHAP output semantics are passed through untouched: `shap_value` keeps its sign, and
`impact_direction` keeps the RISK_INCREASING / RISK_DECREASING convention set by
`ml.models.explain_shap`.
"""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.core.errors import NoPredictionError, StudentNotFoundError
from backend.app.db.session import get_db
from backend.app.schemas.explanation import ExplanationResponse
from backend.app.services import student_service
from backend.app.services.ml_service import ml_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/students")


@router.get(
    "/{student_id}/explanation",
    response_model=ExplanationResponse,
    summary="Top SHAP drivers behind the student's latest prediction",
)
def get_explanation(
    student_id: str,
    top_k: int = Query(default=5, ge=1, le=37, description="Number of drivers to return."),
    db: Session = Depends(get_db),
) -> ExplanationResponse:
    """
    Returns drivers stored with the prediction when available, so the explanation shown
    is byte-for-byte the one recorded at scoring time. Otherwise they are recomputed
    lazily from the persisted input snapshot, which yields the same result because the
    snapshot is immutable.
    """
    student = student_service.get_by_student_id(db, student_id)
    if student is None:
        raise StudentNotFoundError(student_id)

    prediction = student_service.latest_prediction(db, student)
    if prediction is None:
        raise NoPredictionError(student_id)

    stored = prediction.top_drivers
    if stored and len(stored) >= top_k:
        drivers = stored[:top_k]
    else:
        # Recompute from the immutable snapshot — never from student.features.
        drivers = ml_service.explain_from_snapshot(prediction.input_features, top_k=top_k)

    return ExplanationResponse(
        student_id=student_id,
        risk_probability=prediction.calibrated_risk_probability,
        risk_tier=prediction.risk_tier,
        top_drivers=drivers,
    )
