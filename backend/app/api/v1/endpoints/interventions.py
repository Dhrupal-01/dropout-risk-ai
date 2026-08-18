"""
Intervention endpoints.

    GET  /api/v1/students/{student_id}/interventions   recommendations + recourse
    POST /api/v1/interventions/log                     assign or advance a log
    GET  /api/v1/interventions/catalog                 the 12 codified actions

The catalog lives in ml/artifacts/interventions.json and is never duplicated into
PostgreSQL; the database records only what happened to a student.

Both `map_shap_drivers_to_interventions` and `CounterfactualRecourseEngine` are called
as-is — no ranking, mapping or recourse logic is reimplemented here.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.core.errors import NoPredictionError, StudentNotFoundError
from backend.app.db.session import get_db
from backend.app.schemas.intervention import (
    InterventionCatalogItem,
    InterventionLogCreate,
    InterventionLogRead,
    InterventionLogResponse,
    InterventionPlanResponse,
)
from backend.app.services import intervention_service, student_service
from backend.app.services.ml_service import ml_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interventions")

# Mounted separately so the path matches the handover spec exactly.
student_router = APIRouter(prefix="/students")


@student_router.get(
    "/{student_id}/interventions",
    response_model=InterventionPlanResponse,
    summary="Recommended support actions and a counterfactual projection",
)
def get_student_interventions(
    student_id: str,
    max_recommendations: int = 3,
    top_k_drivers: int = 4,
    db: Session = Depends(get_db),
) -> InterventionPlanResponse:
    """
    Built from the student's LATEST prediction and its exact input snapshot, so the
    recommendations explain the same score the dashboard displays.

    Recommendations are suggestions for a mentor; the recourse plan is a simulation, not
    a promise. Neither triggers any action on its own.
    """
    student = student_service.get_by_student_id(db, student_id)
    if student is None:
        raise StudentNotFoundError(student_id)

    prediction = student_service.latest_prediction(db, student)
    if prediction is None:
        raise NoPredictionError(student_id)

    snapshot = prediction.input_features

    # Reuse drivers stored at scoring time when they are detailed enough; otherwise
    # recompute from the immutable snapshot, which yields the same values.
    stored = prediction.top_drivers
    drivers = (
        stored[:top_k_drivers]
        if stored and len(stored) >= top_k_drivers
        else ml_service.explain_from_snapshot(snapshot, top_k=top_k_drivers)
    )

    recommendations = ml_service.recommend_interventions(
        drivers, max_recommendations=max_recommendations
    )

    recourse = ml_service.generate_counterfactual_from_snapshot(
        snapshot, top_shap_drivers=drivers, student_id=student_id
    )

    return InterventionPlanResponse(
        student_id=student_id,
        current_risk_probability=prediction.calibrated_risk_probability,
        current_risk_tier=prediction.risk_tier,
        evaluated_at=prediction.evaluated_at,
        model_version=prediction.model_version,
        recommended_interventions=recommendations,
        counterfactual_recourse=recourse,
    )


@router.get(
    "/catalog",
    response_model=List[InterventionCatalogItem],
    summary="The codified intervention catalog (source of truth: ML artifact)",
)
def get_catalog() -> List[InterventionCatalogItem]:
    return intervention_service.catalog_items()


@router.post(
    "/log",
    response_model=InterventionLogResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Assign an intervention, or advance one already assigned",
)
def log_intervention(
    payload: InterventionLogCreate, db: Session = Depends(get_db)
) -> InterventionLogResponse:
    """
    Creates a log when the student has no open one for this intervention; otherwise
    advances the existing log through the validated lifecycle
    (ASSIGNED -> IN_PROGRESS -> APPLIED -> COMPLETED).

    Backward transitions and changes to a COMPLETED log are rejected with 409.

    Runs in one transaction: the validation, the write and the outcome recomputation
    either all land or none do.
    """
    student = student_service.get_by_student_id(db, payload.student_id)
    if student is None:
        raise StudentNotFoundError(payload.student_id)

    try:
        log, created = intervention_service.log_or_advance(
            db,
            student=student,
            intervention_id=payload.intervention_id,
            status=payload.status,
            assigned_faculty_id=payload.assigned_faculty_id,
            notes=payload.notes,
            scheduled_followup_date=payload.scheduled_followup_date,
            baseline_risk_probability=payload.baseline_risk_probability,
            post_intervention_risk_probability=payload.post_intervention_risk_probability,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(log)
    return InterventionLogResponse(created=created, log=_enrich(log))


@student_router.get(
    "/{student_id}/interventions/history",
    response_model=List[InterventionLogRead],
    summary="Every intervention logged for this student",
)
def get_student_intervention_history(
    student_id: str, db: Session = Depends(get_db)
) -> List[InterventionLogRead]:
    student = student_service.get_by_student_id(db, student_id)
    if student is None:
        raise StudentNotFoundError(student_id)
    return [_enrich(log) for log in intervention_service.list_for_student(db, student)]


def _enrich(log) -> InterventionLogRead:
    """Attach catalog metadata so the UI needs no second lookup."""
    item = intervention_service.load_intervention_catalog().get(log.intervention_id, {})
    return InterventionLogRead(
        id=log.id,
        student_id=log.student_id,
        intervention_id=log.intervention_id,
        assigned_faculty_id=log.assigned_faculty_id,
        status=log.status,
        outcome_status=log.outcome_status,
        baseline_risk_probability=log.baseline_risk_probability,
        post_intervention_risk_probability=log.post_intervention_risk_probability,
        risk_delta=log.risk_delta,
        notes=log.notes,
        scheduled_followup_date=log.scheduled_followup_date,
        created_at=log.created_at,
        updated_at=log.updated_at,
        title=item.get("title"),
        pillar=item.get("pillar"),
        urgency=item.get("urgency"),
    )
