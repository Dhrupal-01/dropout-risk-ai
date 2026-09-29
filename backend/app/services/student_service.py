"""Student persistence helpers."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.latest_prediction import LatestPrediction
from backend.app.models.prediction import Prediction
from backend.app.models.student import Student
from backend.app.services.ml_service import LABEL_COLUMNS
from backend.app.services.priority_scoring import compute_priority_score


def _reject_labels(features: Dict[str, Any]) -> Dict[str, Any]:
    """Guard rail: labels must never be persisted as inference features."""
    leaked = LABEL_COLUMNS.intersection(features)
    if leaked:
        raise ValueError(f"Refusing to store label column(s) as features: {sorted(leaked)}")
    return features


def get_by_student_id(db: Session, student_id: str) -> Optional[Student]:
    return db.execute(select(Student).where(Student.student_id == student_id)).scalar_one_or_none()


def list_students(
    db: Session, *, department: Optional[str] = None, limit: int = 50, offset: int = 0
) -> Sequence[Student]:
    stmt = select(Student).order_by(Student.student_id)
    if department:
        stmt = stmt.where(Student.department == department)
    return db.execute(stmt.limit(limit).offset(offset)).scalars().all()


def upsert_student(
    db: Session,
    *,
    student_id: str,
    features: Dict[str, Any],
    name: Optional[str] = None,
    department: Optional[str] = None,
    assigned_mentor_id: Optional[str] = None,
    flush: bool = True,
) -> Student:
    """Insert or update by institutional roll number. Caller commits."""
    _reject_labels(features)

    student = get_by_student_id(db, student_id)
    if student is None:
        student = Student(id=uuid.uuid4(), student_id=student_id)
        db.add(student)

    student.features = features
    if name is not None:
        student.name = name
    if department is not None:
        student.department = department
    if assigned_mentor_id is not None:
        student.assigned_mentor_id = assigned_mentor_id

    # If student metadata changes and latest_prediction exists, sync department/mentor
    if student.latest_prediction is not None:
        if department is not None:
            student.latest_prediction.department = department
        if assigned_mentor_id is not None:
            student.latest_prediction.assigned_mentor_id = assigned_mentor_id

    if flush:
        db.flush()
    return student


def record_prediction(
    db: Session,
    *,
    student: Student,
    calibrated_risk_probability: float,
    risk_tier: str,
    risk_score_percentage: float,
    input_features: Dict[str, Any],
    model_version: Optional[str] = None,
    top_drivers: Optional[List[Dict[str, Any]]] = None,
    flush: bool = True,
) -> Prediction:
    """Append to prediction history and upsert latest_prediction pointer. Caller commits."""
    now_dt = datetime.now(timezone.utc)
    pred_id = uuid.uuid4()
    prediction = Prediction(
        id=pred_id,
        student_id=student.id,
        calibrated_risk_probability=calibrated_risk_probability,
        risk_tier=risk_tier,
        risk_score_percentage=risk_score_percentage,
        model_version=model_version,
        input_features=input_features,
        top_drivers=top_drivers,
        evaluated_at=now_dt,
    )
    db.add(prediction)

    # Compute deterministic priority score
    priority_score = compute_priority_score(
        calibrated_risk_probability=calibrated_risk_probability,
        backlog_count=input_features.get("backlog_count", 0),
        attendance_percentage=input_features.get("attendance_percentage", 100.0),
    )

    # Upsert latest_predictions in the same transaction
    latest = db.get(LatestPrediction, student.id)
    if latest is None:
        latest = LatestPrediction(
            student_id=student.id,
            prediction_id=pred_id,
            calibrated_risk_probability=calibrated_risk_probability,
            risk_tier=risk_tier,
            priority_score=priority_score,
            evaluated_at=now_dt,
            department=student.department,
            assigned_mentor_id=student.assigned_mentor_id,
        )
        db.add(latest)
    else:
        latest.prediction_id = pred_id
        latest.calibrated_risk_probability = calibrated_risk_probability
        latest.risk_tier = risk_tier
        latest.priority_score = priority_score
        latest.evaluated_at = now_dt
        latest.department = student.department
        latest.assigned_mentor_id = student.assigned_mentor_id

    if flush:
        db.flush()
    return prediction


def latest_prediction(db: Session, student: Student) -> Optional[Prediction]:
    """Most recent score for a student, served by latest_predictions table."""
    latest = db.get(LatestPrediction, student.id)
    if latest is not None:
        return db.get(Prediction, latest.prediction_id)

    stmt = (
        select(Prediction)
        .where(Prediction.student_id == student.id)
        .order_by(Prediction.evaluated_at.desc(), Prediction.id.desc())
        .limit(1)
    )
    return db.execute(stmt).scalar_one_or_none()
