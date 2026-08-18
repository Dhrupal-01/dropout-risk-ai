"""Student persistence helpers."""

from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.prediction import Prediction
from backend.app.models.student import Student
from backend.app.services.ml_service import LABEL_COLUMNS


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
        student = Student(student_id=student_id)
        db.add(student)

    student.features = features
    if name is not None:
        student.name = name
    if department is not None:
        student.department = department
    if assigned_mentor_id is not None:
        student.assigned_mentor_id = assigned_mentor_id

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
    """Append to prediction history. Existing rows are never modified. Caller commits."""
    prediction = Prediction(
        student_id=student.id,
        calibrated_risk_probability=calibrated_risk_probability,
        risk_tier=risk_tier,
        risk_score_percentage=risk_score_percentage,
        model_version=model_version,
        input_features=input_features,
        top_drivers=top_drivers,
    )
    db.add(prediction)
    if flush:
        db.flush()
    return prediction


def latest_prediction(db: Session, student: Student) -> Optional[Prediction]:
    """Most recent score for a student, served by ix_predictions_student_id_evaluated_at."""
    # id DESC is a deterministic tiebreaker: without it, two rows sharing an
    # evaluated_at would return in arbitrary order and "latest" would be ambiguous.
    stmt = (
        select(Prediction)
        .where(Prediction.student_id == student.id)
        .order_by(Prediction.evaluated_at.desc(), Prediction.id.desc())
        .limit(1)
    )
    return db.execute(stmt).scalar_one_or_none()
