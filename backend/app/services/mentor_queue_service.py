"""
Mentor triage queue.

Ranking is delegated to `ml.intervention.engine.build_prioritized_mentor_queue`, which
remains the correct interface: it sorts by calibrated risk descending, breaks ties by
backlog count then attendance deficit, and stamps `queue_rank`. This module's job is to
get the right rows out of PostgreSQL and shape them into the columns that function reads.

LATEST PREDICTION ONLY
`predictions` is append-only, so a student scored five times has five rows. The queue must
rank each student once, on their most recent score. That is done in SQL with
`DISTINCT ON (student_id) ... ORDER BY student_id, evaluated_at DESC`, served by
ix_predictions_student_id_evaluated_at.

NO GROUND TRUTH IN RANKING
`build_prioritized_mentor_queue` picks its sort column as
`calibrated_prob`, else `ground_truth_risk_prob`, else none. Ranking students by a
ground-truth label would be both a leak and ethically indefensible, so this module always
supplies `calibrated_prob` and never places any label column in the frame. The invariant
is asserted before the call rather than assumed.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from backend.app.models.intervention_log import InterventionLog
from backend.app.models.prediction import Prediction
from backend.app.models.student import Student
from backend.app.services.ml_service import LABEL_COLUMNS

logger = logging.getLogger(__name__)

# Tie-breaker columns build_prioritized_mentor_queue sorts on, plus the fields the
# dashboard shows. Sourced from the prediction's own input snapshot so every number
# displayed belongs to the score displayed beside it.
SNAPSHOT_FIELDS = (
    "attendance_percentage",
    "current_cgpa",
    "backlog_count",
    "fee_payment_delay_days",
)


def _latest_prediction_subquery() -> Select:
    """One row per student: their most recent prediction."""
    return (
        select(Prediction)
        .distinct(Prediction.student_id)
        .order_by(Prediction.student_id, Prediction.evaluated_at.desc())
        .subquery()
    )


def _latest_intervention_map(db: Session, student_ids: List[Any]) -> Dict[Any, InterventionLog]:
    """Most recently updated intervention log per student, for the given students."""
    if not student_ids:
        return {}
    rows = (
        db.execute(
            select(InterventionLog)
            .where(InterventionLog.student_id.in_(student_ids))
            .order_by(InterventionLog.student_id, InterventionLog.updated_at.desc())
        )
        .scalars()
        .all()
    )
    latest: Dict[Any, InterventionLog] = {}
    for row in rows:
        latest.setdefault(row.student_id, row)
    return latest


def build_queue(
    db: Session,
    *,
    department: Optional[str] = None,
    risk_tier: Optional[str] = None,
    assigned_mentor_id: Optional[str] = None,
    limit: int = 25,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Return (page_of_rows, total_matching).

    Filtering happens in SQL to keep the working set small; ranking then runs through the
    existing ML function over the filtered set, so `priority_rank` is a true cohort-wide
    rank rather than a per-page artefact. Pagination slices after ranking.
    """
    latest = _latest_prediction_subquery()

    stmt = select(Student, latest).join(latest, Student.id == latest.c.student_id)
    if department:
        stmt = stmt.where(func.lower(Student.department) == department.lower())
    if assigned_mentor_id:
        stmt = stmt.where(Student.assigned_mentor_id == assigned_mentor_id)
    if risk_tier:
        stmt = stmt.where(func.lower(latest.c.risk_tier) == risk_tier.lower())

    rows = db.execute(stmt).all()
    if not rows:
        return [], 0

    students = [row[0] for row in rows]
    intervention_by_student = _latest_intervention_map(db, [s.id for s in students])

    records: List[Dict[str, Any]] = []
    for row in rows:
        student = row[0]
        snapshot = row._mapping["input_features"] or {}
        log = intervention_by_student.get(student.id)

        records.append(
            {
                "student_id": student.student_id,
                "name": student.name,
                "department": student.department,
                "assigned_mentor_id": student.assigned_mentor_id,
                # The column name build_prioritized_mentor_queue sorts on.
                "calibrated_prob": float(row._mapping["calibrated_risk_probability"]),
                "risk_tier": row._mapping["risk_tier"],
                "risk_score_percentage": float(row._mapping["risk_score_percentage"]),
                "evaluated_at": row._mapping["evaluated_at"],
                "model_version": row._mapping["model_version"],
                "attendance_percentage": _snapshot_value(snapshot, "attendance_percentage"),
                "current_cgpa": _snapshot_value(snapshot, "current_cgpa"),
                "backlog_count": _snapshot_value(snapshot, "backlog_count"),
                "fee_payment_delay_days": _snapshot_value(snapshot, "fee_payment_delay_days"),
                "primary_intervention": log.intervention_id if log else None,
                "intervention_status": log.status if log else None,
                "intervention_outcome_status": log.outcome_status if log else None,
            }
        )

    frame = pd.DataFrame(records)

    # Guard the ML function's fallback: it would rank by ground_truth_risk_prob if
    # calibrated_prob were missing. Neither condition may ever hold.
    assert "calibrated_prob" in frame.columns, "calibrated_prob missing - ranking would fall back"
    leaked = LABEL_COLUMNS.intersection(frame.columns)
    if leaked:
        raise RuntimeError(f"Label column(s) {sorted(leaked)} must never enter the mentor queue")

    from ml.intervention.engine import build_prioritized_mentor_queue

    # Filters were already applied in SQL; passing None avoids re-filtering the same data.
    ranked = build_prioritized_mentor_queue(frame)

    total = len(ranked)
    page = ranked.iloc[offset : offset + limit]

    # Read ONLY the ordering back out of pandas, then emit from the original Python
    # dicts. Round-tripping values through a DataFrame silently coerces None to NaN in
    # mixed columns -- a student with no name would come back as float('nan') and fail
    # Optional[str] validation. Keeping the native records avoids every such dtype
    # surprise for display fields.
    by_student_id = {record["student_id"]: record for record in records}

    results: List[Dict[str, Any]] = []
    for student_id, queue_rank in zip(page["student_id"], page["queue_rank"]):
        record = by_student_id[student_id]
        results.append(
            {
                "priority_rank": int(queue_rank),
                "student_id": record["student_id"],
                "name": record["name"],
                "department": record["department"],
                "assigned_mentor_id": record["assigned_mentor_id"],
                "risk_probability": round(record["calibrated_prob"], 4),
                "risk_tier": record["risk_tier"],
                "risk_score_percentage": round(record["risk_score_percentage"], 2),
                "attendance": _optional_float(record["attendance_percentage"]),
                "cgpa": _optional_float(record["current_cgpa"]),
                "backlogs": _optional_int(record["backlog_count"]),
                "fee_delay_days": _optional_int(record["fee_payment_delay_days"]),
                "primary_intervention": record["primary_intervention"],
                "intervention_status": record["intervention_status"],
                "intervention_outcome_status": record["intervention_outcome_status"],
                "evaluated_at": record["evaluated_at"],
            }
        )

    return results, total


def _snapshot_value(snapshot: Dict[str, Any], key: str) -> Optional[float]:
    value = snapshot.get(key)
    return None if value is None else float(value)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None or pd.isna(value) else round(float(value), 2)


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None or pd.isna(value) else int(value)
