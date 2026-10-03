"""
Mentor triage queue service.
Context: DropoutGuard Phase 5 SQL-Native Queue & Database Optimization.

Performance Optimization:
- Pure SQL filtering, ordering, and pagination.
- Zero pandas in request path.
- Served by `latest_predictions` materialized pointer table and composite indexes.
- Keyset cursor pagination (optional) + traditional limit/offset pagination.
- Total count computed from COUNT query on identical filter predicates.
"""

import base64
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.orm import Session

from backend.app.models.intervention_log import InterventionLog
from backend.app.models.latest_prediction import LatestPrediction
from backend.app.models.prediction import Prediction
from backend.app.models.student import Student

logger = logging.getLogger(__name__)

SNAPSHOT_FIELDS = (
    "attendance_percentage",
    "current_cgpa",
    "backlog_count",
    "fee_payment_delay_days",
)


def _encode_cursor(priority_score: float, student_id: str) -> str:
    raw = f"{float(priority_score).hex()}::{student_id}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


class InvalidCursorError(Exception):
    """Raised when a keyset cursor cannot be decoded; the client must restart from page 1 itself."""

    def __init__(self, cursor: str) -> None:
        self.cursor = cursor
        super().__init__("The pagination cursor is not valid; request the first page without a cursor.")


def _decode_cursor(cursor_str: str) -> Tuple[float, str]:
    """Inverse of _encode_cursor. Any malformed cursor raises InvalidCursorError (never page 1)."""
    try:
        raw = base64.urlsafe_b64decode(cursor_str.encode("ascii")).decode("utf-8")
        score_hex, student_id = raw.split("::", 1)
        return float.fromhex(score_hex), student_id
    except (ValueError, UnicodeError) as exc:
        raise InvalidCursorError(cursor_str) from exc


def _latest_intervention_map(db: Session, student_uuids: List[Any]) -> Dict[Any, InterventionLog]:
    """Most recently updated intervention log per student for the page."""
    if not student_uuids:
        return {}
    rows = (
        db.execute(
            select(InterventionLog)
            .where(InterventionLog.student_id.in_(student_uuids))
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
    search: Optional[str] = None,
    cursor: Optional[str] = None,
    limit: int = 25,
    offset: int = 0,
    return_cursor: bool = False,
) -> Union[Tuple[List[Dict[str, Any]], int], Tuple[List[Dict[str, Any]], int, Optional[str]]]:
    """
    Return (page_of_rows, total_matching) or (page_of_rows, total_matching, next_cursor).
    Executes entirely in PostgreSQL using indexes on latest_predictions and students.
    """
    # Base filter criteria
    filters = []
    if department:
        filters.append(func.lower(LatestPrediction.department) == department.lower())
    if assigned_mentor_id:
        filters.append(LatestPrediction.assigned_mentor_id == assigned_mentor_id)
    if risk_tier:
        filters.append(func.lower(LatestPrediction.risk_tier) == risk_tier.lower())
    if search and search.strip():
        escaped = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        filters.append(
            or_(
                Student.student_id.ilike(pattern, escape="\\"),
                Student.name.ilike(pattern, escape="\\"),
            )
        )

    # 1. Total matching count query
    count_stmt = (
        select(func.count())
        .select_from(LatestPrediction)
        .join(Student, Student.id == LatestPrediction.student_id)
    )
    if filters:
        count_stmt = count_stmt.where(and_(*filters))

    total = db.scalar(count_stmt) or 0
    if total == 0:
        return ([], 0, None) if return_cursor else ([], 0)

    # 2. Main paginated query
    query = (
        select(
            Student.id.label("student_uuid"),
            Student.student_id.label("student_id"),
            Student.name.label("name"),
            LatestPrediction.department.label("department"),
            LatestPrediction.assigned_mentor_id.label("assigned_mentor_id"),
            LatestPrediction.calibrated_risk_probability.label("calibrated_prob"),
            LatestPrediction.risk_tier.label("risk_tier"),
            LatestPrediction.priority_score.label("priority_score"),
            LatestPrediction.evaluated_at.label("evaluated_at"),
            Prediction.risk_score_percentage.label("risk_score_percentage"),
            Prediction.input_features.label("input_features"),
        )
        .join(Student, Student.id == LatestPrediction.student_id)
        .join(Prediction, Prediction.id == LatestPrediction.prediction_id)
    )

    if filters:
        query = query.where(and_(*filters))

    # Determine starting rank offset
    rank_offset = offset

    # Keyset cursor pagination
    decoded_cursor = _decode_cursor(cursor) if cursor is not None else None
    if decoded_cursor:
        c_score, c_sid = decoded_cursor
        query = query.where(
            or_(
                LatestPrediction.priority_score < c_score,
                and_(
                    LatestPrediction.priority_score == c_score,
                    Student.student_id > c_sid,
                ),
            )
        )
        # Compute exact rank offset before cursor for consistent continuous priority_rank
        rank_count_stmt = count_stmt.where(
            or_(
                LatestPrediction.priority_score > c_score,
                and_(
                    LatestPrediction.priority_score == c_score,
                    Student.student_id <= c_sid,
                ),
            )
        )
        rank_offset = db.scalar(rank_count_stmt) or 0
    else:
        query = query.offset(offset)

    query = query.order_by(
        desc(LatestPrediction.priority_score),
        Student.student_id.asc(),
    ).limit(limit)

    rows = db.execute(query).all()
    if not rows:
        return ([], total, None) if return_cursor else ([], total)

    # 3. Fetch latest interventions only for this page
    student_uuids = [row.student_uuid for row in rows]
    intervention_by_student = _latest_intervention_map(db, student_uuids)

    # 4. Format page results
    results: List[Dict[str, Any]] = []
    last_score: Optional[float] = None
    last_sid: Optional[str] = None

    for idx, row in enumerate(rows):
        snapshot = row.input_features or {}
        log = intervention_by_student.get(row.student_uuid)
        current_rank = rank_offset + idx + 1
        last_score = row.priority_score
        last_sid = row.student_id

        results.append(
            {
                "priority_rank": int(current_rank),
                "student_id": row.student_id,
                "name": row.name,
                "department": row.department,
                "assigned_mentor_id": row.assigned_mentor_id,
                "risk_probability": round(float(row.calibrated_prob), 4),
                "risk_tier": row.risk_tier,
                "risk_score_percentage": round(float(row.risk_score_percentage), 2),
                "attendance": _snapshot_value(snapshot, "attendance_percentage"),
                "cgpa": _snapshot_value(snapshot, "current_cgpa"),
                "backlogs": _snapshot_int(snapshot, "backlog_count"),
                "fee_delay_days": _snapshot_int(snapshot, "fee_payment_delay_days"),
                "primary_intervention": log.intervention_id if log else None,
                "intervention_status": log.status if log else None,
                "intervention_outcome_status": log.outcome_status if log else None,
                "evaluated_at": row.evaluated_at,
            }
        )

    next_cursor = None
    if len(rows) == limit and last_score is not None and last_sid is not None:
        next_cursor = _encode_cursor(last_score, last_sid)

    if return_cursor:
        return results, total, next_cursor
    return results, total


def _snapshot_value(snapshot: Dict[str, Any], key: str) -> Optional[float]:
    val = snapshot.get(key)
    if val is None or val == "":
        return None
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return None


def _snapshot_int(snapshot: Dict[str, Any], key: str) -> Optional[int]:
    val = snapshot.get(key)
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def get_filters(db: Session) -> Dict[str, List[str]]:
    """Return distinct, non-null, sorted departments and assigned mentor IDs."""
    dept_rows = (
        db.execute(
            select(LatestPrediction.department)
            .where(LatestPrediction.department.isnot(None), LatestPrediction.department != "")
            .distinct()
        )
        .scalars()
        .all()
    )
    departments = sorted(dept_rows)

    mentor_rows = (
        db.execute(
            select(LatestPrediction.assigned_mentor_id)
            .where(LatestPrediction.assigned_mentor_id.isnot(None), LatestPrediction.assigned_mentor_id != "")
            .distinct()
        )
        .scalars()
        .all()
    )
    mentor_ids = sorted(mentor_rows)

    return {
        "departments": departments,
        "mentor_ids": mentor_ids,
    }
