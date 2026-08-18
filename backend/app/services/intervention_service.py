"""
Intervention persistence and lifecycle.

CATALOG OWNERSHIP
`ml/artifacts/interventions.json` is the single source of truth for what an intervention
IS (id, title, description, pillar, urgency, duration). PostgreSQL stores only what
HAPPENED: assignment, status, notes, follow-up dates, outcome. There is deliberately no
interventions table — a second catalog would drift from the artifact.

LIFECYCLE ENFORCEMENT — READ THIS BEFORE CHANGING IT
`ml.intervention.engine.InterventionStatusTracker.update_intervention_status` performs NO
validation whatsoever: it executes `rec["status"] = new_status` for any string handed to
it, and there is no transition table anywhere in the ML package. So there are no existing
transition rules to reuse.

The rules below are therefore an ADDITION made at the persistence layer, derived from the
lifecycle the tracker's own class docstring documents:

    ASSIGNED -> IN_PROGRESS -> APPLIED -> COMPLETED

Forward moves and same-state updates are allowed; backward moves are rejected, and
COMPLETED is terminal. This satisfies the requirement not to silently accept invalid
transitions while leaving the ML tracker itself untouched.

OUTCOME CLASSIFICATION is reused exactly: the +/-0.05 risk-delta thresholds mirror
`update_intervention_status`, so the database and the in-memory tracker never disagree.
"""

import json
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import INTERVENTION_CATALOG_PATH
from backend.app.models.intervention_log import InterventionLog
from backend.app.models.student import Student

logger = logging.getLogger(__name__)

# Same threshold the ML tracker uses to classify an outcome.
RISK_DELTA_THRESHOLD = 0.05

# Ordered lifecycle from the tracker's class docstring. Index = progression rank.
LIFECYCLE_ORDER: Dict[str, int] = {
    "ASSIGNED": 0,
    "IN_PROGRESS": 1,
    "APPLIED": 2,
    "COMPLETED": 3,
}
TERMINAL_STATUS = "COMPLETED"


class InvalidLifecycleTransition(Exception):
    """Raised when a status change would move an intervention backwards or past COMPLETED."""

    def __init__(self, current: str, requested: str) -> None:
        self.current = current
        self.requested = requested
        if current == TERMINAL_STATUS:
            reason = f"'{current}' is terminal; the intervention is already closed"
        else:
            reason = (
                "lifecycle only moves forward: "
                + " -> ".join(sorted(LIFECYCLE_ORDER, key=LIFECYCLE_ORDER.get))
            )
        super().__init__(f"Cannot change status from '{current}' to '{requested}' - {reason}.")


@lru_cache
def load_intervention_catalog() -> Dict[str, Dict[str, Any]]:
    """Read the codified catalog artifact once per process."""
    if not INTERVENTION_CATALOG_PATH.exists():
        return {}
    return json.loads(INTERVENTION_CATALOG_PATH.read_text(encoding="utf-8"))


def catalog_items() -> List[Dict[str, Any]]:
    return list(load_intervention_catalog().values())


def is_known_intervention(intervention_id: str) -> bool:
    catalog = load_intervention_catalog()
    return not catalog or intervention_id in catalog


def validate_transition(current: str, requested: str) -> None:
    """Allow same-state and forward moves only. COMPLETED is terminal."""
    if current == requested and current != TERMINAL_STATUS:
        return
    if current == TERMINAL_STATUS:
        raise InvalidLifecycleTransition(current, requested)
    if LIFECYCLE_ORDER[requested] < LIFECYCLE_ORDER[current]:
        raise InvalidLifecycleTransition(current, requested)


def classify_outcome(baseline: Optional[float], post: Optional[float]) -> str:
    """
    Mirror of the tracker's branch:
        delta > 0.05  -> IMPROVED
        delta < -0.05 -> DETERIORATED
        otherwise     -> NO_CHANGE
    """
    if baseline is None or post is None:
        return "PENDING_EVALUATION"
    delta = round(baseline - post, 4)
    if delta > RISK_DELTA_THRESHOLD:
        return "IMPROVED"
    if delta < -RISK_DELTA_THRESHOLD:
        return "DETERIORATED"
    return "NO_CHANGE"


def get_active_log(
    db: Session, student: Student, intervention_id: str
) -> Optional[InterventionLog]:
    """
    The open log for this student+intervention, if any.

    Matches the tracker's model of one mutable record per assignment: a COMPLETED log is
    closed, so assigning the same intervention again starts a fresh cycle.
    """
    stmt = (
        select(InterventionLog)
        .where(
            InterventionLog.student_id == student.id,
            InterventionLog.intervention_id == intervention_id,
            InterventionLog.status != TERMINAL_STATUS,
        )
        .order_by(InterventionLog.created_at.desc())
        .limit(1)
    )
    return db.execute(stmt).scalar_one_or_none()


def create_log(
    db: Session,
    *,
    student: Student,
    intervention_id: str,
    assigned_faculty_id: Optional[str] = None,
    status: str = "ASSIGNED",
    notes: Optional[str] = None,
    scheduled_followup_date=None,
    baseline_risk_probability: Optional[float] = None,
    flush: bool = True,
) -> InterventionLog:
    """Assign an intervention. Caller commits."""
    if not is_known_intervention(intervention_id):
        raise ValueError(
            f"Unknown intervention_id '{intervention_id}'. "
            f"Known: {sorted(load_intervention_catalog())}"
        )

    log = InterventionLog(
        student_id=student.id,
        intervention_id=intervention_id,
        assigned_faculty_id=assigned_faculty_id,
        status=status,
        outcome_status="PENDING_EVALUATION",
        notes=notes,
        scheduled_followup_date=scheduled_followup_date,
        baseline_risk_probability=baseline_risk_probability,
    )
    db.add(log)
    if flush:
        db.flush()
    return log


def update_log(
    db: Session,
    *,
    log: InterventionLog,
    status: Optional[str] = None,
    outcome_status: Optional[str] = None,
    assigned_faculty_id: Optional[str] = None,
    notes: Optional[str] = None,
    scheduled_followup_date=None,
    post_intervention_risk_probability: Optional[float] = None,
    flush: bool = True,
) -> InterventionLog:
    """
    Advance a log in place — architecture (A), matching the tracker, which mutates one
    record per assignment rather than appending event rows.

    `created_at` is never touched, so the original assignment time survives; `updated_at`
    is bumped by the database.

    When a post-intervention probability arrives the outcome is recomputed from the risk
    delta, overriding any explicitly supplied outcome_status — the same precedence the ML
    tracker applies.
    """
    if status is not None:
        validate_transition(log.status, status)
        log.status = status
    if assigned_faculty_id is not None:
        log.assigned_faculty_id = assigned_faculty_id
    if scheduled_followup_date is not None:
        log.scheduled_followup_date = scheduled_followup_date
    if notes:
        # The tracker appends with a ' | ' separator rather than overwriting.
        log.notes = f"{log.notes} | {notes}".strip(" |") if log.notes else notes

    if outcome_status is not None:
        log.outcome_status = outcome_status

    if post_intervention_risk_probability is not None:
        log.post_intervention_risk_probability = post_intervention_risk_probability
        log.outcome_status = classify_outcome(
            log.baseline_risk_probability, post_intervention_risk_probability
        )

    if flush:
        db.flush()
    return log


def log_or_advance(
    db: Session,
    *,
    student: Student,
    intervention_id: str,
    status: str = "ASSIGNED",
    assigned_faculty_id: Optional[str] = None,
    notes: Optional[str] = None,
    scheduled_followup_date=None,
    baseline_risk_probability: Optional[float] = None,
    post_intervention_risk_probability: Optional[float] = None,
) -> tuple[InterventionLog, bool]:
    """
    Single entry point for POST /interventions/log.

    Creates a log when the student has no open one for this intervention, otherwise
    advances the existing one through the validated lifecycle.

    Returns (log, created).
    """
    if not is_known_intervention(intervention_id):
        raise ValueError(
            f"Unknown intervention_id '{intervention_id}'. "
            f"Known: {sorted(load_intervention_catalog())}"
        )

    existing = get_active_log(db, student, intervention_id)
    if existing is None:
        log = create_log(
            db,
            student=student,
            intervention_id=intervention_id,
            assigned_faculty_id=assigned_faculty_id,
            status=status,
            notes=notes,
            scheduled_followup_date=scheduled_followup_date,
            baseline_risk_probability=baseline_risk_probability,
        )
        if post_intervention_risk_probability is not None:
            update_log(
                db,
                log=log,
                post_intervention_risk_probability=post_intervention_risk_probability,
            )
        return log, True

    update_log(
        db,
        log=existing,
        status=status,
        assigned_faculty_id=assigned_faculty_id,
        notes=notes,
        scheduled_followup_date=scheduled_followup_date,
        post_intervention_risk_probability=post_intervention_risk_probability,
    )
    return existing, False


def list_for_student(db: Session, student: Student) -> Sequence[InterventionLog]:
    stmt = (
        select(InterventionLog)
        .where(InterventionLog.student_id == student.id)
        .order_by(InterventionLog.updated_at.desc())
    )
    return db.execute(stmt).scalars().all()


def list_for_faculty(
    db: Session, faculty_id: str, *, status: Optional[str] = None
) -> Sequence[InterventionLog]:
    """Served by ix_intervention_logs_faculty_status."""
    stmt = select(InterventionLog).where(InterventionLog.assigned_faculty_id == faculty_id)
    if status:
        stmt = stmt.where(InterventionLog.status == status)
    return db.execute(stmt.order_by(InterventionLog.updated_at.desc())).scalars().all()
