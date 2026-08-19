"""
Student reference cohort.

`features` holds the canonical RAW inference input only — the 28 raw model columns plus
`hostel_status` (from which `is_hosteler` is derived). The 9 engineered columns are NOT
stored: they are deterministically recomputed by `build_engineered_features` at predict
time, so persisting them would create a second source of truth that can silently drift.

`is_dropout` and `ground_truth_risk_prob` are labels, not features. They are never stored
here and never reach `predict_proba`.
"""

import uuid
from typing import Any, Dict, List

from sqlalchemy import Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin, uuid_pk


class Student(Base, TimestampMixin):
    __tablename__ = "students"

    id: Mapped[uuid.UUID] = uuid_pk()

    # Institutional roll number, e.g. "IND_2026_0001". The business key the UI works with.
    student_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # Display metadata. Absent from the ML dataset, so it is synthesised for the dashboard
    # and deliberately kept out of `features`.
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    department: Mapped[str | None] = mapped_column(String(64), nullable=True)
    assigned_mentor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    features: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # NO delete cascade, deliberately. Prediction and intervention history is an audit
    # record: deleting a student must not silently erase why they were ever flagged or
    # what support they were offered. The database enforces this with ON DELETE RESTRICT,
    # so removing a student with history requires an explicit, considered archival step
    # rather than happening as a side effect of an ORM delete.
    predictions: Mapped[List["Prediction"]] = relationship(  # noqa: F821
        back_populates="student", passive_deletes=True
    )
    intervention_logs: Mapped[List["InterventionLog"]] = relationship(  # noqa: F821
        back_populates="student", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_students_student_id", "student_id", unique=True),
        Index("ix_students_department", "department"),
        Index("ix_students_assigned_mentor_id", "assigned_mentor_id"),
    )

    def __repr__(self) -> str:
        return f"<Student {self.student_id}>"
