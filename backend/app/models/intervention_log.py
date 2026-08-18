"""
Operational intervention lifecycle.

STATUS VALUES ARE TAKEN FROM THE CODE, NOT THE PROSE. See
`ml.intervention.engine.InterventionStatusTracker`:

  * The tracker performs NO validation — `update_intervention_status` writes whatever
    string it is handed. There is no runtime allowlist to import, so the values below are
    the ones the implementation actually documents and emits.
  * Lifecycle statuses come from the class docstring (`ASSIGNED -> IN_PROGRESS -> APPLIED
    -> COMPLETED`); `ASSIGNED` is the only one the code writes literally, at assignment.
  * Outcome statuses are those the code actually PRODUCES: `PENDING_EVALUATION` on
    creation, then `IMPROVED` / `NO_CHANGE` / `DETERIORATED` from the risk-delta branch.

Two documented discrepancies (code wins, per the handover rule):
  1. The docstring advertises `RESOLVED`, which no code path can ever produce.
  2. The code produces `DETERIORATED` (risk_delta < -0.05), which the docstring omits.
"""

import uuid
from datetime import date
from typing import Optional

from sqlalchemy import CheckConstraint, Date, Float, ForeignKey, Index, String, Text, desc
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin, uuid_pk

INTERVENTION_STATUSES = ("ASSIGNED", "IN_PROGRESS", "APPLIED", "COMPLETED")
INTERVENTION_OUTCOME_STATUSES = ("PENDING_EVALUATION", "IMPROVED", "NO_CHANGE", "DETERIORATED")


class InterventionLog(Base, TimestampMixin):
    __tablename__ = "intervention_logs"

    id: Mapped[uuid.UUID] = uuid_pk()

    student_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("students.id", ondelete="RESTRICT"), nullable=False
    )

    # Catalog key from ml/artifacts/interventions.json, e.g. "INT_ATT_01".
    intervention_id: Mapped[str] = mapped_column(String(32), nullable=False)

    assigned_faculty_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ASSIGNED")

    # Mirrors the tracker's second axis; required to represent its lifecycle faithfully.
    outcome_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PENDING_EVALUATION"
    )
    baseline_risk_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    post_intervention_risk_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    scheduled_followup_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    student: Mapped["Student"] = relationship(back_populates="intervention_logs")  # noqa: F821

    __table_args__ = (
        CheckConstraint(
            "status IN ('ASSIGNED', 'IN_PROGRESS', 'APPLIED', 'COMPLETED')",
            name="ck_intervention_logs_status",
        ),
        CheckConstraint(
            "outcome_status IN ('PENDING_EVALUATION', 'IMPROVED', 'NO_CHANGE', 'DETERIORATED')",
            name="ck_intervention_logs_outcome_status",
        ),
        Index("ix_intervention_logs_student_id_updated_at", "student_id", desc("updated_at")),
        Index("ix_intervention_logs_faculty_status", "assigned_faculty_id", "status"),
    )

    @property
    def risk_delta(self) -> Optional[float]:
        """Baseline minus post-intervention risk, matching the tracker's sign convention."""
        if self.baseline_risk_probability is None or self.post_intervention_risk_probability is None:
            return None
        return round(self.baseline_risk_probability - self.post_intervention_risk_probability, 4)

    def __repr__(self) -> str:
        return f"<InterventionLog {self.intervention_id} {self.status}>"
