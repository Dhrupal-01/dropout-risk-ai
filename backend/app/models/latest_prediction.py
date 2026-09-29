"""
Materialized Latest Prediction Pointer Table.
Context: DropoutGuard Phase 5 Backend Performance & SQL-Native Queue.

Maintains exactly one row per student containing their latest prediction state and priority score.
Accelerates mentor triage queue, filtering, and stats aggregations from O(N) in-memory pandas
to O(1) indexed SQL queries.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, desc
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


class LatestPrediction(Base):
    __tablename__ = "latest_predictions"

    student_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("predictions.id", ondelete="RESTRICT"), nullable=False
    )

    calibrated_risk_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    department: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    assigned_mentor_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    student: Mapped["Student"] = relationship(back_populates="latest_prediction")  # noqa: F821
    prediction: Mapped["Prediction"] = relationship()  # noqa: F821

    __table_args__ = (
        Index("ix_latest_predictions_priority_score_student_id", desc("priority_score"), "student_id"),
        Index("ix_latest_predictions_risk_tier_priority_score", "risk_tier", desc("priority_score")),
        Index("ix_latest_predictions_department", "department"),
        Index("ix_latest_predictions_assigned_mentor_id", "assigned_mentor_id"),
    )

    def __repr__(self) -> str:
        return f"<LatestPrediction student_id={self.student_id} tier={self.risk_tier} priority={self.priority_score:.6f}>"
