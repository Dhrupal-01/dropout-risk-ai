"""
Immutable prediction history.

Every inference is appended, never updated in place: `input_features` stores the exact
37-column snapshot fed to `predict_proba`, so any past score can be reproduced against
the model version that produced it.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Index, String, desc, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, uuid_pk

# Mirrors ml.config.get_risk_tier, which is the only function allowed to assign a tier.
RISK_TIERS = ("Low", "Medium", "High")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = uuid_pk()

    student_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("students.id", ondelete="RESTRICT"), nullable=False
    )

    calibrated_risk_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    risk_score_percentage: Mapped[float] = mapped_column(Float, nullable=False)

    # Fingerprint of the artifact that produced this row (see MLService.model_version).
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Exact ordered 37-feature vector used for this inference.
    input_features: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # Top SHAP drivers, when the explanation was computed alongside the prediction.
    # none_as_null=True is required: SQLAlchemy's default maps Python None to the JSON
    # scalar `null` rather than SQL NULL, so `WHERE top_drivers IS NULL` would match
    # nothing and jsonb_array_length() would fail on those rows.
    top_drivers: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )

    # clock_timestamp(), not now(): now() returns the TRANSACTION start time, so a batch of
    # predictions written in one transaction would share an identical evaluated_at and the
    # "most recent prediction" ordering would be ambiguous. clock_timestamp() advances per
    # statement, keeping this append-only history strictly ordered.
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.clock_timestamp()
    )

    student: Mapped["Student"] = relationship(back_populates="predictions")  # noqa: F821

    __table_args__ = (
        CheckConstraint(
            "calibrated_risk_probability >= 0.0 AND calibrated_risk_probability <= 1.0",
            name="ck_predictions_probability_range",
        ),
        CheckConstraint(
            "risk_tier IN ('Low', 'Medium', 'High')",
            name="ck_predictions_risk_tier",
        ),
        # Latest-prediction-per-student lookups.
        Index("ix_predictions_student_id_evaluated_at", "student_id", desc("evaluated_at")),
        # Mentor queue: "all High-risk students, most recent first".
        Index("ix_predictions_risk_tier_evaluated_at", "risk_tier", desc("evaluated_at")),
    )

    def __repr__(self) -> str:
        return f"<Prediction {self.student_id} {self.risk_tier} {self.calibrated_risk_probability:.4f}>"
