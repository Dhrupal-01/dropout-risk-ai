"""
Prioritised mentor worklist schemas (triage dashboard).

Ordering comes from `ml.intervention.engine.build_prioritized_mentor_queue`. Ranks are
cohort-wide over the filtered set, not per page, so page 2 continues from page 1.

Responsible AI: this is a SUPPORT worklist. `priority_rank` orders who a mentor should
reach out to first; it is not a ranking of who will drop out, and it authorises no
academic penalty.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from backend.app.schemas.prediction import RiskTier

QUEUE_DISCLAIMER = (
    "Support triage order based on model risk estimates. Not a prediction of certain "
    "dropout and not grounds for any punitive or automated academic action."
)


class MentorQueueItem(BaseModel):
    """One row of the triage queue."""

    priority_rank: int = Field(..., ge=1, description="Cohort-wide rank, highest risk first.")
    student_id: str
    name: Optional[str] = None
    department: Optional[str] = None
    assigned_mentor_id: Optional[str] = None

    risk_probability: float = Field(..., ge=0.0, le=1.0)
    risk_tier: RiskTier
    risk_score_percentage: float = Field(..., ge=0.0, le=100.0)
    evaluated_at: datetime

    # Drawn from the prediction's own input snapshot, so each number belongs to the score
    # shown beside it.
    attendance: Optional[float] = None
    cgpa: Optional[float] = None
    backlogs: Optional[int] = None
    fee_delay_days: Optional[int] = None

    primary_intervention: Optional[str] = None
    intervention_status: Optional[str] = None
    intervention_outcome_status: Optional[str] = None


class MentorQueueResponse(BaseModel):
    """Paginated queue payload."""

    items: List[MentorQueueItem]
    total: int = Field(..., description="Students matching the filters, before pagination.")
    limit: int = Field(..., ge=1, le=200)
    offset: int = Field(..., ge=0)
    department: Optional[str] = None
    risk_tier: Optional[RiskTier] = None
    assigned_mentor_id: Optional[str] = None
    disclaimer: str = QUEUE_DISCLAIMER
