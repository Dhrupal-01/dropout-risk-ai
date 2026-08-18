"""
Intervention schemas.

RESPONSIBLE AI FRAMING (docs/ethics_and_fairness.md)
Every field here describes a RISK ESTIMATE and RECOMMENDED SUPPORT. Nothing in this API
represents a decision, a verdict, or an automated action against a student:

  * `recommended_interventions` are suggestions for a human mentor to consider, never
    actions the system takes by itself.
  * `counterfactual_recourse` is a model SIMULATION — "if these values changed, the model
    would score differently". It is not a causal guarantee that doing so will change a
    real outcome, and it is labelled as a projection in the response.
  * No field authorises debarment, penalty, or any punitive academic action. Attendance
    debarment risk is surfaced as an existing statutory fact for a mentor to act on, not
    as something this system triggers.

Status literals come from the ACTUAL behaviour of
`ml.intervention.engine.InterventionStatusTracker` — see
`backend/app/services/intervention_service.py` for the recorded discrepancies.
"""

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

InterventionStatus = Literal["ASSIGNED", "IN_PROGRESS", "APPLIED", "COMPLETED"]
InterventionOutcome = Literal["PENDING_EVALUATION", "IMPROVED", "NO_CHANGE", "DETERIORATED"]
Urgency = Literal["HIGH", "MEDIUM", "LOW"]
RiskTier = Literal["Low", "Medium", "High"]

RECOMMENDATION_DISCLAIMER = (
    "Risk estimate and recommended support only. This is a decision-support signal for a "
    "human mentor, not a prediction of certain dropout and not an automated academic "
    "decision. No punitive action is triggered by this system."
)

RECOURSE_DISCLAIMER = (
    "Projection from a model simulation, not a causal guarantee. It shows how the model's "
    "estimate would change if these values changed; real-world outcomes depend on factors "
    "the model does not observe."
)


class InterventionCatalogItem(BaseModel):
    """One entry of the 12-item catalog in ml/artifacts/interventions.json."""

    intervention_id: str
    pillar: str
    title: str
    description: str
    action_type: str
    urgency: Urgency
    target_pillar: str
    suggested_duration_days: int = Field(..., gt=0)
    recommended_for: List[str] = Field(default_factory=list)


class RecourseAction(BaseModel):
    """One suggested step in a counterfactual plan."""

    feature_name: str
    current_value: float
    target_value: float
    plain_language_action: str


class CounterfactualRecourse(BaseModel):
    """
    Output of `CounterfactualRecourseEngine.generate_counterfactual`, passed through with
    its original field names.
    """

    student_id: Optional[str] = None
    current_risk_prob: float
    current_risk_tier: str
    projected_risk_prob: float
    projected_risk_tier: str
    risk_reduction_pct: float
    target_reached: bool
    intervention_plan_name: str
    required_actions: List[RecourseAction] = Field(default_factory=list)
    counselor_summary: str
    status_message: Optional[str] = None
    is_projection: bool = Field(
        default=True, description="Always true - a simulation, never a guaranteed outcome."
    )
    disclaimer: str = RECOURSE_DISCLAIMER


class InterventionPlanResponse(BaseModel):
    """Combined recommendations + counterfactual projection for one student."""

    student_id: str
    current_risk_probability: float = Field(..., ge=0.0, le=1.0)
    current_risk_tier: RiskTier
    evaluated_at: datetime
    model_version: Optional[str] = None

    # Passed through from map_shap_drivers_to_interventions, which enriches each catalog
    # item with the SHAP driver that triggered it.
    recommended_interventions: List[Dict[str, Any]]
    counterfactual_recourse: Optional[CounterfactualRecourse] = None

    disclaimer: str = RECOMMENDATION_DISCLAIMER


class InterventionLogCreate(BaseModel):
    """Assign an intervention, or advance one already assigned."""

    model_config = ConfigDict(extra="forbid")

    student_id: str = Field(..., max_length=64, description="Institutional roll number.")
    intervention_id: str = Field(
        ..., max_length=32, description="Must exist in ml/artifacts/interventions.json."
    )
    assigned_faculty_id: Optional[str] = Field(default=None, max_length=64)
    status: InterventionStatus = "ASSIGNED"
    notes: Optional[str] = None
    scheduled_followup_date: Optional[date] = None
    baseline_risk_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    post_intervention_risk_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class InterventionLogUpdate(BaseModel):
    """Advance an intervention through its lifecycle."""

    model_config = ConfigDict(extra="forbid")

    status: Optional[InterventionStatus] = None
    outcome_status: Optional[InterventionOutcome] = None
    assigned_faculty_id: Optional[str] = Field(default=None, max_length=64)
    notes: Optional[str] = None
    scheduled_followup_date: Optional[date] = None
    post_intervention_risk_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class InterventionLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    student_id: uuid.UUID
    intervention_id: str
    assigned_faculty_id: Optional[str] = None
    status: InterventionStatus
    outcome_status: InterventionOutcome
    baseline_risk_probability: Optional[float] = None
    post_intervention_risk_probability: Optional[float] = None
    risk_delta: Optional[float] = None
    notes: Optional[str] = None
    scheduled_followup_date: Optional[date] = None
    created_at: datetime
    updated_at: datetime

    # Enrichment from the catalog artifact, so the UI needs no second lookup.
    title: Optional[str] = None
    pillar: Optional[str] = None
    urgency: Optional[Urgency] = None


class InterventionLogResponse(BaseModel):
    """Result of POST /interventions/log."""

    created: bool = Field(..., description="True if newly assigned, false if advanced.")
    log: InterventionLogRead
    disclaimer: str = RECOMMENDATION_DISCLAIMER
