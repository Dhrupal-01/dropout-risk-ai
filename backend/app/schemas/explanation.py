"""
SHAP explanation schemas.

Field names mirror the dicts returned by
`ml.models.explain_shap.SHAPExplainerService.explain_local_student` exactly, so responses
pass through without remapping.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ImpactDirection = Literal["RISK_INCREASING", "RISK_DECREASING"]


class ShapDriver(BaseModel):
    """One local SHAP attribution."""

    feature_name: str
    display_name: str
    feature_value: float
    shap_value: float
    impact_direction: ImpactDirection
    risk_delta_percentage_points: float = Field(
        ..., description="Signed marginal probability change, in percentage points."
    )
    plain_language_explanation: str


class GlobalDriver(BaseModel):
    """One cohort-level mean |SHAP| importance entry."""

    feature_name: str
    display_name: str
    mean_abs_shap: float
    importance_percentage: float


class ExplanationResponse(BaseModel):
    student_id: Optional[str] = None
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    risk_tier: str
    top_drivers: List[ShapDriver]
