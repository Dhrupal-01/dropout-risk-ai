"""
Prediction request/response schemas.

ON `confidence_interval` / `confidence`
--------------------------------------
Section 7.3 of docs/ml_architecture_and_pipeline.md shows a `confidence` field in its
example response. A search of the entire ml/ package found NO implementation of a
confidence interval, standard error, bootstrap, or predictive-variance estimator:

    grep -rniE "confidence_interval|std_err|bootstrap|uncertainty" ml/   ->  no hits

`CalibratedClassifierCV` returns a calibrated point probability only; it exposes no
interval. Fabricating one (e.g. a normal approximation over the 5 calibrated folds)
would be a statistical claim the ML team never made and never validated, and mentors
would read it as a real uncertainty bound.

DECISION: omitted from the v1 response. It is deliberately absent rather than null, so
no client can mistake a placeholder for a measured value. If the ML team later adds a
genuine estimator, add the field then.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.explanation import ShapDriver
from backend.app.schemas.student import StudentBase, StudentFeatureInput

RiskTier = Literal["Low", "Medium", "High"]

# Guard against unbounded request bodies; batch inference is a single vectorised call.
MAX_BATCH_SIZE = 1000


class PredictionRequest(StudentBase):
    """
    Score one student and append the result to their history.

    Inherits the optional display metadata (name/department/assigned_mentor_id) so a
    caller can create a previously unknown student in one call. That metadata is stored
    alongside the student record and never reaches the model.
    """

    student_id: str = Field(..., min_length=1, max_length=64)
    features: StudentFeatureInput
    include_explanation: bool = Field(
        default=False, description="Include top SHAP drivers in the response."
    )


class PredictionResponse(BaseModel):
    """
    Scoring result.

    `risk_tier` comes from `ml.config.get_risk_tier` via
    `ml.models.calibrate.predict_student_risk` — the backend never derives tiers itself.
    """

    student_id: str
    calibrated_risk_probability: float = Field(..., ge=0.0, le=1.0)
    risk_tier: RiskTier
    risk_score_percentage: float = Field(..., ge=0.0, le=100.0)
    evaluation_timestamp: datetime
    model_version: Optional[str] = None
    prediction_id: Optional[uuid.UUID] = None
    top_drivers: Optional[List[ShapDriver]] = None


class BatchPredictionRequest(BaseModel):
    """JSON batch. Scored with one vectorised inference call, not a per-row loop."""

    students: List[PredictionRequest] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    sort_by_risk_desc: bool = Field(
        default=False, description="Return results highest-risk first (triage order)."
    )
    include_explanations: bool = Field(
        default=False,
        description="Compute SHAP per student. Adds roughly 43ms each — off by default.",
    )


class BatchPredictionResponse(BaseModel):
    """Batch result with tier rollup."""

    total_students_evaluated: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    model_version: Optional[str] = None
    results: List[PredictionResponse]


class PredictionRead(BaseModel):
    """A persisted prediction row, including the reproducibility snapshot."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    student_id: uuid.UUID
    calibrated_risk_probability: float
    risk_tier: RiskTier
    risk_score_percentage: float
    model_version: Optional[str] = None
    input_features: Dict[str, Any]
    top_drivers: Optional[List[Dict[str, Any]]] = None
    evaluated_at: datetime
