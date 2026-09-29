"""
Universal Multi-Tier Dropout Prediction Endpoint.
Supports all 5 educational stages (Pre-10th, Higher Secondary, UG, PG, PhD)
with geographic socio-economic risk stratification.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
from ml.models.universal_engine import universal_engine

router = APIRouter(prefix="/universal", tags=["universal-prediction"])

class UniversalStudentRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    student_id: str = Field(..., json_schema_extra={"example": "STU_SCH_001"})
    educational_tier: str = Field(
        ...,
        description="One of: PRE_10TH, HIGHER_SECONDARY, UNDERGRADUATE, POSTGRADUATE, DOCTORATE",
        json_schema_extra={"example": "PRE_10TH"}
    )
    state: str = Field("Bihar", json_schema_extra={"example": "Bihar"})
    district: str = Field("Purnia", json_schema_extra={"example": "Purnia"})
    
    # Optional polymorphic feature payload
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value dictionary of domain features for the specific educational stage."
    )

@router.post(
    "/predict",
    summary="Predict dropout risk for any student across all 5 educational stages",
    status_code=status.HTTP_200_OK
)
def predict_universal(payload: UniversalStudentRequest) -> Dict[str, Any]:
    try:
        # Merge top-level attributes and feature dictionary
        data_dict = payload.model_dump()
        features_dict = data_dict.pop("features", {}) or {}
        data_dict.update(features_dict)

        result = universal_engine.predict_universal_student(data_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")
