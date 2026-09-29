"""
Stats summary schemas.
Phase 5 SQL-native aggregations for cohort risk and department breakdowns.
"""

from typing import Dict
from pydantic import BaseModel, Field


class StatsSummaryResponse(BaseModel):
    total: int = Field(..., description="Total number of students with an active prediction")
    by_tier: Dict[str, int] = Field(
        default_factory=lambda: {"High": 0, "Medium": 0, "Low": 0},
        description="Count of students by risk tier",
    )
    by_department: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of students by department",
    )
