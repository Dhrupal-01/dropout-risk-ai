"""
Stats summary schemas.
Phase 5 SQL-native aggregations for cohort risk and department breakdowns.
"""

from typing import Dict, List, Optional
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
    by_department_tier: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Count of students by department and risk tier; every department has High, Medium and Low",
    )


# ---------------------------------------------------------------------------------------------
# Overview dashboard aggregates (read-only). Every field is a count or a bin edge: no student
# identifiers or personal fields are ever returned.
# ---------------------------------------------------------------------------------------------


class RiskScoreBin(BaseModel):
    lower: float = Field(..., description="Inclusive lower edge of the calibrated-probability bin")
    upper: float = Field(..., description="Upper edge; exclusive except for the last bin, which includes 1.0")
    count: int = Field(..., description="Students whose latest calibrated risk probability falls in the bin")


class StatsDistributionResponse(BaseModel):
    total: int = Field(..., description="Students with a latest prediction (sum of all bin counts)")
    bin_count: int = Field(..., description="Number of equal-width bins over [0, 1]")
    bins: List[RiskScoreBin]


class DriverCount(BaseModel):
    feature_name: str = Field(..., description="Model feature name")
    display_name: str = Field(..., description="Human-readable feature name stored with the SHAP driver")
    student_count: int = Field(..., description="Students whose stored top drivers include this feature as risk-increasing")


class StatsDriversResponse(BaseModel):
    risk_tier: Optional[str] = Field(None, description="Tier filter applied, or null for all students")
    students_considered: int = Field(..., description="Students with a latest prediction (after the tier filter)")
    students_with_drivers: int = Field(..., description="Of those, students whose latest prediction has stored top drivers")
    drivers: List[DriverCount] = Field(..., description="Most common risk-increasing drivers, most frequent first")


class StatsInterventionsResponse(BaseModel):
    total: int = Field(..., description="All intervention log entries")
    open: int = Field(..., description="Entries whose status is not COMPLETED")
    students_with_open_interventions: int
    by_status: Dict[str, int] = Field(..., description="Count per lifecycle status; every status key is present")
    by_outcome_status: Dict[str, int] = Field(..., description="Count per outcome status; every key is present")


class AlertCount(BaseModel):
    code: str = Field(..., description="Rule code, as in /students/{id}/interventions rule_based_alerts")
    student_count: int


class StatsAlertsResponse(BaseModel):
    students_considered: int = Field(..., description="Students with a latest prediction")
    students_with_any_alert: int
    attendance_threshold: float = Field(..., description="Attendance percentage below which the attendance alert fires (ml.config.ATTENDANCE_THRESHOLD)")
    alerts: List[AlertCount]
