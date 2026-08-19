"""
Student and feature-input schemas.

`StudentFeatureInput` is the API's inference contract. It was derived from the ACTUAL
code — `ml/artifacts/feature_names.json`, the trained `XGBClassifier.feature_names_in_`,
and `ml.data_pipeline.feature_engineering.build_engineered_features` — not from the
Markdown example payloads.

The four categories a caller must distinguish:

  A. RAW INPUTS ACCEPTED HERE (28)
     The model features the caller actually supplies, plus `hostel_status`.

  B. ENGINEERED — COMPUTED BY THE ML PIPELINE, NEVER ACCEPTED (9)
     subject_attendance_std, academic_crisis_flag, behavioral_disengagement_index,
     is_hosteler, financial_stress_index, and the four interaction_* terms. These are
     produced exclusively by `build_engineered_features`, the same function that built
     the training set, so a caller can never supply a value contradicting the trained
     formula.

  C. METADATA — NOT MODEL INPUT
     name, department, assigned_mentor_id (display only); gender, category,
     family_income_slab (excluded by ml.config.EXCLUDED_FEATURES).

  D. LABELS — FORBIDDEN DURING INFERENCE
     is_dropout, ground_truth_risk_prob. Rejected explicitly with a clear message.

RANGE VALIDATION
Bounds follow docs/data_dictionary.md as guidance, widened where the real cohort or the
generating code intentionally exceeds it. Documented ranges too narrow for data the ML
code actually produces:
  * fee_payment_delay_days         — doc 0-120,     actual max 124
  * assignment_submission_lag_days — doc -5..+15,   actual min -7.8
  * age                            — absent from the data dictionary entirely, yet it is
                                     model feature index 0 (actual 17.5-24.6)
Enforcing the documented values would reject rows from the project's own features.csv,
so the bounds below are the union of documented intent and real data.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HostelStatus = Literal["Hosteler", "Day Scholar"]

# Category D. Never predictors — see ml.config.EXCLUDED_FEATURES.
FORBIDDEN_LABEL_FIELDS = ("is_dropout", "ground_truth_risk_prob")

# Category B. Derived by build_engineered_features; never accepted from a caller.
REJECTED_ENGINEERED_FIELDS = (
    "subject_attendance_std",
    "academic_crisis_flag",
    "behavioral_disengagement_index",
    "financial_stress_index",
    "interaction_att_x_fee",
    "interaction_cgpa_x_backlog",
    "interaction_firstgen_x_inactivity",
    "interaction_att_x_cgpa_drop",
)


class StudentFeatureInput(BaseModel):
    """The 28 raw model inputs plus residency status."""

    model_config = ConfigDict(extra="forbid")

    # --- Demographic / socio-economic -------------------------------------------------
    # `age` IS model feature index 0, despite being filed under "demographics_protected"
    # in feature_metadata.json and omitted from docs/data_dictionary.md entirely.
    age: float = Field(..., ge=15.0, le=60.0, description="Years. Model feature index 0.")
    commute_distance_km: float = Field(..., ge=0.0, le=100.0)
    income_slab_idx: int = Field(..., ge=0, le=3, description="0=<2 LPA, 1=2-5, 2=5-8, 3=>8 LPA")
    is_first_generation: int = Field(..., ge=0, le=1)
    has_scholarship: int = Field(..., ge=0, le=1)
    fee_payment_delay_days: int = Field(
        ..., ge=0, le=365, description="Doc says 0-120; the real cohort reaches 124."
    )

    # Source for the engineered `is_hosteler`. Supply either this or is_hosteler.
    hostel_status: Optional[HostelStatus] = None
    is_hosteler: Optional[int] = Field(default=None, ge=0, le=1)

    # --- Pillar 1: attendance ---------------------------------------------------------
    att_core1: float = Field(..., ge=0.0, le=100.0)
    att_core2: float = Field(..., ge=0.0, le=100.0)
    att_lab: float = Field(..., ge=0.0, le=100.0)
    att_elective: float = Field(..., ge=0.0, le=100.0)
    attendance_month_1: float = Field(..., ge=0.0, le=100.0)
    attendance_month_2: float = Field(..., ge=0.0, le=100.0)
    attendance_month_3: float = Field(..., ge=0.0, le=100.0)
    attendance_percentage: float = Field(..., ge=0.0, le=100.0)
    consecutive_absences: int = Field(..., ge=0, le=180)

    # Derivable by build_engineered_features when omitted. Bounds follow from the
    # formula's own domain: (M3 - M1) / 2 over two 0-100 percentages.
    attendance_3m_trend: Optional[float] = Field(default=None, ge=-50.0, le=50.0)
    attendance_risk_flag: Optional[int] = Field(default=None, ge=0, le=1)

    # --- Pillar 2: academic -----------------------------------------------------------
    prev_sem_cgpa: float = Field(..., ge=0.0, le=10.0)
    current_cgpa: float = Field(..., ge=0.0, le=10.0)
    # Domain of current_cgpa - prev_sem_cgpa on a 0-10 scale.
    cgpa_delta: Optional[float] = Field(default=None, ge=-10.0, le=10.0)
    backlog_count: int = Field(..., ge=0, le=20)
    internal_exam_score_pct: float = Field(..., ge=0.0, le=100.0)
    stem_core_fail_flag: int = Field(..., ge=0, le=1)

    # --- Pillar 3: learning behaviour -------------------------------------------------
    lms_logins_per_week: float = Field(..., ge=0.0, le=50.0)
    assignment_submission_lag_days: float = Field(
        ...,
        ge=-30.0,
        le=90.0,
        description="Negative = early. Doc says -5..+15; the cohort reaches -7.8.",
    )
    resource_access_count: int = Field(..., ge=0, le=1000)
    days_since_last_lms_activity: int = Field(..., ge=0, le=365)
    forum_participation_count: int = Field(..., ge=0, le=200)

    @model_validator(mode="before")
    @classmethod
    def _reject_labels_and_engineered(cls, data: Any) -> Any:
        """
        Fail loudly and specifically. `extra="forbid"` would already reject these, but
        with a generic message; label leakage deserves an unmistakable one.
        """
        if not isinstance(data, dict):
            return data
        leaked = [f for f in FORBIDDEN_LABEL_FIELDS if f in data]
        if leaked:
            raise ValueError(
                f"Label field(s) {leaked} must never be sent as inference input - "
                "they are ground-truth targets, not predictors."
            )
        supplied = [f for f in REJECTED_ENGINEERED_FIELDS if f in data]
        if supplied:
            raise ValueError(
                f"Engineered feature(s) {supplied} are computed by the ML pipeline "
                "(build_engineered_features) and must not be supplied by the caller."
            )
        return data

    @model_validator(mode="after")
    def _residency_supplied(self) -> "StudentFeatureInput":
        if self.hostel_status is None and self.is_hosteler is None:
            raise ValueError("Supply either 'hostel_status' or 'is_hosteler'")
        return self

    @model_validator(mode="after")
    def _fill_derivable(self) -> "StudentFeatureInput":
        """
        Derive the optional columns using the SAME formulas as
        `build_engineered_features`, so the stored snapshot matches what the model saw.
        """
        if self.cgpa_delta is None:
            self.cgpa_delta = round(self.current_cgpa - self.prev_sem_cgpa, 2)
        if self.attendance_3m_trend is None:
            self.attendance_3m_trend = round(
                (self.attendance_month_3 - self.attendance_month_1) / 2.0, 2
            )
        if self.attendance_risk_flag is None:
            self.attendance_risk_flag = int(self.attendance_percentage < 75.0)
        if self.is_hosteler is None:
            self.is_hosteler = int(self.hostel_status == "Hosteler")
        return self

    def to_model_input(self) -> Dict[str, Any]:
        """Raw dict for MLService.build_feature_frame."""
        return self.model_dump(exclude_none=True)


# Backwards-compatible alias for the Stage 1 name.
RawStudentFeatures = StudentFeatureInput


class StudentBase(BaseModel):
    """Category C. Display metadata, never fed to the model."""

    name: Optional[str] = Field(default=None, max_length=128)
    department: Optional[str] = Field(default=None, max_length=64)
    assigned_mentor_id: Optional[str] = Field(default=None, max_length=64)


class StudentCreate(StudentBase):
    student_id: str = Field(..., min_length=1, max_length=64)
    features: StudentFeatureInput

    @field_validator("student_id")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("student_id must not be blank")
        return v


class StudentUpdate(StudentBase):
    features: Optional[StudentFeatureInput] = None


class StudentRead(StudentBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    student_id: str
    features: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
