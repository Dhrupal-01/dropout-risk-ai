"""
SHAP Explainability Layer for DropoutGuard
Provides:
1. Global Feature Importance (Cohort-level driver rankings)
2. Local Per-Student SHAP Explanations with signed contributions and plain-language sentences.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
import shap

from ml.config import (
    PROCESSED_DATA_PATH,
    BASE_MODEL_PATH,
    MODEL_ARTIFACT_PATH,
    FEATURE_NAMES_PATH,
    SHAP_EXPLAINER_PATH,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Feature Display Names for Mentors & Dashboard
FEATURE_DISPLAY_NAMES = {
    "attendance_percentage": "Overall 3-Month Attendance",
    "attendance_month_1": "Month 1 Attendance (2 Months Prior)",
    "attendance_month_2": "Month 2 Attendance (Last Month)",
    "attendance_month_3": "Current Month Attendance (Recent Month)",
    "attendance_3m_trend": "3-Month Attendance Trend Slope",
    "consecutive_absences": "Consecutive Absent Days Streak",
    "attendance_risk_flag": "Attendance Debarment Risk (<75% Rule)",
    "subject_attendance_std": "Subject Attendance Variance",
    "att_core1": "Core Mathematics/Theory Course Attendance",
    "att_core2": "Department Major Core Course Attendance",
    "att_lab": "Practical Laboratory Course Attendance",
    "att_elective": "Elective Course Attendance",
    "current_cgpa": "Current Semester CGPA",
    "prev_sem_cgpa": "Previous Semester CGPA",
    "cgpa_delta": "Semester CGPA Trajectory",
    "backlog_count": "Uncleared Backlog Count",
    "internal_exam_score_pct": "Internal Continuous Assessment Marks",
    "stem_core_fail_flag": "Core Course Failure Status",
    "academic_crisis_flag": "Academic Crisis Indicator",
    "lms_logins_per_week": "LMS Weekly Login Frequency",
    "assignment_submission_lag_days": "Assignment Submission Delay",
    "resource_access_count": "Digital Resource Access Count",
    "days_since_last_lms_activity": "LMS Inactivity Recency",
    "forum_participation_count": "Discussion Forum Activity",
    "behavioral_disengagement_index": "Composite Behavioral Disengagement Index",
    "family_income_slab": "Family Income Bracket",
    "income_slab_idx": "Income Bracket Level",
    "is_first_generation": "First-Generation College Learner",
    "fee_payment_delay_days": "Tuition Fee Payment Overdue Days",
    "has_scholarship": "Financial Scholarship Buffer",
    "is_hosteler": "Hosteler Status",
    "commute_distance_km": "Daily Commute Distance",
    "financial_stress_index": "Composite Financial Stress Index",
    "interaction_att_x_fee": "Compounded Absenteeism x Fee Delay",
    "interaction_cgpa_x_backlog": "Compounded Academic Deficit x Backlogs",
    "interaction_firstgen_x_inactivity": "First-Gen Status x LMS Inactivity",
    "interaction_att_x_cgpa_drop": "Attendance Collapse x CGPA Drop",
    "age": "Student Age"
}


def logit_to_prob(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))


def build_plain_language_sentence(
    feature_name: str,
    feature_val: float,
    shap_val: float,
    pct_impact: float
) -> str:
    """
    Translates numerical SHAP contribution into an actionable, counselor-friendly plain-language sentence.
    Explicitly distinguishes overall attendance percentage from recent-month attendance.
    """
    impact_dir = "increasing" if shap_val > 0 else "reducing"
    display = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name.replace("_", " ").title())

    if feature_name == "attendance_percentage":
        if shap_val > 0:
            return f"Low overall attendance ({feature_val:.1f}%) is increasing risk by {pct_impact:.1f} percentage points (violates 75% minimum requirement)."
        else:
            return f"Consistent overall attendance ({feature_val:.1f}%) is reducing risk by {pct_impact:.1f} percentage points."

    elif feature_name == "attendance_month_3":
        if shap_val > 0:
            return f"Low recent-month (Month 3) attendance ({feature_val:.1f}%) is increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"High recent-month (Month 3) attendance ({feature_val:.1f}%) is reducing risk by {pct_impact:.1f} percentage points."

    elif feature_name in ["attendance_month_1", "attendance_month_2"]:
        month_label = "Month 1 (2 months prior)" if feature_name == "attendance_month_1" else "Month 2 (last month)"
        if shap_val > 0:
            return f"Low {month_label} attendance ({feature_val:.1f}%) is increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"High {month_label} attendance ({feature_val:.1f}%) is reducing risk by {pct_impact:.1f} percentage points."

    elif feature_name in ["att_core1", "att_core2", "att_lab", "att_elective"]:
        subj_name = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)
        if shap_val > 0:
            return f"Lagging attendance in {subj_name} ({feature_val:.1f}%) is increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"Strong attendance in {subj_name} ({feature_val:.1f}%) is reducing risk by {pct_impact:.1f} percentage points."

    elif feature_name == "attendance_3m_trend":
        if feature_val < 0:
            return f"Sharp attendance decline ({feature_val:+.1f}%/month over 3 months) is increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"Improving attendance trajectory ({feature_val:+.1f}%/month) is reducing risk by {pct_impact:.1f} percentage points."

    elif feature_name == "consecutive_absences":
        if shap_val > 0:
            return f"Recent streak of {int(feature_val)} consecutive absent days is driving up risk by {pct_impact:.1f} percentage points."
        else:
            return f"Minimal absence streaks ({int(feature_val)} days) are keeping risk low by {pct_impact:.1f} percentage points."

    elif feature_name == "current_cgpa":
        if shap_val > 0:
            return f"Low current CGPA ({feature_val:.2f}/10.0) is increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"Strong academic standing (CGPA {feature_val:.2f}/10.0) is lowering risk by {pct_impact:.1f} percentage points."

    elif feature_name == "cgpa_delta":
        if feature_val < 0:
            return f"Drop in semester CGPA ({feature_val:+.2f} grade points) is increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"Positive semester grade progression ({feature_val:+.2f} points) is lowering risk by {pct_impact:.1f} percentage points."

    elif feature_name == "backlog_count":
        if feature_val > 0:
            return f"{int(feature_val)} uncleared exam backlog(s) are increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"Clean academic record with 0 backlogs is reducing risk by {pct_impact:.1f} percentage points."

    elif feature_name == "fee_payment_delay_days":
        if feature_val > 0:
            return f"Tuition fee delay of {int(feature_val)} days is causing financial distress, increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"Tuition fees are fully up to date, protecting the student by {pct_impact:.1f} percentage points."

    elif feature_name == "has_scholarship":
        if feature_val == 1:
            return f"Active scholarship provides financial security, reducing risk by {pct_impact:.1f} percentage points."
        else:
            return f"Lack of scholarship support increases financial vulnerability by {pct_impact:.1f} percentage points."

    elif feature_name == "days_since_last_lms_activity":
        if feature_val > 7:
            return f"LMS digital inactivity for {int(feature_val)} consecutive days is increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"Active daily LMS participation is lowering risk by {pct_impact:.1f} percentage points."

    elif feature_name == "lms_logins_per_week":
        if shap_val > 0:
            return f"Infrequent LMS logins ({feature_val:.1f} times/week) are increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"High digital engagement ({feature_val:.1f} logins/week) is reducing risk by {pct_impact:.1f} percentage points."

    elif feature_name == "assignment_submission_lag_days":
        if feature_val > 0:
            return f"Submitting assignments {feature_val:.1f} days late on average is increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"Consistently submitting assignments on or before deadlines is reducing risk by {pct_impact:.1f} percentage points."

    elif feature_name == "behavioral_disengagement_index":
        if shap_val > 0:
            return f"High behavioral disengagement index ({feature_val:.2f}) is increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"Low behavioral disengagement index ({feature_val:.2f}) is reducing risk by {pct_impact:.1f} percentage points."

    elif feature_name == "financial_stress_index":
        if shap_val > 0:
            return f"High financial stress index ({feature_val:.2f}) is increasing risk by {pct_impact:.1f} percentage points."
        else:
            return f"Low financial stress index ({feature_val:.2f}) is reducing risk by {pct_impact:.1f} percentage points."

    elif feature_name.startswith("interaction_"):
        return f"Compound risk factor ({display}) is {impact_dir} overall dropout risk by {pct_impact:.1f} percentage points."

    # Generic Fallback
    return f"{display} (value: {feature_val}) is {impact_dir} this student's risk by {pct_impact:.1f} percentage points."


class SHAPExplainerService:
    """
    Service for extracting and explaining SHAP contributions for the tree-based model.
    """

    def __init__(self, explainer_path: Path = SHAP_EXPLAINER_PATH):
        self.explainer_path = Path(explainer_path)
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        self._load_or_initialize()

    def _load_or_initialize(self):
        # Load feature names
        if FEATURE_NAMES_PATH.exists():
            with open(FEATURE_NAMES_PATH, "r") as f:
                self.feature_names = json.load(f)

        if self.explainer_path.exists():
            try:
                self.explainer = joblib.load(self.explainer_path)
                logger.info("Loaded pre-computed SHAP TreeExplainer from %s", self.explainer_path)
            except Exception as e:
                logger.warning("Could not load SHAP explainer from artifact (%s). Rebuilding...", e)
                self.build_and_save_explainer()
        else:
            self.build_and_save_explainer()

    def build_and_save_explainer(self):
        """
        Builds a TreeExplainer from the underlying XGBoost model and saves artifact.
        """
        if not BASE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Base model artifact not found at {BASE_MODEL_PATH}. Train model first.")

        base_model = joblib.load(BASE_MODEL_PATH)
        # For CalibratedClassifierCV, extract underlying estimator if needed
        underlying_model = base_model
        if hasattr(base_model, "calibrated_classifiers_"):
            underlying_model = base_model.calibrated_classifiers_[0].estimator

        logger.info("Fitting SHAP TreeExplainer on underlying XGBoost model...")
        self.explainer = shap.TreeExplainer(underlying_model)
        self.explainer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.explainer, self.explainer_path)
        logger.info("Saved SHAP TreeExplainer artifact to %s", self.explainer_path)

    def explain_global(self, X_sample: pd.DataFrame, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Computes global mean |SHAP| feature importances across a representative sample.
        """
        if self.explainer is None:
            self.build_and_save_explainer()

        shap_values = self.explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class for binary

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1][:top_k]

        global_drivers = []
        for idx in sorted_indices:
            feat = self.feature_names[idx]
            global_drivers.append({
                "feature_name": feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title()),
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
                "importance_percentage": round(float(mean_abs_shap[idx] / np.sum(mean_abs_shap) * 100), 2)
            })

        return global_drivers

    def explain_local_student(
        self,
        student_features: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generates local SHAP explanation for a single student row:
        Returns top k features with signed contribution, percentage points, and plain-language sentence.
        """
        if self.explainer is None:
            self.build_and_save_explainer()

        if isinstance(student_features, dict):
            student_features = pd.DataFrame([student_features])
        elif isinstance(student_features, pd.Series):
            student_features = pd.DataFrame([student_features])

        # Ensure correct column ordering
        aligned_df = student_features[self.feature_names].copy()
        raw_vals = aligned_df.iloc[0].values

        # Compute SHAP values for single instance
        shap_values = self.explainer.shap_values(aligned_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_row = shap_values[0]

        # Rank by absolute magnitude of contribution
        top_indices = np.argsort(np.abs(shap_row))[::-1][:top_k]

        # Baseline expected value of model
        base_val = float(self.explainer.expected_value) if hasattr(self.explainer, "expected_value") else -0.50
        if isinstance(base_val, np.ndarray) and len(base_val) > 1:
            base_val = float(base_val[1])

        explanations = []
        for idx in top_indices:
            feat_name = self.feature_names[idx]
            feat_val = float(raw_vals[idx])
            s_val = float(shap_row[idx])
            impact_dir = "RISK_INCREASING" if s_val > 0 else "RISK_DECREASING"

            # Compute marginal probability delta in percentage points
            base_p = logit_to_prob(base_val)
            new_p = logit_to_prob(base_val + s_val)
            pct_impact = abs(new_p - base_p) * 100.0

            sentence = build_plain_language_sentence(feat_name, feat_val, s_val, pct_impact)

            explanations.append({
                "feature_name": feat_name,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat_name, feat_name.replace("_", " ").title()),
                "feature_value": round(feat_val, 2),
                "shap_value": round(s_val, 4),
                "impact_direction": impact_dir,
                "risk_delta_percentage_points": round((new_p - base_p) * 100.0, 2),
                "plain_language_explanation": sentence
            })

        return explanations


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_PATH)
    service = SHAPExplainerService()
    with open(FEATURE_NAMES_PATH, "r") as f:
        feat_names = json.load(f)

    print("Global Top 10 Feature Drivers:")
    global_importance = service.explain_global(df[feat_names].head(200), top_k=10)
    for item in global_importance:
        print(f" - {item['display_name']} ({item['feature_name']}): {item['mean_abs_shap']} ({item['importance_percentage']}%)")

    print("\nSample Local Explanation (Student 0):")
    local_exp = service.explain_local_student(df[feat_names].iloc[0], top_k=5)
    for exp in local_exp:
        print(f" * [{exp['impact_direction']}] {exp['plain_language_explanation']}")
