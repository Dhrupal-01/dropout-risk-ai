"""
Intervention Recommendation Engine & Counterfactual Recourse for DropoutGuard
Context: SIH 2026 PSID 7-L / SDG 4 (Quality Education)

Components:
1. Structured Intervention Catalog (Categorized by the 4 pillars)
2. SHAP-to-Intervention Mapper & Ranker (Selects top 1-3 targeted actions)
3. Actionable Counterfactual Recourse Generator (Quantified path to lower risk tier)
4. Prioritized Mentor Worklist Queue (Ranked by clinical risk score)
5. Intervention Status & Feedback Tracking Lifecycle (Applied / No Change / Improved)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

from ml.config import (
    MODEL_ARTIFACT_PATH,
    FEATURE_NAMES_PATH,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH,
    get_risk_tier
)
from ml.models.calibrate import predict_student_risk
from ml.models.explain_shap import SHAPExplainerService, FEATURE_DISPLAY_NAMES
from ml.data_pipeline.feature_engineering import build_engineered_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INTERVENTION_CATALOG_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "interventions.json"

# =============================================================================
# 1. STRUCTURED INTERVENTION CATALOG
# =============================================================================
INTERVENTION_CATALOG = {
    # ------------------ Pillar 1: Attendance Interventions ------------------
    "INT_ATT_01": {
        "intervention_id": "INT_ATT_01",
        "pillar": "attendance",
        "title": "Mandatory Attendance Counseling & Faculty Mentor Check-in",
        "description": "Schedule a 1-on-1 mentor session to identify root causes of absenteeism (commute, medical, health) and create a daily check-in schedule.",
        "action_type": "MENTOR_CHECKIN",
        "urgency": "HIGH",
        "target_pillar": "Attendance Dynamics",
        "suggested_duration_days": 14,
        "recommended_for": ["attendance_percentage", "attendance_month_3", "attendance_risk_flag", "attendance_3m_trend"]
    },
    "INT_ATT_02": {
        "intervention_id": "INT_ATT_02",
        "pillar": "attendance",
        "title": "Parent / Guardian Attendance Advisory Notification",
        "description": "Send official institutional alert regarding 75% AICTE/UGC debarment risk to parent/guardian with automated attendance tracking updates.",
        "action_type": "PARENT_NOTIFICATION",
        "urgency": "HIGH",
        "target_pillar": "Attendance Dynamics",
        "suggested_duration_days": 3,
        "recommended_for": ["consecutive_absences", "attendance_risk_flag", "attendance_percentage"]
    },
    "INT_ATT_03": {
        "intervention_id": "INT_ATT_03",
        "pillar": "attendance",
        "title": "Subject-Specific Make-up Class & Lab Session Allocation",
        "description": "Assign dedicated weekend laboratory slots and remedial theory attendance reconciliation sessions for lagging subjects.",
        "action_type": "LAB_MAKEUP",
        "urgency": "MEDIUM",
        "target_pillar": "Attendance Dynamics",
        "suggested_duration_days": 21,
        "recommended_for": ["att_core1", "att_core2", "att_lab", "subject_attendance_std"]
    },

    # ------------------ Pillar 2: Academic Interventions --------------------
    "INT_ACAD_01": {
        "intervention_id": "INT_ACAD_01",
        "pillar": "academic",
        "title": "Peer Tutoring & Department Remedial Session Assignment",
        "description": "Pair student with a high-performing departmental peer mentor and enroll in weekly remedial coaching for core engineering subjects.",
        "action_type": "PEER_TUTORING",
        "urgency": "HIGH",
        "target_pillar": "Academic Performance",
        "suggested_duration_days": 30,
        "recommended_for": ["current_cgpa", "cgpa_delta", "internal_exam_score_pct", "stem_core_fail_flag"]
    },
    "INT_ACAD_02": {
        "intervention_id": "INT_ACAD_02",
        "pillar": "academic",
        "title": "Backlog Clearance Action Plan & Academic Strategy Roadmap",
        "description": "Formulate a customized semester study roadmap with the Head of Department (HOD) prioritizing uncleared prerequisite backlogs.",
        "action_type": "BACKLOG_REMEDIATION",
        "urgency": "HIGH",
        "target_pillar": "Academic Performance",
        "suggested_duration_days": 45,
        "recommended_for": ["backlog_count", "interaction_cgpa_x_backlog", "academic_crisis_flag"]
    },
    "INT_ACAD_03": {
        "intervention_id": "INT_ACAD_03",
        "pillar": "academic",
        "title": "Continuous Internal Assessment (CIA) Retest Opportunity",
        "description": "Grant eligibility for internal improvement quizzes and supplementary assignment submissions to recover internal marks deficit.",
        "action_type": "INTERNAL_RETEST",
        "urgency": "MEDIUM",
        "target_pillar": "Academic Performance",
        "suggested_duration_days": 14,
        "recommended_for": ["internal_exam_score_pct", "cgpa_delta"]
    },

    # ------------------ Pillar 3: Socio-Economic & Financial Interventions --
    "INT_FIN_01": {
        "intervention_id": "INT_FIN_01",
        "pillar": "financial",
        "title": "Institutional Fee-Waiver & Emergency Financial Aid Desk Referral",
        "description": "Fast-track application for institute emergency hardship funds, alumni tuition grants, and flexible installment fee payment schedules.",
        "action_type": "FINANCIAL_DESK",
        "urgency": "HIGH",
        "target_pillar": "Socio-Economic Indicators",
        "suggested_duration_days": 7,
        "recommended_for": ["fee_payment_delay_days", "financial_stress_index", "interaction_att_x_fee"]
    },
    "INT_FIN_02": {
        "intervention_id": "INT_FIN_02",
        "pillar": "financial",
        "title": "State/National Post-Matric Scholarship Desk Assistance",
        "description": "Assign institutional scholarship coordinator to resolve pending documentation and disburse government welfare subsidies (NSP, PMSSS, SC/ST/EWS).",
        "action_type": "SCHOLARSHIP_DESK",
        "urgency": "MEDIUM",
        "target_pillar": "Socio-Economic Indicators",
        "suggested_duration_days": 15,
        "recommended_for": ["has_scholarship", "income_slab_idx", "financial_stress_index"]
    },
    "INT_FIN_03": {
        "intervention_id": "INT_FIN_03",
        "pillar": "financial",
        "title": "Hostel Subsidized Accommodation & Commute Relief Support",
        "description": "Evaluate long-distance day-scholar transit burden and allocate emergency subsidized campus hostel room or college bus pass concession.",
        "action_type": "HOSTEL_COMMUTE_RELIEF",
        "urgency": "LOW",
        "target_pillar": "Socio-Economic Indicators",
        "suggested_duration_days": 10,
        "recommended_for": ["commute_distance_km", "is_hosteler"]
    },

    # ------------------ Pillar 4: Learning Behavior & Engagement -----------
    "INT_BEH_01": {
        "intervention_id": "INT_BEH_01",
        "pillar": "engagement",
        "title": "Professional Student Counselor & Wellbeing Consultation",
        "description": "Refer student to campus counseling center for confidential mental health support, motivation assessment, and personal guidance.",
        "action_type": "COUNSELOR_CONSULT",
        "urgency": "HIGH",
        "target_pillar": "Learning Behavior",
        "suggested_duration_days": 7,
        "recommended_for": ["days_since_last_lms_activity", "behavioral_disengagement_index", "interaction_firstgen_x_inactivity"]
    },
    "INT_BEH_02": {
        "intervention_id": "INT_BEH_02",
        "pillar": "engagement",
        "title": "Digital LMS Onboarding & Learning Resource Assistance",
        "description": "Provide dedicated orientation on accessing digital lecture recordings, e-library repositories, and interactive quiz modules.",
        "action_type": "LMS_ONBOARDING",
        "urgency": "MEDIUM",
        "target_pillar": "Learning Behavior",
        "suggested_duration_days": 5,
        "recommended_for": ["lms_logins_per_week", "resource_access_count", "assignment_submission_lag_days"]
    },
    "INT_BEH_03": {
        "intervention_id": "INT_BEH_03",
        "pillar": "engagement",
        "title": "First-Generation Learner Academic Navigation Mentorship",
        "description": "Connect student with senior student mentors and faculty advisers experienced in guiding first-generation collegiate scholars.",
        "action_type": "FIRST_GEN_MENTORSHIP",
        "urgency": "MEDIUM",
        "target_pillar": "Learning Behavior",
        "suggested_duration_days": 21,
        "recommended_for": ["is_first_generation", "interaction_firstgen_x_inactivity"]
    }
}


def save_intervention_catalog(output_path: Path = INTERVENTION_CATALOG_PATH):
    """Persists the structured catalog to JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(INTERVENTION_CATALOG, f, indent=2)
    logger.info("Saved intervention catalog to %s", output_path)


# Ensure catalog exists on import
save_intervention_catalog()


# =============================================================================
# 2. SHAP-TO-INTERVENTION MAPPER & RANKER
# =============================================================================
FEATURE_TO_PILLAR_MAP = {
    # Attendance
    "attendance_percentage": "attendance",
    "attendance_month_1": "attendance",
    "attendance_month_2": "attendance",
    "attendance_month_3": "attendance",
    "attendance_3m_trend": "attendance",
    "consecutive_absences": "attendance",
    "attendance_risk_flag": "attendance",
    "att_core1": "attendance",
    "att_core2": "attendance",
    "att_lab": "attendance",
    "att_elective": "attendance",
    "subject_attendance_std": "attendance",
    # Academic
    "current_cgpa": "academic",
    "prev_sem_cgpa": "academic",
    "cgpa_delta": "academic",
    "backlog_count": "academic",
    "internal_exam_score_pct": "academic",
    "stem_core_fail_flag": "academic",
    "academic_crisis_flag": "academic",
    "interaction_cgpa_x_backlog": "academic",
    "interaction_att_x_cgpa_drop": "academic",
    # Financial / Socio-Economic
    "fee_payment_delay_days": "financial",
    "has_scholarship": "financial",
    "income_slab_idx": "financial",
    "financial_stress_index": "financial",
    "interaction_att_x_fee": "financial",
    "commute_distance_km": "financial",
    # Engagement / Behavior
    "lms_logins_per_week": "engagement",
    "assignment_submission_lag_days": "engagement",
    "resource_access_count": "engagement",
    "days_since_last_lms_activity": "engagement",
    "forum_participation_count": "engagement",
    "behavioral_disengagement_index": "engagement",
    "is_first_generation": "engagement",
    "interaction_firstgen_x_inactivity": "engagement"
}


def map_shap_drivers_to_interventions(
    shap_explanations: List[Dict[str, Any]],
    max_recommendations: int = 3
) -> List[Dict[str, Any]]:
    """
    Given a student's local SHAP drivers, selects and ranks 1-3 targeted interventions.
    Ranks by:
    1. Highest positive SHAP contribution (greatest risk driver)
    2. Exact feature-to-intervention match
    3. Action urgency
    """
    # Filter for risk-increasing drivers (positive SHAP contribution)
    risk_drivers = [d for d in shap_explanations if d.get("impact_direction") == "RISK_INCREASING" or d.get("shap_value", 0) > 0]
    
    if not risk_drivers:
        # If student is very low risk, recommend general supportive check-in
        return [{
            **INTERVENTION_CATALOG["INT_BEH_02"],
            "matched_driver": "General Maintenance",
            "driver_impact_pct": 0.0,
            "rationale": "Student is currently at Low Risk. Maintain regular digital LMS resource utilization."
        }]

    # Sort drivers by absolute SHAP magnitude descending
    risk_drivers = sorted(risk_drivers, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)

    recommended_interventions = []
    seen_intervention_ids = set()
    seen_pillars = set()

    for driver in risk_drivers:
        feat_name = driver["feature_name"]
        pillar = FEATURE_TO_PILLAR_MAP.get(feat_name, "academic")
        shap_mag = abs(driver.get("shap_value", 0))
        delta_pct = driver.get("risk_delta_percentage_points", shap_mag * 100.0)

        # Find best matching intervention in this pillar
        best_candidate = None
        for int_id, int_data in INTERVENTION_CATALOG.items():
            if int_id in seen_intervention_ids:
                continue
            if feat_name in int_data.get("recommended_for", []):
                best_candidate = int_data
                break
        
        # Fallback to pillar match if exact feature match not found
        if best_candidate is None:
            for int_id, int_data in INTERVENTION_CATALOG.items():
                if int_id in seen_intervention_ids:
                    continue
                if int_data.get("pillar") == pillar:
                    best_candidate = int_data
                    break

        if best_candidate and best_candidate["intervention_id"] not in seen_intervention_ids:
            seen_intervention_ids.add(best_candidate["intervention_id"])
            seen_pillars.add(pillar)

            rec_item = {
                **best_candidate,
                "matched_driver_feature": feat_name,
                "matched_driver_display": driver.get("display_name", feat_name),
                "driver_impact_percentage_points": delta_pct,
                "rationale": f"Triggered by {driver.get('display_name', feat_name)} (+{delta_pct:.1f}% risk impact): {driver.get('plain_language_explanation', '')}"
            }
            recommended_interventions.append(rec_item)

            if len(recommended_interventions) >= max_recommendations:
                break

    return recommended_interventions


# =============================================================================
# 3. COUNTERFACTUAL RECOURSE GENERATOR ("Path to Improvement")
# =============================================================================
class CounterfactualRecourseEngine:
    """
    Computes minimal actionable changes to student features that reliably shift
    the student to a lower risk tier (e.g. High -> Medium, or Medium -> Low).
    """

    def __init__(self, model_path: Path = MODEL_ARTIFACT_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = []
        self._load()

    def _load(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        if FEATURE_NAMES_PATH.exists():
            with open(FEATURE_NAMES_PATH, "r") as f:
                self.feature_names = json.load(f)

    def generate_counterfactual(
        self,
        student_features: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        target_tier: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Performs grid search over realistic actionable levers:
        - Attendance Improvement: (e.g., Target 78%, 82%, 88%)
        - Fee Payment Resolution: (Delay -> 0 days)
        - Backlog Clearance: (Backlogs -> max(0, backlogs - 1 or 2))
        - LMS Engagement Boost: (Logins -> 6.0/wk, Inactivity -> 1 day)
        """
        if self.model is None:
            self._load()

        if isinstance(student_features, dict):
            student_df = pd.DataFrame([student_features])
        elif isinstance(student_features, pd.Series):
            student_df = pd.DataFrame([student_features])
        else:
            student_df = student_features.copy()

        # Ensure all columns present
        base_df = student_df.copy()
        
        # Calculate baseline probability & tier
        aligned_base = base_df[self.feature_names]
        base_prob = float(self.model.predict_proba(aligned_base)[:, 1][0])
        current_tier = get_risk_tier(base_prob)

        # Determine target tier if not explicitly specified
        if target_tier is None:
            if current_tier == "High":
                target_tier = "Medium"
                target_threshold = RISK_THRESHOLD_HIGH  # e.g. <= 0.66
            elif current_tier == "Medium":
                target_tier = "Low"
                target_threshold = RISK_THRESHOLD_LOW   # e.g. < 0.33
            else:
                target_tier = "Low"
                target_threshold = RISK_THRESHOLD_LOW

        if current_tier == "Low":
            return {
                "student_id": student_df.get("student_id", ["STUDENT"])[0] if "student_id" in student_df else "STUDENT",
                "current_risk_prob": round(base_prob, 4),
                "current_risk_tier": current_tier,
                "projected_risk_prob": round(base_prob, 4),
                "projected_risk_tier": "Low",
                "risk_reduction_pct": 0.0,
                "target_reached": True,
                "intervention_plan_name": "Low-Risk Academic Maintenance Plan",
                "status_message": "Student is already in the lowest risk tier. No urgent intervention required.",
                "required_actions": [],
                "counselor_summary": "Student is performing exceptionally well across all 4 pillars and remains in the Low Risk tier (prob < 33%). Continue regular progress monitoring."
            }

        # Plausible Intervention Simulation Scenarios
        scenarios = []

        curr_att = float(base_df["attendance_percentage"].iloc[0]) if "attendance_percentage" in base_df else 75.0
        curr_fee_delay = float(base_df["fee_payment_delay_days"].iloc[0]) if "fee_payment_delay_days" in base_df else 0.0
        curr_backlogs = int(base_df["backlog_count"].iloc[0]) if "backlog_count" in base_df else 0
        curr_logins = float(base_df["lms_logins_per_week"].iloc[0]) if "lms_logins_per_week" in base_df else 4.0
        curr_cgpa = float(base_df["current_cgpa"].iloc[0]) if "current_cgpa" in base_df else 6.0

        # Scenario 1: Attendance Remediation to 78% (clears 75% violation)
        if curr_att < 78.0:
            target_att = 80.0
            s1_df = base_df.copy()
            s1_df["attendance_percentage"] = target_att
            s1_df["attendance_month_3"] = max(target_att + 2.0, float(s1_df.get("attendance_month_3", [target_att])[0]))
            s1_df["attendance_3m_trend"] = 3.5  # Improving slope
            s1_df["consecutive_absences"] = 1
            s1_df["attendance_risk_flag"] = 0
            if "att_core1" in s1_df:
                s1_df["att_core1"] = max(float(s1_df["att_core1"].iloc[0]), target_att)
            s1_df = build_engineered_features(s1_df)
            p1 = float(self.model.predict_proba(s1_df[self.feature_names])[:, 1][0])
            scenarios.append({
                "name": "Attendance Recovery",
                "prob": p1,
                "delta": base_prob - p1,
                "actions": [{
                    "feature_name": "attendance_percentage",
                    "current_value": round(curr_att, 1),
                    "target_value": round(target_att, 1),
                    "plain_language_action": f"Improve overall class attendance from {curr_att:.1f}% to {target_att:.1f}% (+{target_att - curr_att:.1f}% by attending all lectures in the next 30 days)."
                }]
            })

        # Scenario 2: Fee Relief / Settlement (Delay -> 0 days + Emergency Scholarship)
        if curr_fee_delay > 0:
            s2_df = base_df.copy()
            s2_df["fee_payment_delay_days"] = 0
            s2_df["has_scholarship"] = 1
            s2_df = build_engineered_features(s2_df)
            p2 = float(self.model.predict_proba(s2_df[self.feature_names])[:, 1][0])
            scenarios.append({
                "name": "Fee Clearance & Scholarship",
                "prob": p2,
                "delta": base_prob - p2,
                "actions": [{
                    "feature_name": "fee_payment_delay_days",
                    "current_value": int(curr_fee_delay),
                    "target_value": 0,
                    "plain_language_action": f"Clear overdue tuition fees ({int(curr_fee_delay)} days pending) via financial aid desk installment plan."
                }]
            })

        # Scenario 3: Backlog Clearance (Clear 1-2 backlogs)
        if curr_backlogs > 0:
            target_backlogs = max(0, curr_backlogs - 1)
            s3_df = base_df.copy()
            s3_df["backlog_count"] = target_backlogs
            s3_df["internal_exam_score_pct"] = min(95.0, float(s3_df.get("internal_exam_score_pct", [50.0])[0]) + 15.0)
            s3_df = build_engineered_features(s3_df)
            p3 = float(self.model.predict_proba(s3_df[self.feature_names])[:, 1][0])
            scenarios.append({
                "name": "Backlog Clearance",
                "prob": p3,
                "delta": base_prob - p3,
                "actions": [{
                    "feature_name": "backlog_count",
                    "current_value": curr_backlogs,
                    "target_value": target_backlogs,
                    "plain_language_action": f"Clear {curr_backlogs - target_backlogs} uncleared subject backlog(s) in upcoming remedial supplementary exams."
                }]
            })

        # Scenario 4: Digital Re-engagement (Daily LMS Logins + On-Time Submissions)
        s4_df = base_df.copy()
        s4_df["lms_logins_per_week"] = max(curr_logins, 7.5)
        s4_df["days_since_last_lms_activity"] = 1
        s4_df["assignment_submission_lag_days"] = min(0.0, float(s4_df.get("assignment_submission_lag_days", [0.0])[0]))
        s4_df = build_engineered_features(s4_df)
        p4 = float(self.model.predict_proba(s4_df[self.feature_names])[:, 1][0])
        scenarios.append({
            "name": "Digital LMS Engagement",
            "prob": p4,
            "delta": base_prob - p4,
            "actions": [{
                "feature_name": "lms_logins_per_week",
                "current_value": round(curr_logins, 1),
                "target_value": 7.5,
                "plain_language_action": f"Increase LMS logins to 7-8 times/week and submit assignments before due dates."
            }]
        })

        # Scenario 5: Comprehensive Institutional Support Package
        # (Attendance Recovery + Academic Remediation + Financial Aid + Digital Re-engagement)
        s5_df = base_df.copy()
        target_att_s5 = max(80.0, curr_att + 25.0)
        s5_df["attendance_percentage"] = target_att_s5
        s5_df["attendance_month_3"] = target_att_s5 + 2.0
        s5_df["attendance_3m_trend"] = 4.5
        s5_df["consecutive_absences"] = 1
        s5_df["attendance_risk_flag"] = 0
        for col in ["att_core1", "att_core2", "att_lab", "att_elective"]:
            if col in s5_df:
                s5_df[col] = max(78.0, float(s5_df[col].iloc[0]) + 30.0)

        s5_df["fee_payment_delay_days"] = 0
        s5_df["has_scholarship"] = 1
        
        target_backlogs_s5 = max(0, curr_backlogs - 2) if curr_backlogs >= 2 else 0
        s5_df["backlog_count"] = target_backlogs_s5
        s5_df["internal_exam_score_pct"] = min(95.0, float(s5_df.get("internal_exam_score_pct", [50.0])[0]) + 25.0)
        s5_df["stem_core_fail_flag"] = 0
        s5_df["cgpa_delta"] = 0.50
        s5_df["current_cgpa"] = min(10.0, curr_cgpa + 1.2)
        
        s5_df["days_since_last_lms_activity"] = 1
        s5_df["lms_logins_per_week"] = max(curr_logins, 7.5)
        s5_df["assignment_submission_lag_days"] = -1.0
        s5_df = build_engineered_features(s5_df)
        p5 = float(self.model.predict_proba(s5_df[self.feature_names])[:, 1][0])

        combined_actions = []
        if curr_att < 75.0:
            combined_actions.append({
                "feature_name": "attendance_percentage",
                "current_value": round(curr_att, 1),
                "target_value": round(target_att_s5, 1),
                "plain_language_action": f"Raise class attendance from {curr_att:.1f}% to {target_att_s5:.1f}% (+{target_att_s5 - curr_att:.1f}% via regular attendance & lab make-up)."
            })
        if curr_fee_delay > 0:
            combined_actions.append({
                "feature_name": "fee_payment_delay_days",
                "current_value": int(curr_fee_delay),
                "target_value": 0,
                "plain_language_action": "Settle overdue tuition fee arrears with financial aid desk installment plan."
            })
        if curr_backlogs > 0:
            cleared = curr_backlogs - target_backlogs_s5
            combined_actions.append({
                "feature_name": "backlog_count",
                "current_value": curr_backlogs,
                "target_value": target_backlogs_s5,
                "plain_language_action": f"Clear {cleared} backlog course(s) through department remedial coaching & supplementary exams."
            })

        scenarios.append({
            "name": "Comprehensive Multi-Pillar Support Package",
            "prob": p5,
            "delta": base_prob - p5,
            "actions": combined_actions
        })

        # Pick the scenario that reaches target tier with minimal actions, or maximum risk drop
        target_reaching = [s for s in scenarios if get_risk_tier(s["prob"]) == target_tier]
        if target_reaching:
            best_scenario = sorted(target_reaching, key=lambda s: len(s["actions"]))[0]
        else:
            # Pick scenario with maximum risk reduction
            best_scenario = sorted(scenarios, key=lambda s: s["delta"], reverse=True)[0]

        projected_prob = float(best_scenario["prob"])
        projected_tier = get_risk_tier(projected_prob)

        return {
            "student_id": str(student_df.get("student_id", ["STUDENT"])[0] if "student_id" in student_df else "STUDENT"),
            "current_risk_prob": round(base_prob, 4),
            "current_risk_tier": current_tier,
            "projected_risk_prob": round(projected_prob, 4),
            "projected_risk_tier": projected_tier,
            "risk_reduction_pct": round((base_prob - projected_prob) * 100.0, 1),
            "target_reached": (projected_tier == target_tier or projected_prob < base_prob),
            "intervention_plan_name": best_scenario["name"],
            "required_actions": best_scenario["actions"],
            "counselor_summary": f"If the student executes the '{best_scenario['name']}' plan, their dropout risk is projected to decrease from {base_prob*100:.1f}% ({current_tier}) down to {projected_prob*100:.1f}% ({projected_tier}), achieving a {((base_prob - projected_prob)*100):.1f}% reduction."
        }


# =============================================================================
# 4. PRIORITIZED MENTOR QUEUE
# =============================================================================
def build_prioritized_mentor_queue(
    students_df: pd.DataFrame,
    department: Optional[str] = None,
    risk_tier_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Sorts and filters students for the mentor dashboard worklist:
    - Sorts descending by calibrated dropout risk score (highest risk first)
    - Breaks ties by active backlogs and attendance deficit
    """
    df = students_df.copy()

    # Optional department filter (if present)
    if department and "department" in df.columns:
        df = df[df["department"].astype(str).str.lower() == department.lower()]

    # Optional risk tier filter
    if risk_tier_filter and "risk_tier" in df.columns:
        df = df[df["risk_tier"].astype(str).str.lower() == risk_tier_filter.lower()]

    # Sort priority:
    # 1. ground_truth_risk_prob or calibrated_prob descending
    sort_col = "calibrated_prob" if "calibrated_prob" in df.columns else ("ground_truth_risk_prob" if "ground_truth_risk_prob" in df.columns else None)
    
    if sort_col:
        df = df.sort_values(by=[sort_col, "backlog_count", "attendance_percentage"], ascending=[False, False, True])
    
    # Assign queue priority rank
    df["queue_rank"] = np.arange(1, len(df) + 1)
    return df


# =============================================================================
# 5. INTERVENTION FEEDBACK & STATUS TRACKER
# =============================================================================
class InterventionStatusTracker:
    """
    Manages the operational lifecycle of student interventions:
    Status: ASSIGNED -> IN_PROGRESS -> APPLIED -> COMPLETED
    Outcome: PENDING_EVALUATION -> IMPROVED / NO_CHANGE / RESOLVED
    """

    def __init__(self):
        self.records: Dict[str, Dict[str, Any]] = {}

    def log_intervention_assignment(
        self,
        student_id: str,
        intervention_id: str,
        mentor_name: str,
        baseline_risk_prob: float,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Logs a new intervention assignment for a student."""
        rec_id = f"{student_id}_{intervention_id}_{int(datetime.now().timestamp())}"
        int_info = INTERVENTION_CATALOG.get(intervention_id, {})
        
        record = {
            "record_id": rec_id,
            "student_id": student_id,
            "intervention_id": intervention_id,
            "title": int_info.get("title", intervention_id),
            "pillar": int_info.get("pillar", "general"),
            "mentor_name": mentor_name,
            "assigned_date": datetime.now().isoformat(),
            "status": "ASSIGNED",
            "outcome_status": "PENDING_EVALUATION",
            "baseline_risk_prob": round(baseline_risk_prob, 4),
            "post_intervention_risk_prob": None,
            "risk_delta": None,
            "notes": notes
        }
        self.records[rec_id] = record
        logger.info("Logged intervention %s for student %s by mentor %s", intervention_id, student_id, mentor_name)
        return record

    def update_intervention_status(
        self,
        record_id: str,
        new_status: str,
        outcome_status: Optional[str] = None,
        post_intervention_risk_prob: Optional[float] = None,
        notes: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Updates the status and logs outcome of an ongoing intervention."""
        if record_id not in self.records:
            return None

        rec = self.records[record_id]
        rec["status"] = new_status
        if outcome_status:
            rec["outcome_status"] = outcome_status
        if post_intervention_risk_prob is not None:
            rec["post_intervention_risk_prob"] = round(post_intervention_risk_prob, 4)
            rec["risk_delta"] = round(rec["baseline_risk_prob"] - post_intervention_risk_prob, 4)
            if rec["risk_delta"] > 0.05:
                rec["outcome_status"] = "IMPROVED"
            elif rec["risk_delta"] < -0.05:
                rec["outcome_status"] = "DETERIORATED"
            else:
                rec["outcome_status"] = "NO_CHANGE"

        if notes:
            rec["notes"] = f"{rec.get('notes', '')} | {notes}".strip(" |")

        return rec

    def get_student_interventions(self, student_id: str) -> List[Dict[str, Any]]:
        return [r for r in self.records.values() if r["student_id"] == student_id]

    def get_all_records(self) -> List[Dict[str, Any]]:
        return list(self.records.values())


# Instantiate global tracker singleton
intervention_tracker = InterventionStatusTracker()


if __name__ == "__main__":
    from ml.config import PROCESSED_DATA_PATH
    df = pd.read_csv(PROCESSED_DATA_PATH)
    explainer_service = SHAPExplainerService()
    recourse_engine = CounterfactualRecourseEngine()

    sample_student = df.iloc[0]
    print(f"Testing Student: {sample_student['student_id']}")
    print(f"Current Features -> Attendance: {sample_student['attendance_percentage']}%, CGPA: {sample_student['current_cgpa']}, Backlogs: {sample_student['backlog_count']}, Fee Delay: {sample_student['fee_payment_delay_days']} days")

    # 1. SHAP Local Drivers
    shap_drivers = explainer_service.explain_local_student(sample_student, top_k=4)
    print("\n--- SHAP Top Drivers ---")
    for d in shap_drivers:
        print(f"[{d['impact_direction']}] {d['plain_language_explanation']}")

    # 2. Selected Interventions
    interventions = map_shap_drivers_to_interventions(shap_drivers)
    print("\n--- Recommended Interventions ---")
    for intv in interventions:
        print(f"[{intv['urgency']}] {intv['title']} (Pillar: {intv['pillar']})")
        print(f"  Action: {intv['description']}")
        print(f"  Rationale: {intv['rationale']}\n")

    # 3. Counterfactual Path
    recourse = recourse_engine.generate_counterfactual(sample_student)
    print("--- Counterfactual Recourse Plan ---")
    print(f"Plan: {recourse['intervention_plan_name']}")
    print(f"Current Risk: {recourse['current_risk_prob']*100:.1f}% ({recourse['current_risk_tier']}) -> Projected Risk: {recourse['projected_risk_prob']*100:.1f}% ({recourse['projected_risk_tier']})")
    print(f"Reduction: -{recourse['risk_reduction_pct']}%")
    print("Action Steps:")
    for act in recourse['required_actions']:
        print(f" - {act['plain_language_action']}")
