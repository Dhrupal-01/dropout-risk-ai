"""
Universal Multi-Tier Pipeline Validation Script for DropoutGuard.
Validates end-to-end inference, SHAP explanations, and intervention mapping across
5 distinct archetypes (Pre-10th, Higher Secondary, Undergraduate, Postgraduate, Doctorate)
with regional/district socio-economic variation.
"""

import json
from ml.models.universal_engine import universal_engine

SAMPLE_STUDENTS = [
    {
        "student_id": "STU_SCH_01_BIHAR_PURNEA",
        "educational_tier": "PRE_10TH",
        "gender": "Female",
        "age": 12,
        "state": "Bihar",
        "district": "Purnia",
        "is_aspirational_district": 1,
        "district_mpi_pct": 50.4,
        "state_literacy_rate": 61.8,
        "district_rurality_pct": 89.2,
        "district_ptr_ratio": 42.1,
        "distance_to_school_km": 6.5,
        "attendance_percentage": 52.0,
        "attendance_risk_flag": 1,
        "mid_day_meal_attendance_pct": 48.0,
        "parental_literacy_years": 2,
        "seasonal_migration_flag": 1,
        "fln_score_pct": 34.0,
        "sibling_count_under_5": 2,
        "toilets_available_flag": 0
    },
    {
        "student_id": "STU_HSC_02_UP_BAHRAICH",
        "educational_tier": "HIGHER_SECONDARY",
        "gender": "Male",
        "age": 17,
        "state": "Uttar Pradesh",
        "district": "Bahraich",
        "is_aspirational_district": 1,
        "district_mpi_pct": 54.4,
        "state_literacy_rate": 67.7,
        "district_rurality_pct": 92.1,
        "class_10_board_pct": 58.0,
        "attendance_percentage": 48.5,
        "attendance_risk_flag": 1,
        "stream_preference_match": 0,
        "coaching_financial_strain": 0.85,
        "mid_term_board_sim_score": 38.0,
        "vocational_practical_att_pct": 42.0,
        "family_debt_pressure_flag": 1,
        "consecutive_absences": 8
    },
    {
        "student_id": "STU_UG_03_MAHA_PUNE",
        "educational_tier": "UNDERGRADUATE",
        "gender": "Male",
        "age": 20,
        "state": "Maharashtra",
        "district": "Pune",
        "is_aspirational_district": 0,
        "district_mpi_pct": 3.9,
        "state_literacy_rate": 82.3,
        "attendance_percentage": 42.5,
        "attendance_month_1": 65.0,
        "attendance_month_2": 45.0,
        "attendance_month_3": 30.0,
        "attendance_3m_trend": -17.5,
        "consecutive_absences": 6,
        "attendance_risk_flag": 1,
        "subject_attendance_std": 8.5,
        "att_core1": 35.0,
        "att_core2": 40.0,
        "att_lab": 50.0,
        "att_elective": 45.0,
        "current_cgpa": 4.85,
        "prev_sem_cgpa": 6.50,
        "cgpa_delta": -1.65,
        "backlog_count": 3,
        "internal_exam_score_pct": 44.0,
        "stem_core_fail_flag": 1,
        "academic_crisis_flag": 1,
        "lms_logins_per_week": 1.2,
        "assignment_submission_lag_days": 6.5,
        "resource_access_count": 8,
        "days_since_last_lms_activity": 26,
        "forum_participation_count": 0,
        "behavioral_disengagement_index": 0.88,
        "income_slab_idx": 1,
        "is_first_generation": 1,
        "fee_payment_delay_days": 55,
        "has_scholarship": 0,
        "is_hosteler": 0,
        "commute_distance_km": 18.0,
        "financial_stress_index": 0.72,
        "interaction_att_x_fee": 105.4,
        "interaction_cgpa_x_backlog": 15.45,
        "interaction_firstgen_x_inactivity": 1.0,
        "interaction_att_x_cgpa_drop": 1.65
    },
    {
        "student_id": "STU_PG_04_HARYANA_NUH",
        "educational_tier": "POSTGRADUATE",
        "gender": "Female",
        "age": 23,
        "state": "Haryana",
        "district": "Nuh / Mewat",
        "is_aspirational_district": 1,
        "district_mpi_pct": 44.8,
        "state_literacy_rate": 75.6,
        "ug_graduation_cgpa": 6.8,
        "current_pg_cgpa": 5.1,
        "attendance_percentage": 58.0,
        "dissertation_milestone_pct": 25.0,
        "education_loan_burden_inr": 650000,
        "placement_readiness_score": 32.0,
        "part_time_work_hours_per_week": 24,
        "backlog_count": 2,
        "fee_payment_delay_days": 45
    },
    {
        "student_id": "STU_PHD_05_TAMILNADU_RAMANATH",
        "educational_tier": "DOCTORATE",
        "gender": "Male",
        "age": 28,
        "state": "Tamil Nadu",
        "district": "Ramanathapuram",
        "is_aspirational_district": 1,
        "district_mpi_pct": 7.8,
        "state_literacy_rate": 80.1,
        "phd_tenure_months": 42,
        "fellowship_disbursement_delay_days": 110,
        "advisor_meeting_frequency_per_month": 0,
        "peer_reviewed_submissions": 0,
        "comprehensive_exam_attempts": 2,
        "research_stagnation_index": 0.88,
        "stipend_adequacy_ratio": 0.35,
        "lab_isolation_index": 0.92
    }
]

def run_universal_validation():
    print("=" * 80)
    print("      DROPOUTGUARD — UNIVERSAL MULTI-TIER & GEO PIPELINE VALIDATION     ")
    print("=" * 80)

    for i, student in enumerate(SAMPLE_STUDENTS, 1):
        tier = student["educational_tier"]
        stu_id = student["student_id"]
        state = student["state"]
        district = student["district"]
        asp = " (NITI Aayog Aspirational District)" if student.get("is_aspirational_district") else ""

        print(f"\n[{i}/5] EVALUATING ARCHETYPE: {stu_id}")
        print(f" • Educational Stage: {tier}")
        print(f" • Geographic Context: {district}, {state}{asp}")
        print("-" * 80)

        result = universal_engine.predict_universal_student(student)
        risk_pct = result["risk_score_percentage"]
        tier_lbl = result["risk_tier"]
        print(f" AI INFERENCE -> Calibrated Dropout Risk: {risk_pct}% | Risk Tier: [{tier_lbl}]")
        print("\n Top SHAP Root Cause Drivers:")
        for d in result["top_drivers"]:
            print(f"   - {d['plain_language_explanation']}")

        print("\n Recommended Tier-Specific Interventions:")
        for intv in result["recommended_interventions"]:
            print(f"   * [{intv.get('urgency', 'HIGH')}] {intv.get('title')}")
            print(f"     Action: {intv.get('action_description')}")
        print("=" * 80)

if __name__ == "__main__":
    run_universal_validation()
