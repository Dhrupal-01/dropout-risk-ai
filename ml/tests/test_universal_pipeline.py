"""
Automated unit test suite for Universal Multi-Tier Pipeline in DropoutGuard.
Tests all 5 educational stages (Pre-10th, Higher Secondary, UG, PG, PhD)
and macro-geographic contextual inference.
"""

import pytest
import numpy as np
import pandas as pd
from ml.models.universal_engine import UniversalModelEngine, TIER_CONFIG

@pytest.fixture(scope="module")
def universal_engine():
    engine = UniversalModelEngine()
    engine.load_all_artifacts()
    return engine

class TestUniversalMultiTierPipeline:
    def test_all_tier_artifacts_loaded(self, universal_engine):
        for tier in ["PRE_10TH", "HIGHER_SECONDARY", "UNDERGRADUATE", "POSTGRADUATE", "DOCTORATE"]:
            assert tier in universal_engine.models
            assert tier in universal_engine.explainers
            assert tier in universal_engine.feature_sets
            assert len(universal_engine.feature_sets[tier]) > 0

    def test_pre_10th_inference_and_shap(self, universal_engine):
        student = {
            "student_id": "TEST_SCH_01",
            "educational_tier": "PRE_10TH",
            "distance_to_school_km": 8.0,
            "attendance_percentage": 45.0,
            "attendance_risk_flag": 1,
            "mid_day_meal_attendance_pct": 40.0,
            "parental_literacy_years": 0,
            "seasonal_migration_flag": 1,
            "fln_score_pct": 25.0,
            "sibling_count_under_5": 2,
            "toilets_available_flag": 0,
            "state_literacy_rate": 61.8,
            "district_mpi_pct": 50.4,
            "is_aspirational_district": 1,
            "district_rurality_pct": 92.0,
            "district_ptr_ratio": 44.0
        }
        res = universal_engine.predict_universal_student(student)
        assert res["educational_tier"] == "PRE_10TH"
        assert 0.0 <= res["calibrated_risk_probability"] <= 1.0
        assert res["risk_tier"] == "High"
        assert len(res["top_drivers"]) >= 3
        assert len(res["recommended_interventions"]) >= 1

    def test_phd_inference_and_shap(self, universal_engine):
        scholar = {
            "student_id": "TEST_PHD_01",
            "educational_tier": "DOCTORATE",
            "phd_tenure_months": 48,
            "fellowship_disbursement_delay_days": 120,
            "advisor_meeting_frequency_per_month": 0,
            "peer_reviewed_submissions": 0,
            "comprehensive_exam_attempts": 2,
            "research_stagnation_index": 0.90,
            "stipend_adequacy_ratio": 0.30,
            "lab_isolation_index": 0.95,
            "state_literacy_rate": 80.0,
            "district_mpi_pct": 10.0,
            "is_aspirational_district": 0
        }
        res = universal_engine.predict_universal_student(scholar)
        assert res["educational_tier"] == "DOCTORATE"
        assert res["calibrated_risk_probability"] > 0.66
        assert res["risk_tier"] == "High"
        assert any("fellowship" in d["plain_language_explanation"].lower() for d in res["top_drivers"])
        assert any("INT_PHD" in intv["intervention_id"] for intv in res["recommended_interventions"])

    def test_low_risk_undergraduate_stays_low(self, universal_engine):
        ug_student = {
            "student_id": "TEST_UG_SAFE",
            "educational_tier": "UNDERGRADUATE",
            "attendance_percentage": 92.0,
            "attendance_month_1": 90.0,
            "attendance_month_2": 92.0,
            "attendance_month_3": 94.0,
            "attendance_3m_trend": 4.0,
            "consecutive_absences": 0,
            "attendance_risk_flag": 0,
            "subject_attendance_std": 2.1,
            "att_core1": 92.0,
            "att_core2": 95.0,
            "att_lab": 96.0,
            "att_elective": 90.0,
            "current_cgpa": 8.9,
            "prev_sem_cgpa": 8.7,
            "cgpa_delta": 0.2,
            "backlog_count": 0,
            "internal_exam_score_pct": 88.0,
            "stem_core_fail_flag": 0,
            "academic_crisis_flag": 0,
            "lms_logins_per_week": 8.5,
            "assignment_submission_lag_days": -1.5,
            "resource_access_count": 45,
            "days_since_last_lms_activity": 1,
            "forum_participation_count": 5,
            "behavioral_disengagement_index": 0.05,
            "income_slab_idx": 2,
            "is_first_generation": 0,
            "fee_payment_delay_days": 0,
            "has_scholarship": 1,
            "is_hosteler": 1,
            "commute_distance_km": 2.0,
            "financial_stress_index": 0.0,
            "interaction_att_x_fee": 0.0,
            "interaction_cgpa_x_backlog": 0.0,
            "interaction_firstgen_x_inactivity": 0.0,
            "interaction_att_x_cgpa_drop": 0.0,
            "age": 19,
            "state_literacy_rate": 82.0,
            "district_mpi_pct": 4.0,
            "is_aspirational_district": 0
        }
        res = universal_engine.predict_universal_student(ug_student)
        assert res["calibrated_risk_probability"] < 0.33
        assert res["risk_tier"] == "Low"
