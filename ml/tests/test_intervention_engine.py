"""
Unit & Integration Tests for DropoutGuard Intervention Engine
Tests:
- Structured catalog integrity across 4 pillars
- SHAP driver to intervention selection and ranking
- Counterfactual recourse generation (verifies risk probability reduction)
- Prioritized mentor queue sorting
- Intervention status tracking lifecycle
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.intervention.engine import (
    INTERVENTION_CATALOG,
    map_shap_drivers_to_interventions,
    CounterfactualRecourseEngine,
    build_prioritized_mentor_queue,
    InterventionStatusTracker
)
from ml.config import PROCESSED_DATA_PATH


class TestInterventionCatalog:
    """Tests for intervention catalog schema and completeness."""

    def test_catalog_contains_all_four_pillars(self):
        """Verify all 4 core pillars are represented in the intervention catalog."""
        pillars = {intv["pillar"] for intv in INTERVENTION_CATALOG.values()}
        assert "attendance" in pillars
        assert "academic" in pillars
        assert "financial" in pillars
        assert "engagement" in pillars

    def test_catalog_item_schema(self):
        """Verify each intervention item has required metadata fields."""
        for int_id, intv in INTERVENTION_CATALOG.items():
            assert "intervention_id" in intv
            assert "title" in intv
            assert "description" in intv
            assert "action_type" in intv
            assert "urgency" in intv
            assert intv["urgency"] in ["HIGH", "MEDIUM", "LOW"]
            assert "suggested_duration_days" in intv
            assert intv["suggested_duration_days"] > 0


class TestSHAPToInterventionMapping:
    """Tests for mapping SHAP drivers to appropriate intervention actions."""

    def test_attendance_driver_mapping(self):
        """Verify attendance deficit SHAP driver triggers attendance counseling."""
        mock_shap = [
            {
                "feature_name": "attendance_percentage",
                "display_name": "Overall Attendance Rate",
                "feature_value": 48.0,
                "shap_value": 0.35,
                "impact_direction": "RISK_INCREASING",
                "risk_delta_percentage_points": 35.0,
                "plain_language_explanation": "Low attendance is increasing risk."
            }
        ]
        recs = map_shap_drivers_to_interventions(mock_shap, max_recommendations=2)
        assert len(recs) >= 1
        assert recs[0]["pillar"] == "attendance"
        assert recs[0]["intervention_id"] in ["INT_ATT_01", "INT_ATT_02"]

    def test_academic_backlog_driver_mapping(self):
        """Verify active backlogs SHAP driver triggers academic remedial support."""
        mock_shap = [
            {
                "feature_name": "backlog_count",
                "display_name": "Uncleared Backlog Count",
                "feature_value": 3.0,
                "shap_value": 0.28,
                "impact_direction": "RISK_INCREASING",
                "risk_delta_percentage_points": 28.0,
                "plain_language_explanation": "3 backlogs are increasing risk."
            }
        ]
        recs = map_shap_drivers_to_interventions(mock_shap, max_recommendations=2)
        assert len(recs) >= 1
        assert recs[0]["pillar"] == "academic"
        assert recs[0]["intervention_id"] in ["INT_ACAD_01", "INT_ACAD_02"]

    def test_financial_fee_delay_driver_mapping(self):
        """Verify fee payment delay SHAP driver triggers financial aid desk."""
        mock_shap = [
            {
                "feature_name": "fee_payment_delay_days",
                "display_name": "Tuition Fee Payment Overdue Days",
                "feature_value": 60.0,
                "shap_value": 0.22,
                "impact_direction": "RISK_INCREASING",
                "risk_delta_percentage_points": 22.0,
                "plain_language_explanation": "60 days fee delay is increasing risk."
            }
        ]
        recs = map_shap_drivers_to_interventions(mock_shap, max_recommendations=2)
        assert len(recs) >= 1
        assert recs[0]["pillar"] == "financial"
        assert recs[0]["intervention_id"] in ["INT_FIN_01", "INT_FIN_02"]

    def test_engagement_inactivity_driver_mapping(self):
        """Verify LMS disengagement SHAP driver triggers counseling/onboarding."""
        mock_shap = [
            {
                "feature_name": "days_since_last_lms_activity",
                "display_name": "LMS Inactivity Recency",
                "feature_value": 25.0,
                "shap_value": 0.19,
                "impact_direction": "RISK_INCREASING",
                "risk_delta_percentage_points": 19.0,
                "plain_language_explanation": "25 days inactivity."
            }
        ]
        recs = map_shap_drivers_to_interventions(mock_shap, max_recommendations=2)
        assert len(recs) >= 1
        assert recs[0]["pillar"] == "engagement"
        assert recs[0]["intervention_id"] in ["INT_BEH_01", "INT_BEH_02"]


@pytest.fixture(scope="module")
def recourse_engine():
    return CounterfactualRecourseEngine()


class TestCounterfactualRecourse:
    """Tests for counterfactual recourse and path-to-improvement generation."""

    def test_counterfactual_reduces_predicted_risk(self, recourse_engine):
        """Verify counterfactual plan measurably decreases predicted risk probability."""
        df = pd.read_csv(PROCESSED_DATA_PATH)
        # Find a high-risk student
        high_risk_candidates = df[df["ground_truth_risk_prob"] > 0.75]
        assert len(high_risk_candidates) > 0

        target_student = high_risk_candidates.iloc[0]
        recourse = recourse_engine.generate_counterfactual(target_student)

        assert "current_risk_prob" in recourse
        assert "projected_risk_prob" in recourse
        assert "risk_reduction_pct" in recourse
        assert "required_actions" in recourse

        # Projected risk must be lower than current risk
        assert recourse["projected_risk_prob"] < recourse["current_risk_prob"]
        assert recourse["risk_reduction_pct"] > 0.0
        assert len(recourse["required_actions"]) > 0


class TestPrioritizedMentorQueue:
    """Tests for sorting and queue generation."""

    def test_queue_sorting_descending_risk(self):
        """Verify mentor worklist is strictly sorted with highest risk students first."""
        dummy_students = pd.DataFrame({
            "student_id": ["S1", "S2", "S3", "S4"],
            "calibrated_prob": [0.15, 0.92, 0.45, 0.88],
            "risk_tier": ["Low", "High", "Medium", "High"],
            "backlog_count": [0, 2, 0, 1],
            "attendance_percentage": [92.0, 45.0, 78.0, 52.0]
        })
        queue = build_prioritized_mentor_queue(dummy_students)

        # Expected order: S2 (0.92), S4 (0.88), S3 (0.45), S1 (0.15)
        assert list(queue["student_id"]) == ["S2", "S4", "S3", "S1"]
        assert list(queue["queue_rank"]) == [1, 2, 3, 4]


class TestInterventionTracker:
    """Tests for operational intervention logging and status tracking."""

    def test_lifecycle_transitions(self):
        """Verify recording, status update, and outcome evaluation."""
        tracker = InterventionStatusTracker()
        
        # 1. Log assignment
        rec = tracker.log_intervention_assignment(
            student_id="IND_2026_9999",
            intervention_id="INT_ATT_01",
            mentor_name="Dr. Sharma",
            baseline_risk_prob=0.88,
            notes="Initial attendance warning"
        )
        rec_id = rec["record_id"]
        assert rec["status"] == "ASSIGNED"
        assert rec["baseline_risk_prob"] == 0.88

        # 2. Update to APPLIED with improved outcome
        updated = tracker.update_intervention_status(
            record_id=rec_id,
            new_status="APPLIED",
            post_intervention_risk_prob=0.35,
            notes="Attendance recovered to 82%"
        )
        assert updated is not None
        assert updated["status"] == "APPLIED"
        assert updated["outcome_status"] == "IMPROVED"
        assert updated["risk_delta"] == 0.53  # 0.88 - 0.35
