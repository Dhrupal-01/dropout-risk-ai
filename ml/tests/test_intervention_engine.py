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
import pandas as pd

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
        for intv in INTERVENTION_CATALOG.values():
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
def recourse_engine(simulated_artifacts):
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


BANNED_PHRASES = ("performing exceptionally well", "No urgent intervention required")


def _first_catalog_item_for(flag):
    return next(i for i, item in INTERVENTION_CATALOG.items() if flag in item["recommended_for"])


class TestRuleBasedAlerts:
    """Alerts come from the student's values and the configured threshold, for every tier."""

    def test_attendance_alert_uses_configured_threshold(self, monkeypatch):
        import ml.intervention.engine as engine

        monkeypatch.setattr(engine, "ATTENDANCE_THRESHOLD", 80.0)
        alerts = engine.rule_based_alerts({"attendance_percentage": 78.0, "attendance_risk_flag": 0})
        assert [a["code"] for a in alerts] == ["ATTENDANCE_BELOW_REQUIREMENT"]
        assert alerts[0]["threshold"] == 80.0
        assert alerts[0]["message"] == "attendance is 78.0%, below the 80% requirement"

    def test_threshold_is_the_assumptions_yaml_value(self):
        import yaml
        from ml.config import ASSUMPTIONS_PATH, ATTENDANCE_THRESHOLD

        raw = yaml.safe_load(ASSUMPTIONS_PATH.read_text(encoding="utf-8"))
        assert ATTENDANCE_THRESHOLD == float(raw["regulations_and_thresholds"]["mandatory_attendance_threshold"]["value"])

    def test_no_alert_at_or_above_threshold(self):
        from ml.config import ATTENDANCE_THRESHOLD
        from ml.intervention.engine import rule_based_alerts

        assert rule_based_alerts({"attendance_percentage": ATTENDANCE_THRESHOLD, "attendance_risk_flag": 0,
                                  "academic_crisis_flag": 0}) == []

    def test_recommended_actions_come_from_the_catalog(self):
        from ml.intervention.engine import rule_based_alerts

        alerts = rule_based_alerts({"attendance_percentage": 40.0, "attendance_risk_flag": 1,
                                    "academic_crisis_flag": 1, "current_cgpa": 4.5, "backlog_count": 3})
        by_code = {a["code"]: a for a in alerts}
        att, acad = by_code["ATTENDANCE_BELOW_REQUIREMENT"], by_code["ACADEMIC_CRISIS_FLAG"]
        assert att["recommended_intervention_id"] == _first_catalog_item_for("attendance_risk_flag")
        assert acad["recommended_intervention_id"] == _first_catalog_item_for("academic_crisis_flag")
        assert att["recommended_intervention_title"] == INTERVENTION_CATALOG[att["recommended_intervention_id"]]["title"]
        assert acad["message"] == "academic crisis flag is set (CGPA 4.50, 3 active backlogs)"

    def test_low_tier_student_below_threshold_gets_the_alert(self, recourse_engine):
        from ml.config import ATTENDANCE_THRESHOLD, get_risk_tier

        df = pd.read_csv(PROCESSED_DATA_PATH)
        below = df[df["attendance_percentage"] < ATTENDANCE_THRESHOLD]
        probs = recourse_engine.model.predict_proba(below[recourse_engine.feature_names])[:, 1]
        low = below[[get_risk_tier(p) == "Low" for p in probs]]
        assert len(low) > 0, "no Low-tier student below the attendance threshold in the simulated cohort"

        student = low.iloc[0]
        recourse = recourse_engine.generate_counterfactual(student)
        assert recourse["current_risk_tier"] == "Low"
        att = [a for a in recourse["rule_based_alerts"] if a["code"] == "ATTENDANCE_BELOW_REQUIREMENT"]
        assert len(att) == 1
        assert recourse["counselor_summary"].startswith("Low model risk")
        assert att[0]["message"] in recourse["counselor_summary"]
        assert att[0]["recommended_intervention_title"] in recourse["counselor_summary"]
        for phrase in BANNED_PHRASES:
            assert phrase not in str(recourse)

    def test_shap_failure_is_logged_and_reported(self, recourse_engine, monkeypatch, caplog):
        import logging
        import ml.intervention.engine as engine

        def boom(self, *args, **kwargs):
            raise RuntimeError("explainer exploded")

        monkeypatch.setattr(engine.SHAPExplainerService, "explain_local_student", boom)
        student = pd.read_csv(PROCESSED_DATA_PATH).iloc[0]
        with caplog.at_level(logging.ERROR, logger=engine.logger.name):
            recourse = recourse_engine.generate_counterfactual(student, top_shap_drivers=None)
        assert recourse["drivers_available"] is False
        assert any("SHAP drivers unavailable" in r.getMessage() and r.exc_info for r in caplog.records)

    def test_passed_drivers_are_available(self, recourse_engine):
        student = pd.read_csv(PROCESSED_DATA_PATH).iloc[0]
        assert recourse_engine.generate_counterfactual(student, top_shap_drivers=[])["drivers_available"] is True


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
