"""
Unit and integration tests for Universal Multi-Tier and Geo-Analytics Endpoints.
"""

import pytest
from backend.app.main import app

class TestGeoAnalyticsAPI:
    def test_get_states_overview(self, client):
        response = client.get("/api/v1/analytics/geo/states")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 8
        first_state = data[0]
        assert "state_name" in first_state
        assert "literacy_rate" in first_state
        assert "mpi_poverty_pct" in first_state

    def test_get_districts_by_state(self, client):
        response = client.get("/api/v1/analytics/geo/districts?state=Bihar")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 4
        purnia = next((d for d in data if d["district_name"] == "Purnia"), None)
        assert purnia is not None
        assert purnia["is_aspirational"] is True

    def test_get_national_heatmap(self, client):
        response = client.get("/api/v1/analytics/geo/heatmap")
        assert response.status_code == 200
        data = response.json()
        assert "total_states_monitored" in data
        assert "states" in data
        assert len(data["states"]) >= 8

class TestUniversalPredictionAPI:
    def test_predict_pre_10th(self, client):
        payload = {
            "student_id": "TEST_SCH_001",
            "educational_tier": "PRE_10TH",
            "state": "Bihar",
            "district": "Purnia",
            "features": {
                "distance_to_school_km": 7.0,
                "attendance_percentage": 50.0,
                "attendance_risk_flag": 1,
                "mid_day_meal_attendance_pct": 45.0,
                "parental_literacy_years": 1,
                "seasonal_migration_flag": 1,
                "fln_score_pct": 30.0,
                "sibling_count_under_5": 2,
                "toilets_available_flag": 0,
                "state_literacy_rate": 61.8,
                "district_mpi_pct": 50.4,
                "is_aspirational_district": 1,
                "district_rurality_pct": 89.2,
                "district_ptr_ratio": 42.1
            }
        }
        response = client.post("/api/v1/universal/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["student_id"] == "TEST_SCH_001"
        assert data["educational_tier"] == "PRE_10TH"
        assert data["risk_tier"] == "High"
        assert len(data["top_drivers"]) >= 3
        assert len(data["recommended_interventions"]) >= 1

    def test_predict_doctorate_phd(self, client):
        payload = {
            "student_id": "TEST_PHD_002",
            "educational_tier": "DOCTORATE",
            "state": "Tamil Nadu",
            "district": "Ramanathapuram",
            "features": {
                "phd_tenure_months": 40,
                "fellowship_disbursement_delay_days": 100,
                "advisor_meeting_frequency_per_month": 0,
                "peer_reviewed_submissions": 0,
                "comprehensive_exam_attempts": 2,
                "research_stagnation_index": 0.85,
                "stipend_adequacy_ratio": 0.40,
                "lab_isolation_index": 0.90,
                "state_literacy_rate": 80.1,
                "district_mpi_pct": 7.8,
                "is_aspirational_district": 1
            }
        }
        response = client.post("/api/v1/universal/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["educational_tier"] == "DOCTORATE"
        assert data["risk_tier"] == "High"
        assert any("fellowship" in d["plain_language_explanation"].lower() for d in data["top_drivers"])
