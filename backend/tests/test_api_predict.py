"""
API tests for POST /api/v1/predict, /predict/batch and /predict/batch/csv.

Live-database tests are guarded by `requires_db`; the schema comes from the real Alembic
migration (see conftest).
"""

import io

import pandas as pd
import pytest
from sqlalchemy import func, select

from backend.app.models.prediction import Prediction
from backend.app.models.student import Student
from backend.app.services.ml_service import (
    ENGINEERED_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
)
from backend.tests.conftest import requires_db

PREDICT_URL = "/api/v1/predict"
BATCH_URL = "/api/v1/predict/batch"
CSV_URL = "/api/v1/predict/batch/csv"


# --------------------------------------------------------------------------- payloads

@pytest.fixture
def high_risk_features(sample_raw_features):
    return dict(sample_raw_features)


@pytest.fixture
def low_risk_features(sample_raw_features):
    """Strong attendance, high CGPA, no arrears."""
    return dict(
        sample_raw_features,
        att_core1=95.0, att_core2=94.0, att_lab=98.0, att_elective=92.0,
        attendance_month_1=92.0, attendance_month_2=95.0, attendance_month_3=96.0,
        attendance_percentage=94.5, attendance_3m_trend=2.0,
        consecutive_absences=0, attendance_risk_flag=0,
        prev_sem_cgpa=8.5, current_cgpa=8.9, cgpa_delta=0.4,
        backlog_count=0, internal_exam_score_pct=92.0, stem_core_fail_flag=0,
        lms_logins_per_week=11.5, assignment_submission_lag_days=-2.5,
        resource_access_count=95, days_since_last_lms_activity=1,
        forum_participation_count=12,
        fee_payment_delay_days=0, has_scholarship=1, is_first_generation=0,
        income_slab_idx=3, hostel_status="Hosteler", commute_distance_km=0.5,
    )


@pytest.fixture
def medium_risk_features():
    """
    A genuine Medium-tier profile taken from the real cohort (calibrated p = 0.5023):
    slipping attendance just under the 75% bar, flat CGPA, no backlogs, fees clear.
    Medium is a narrow band (92 of 2000 students), so this is drawn from real data
    rather than hand-tuned.
    """
    return {
        "age": 20.7,
        "commute_distance_km": 0.5,
        "income_slab_idx": 0,
        "is_first_generation": 1,
        "has_scholarship": 1,
        "fee_payment_delay_days": 0,
        "att_core1": 63.1, "att_core2": 75.6, "att_lab": 73.7, "att_elective": 62.4,
        "attendance_month_1": 81.4, "attendance_month_2": 73.3, "attendance_month_3": 65.7,
        "attendance_percentage": 72.3, "attendance_3m_trend": -7.89,
        "consecutive_absences": 2, "attendance_risk_flag": 1,
        "prev_sem_cgpa": 6.32, "current_cgpa": 6.25, "cgpa_delta": -0.07,
        "backlog_count": 0, "internal_exam_score_pct": 46.2, "stem_core_fail_flag": 0,
        "lms_logins_per_week": 6.6, "assignment_submission_lag_days": -2.1,
        "resource_access_count": 36, "days_since_last_lms_activity": 9,
        "forum_participation_count": 3,
        "hostel_status": "Hosteler",
    }


def body(student_id, features, **extra):
    return {"student_id": student_id, "features": features, **extra}


# --------------------------------------------------------------- validation (no DB)

class TestInputValidation:
    def test_missing_required_field_is_422(self, client, high_risk_features):
        features = dict(high_risk_features)
        del features["current_cgpa"]
        response = client.post(PREDICT_URL, json=body("V_001", features))
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"] == "validation_error"
        assert any("current_cgpa" in d["field"] for d in payload["details"])

    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("current_cgpa", 15.0),          # CGPA is 0-10
            ("attendance_percentage", 150.0),
            ("income_slab_idx", 9),          # ordinal 0-3
            ("has_scholarship", 2),          # binary
            ("backlog_count", -1),
            ("age", 5.0),
        ],
    )
    def test_out_of_range_values_are_422(self, client, high_risk_features, field, bad_value):
        response = client.post(
            PREDICT_URL, json=body("V_002", dict(high_risk_features, **{field: bad_value}))
        )
        assert response.status_code == 422
        assert any(field in d["field"] for d in response.json()["details"])

    @pytest.mark.parametrize("label", ["is_dropout", "ground_truth_risk_prob"])
    def test_label_leakage_is_rejected(self, client, high_risk_features, label):
        """Ground-truth targets must never be accepted as predictors."""
        response = client.post(
            PREDICT_URL, json=body("V_003", dict(high_risk_features, **{label: 1}))
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("engineered", list(ENGINEERED_FEATURE_COLUMNS))
    def test_engineered_features_cannot_be_supplied(self, client, high_risk_features, engineered):
        """The pipeline owns these; a caller must not override the trained formula."""
        response = client.post(
            PREDICT_URL, json=body("V_004", dict(high_risk_features, **{engineered: 0.5}))
        )
        assert response.status_code == 422

    def test_unknown_field_is_rejected(self, client, high_risk_features):
        response = client.post(
            PREDICT_URL, json=body("V_005", dict(high_risk_features, made_up_feature=1))
        )
        assert response.status_code == 422

    def test_missing_residency_is_422(self, client, high_risk_features):
        features = dict(high_risk_features)
        features.pop("hostel_status", None)
        features.pop("is_hosteler", None)
        response = client.post(PREDICT_URL, json=body("V_006", features))
        assert response.status_code == 422

    def test_errors_never_leak_tracebacks(self, client, high_risk_features):
        response = client.post(PREDICT_URL, json=body("V_007", {"age": 20.0}))
        raw = response.text.lower()
        for marker in ("traceback", "file \"", ".py\", line", "site-packages"):
            assert marker not in raw


# ------------------------------------------------------------------ scoring (live DB)

@requires_db
class TestSinglePrediction:
    def test_high_risk_student(self, client, db_engine, high_risk_features):
        response = client.post(PREDICT_URL, json=body("API_HIGH", high_risk_features))
        assert response.status_code == 201
        payload = response.json()
        assert payload["student_id"] == "API_HIGH"
        assert payload["risk_tier"] == "High"
        assert 0.0 <= payload["calibrated_risk_probability"] <= 1.0

    def test_low_risk_student(self, client, db_engine, low_risk_features):
        payload = client.post(PREDICT_URL, json=body("API_LOW", low_risk_features)).json()
        assert payload["risk_tier"] == "Low"

    def test_medium_risk_student(self, client, db_engine, medium_risk_features):
        """A real borderline student must land in the Medium band, not be forced to an extreme."""
        from ml.config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW

        payload = client.post(PREDICT_URL, json=body("API_MED", medium_risk_features)).json()
        assert payload["risk_tier"] == "Medium", payload
        assert RISK_THRESHOLD_LOW <= payload["calibrated_risk_probability"] <= RISK_THRESHOLD_HIGH

    def test_tiers_are_ordered_low_medium_high(
        self, client, db_engine, low_risk_features, medium_risk_features, high_risk_features
    ):
        """Probabilities must increase monotonically across the three archetypes."""
        low = client.post(PREDICT_URL, json=body("API_ORD_L", low_risk_features)).json()
        medium = client.post(PREDICT_URL, json=body("API_ORD_M", medium_risk_features)).json()
        high = client.post(PREDICT_URL, json=body("API_ORD_H", high_risk_features)).json()
        assert (
            low["calibrated_risk_probability"]
            < medium["calibrated_risk_probability"]
            < high["calibrated_risk_probability"]
        )

    def test_probability_bounds_and_percentage_agree(self, client, db_engine, high_risk_features):
        payload = client.post(PREDICT_URL, json=body("API_BOUNDS", high_risk_features)).json()
        probability = payload["calibrated_risk_probability"]
        assert 0.0 <= probability <= 1.0
        assert payload["risk_score_percentage"] == pytest.approx(probability * 100, abs=0.01)

    def test_tier_matches_ml_core_thresholds(self, client, db_engine, high_risk_features):
        """The API must not fork Low/Medium/High from ml.config.get_risk_tier."""
        from ml.config import get_risk_tier

        payload = client.post(PREDICT_URL, json=body("API_TIER", high_risk_features)).json()
        assert payload["risk_tier"] == get_risk_tier(payload["calibrated_risk_probability"])

    def test_response_contains_required_fields(self, client, db_engine, high_risk_features):
        payload = client.post(PREDICT_URL, json=body("API_FIELDS", high_risk_features)).json()
        for field in (
            "student_id",
            "calibrated_risk_probability",
            "risk_tier",
            "risk_score_percentage",
            "evaluation_timestamp",
        ):
            assert field in payload

    def test_confidence_interval_is_absent(self, client, db_engine, high_risk_features):
        """
        No confidence-interval estimator exists in ml/, so the field is deliberately
        omitted rather than fabricated. See schemas/prediction.py.
        """
        payload = client.post(PREDICT_URL, json=body("API_CI", high_risk_features)).json()
        assert "confidence_interval" not in payload
        assert "confidence" not in payload

    def test_explanation_included_on_request(self, client, db_engine, high_risk_features):
        payload = client.post(
            PREDICT_URL, json=body("API_EXPL", high_risk_features, include_explanation=True)
        ).json()
        assert payload["top_drivers"] and len(payload["top_drivers"]) == 5

    def test_explanation_omitted_by_default(self, client, db_engine, high_risk_features):
        payload = client.post(PREDICT_URL, json=body("API_NOEXPL", high_risk_features)).json()
        assert payload["top_drivers"] is None


@requires_db
class TestPersistence:
    def test_student_and_prediction_are_persisted(self, client, db_engine, high_risk_features):
        from sqlalchemy.orm import Session

        client.post(PREDICT_URL, json=body("API_PERSIST", high_risk_features))
        with Session(db_engine) as session:
            student = session.execute(
                select(Student).where(Student.student_id == "API_PERSIST")
            ).scalar_one()
            count = session.execute(
                select(func.count()).select_from(Prediction).where(Prediction.student_id == student.id)
            ).scalar()
        assert count == 1

    def test_history_is_appended_not_overwritten(self, client, db_engine, high_risk_features, low_risk_features):
        """Re-scoring must add a row and leave the earlier one untouched."""
        from sqlalchemy.orm import Session

        client.post(PREDICT_URL, json=body("API_HIST", high_risk_features))
        client.post(PREDICT_URL, json=body("API_HIST", low_risk_features))

        with Session(db_engine) as session:
            student = session.execute(
                select(Student).where(Student.student_id == "API_HIST")
            ).scalar_one()
            rows = session.execute(
                select(Prediction)
                .where(Prediction.student_id == student.id)
                .order_by(Prediction.evaluated_at)
            ).scalars().all()

        assert len(rows) == 2
        assert rows[0].risk_tier == "High"   # original preserved
        assert rows[1].risk_tier == "Low"    # newer appended
        assert rows[0].evaluated_at < rows[1].evaluated_at

    def test_snapshot_has_37_features_in_canonical_order(self, client, db_engine, high_risk_features):
        from sqlalchemy.orm import Session

        from backend.app.services.ml_service import ml_service

        client.post(PREDICT_URL, json=body("API_SNAP", high_risk_features))
        with Session(db_engine) as session:
            student = session.execute(
                select(Student).where(Student.student_id == "API_SNAP")
            ).scalar_one()
            prediction = session.execute(
                select(Prediction).where(Prediction.student_id == student.id)
            ).scalars().first()

        # Postgres JSONB normalises key order, so the stored dict's iteration order is
        # NOT the canonical feature order. Only membership survives the round trip --
        # which is exactly why explain_from_snapshot reindexes by feature_names instead
        # of trusting dict order.
        assert set(prediction.input_features) == set(ml_service.feature_names)
        assert len(prediction.input_features) == 37

        # Reconstructing the canonical order from the snapshot must still work.
        frame = ml_service.build_feature_frame_batch([high_risk_features])
        rebuilt = pd.DataFrame([prediction.input_features]).reindex(
            columns=ml_service.feature_names
        )
        assert list(rebuilt.columns) == ml_service.feature_names
        for column in ml_service.feature_names:
            assert rebuilt[column].iloc[0] == pytest.approx(frame[column].iloc[0])

    def test_top_drivers_persisted_for_reproducibility(self, client, db_engine, high_risk_features):
        from sqlalchemy.orm import Session

        client.post(PREDICT_URL, json=body("API_DRIVERS", high_risk_features))
        with Session(db_engine) as session:
            student = session.execute(
                select(Student).where(Student.student_id == "API_DRIVERS")
            ).scalar_one()
            prediction = session.execute(
                select(Prediction).where(Prediction.student_id == student.id)
            ).scalars().first()

        assert prediction.top_drivers
        assert prediction.top_drivers[0]["impact_direction"] in (
            "RISK_INCREASING",
            "RISK_DECREASING",
        )

    def test_absent_top_drivers_are_sql_null_not_json_null(self, client, db_engine, high_risk_features):
        """
        SQLAlchemy maps Python None on a JSONB column to the JSON scalar `null` unless
        none_as_null=True. That made `WHERE top_drivers IS NULL` match nothing and broke
        jsonb_array_length(). Batch without explanations is the path that writes None.
        """
        from sqlalchemy import text

        client.post(
            BATCH_URL,
            json={"students": [body("API_JSONNULL", high_risk_features)],
                  "include_explanations": False},
        )
        with db_engine.connect() as connection:
            json_nulls = connection.execute(
                text("SELECT count(*) FROM predictions WHERE top_drivers = 'null'::jsonb")
            ).scalar()
            typeofs = [
                row[0]
                for row in connection.execute(
                    text("SELECT DISTINCT jsonb_typeof(top_drivers) FROM predictions")
                )
            ]
        assert json_nulls == 0, "Python None must persist as SQL NULL, not JSON 'null'"
        assert "null" not in typeofs

    def test_stored_features_exclude_labels_and_engineered(self, client, db_engine, high_risk_features):
        from sqlalchemy.orm import Session

        client.post(PREDICT_URL, json=body("API_STORED", high_risk_features))
        with Session(db_engine) as session:
            student = session.execute(
                select(Student).where(Student.student_id == "API_STORED")
            ).scalar_one()

        assert "is_dropout" not in student.features
        assert "ground_truth_risk_prob" not in student.features
        for engineered in ENGINEERED_FEATURE_COLUMNS:
            if engineered != "is_hosteler":
                assert engineered not in student.features
        for raw in RAW_FEATURE_COLUMNS:
            assert raw in student.features

    def test_metadata_stored_but_not_scored(self, client, db_engine, high_risk_features):
        from sqlalchemy.orm import Session

        client.post(
            PREDICT_URL,
            json=body(
                "API_META",
                high_risk_features,
                name="Test Student",
                department="Computer Science & Engineering",
                assigned_mentor_id="FAC_001",
            ),
        )
        with Session(db_engine) as session:
            student = session.execute(
                select(Student).where(Student.student_id == "API_META")
            ).scalar_one()

        assert student.name == "Test Student"
        assert student.department == "Computer Science & Engineering"
        assert "name" not in student.features
        assert "department" not in student.features


# ----------------------------------------------------------------------------- batch

@requires_db
class TestBatchPrediction:
    def test_counts_and_total(self, client, db_engine, high_risk_features, low_risk_features):
        payload = {
            "students": [
                body("B_H1", high_risk_features),
                body("B_H2", high_risk_features),
                body("B_L1", low_risk_features),
            ]
        }
        response = client.post(BATCH_URL, json=payload)
        assert response.status_code == 201

        result = response.json()
        assert result["total_students_evaluated"] == 3
        assert result["high_risk_count"] == 2
        assert result["low_risk_count"] == 1
        assert (
            result["high_risk_count"] + result["medium_risk_count"] + result["low_risk_count"]
            == result["total_students_evaluated"]
        )

    def test_preserves_input_order_by_default(self, client, db_engine, high_risk_features, low_risk_features):
        payload = {
            "students": [
                body("B_ORDER_L", low_risk_features),
                body("B_ORDER_H", high_risk_features),
            ]
        }
        results = client.post(BATCH_URL, json=payload).json()["results"]
        assert [r["student_id"] for r in results] == ["B_ORDER_L", "B_ORDER_H"]

    def test_sort_by_risk_desc(self, client, db_engine, high_risk_features, low_risk_features):
        payload = {
            "students": [
                body("B_SORT_L", low_risk_features),
                body("B_SORT_H", high_risk_features),
            ],
            "sort_by_risk_desc": True,
        }
        results = client.post(BATCH_URL, json=payload).json()["results"]
        probabilities = [r["calibrated_risk_probability"] for r in results]
        assert probabilities == sorted(probabilities, reverse=True)
        assert results[0]["student_id"] == "B_SORT_H"

    def test_batch_matches_single_prediction_exactly(self, client, db_engine, high_risk_features):
        """Vectorised inference must produce identical numbers to the single-row path."""
        single = client.post(PREDICT_URL, json=body("B_EQ_SINGLE", high_risk_features)).json()
        batch = client.post(
            BATCH_URL, json={"students": [body("B_EQ_BATCH", high_risk_features)]}
        ).json()["results"][0]
        assert batch["calibrated_risk_probability"] == single["calibrated_risk_probability"]
        assert batch["risk_tier"] == single["risk_tier"]

    def test_all_batch_rows_persisted(self, client, db_engine, high_risk_features):
        from sqlalchemy.orm import Session

        payload = {"students": [body(f"B_PERSIST_{i}", high_risk_features) for i in range(4)]}
        client.post(BATCH_URL, json=payload)
        with Session(db_engine) as session:
            found = session.execute(
                select(func.count()).select_from(Student).where(
                    Student.student_id.like("B_PERSIST_%")
                )
            ).scalar()
        assert found == 4

    def test_duplicate_ids_rejected(self, client, db_engine, high_risk_features):
        payload = {"students": [body("B_DUP", high_risk_features), body("B_DUP", high_risk_features)]}
        response = client.post(BATCH_URL, json=payload)
        assert response.status_code == 422
        assert "Duplicate" in response.json()["message"]

    def test_empty_batch_rejected(self, client):
        assert client.post(BATCH_URL, json={"students": []}).status_code == 422

    def test_one_invalid_row_fails_whole_batch(self, client, db_engine, high_risk_features):
        """A partially-applied batch would leave inconsistent history."""
        bad = dict(high_risk_features, current_cgpa=99.0)
        payload = {"students": [body("B_OK", high_risk_features), body("B_BAD", bad)]}
        assert client.post(BATCH_URL, json=payload).status_code == 422

    def test_explanations_opt_in(self, client, db_engine, high_risk_features):
        payload = {"students": [body("B_EXPL", high_risk_features)], "include_explanations": True}
        result = client.post(BATCH_URL, json=payload).json()["results"][0]
        assert result["top_drivers"] and len(result["top_drivers"]) == 5


# ------------------------------------------------------------------------- CSV batch

@requires_db
class TestCsvBatch:
    def _csv(self, rows):
        import pandas as pd

        buffer = io.StringIO()
        pd.DataFrame(rows).to_csv(buffer, index=False)
        return buffer.getvalue().encode()

    def test_csv_upload_scores_rows(self, client, db_engine, high_risk_features, low_risk_features):
        rows = [
            dict(high_risk_features, student_id="CSV_H"),
            dict(low_risk_features, student_id="CSV_L"),
        ]
        response = client.post(
            CSV_URL, files={"file": ("cohort.csv", self._csv(rows), "text/csv")}
        )
        assert response.status_code == 201
        result = response.json()
        assert result["total_students_evaluated"] == 2
        assert result["high_risk_count"] == 1
        assert result["low_risk_count"] == 1

    def test_csv_ignores_labels_and_engineered_columns(self, client, db_engine, high_risk_features):
        """A genuine features.csv export carries labels; they must be dropped, not fatal."""
        rows = [
            dict(
                high_risk_features,
                student_id="CSV_EXTRA",
                is_dropout=1,
                ground_truth_risk_prob=0.97,
                gender="Female",
                category="SC",
                family_income_slab="<2 LPA",
                financial_stress_index=0.9,
                interaction_att_x_fee=77.0,
            )
        ]
        response = client.post(
            CSV_URL, files={"file": ("export.csv", self._csv(rows), "text/csv")}
        )
        assert response.status_code == 201
        assert response.json()["total_students_evaluated"] == 1

    def test_csv_missing_student_id_rejected(self, client, db_engine, high_risk_features):
        response = client.post(
            CSV_URL, files={"file": ("bad.csv", self._csv([high_risk_features]), "text/csv")}
        )
        assert response.status_code == 422
        assert "student_id" in response.json()["message"]

    def test_non_csv_rejected(self, client):
        response = client.post(
            CSV_URL, files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 422

    def test_real_cohort_slice_scores(self, client, db_engine):
        """
        End-to-end against the project's own features.csv, including its label and
        protected-attribute columns.
        """
        import pandas as pd

        from ml.config import PROCESSED_DATA_PATH

        if not PROCESSED_DATA_PATH.exists():
            pytest.skip("features.csv not generated")

        frame = pd.read_csv(PROCESSED_DATA_PATH).head(5).copy()
        frame["student_id"] = [f"CSV_REAL_{i}" for i in range(len(frame))]
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)

        response = client.post(
            CSV_URL, files={"file": ("features.csv", buffer.getvalue().encode(), "text/csv")}
        )
        assert response.status_code == 201, response.text
        assert response.json()["total_students_evaluated"] == 5
