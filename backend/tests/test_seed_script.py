"""
Seed/import script.

Covers the parts that need no database — payload construction, label exclusion and
deterministic display metadata — plus a live end-to-end import when TEST_DATABASE_URL is set.
"""

import pandas as pd
import pytest

from backend.app.services.ml_service import (
    ENGINEERED_FEATURE_COLUMNS,
    LABEL_COLUMNS,
    RAW_FEATURE_COLUMNS,
)
from backend.scripts.seed_students import (
    STORED_FEATURE_COLUMNS,
    build_feature_payload,
    make_display_metadata,
    seed,
)
from backend.tests.conftest import requires_db
from ml.config import PROCESSED_DATA_PATH

pytestmark = pytest.mark.skipif(
    not PROCESSED_DATA_PATH.exists(),
    reason="features.csv absent; run `python -m ml.data_pipeline.feature_engineering`",
)


@pytest.fixture(scope="module")
def cohort():
    return pd.read_csv(PROCESSED_DATA_PATH)


class TestDatasetAssumptions:
    def test_dataset_has_every_raw_column(self, cohort):
        missing = [c for c in RAW_FEATURE_COLUMNS if c not in cohort.columns]
        assert not missing, f"features.csv is missing: {missing}"

    def test_dataset_uses_the_real_column_names(self, cohort):
        assert "student_id" in cohort.columns
        assert "hostel_status" in cohort.columns

    def test_dataset_has_no_name_or_department(self, cohort):
        """Confirms display metadata genuinely has to be synthesised."""
        assert "name" not in cohort.columns
        assert "department" not in cohort.columns

    def test_labels_are_present_in_the_csv(self, cohort):
        """They exist in the file, which is exactly why the script must exclude them."""
        assert LABEL_COLUMNS.issubset(cohort.columns)


class TestPayloadConstruction:
    def test_stored_columns_are_raw_plus_residency(self):
        assert set(STORED_FEATURE_COLUMNS) == set(RAW_FEATURE_COLUMNS) | {"hostel_status"}

    def test_payload_excludes_labels(self, cohort):
        payload = build_feature_payload(cohort.iloc[0])
        assert not LABEL_COLUMNS.intersection(payload)

    def test_payload_excludes_protected_attributes(self, cohort):
        payload = build_feature_payload(cohort.iloc[0])
        for column in ("gender", "category", "family_income_slab"):
            assert column not in payload

    def test_payload_keeps_the_ordinal_income_encoding(self, cohort):
        """income_slab_idx IS a model feature; only the string slab is excluded."""
        assert "income_slab_idx" in build_feature_payload(cohort.iloc[0])

    def test_payload_excludes_engineered_columns(self, cohort):
        """Engineered values live in the CSV but must be recomputed, not stored."""
        payload = build_feature_payload(cohort.iloc[0])
        for column in ENGINEERED_FEATURE_COLUMNS:
            assert column not in payload

    def test_payload_is_json_serialisable(self, cohort):
        import json

        json.dumps(build_feature_payload(cohort.iloc[0]))

    def test_payload_feeds_the_model_end_to_end(self, cohort, ml):
        """The stored representation must be sufficient to score the student."""
        payload = build_feature_payload(cohort.iloc[0])
        frame = ml.build_feature_frame(payload)
        assert list(frame.columns) == ml.feature_names

    def test_scores_match_the_csv_derived_features(self, cohort, ml):
        """
        Recomputing engineered values from raw inputs must reproduce the CSV's own values,
        proving nothing is lost by not storing them.
        """
        row = cohort.iloc[0]
        frame = ml.build_feature_frame(build_feature_payload(row))
        for column in ENGINEERED_FEATURE_COLUMNS:
            assert frame[column].iloc[0] == pytest.approx(row[column], abs=1e-6), column

    def test_missing_column_raises_clearly(self, cohort):
        row = cohort.iloc[0].drop(labels=["current_cgpa"])
        with pytest.raises(ValueError, match="current_cgpa"):
            build_feature_payload(row)


class TestDisplayMetadata:
    def test_is_deterministic(self):
        assert make_display_metadata("IND_2026_0001") == make_display_metadata("IND_2026_0001")

    def test_differs_across_students(self):
        generated = {make_display_metadata(f"IND_2026_{i:04d}")["name"] for i in range(1, 60)}
        assert len(generated) > 5

    def test_has_the_expected_fields(self):
        meta = make_display_metadata("IND_2026_0001")
        assert set(meta) == {"name", "department", "assigned_mentor_id"}
        assert meta["assigned_mentor_id"].startswith("FAC_")


@requires_db
class TestLiveSeed:
    """Seeds into the TEST database via an injected session factory — never the app database."""

    def test_seed_imports_and_is_idempotent(self, db_engine, db_session_factory):
        from sqlalchemy import func, select
        from sqlalchemy.orm import Session

        from backend.app.models.student import Student

        # Count only the seeded cohort: API tests share this database and create their
        # own students, so a global COUNT(*) would be order-dependent.
        cohort = Student.student_id.like("IND_2026_%")

        assert seed(limit=25, session_factory=db_session_factory) == 25
        with Session(db_engine) as session:
            assert session.execute(
                select(func.count()).select_from(Student).where(cohort)
            ).scalar() == 25

        seed(limit=25, session_factory=db_session_factory)  # re-run must upsert, not duplicate
        with Session(db_engine) as session:
            assert session.execute(
                select(func.count()).select_from(Student).where(cohort)
            ).scalar() == 25

    def test_seed_with_predictions(self, db_engine, db_session_factory):
        from sqlalchemy import func, select
        from sqlalchemy.orm import Session

        from backend.app.models.prediction import Prediction

        seed(limit=10, with_predictions=True, session_factory=db_session_factory)
        with Session(db_engine) as session:
            assert session.execute(select(func.count()).select_from(Prediction)).scalar() >= 10

    def test_predictions_are_distinctly_ordered(self, db_engine, db_session_factory):
        """
        Regression guard: predictions written inside one transaction must still have
        strictly increasing evaluated_at, so 'latest prediction' is unambiguous. This is
        why evaluated_at defaults to clock_timestamp() rather than now().
        """
        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from backend.app.models.prediction import Prediction

        seed(limit=5, with_predictions=True, session_factory=db_session_factory)
        with Session(db_engine) as session:
            stamps = session.execute(select(Prediction.evaluated_at)).scalars().all()
        assert len(set(stamps)) == len(stamps), "evaluated_at collided within a transaction"
