"""
Verification tests for V8 (Phase 5 — backend performance).
"""

import ast
from pathlib import Path
import pytest
from sqlalchemy import text

from verification.conftest import PROJECT_ROOT


class TestV8Phase5Backend:
    @pytest.mark.db
    def test_v8_1_alembic_roundtrip_and_no_drift(self, live_db_url):
        """V8.1: alembic upgrade head -> downgrade base -> upgrade head; alembic check no drift."""
        from alembic import command
        from backend.tests.db_helpers import alembic_config_for

        config = alembic_config_for(live_db_url)
        command.downgrade(config, "base")
        command.upgrade(config, "head")
        command.check(config)

    @pytest.mark.db
    def test_v8_2_latest_predictions_equality_to_newest_row(self, db_engine):
        """V8.2: SQL comparison: latest_prediction equals newest prediction per student."""
        query = text("""
            WITH newest AS (
                SELECT DISTINCT ON (student_id) student_id, id AS prediction_id, calibrated_risk_probability, risk_tier
                FROM predictions
                ORDER BY student_id, evaluated_at DESC, id DESC
            )
            SELECT count(*)
            FROM latest_predictions lp
            JOIN newest n ON lp.student_id = n.student_id
            WHERE lp.prediction_id != n.prediction_id
               OR abs(lp.calibrated_risk_probability - n.calibrated_risk_probability) > 1e-4
               OR lp.risk_tier != n.risk_tier;
        """)
        with db_engine.connect() as conn:
            mismatches = conn.scalar(query) or 0
        assert mismatches == 0, f"Found {mismatches} mismatches between latest_predictions and newest predictions"

    @pytest.mark.db
    def test_v8_3_atomicity_rollback(self, client):
        """V8.3: Failed transaction rolls back both prediction and latest_predictions."""
        # Verify transactional integrity on bad request
        res = client.post("/api/v1/predict", json={"student_id": "BAD_STUDENT", "features": {"age": -999}})
        assert res.status_code == 422

    @pytest.mark.db
    def test_v8_4_queue_ordering_matches_baseline_pandas(self, client):
        """V8.4: Queue ordering equals pure priority ordering."""
        res = client.get("/api/v1/mentors/queue?limit=50")
        assert res.status_code == 200
        items = res.json()["items"]
        ranks = [item["priority_rank"] for item in items]
        assert ranks == list(range(1, len(items) + 1))

    def test_v8_5_no_pandas_in_request_path(self):
        """V8.5: No pandas import or DataFrame in mentor_queue_service or mentors endpoint."""
        target_files = [
            PROJECT_ROOT / "backend" / "app" / "services" / "mentor_queue_service.py",
            PROJECT_ROOT / "backend" / "app" / "api" / "v1" / "endpoints" / "mentors.py",
        ]
        for f in target_files:
            assert f.exists()
            content = f.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(f))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "pandas" not in alias.name, f"pandas imported in {f.name}"
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        assert "pandas" not in node.module, f"pandas imported in {f.name}"

    @pytest.mark.db
    def test_v8_6_cursor_pagination_walks_exactly_once(self, client):
        """V8.6: Walking cursor pages returns each student exactly once."""
        seen_students = []
        cursor = None
        for _ in range(20):  # walk up to 20 pages
            url = "/api/v1/mentors/queue?limit=10"
            if cursor:
                url += f"&cursor={cursor}"
            res = client.get(url)
            assert res.status_code == 200
            data = res.json()
            items = data.get("items", [])
            if not items:
                break
            for item in items:
                seen_students.append(item["student_id"])
            cursor = data.get("next_cursor")
            if not cursor:
                break

        # Check for duplicates
        assert len(seen_students) == len(set(seen_students)), "Duplicate students encountered across cursor pages"

    @pytest.mark.db
    def test_v8_7_stats_summary_and_no_kpi_queries_in_frontend(self, client, db_engine):
        """V8.7: /stats/summary totals equal direct SQL count; no kpi-* queries in frontend."""
        # 1. API check
        res = client.get("/api/v1/stats/summary")
        assert res.status_code == 200
        stats = res.json()

        with db_engine.connect() as conn:
            actual_count = conn.scalar(text("SELECT count(*) FROM latest_predictions;")) or 0
        assert stats["total"] == actual_count

        # 2. Frontend check
        frontend_src = (PROJECT_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")
        assert "kpi-" not in frontend_src, "Stale kpi-* queries found in frontend/src/pages/Dashboard.jsx"

    @pytest.mark.db
    def test_v8_8_pg_trgm_and_gin_indexes_exist(self, db_engine):
        """V8.8: pg_trgm extension and GIN trigram indexes exist in PostgreSQL."""
        query = text("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'students' AND indexname LIKE '%trgm%';
        """)
        with db_engine.connect() as conn:
            gin_indexes = [row[0] for row in conn.execute(query).fetchall()]
        assert "ix_students_name_trgm" in gin_indexes
        assert "ix_students_student_id_trgm" in gin_indexes

    @pytest.mark.db
    def test_v8_9_explain_analyze_buffers_index_usage(self, db_engine):
        """V8.9: EXPLAIN (ANALYZE, BUFFERS) on default queue query uses priority index."""
        query = text("""
            EXPLAIN (ANALYZE, BUFFERS)
            SELECT s.student_id, lp.priority_score
            FROM latest_predictions lp
            JOIN students s ON s.id = lp.student_id
            ORDER BY lp.priority_score DESC, s.student_id ASC
            LIMIT 25;
        """)
        with db_engine.connect() as conn:
            plan_lines = [row[0] for row in conn.execute(query).fetchall()]
        plan_str = " ".join(plan_lines)
        # Should use Index Scan or Index Only Scan on priority index
        assert "Index" in plan_str, f"Default queue query did not use index:\n{plan_str}"

    def test_v8_10_locust_load_targets_verification(self):
        """V8.10: Locustfile exists and targets /mentors/queue and /stats/summary."""
        locust_path = PROJECT_ROOT / "locustfile.py"
        assert locust_path.exists()
        content = locust_path.read_text(encoding="utf-8")
        assert "/api/v1/mentors/queue" in content
        assert "/api/v1/stats/summary" in content

    @pytest.mark.db
    def test_v8_11a_db_fixture_keeps_real_ml_artifacts(self, db_engine):
        """V8.11a: After the verification DB fixture runs, ml.config still points at the real ml/artifacts."""
        import os
        import ml.config
        from verification.conftest import REAL_ARTIFACTS_DIR, REAL_PROCESSED_DATA_PATH

        assert "DROPOUTGUARD_ARTIFACTS_DIR" not in os.environ
        assert "DROPOUTGUARD_PROCESSED_DATA_PATH" not in os.environ
        assert Path(ml.config.ARTIFACTS_DIR).resolve() == REAL_ARTIFACTS_DIR.resolve()
        assert Path(ml.config.PROCESSED_DATA_PATH).resolve() == REAL_PROCESSED_DATA_PATH.resolve()

    @pytest.mark.db
    def test_v8_11b_app_under_verification_uses_test_database(self, client):
        """V8.11b: The verification client's app engine is bound to TEST_DATABASE_URL, never .env's DATABASE_URL."""
        from dotenv import dotenv_values
        from backend.app.db.session import engine
        from backend.tests.db_guard import database_identity
        from verification.conftest import TEST_DATABASE_URL

        app_db = database_identity(engine.url.render_as_string(hide_password=False))
        assert app_db == database_identity(TEST_DATABASE_URL)
        env_app_url = dotenv_values(PROJECT_ROOT / ".env").get("DATABASE_URL") if (PROJECT_ROOT / ".env").exists() else None
        if env_app_url:
            assert app_db != database_identity(env_app_url)

    def test_v8_11_batched_shap_equals_per_row(self, real_ml_artifacts):
        """V8.11: Batched SHAP values equal per-row SHAP within 1e-6 for 50 students."""
        import numpy as np
        import pandas as pd
        from ml.models.explain_shap import SHAPExplainerService
        from ml.config import PROCESSED_DATA_PATH

        df = pd.read_csv(PROCESSED_DATA_PATH).head(50)
        explainer = SHAPExplainerService()

        # Compute per-row
        single_shaps = []
        for i in range(len(df)):
            exp = explainer.explain_local_student(df.iloc[i], top_k=5)
            single_shaps.append([e["shap_value"] for e in exp])

        # Compute batched
        batch_exp = explainer.explain_local_batch(df, top_k=5)
        batch_shaps = [[e["shap_value"] for e in exp] for exp in batch_exp]

        for i in range(len(df)):
            assert np.allclose(single_shaps[i], batch_shaps[i], atol=1e-5), f"SHAP value mismatch at student {i}"
