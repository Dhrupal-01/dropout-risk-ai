"""
Access control, first stage: every write needs `Authorization: Bearer <API_ADMIN_TOKEN>`; reads
are public only while PUBLIC_READ_ONLY is true; production refuses to start without a token.
The shared `client` fixture sends the test token; these tests use a client without it.
"""

import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.core.config import Settings, get_settings
from backend.tests.conftest import TEST_ADMIN_TOKEN, requires_db
from backend.tests.db_guard import PROJECT_ROOT

WRITES = [
    ("/api/v1/predict", {"json": {}}),
    ("/api/v1/predict/batch", {"json": {}}),
    ("/api/v1/predict/batch/csv", {"files": {"file": ("cohort.csv", b"student_id\n", "text/csv")}}),
    ("/api/v1/interventions/log", {"json": {}}),
]
READ = "/api/v1/interventions/catalog"
FAKE_DB = "postgresql+psycopg://u:p@db.invalid/x"


@pytest.fixture(scope="module")
def anon(client):
    """A client that sends no Authorization header. It reuses the app the shared `client` already
    started; it does not run its own lifespan, whose shutdown would unload the shared model."""
    from backend.app.main import app

    return TestClient(app)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


class TestWritesNeedTheAdminToken:
    @pytest.mark.parametrize("url, kwargs", WRITES)
    def test_missing_token_is_401(self, anon, url, kwargs):
        response = anon.post(url, **kwargs)
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorized"
        assert response.headers["www-authenticate"] == "Bearer"

    @pytest.mark.parametrize("url, kwargs", WRITES)
    def test_wrong_token_is_401(self, anon, url, kwargs):
        response = anon.post(url, headers=bearer(TEST_ADMIN_TOKEN + "x"), **kwargs)
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorized"

    @pytest.mark.parametrize("url, kwargs", WRITES)
    def test_right_token_passes_auth(self, anon, url, kwargs):
        """These bodies are invalid on purpose: with the right token the request reaches validation."""
        response = anon.post(url, headers=bearer(TEST_ADMIN_TOKEN), **kwargs)
        assert response.status_code == 422, response.text[:300]

    def test_non_bearer_scheme_is_401(self, anon):
        response = anon.post("/api/v1/predict", headers={"Authorization": f"Basic {TEST_ADMIN_TOKEN}"}, json={})
        assert response.status_code == 401

    def test_comparison_is_constant_time(self, anon, monkeypatch):
        import backend.app.core.security as security

        calls = []
        real = security.hmac.compare_digest
        monkeypatch.setattr(security.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
        anon.post("/api/v1/predict", headers=bearer("guess"), json={})
        assert calls, "token comparison must go through hmac.compare_digest"

    def test_server_without_a_token_refuses_writes(self, anon, monkeypatch):
        monkeypatch.setattr(get_settings(), "API_ADMIN_TOKEN", None)
        response = anon.post("/api/v1/predict", headers=bearer(TEST_ADMIN_TOKEN), json={})
        assert response.status_code == 401
        assert "not configured" in response.json()["message"]


@requires_db
class TestRightTokenWritesForReal:
    def test_scoring_with_the_token_is_201(self, anon, db_engine, sample_raw_features):
        body = {"student_id": "AUTH_OK", "features": dict(sample_raw_features)}
        assert anon.post("/api/v1/predict", headers=bearer(TEST_ADMIN_TOKEN), json=body).status_code == 201
        assert anon.post("/api/v1/predict", json=body).status_code == 401


class TestReads:
    def test_reads_are_public_while_public_read_only(self, anon):
        assert get_settings().PUBLIC_READ_ONLY is True
        assert anon.get(READ).status_code == 200

    def test_reads_need_the_token_when_not_public(self, anon, monkeypatch):
        monkeypatch.setattr(get_settings(), "PUBLIC_READ_ONLY", False)
        response = anon.get(READ)
        assert response.status_code == 401 and response.json()["error"] == "unauthorized"
        assert anon.get(READ, headers=bearer(TEST_ADMIN_TOKEN + "x")).status_code == 401
        assert anon.get(READ, headers=bearer(TEST_ADMIN_TOKEN)).status_code == 200

    def test_health_stays_public(self, anon, monkeypatch):
        monkeypatch.setattr(get_settings(), "PUBLIC_READ_ONLY", False)
        assert anon.get("/health").status_code == 200


class TestProductionNeedsAToken:
    def test_settings_reject_production_without_token(self):
        with pytest.raises(ValidationError, match="requires API_ADMIN_TOKEN"):
            Settings(_env_file=None, DATABASE_URL=FAKE_DB, ENVIRONMENT="production", API_ADMIN_TOKEN="  ")

    def test_settings_accept_production_with_token(self):
        settings = Settings(_env_file=None, DATABASE_URL=FAKE_DB, ENVIRONMENT="production", API_ADMIN_TOKEN="t0ken")
        assert settings.API_ADMIN_TOKEN.get_secret_value() == "t0ken"
        assert "t0ken" not in repr(settings)

    def test_production_server_without_token_fails_at_startup(self):
        env = {**os.environ, "DATABASE_URL": FAKE_DB, "ENVIRONMENT": "production", "API_ADMIN_TOKEN": ""}
        res = subprocess.run(
            [sys.executable, "-c", "import backend.app.main"], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True
        )
        assert res.returncode != 0
        assert "requires API_ADMIN_TOKEN" in res.stderr, res.stderr[-1000:]
