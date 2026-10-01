"""
Verification suite fixtures and helpers.
"""

import os
import subprocess
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "9d6afba"

for env_file in (PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env"):
    if env_file.exists():
        load_dotenv(env_file, override=False)

# Abort before any engine exists if TEST_DATABASE_URL is the application database.
from backend.tests.db_guard import abort_if_test_db_is_app_db  # noqa: E402

abort_if_test_db_is_app_db()


def _psycopg3(url: str | None) -> str | None:
    if not url:
        return url
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


TEST_DATABASE_URL = _psycopg3(os.environ.get("TEST_DATABASE_URL"))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def baseline_commit() -> str:
    return BASELINE_COMMIT


@pytest.fixture(scope="session")
def live_db_url() -> str:
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set; skipping live DB test")
    return TEST_DATABASE_URL


@pytest.fixture(scope="session")
def db_engine(live_db_url):
    from sqlalchemy import create_engine
    from alembic.config import Config
    from alembic import command
    from backend.tests.conftest import _reset_schema, alembic_config_for

    engine = create_engine(live_db_url, pool_pre_ping=True)
    config = alembic_config_for(live_db_url)
    _reset_schema(engine)
    command.upgrade(config, "head")
    try:
        yield engine
    finally:
        _reset_schema(engine)
        engine.dispose()


@pytest.fixture
def client(db_engine):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client
