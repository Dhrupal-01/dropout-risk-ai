"""
Verification suite fixtures and helpers.
"""

import os
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

# Verification audits the real repository artifacts, never the per-session temp copies that
# backend/tests and ml/tests build (their conftests redirect ml.config via these env vars).
REAL_ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
REAL_PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
ARTIFACT_PATH_ENV_VARS = ("DROPOUTGUARD_ARTIFACTS_DIR", "DROPOUTGUARD_PROCESSED_DATA_PATH")

# Bind ml.config to the real paths now, unless something already redirected them (a mixed
# backend+verification session or a shell export); real_ml_artifacts then fails on use. The
# environment is not modified, so subprocesses (V1.1, V1.3) still redirect to their own temp dirs.
PRESET_ARTIFACT_ENV = {k: os.environ[k] for k in ARTIFACT_PATH_ENV_VARS if k in os.environ}
if not PRESET_ARTIFACT_ENV:
    import ml.config  # noqa: E402,F401


def _psycopg3(url: str | None) -> str | None:
    if not url:
        return url
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


TEST_DATABASE_URL = _psycopg3(os.environ.get("TEST_DATABASE_URL"))

# Must precede any `backend.app.*` import: db.session builds its engine from DATABASE_URL at import
# time. The app under verification is pointed at the TEST database explicitly (or at a dummy URL
# when none is set), never at the application database loaded from .env.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL or "postgresql+psycopg://test:test@localhost:5432/test"
os.environ.pop("MIGRATION_DATABASE_URL", None)


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
def real_ml_artifacts() -> Path:
    """Fails unless ml.config points at the repository's ml/artifacts and data/processed."""
    import ml.config

    actual = (Path(ml.config.ARTIFACTS_DIR).resolve(), Path(ml.config.PROCESSED_DATA_PATH).resolve())
    expected = (REAL_ARTIFACTS_DIR.resolve(), REAL_PROCESSED_DATA_PATH.resolve())
    if actual != expected:
        pytest.fail(
            "Verification audits the real ml/artifacts, but ml.config points at "
            f"ARTIFACTS_DIR={actual[0]}, PROCESSED_DATA_PATH={actual[1]} "
            f"(preset env: {sorted(PRESET_ARTIFACT_ENV)}). Run verification in its own pytest session."
        )
    return REAL_ARTIFACTS_DIR


@pytest.fixture(scope="session")
def db_engine(live_db_url, real_ml_artifacts):
    from sqlalchemy import create_engine
    from alembic import command
    from backend.tests.db_helpers import _reset_schema, alembic_config_for

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
    from backend.app.db.session import engine
    from backend.app.main import app
    from backend.tests.db_guard import database_identity

    app_db = database_identity(engine.url.render_as_string(hide_password=False))
    if app_db != database_identity(TEST_DATABASE_URL):
        pytest.exit(
            f"Refusing to run: the app under verification is bound to {app_db[:2]}, not TEST_DATABASE_URL.",
            returncode=pytest.ExitCode.USAGE_ERROR,
        )

    with TestClient(app) as test_client:
        yield test_client
