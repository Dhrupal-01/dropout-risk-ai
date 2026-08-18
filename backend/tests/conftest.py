"""
Shared backend test fixtures.

DATABASE_URL is set before any backend module is imported, because
`backend.app.db.session` builds its engine at import time.

Tests that need a live database are skipped unless TEST_DATABASE_URL is set, so the suite
stays runnable without network access to Neon.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load the same .env files the app uses, so TEST_DATABASE_URL does not have to be exported
# by hand. Real environment variables always win.
for env_file in (PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env"):
    if env_file.exists():
        load_dotenv(env_file, override=False)

# Must precede any `backend.app.*` import: db.session builds its engine at import time.
# The application under test is pointed at the TEST database, never the app database, so
# nothing in the suite can touch real data.
if os.environ.get("TEST_DATABASE_URL"):
    os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
    os.environ.pop("MIGRATION_DATABASE_URL", None)
else:
    os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test")
os.environ["ENVIRONMENT"] = "test"


def _psycopg3(url: str | None) -> str | None:
    """Pin the psycopg3 driver, matching Settings._normalise_driver. Neon URLs are bare
    `postgresql://`, which SQLAlchemy would otherwise route to psycopg2 (not installed)."""
    if not url:
        return url
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


TEST_DATABASE_URL = _psycopg3(os.environ.get("TEST_DATABASE_URL"))
requires_db = pytest.mark.skipif(
    not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set; live-database test skipped"
)


@pytest.fixture(scope="session")
def settings():
    from backend.app.core.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def ml():
    """Process-wide MLService, loaded once (mirrors the app lifespan)."""
    from backend.app.services.ml_service import ml_service

    ml_service.load()
    return ml_service


@pytest.fixture(scope="session")
def client():
    """
    TestClient that runs the real lifespan handler, so ML artifacts load exactly as they
    do in production.
    """
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


def alembic_config_for(url: str):
    """Alembic config pointed at an explicit database."""
    from alembic.config import Config

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", url)
    return config


def _reset_schema(engine) -> None:
    """Drop every table plus Alembic's bookkeeping, leaving a clean slate."""
    from sqlalchemy import text

    import backend.app.models  # noqa: F401 — populate metadata
    from backend.app.db.base import Base

    Base.metadata.drop_all(engine)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS alembic_version"))


@pytest.fixture(scope="session")
def db_engine():
    """
    Live engine against TEST_DATABASE_URL.

    The schema is built by running the real Alembic migration — not
    `Base.metadata.create_all` — so every database test exercises the same schema that
    will be deployed, and a broken migration fails the suite.
    """
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set")

    from alembic import command
    from sqlalchemy import create_engine

    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    config = alembic_config_for(TEST_DATABASE_URL)

    _reset_schema(engine)
    command.upgrade(config, "head")
    try:
        yield engine
    finally:
        _reset_schema(engine)
        engine.dispose()


@pytest.fixture
def db_session_factory(db_engine):
    """Session factory bound to the test database, for code that would otherwise use
    the application's SessionLocal (e.g. the seed script)."""
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(bind=db_engine, autoflush=False, autocommit=False, expire_on_commit=False)


@pytest.fixture
def db_session(db_engine):
    """Function-scoped session rolled back after each test."""
    from sqlalchemy.orm import sessionmaker

    connection = db_engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        # A test that triggered an IntegrityError may already have unwound the
        # transaction; rolling back again would emit a spurious SAWarning.
        if transaction.is_active:
            transaction.rollback()
        connection.close()


@pytest.fixture
def sample_raw_features():
    """A realistic high-risk raw payload: the 28 model inputs plus hostel_status."""
    return {
        "age": 20.5,
        "commute_distance_km": 28.0,
        "income_slab_idx": 0,
        "is_first_generation": 1,
        "has_scholarship": 0,
        "fee_payment_delay_days": 75,
        "hostel_status": "Day Scholar",
        "att_core1": 40.0,
        "att_core2": 42.0,
        "att_lab": 52.0,
        "att_elective": 41.0,
        "attendance_month_1": 56.0,
        "attendance_month_2": 45.0,
        "attendance_month_3": 36.0,
        "attendance_percentage": 44.2,
        "attendance_3m_trend": -10.0,
        "consecutive_absences": 12,
        "attendance_risk_flag": 1,
        "prev_sem_cgpa": 6.20,
        "current_cgpa": 5.05,
        "cgpa_delta": -1.15,
        "backlog_count": 2,
        "internal_exam_score_pct": 42.0,
        "stem_core_fail_flag": 1,
        "lms_logins_per_week": 1.5,
        "assignment_submission_lag_days": 5.8,
        "resource_access_count": 12,
        "days_since_last_lms_activity": 22,
        "forum_participation_count": 0,
    }
