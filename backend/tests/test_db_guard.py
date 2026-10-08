"""
The DB-backed suites must refuse to run when TEST_DATABASE_URL names the application database.
Only fake URLs are used here; no test ever reads or connects to the real .env database.
"""

import os
import subprocess
import sys

import pytest

from backend.tests.db_guard import PROJECT_ROOT, database_identity, find_app_database_conflict

APP = "postgresql://appuser:s3cret-pw@ep-cool-lake-123.ap-south-1.aws.neon.tech/appdb"


@pytest.mark.parametrize("variant", [
    "postgres://other:pw@ep-cool-lake-123.ap-south-1.aws.neon.tech/appdb",
    "postgresql+psycopg://appuser:pw@ep-cool-lake-123.ap-south-1.aws.neon.tech/appdb?sslmode=require&channel_binding=require",
    "postgresql://appuser:pw@EP-COOL-LAKE-123.ap-south-1.aws.neon.tech/appdb?sslmode=require",
    "postgresql://appuser:pw@ep-cool-lake-123-pooler.ap-south-1.aws.neon.tech/appdb",
])
def test_same_database_despite_driver_query_case_or_pooler(variant):
    assert database_identity(variant) == database_identity(APP)


@pytest.mark.parametrize("other", [
    "postgresql://appuser:pw@ep-cool-lake-123.ap-south-1.aws.neon.tech/appdb_test",
    "postgresql://appuser:pw@ep-other-999.ap-south-1.aws.neon.tech/appdb",
    "sqlite:///appdb",
])
def test_different_database(other):
    assert database_identity(other) != database_identity(APP)


def test_sqlite_relative_and_absolute_paths_match():
    absolute = PROJECT_ROOT / "dropoutguard.db"
    assert database_identity("sqlite:///dropoutguard.db") == database_identity(f"sqlite:///{absolute}")


def test_conflict_found_from_env_file_without_leaking_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"DATABASE_URL={APP}\n")
    message = find_app_database_conflict(APP.replace("postgresql://", "postgres://"), [env])
    assert message and "Refusing to run tests" in message and "'appdb'" in message
    assert "s3cret-pw" not in message and "appuser" not in message


def test_no_conflict_for_different_database_or_unset_test_url(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"DATABASE_URL={APP}\n")
    assert find_app_database_conflict(APP.replace("/appdb", "/appdb_test"), [env]) is None
    assert find_app_database_conflict(None, [env]) is None


def test_conflict_with_process_environment_database_url(monkeypatch):
    monkeypatch.delenv("DROPOUTGUARD_APP_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", APP)
    assert find_app_database_conflict(APP, []) is not None


def test_recorded_app_url_wins_after_conftest_overwrites_database_url(monkeypatch):
    # backend/tests/conftest.py sets DATABASE_URL = TEST_DATABASE_URL after the first guard call;
    # a later guard call (verification conftest in the same session) must not see that as a conflict.
    test_url = APP.replace("/appdb", "/appdb_test")
    monkeypatch.setenv("DROPOUTGUARD_APP_DATABASE_URL", APP)
    monkeypatch.setenv("DATABASE_URL", test_url)
    assert find_app_database_conflict(test_url, []) is None
    monkeypatch.setenv("DROPOUTGUARD_APP_DATABASE_URL", test_url)
    assert find_app_database_conflict(test_url, []) is not None


def _collect(target: str, test_url: str, app_url: str, tmp_path) -> subprocess.CompletedProcess:
    env_file = tmp_path / "fake.env"
    env_file.write_text(f"DATABASE_URL={app_url}\n")
    env = {k: v for k, v in os.environ.items() if k not in ("TEST_DATABASE_URL", "DATABASE_URL", "DROPOUTGUARD_APP_DATABASE_URL")}
    env.update({"TEST_DATABASE_URL": test_url, "DROPOUTGUARD_DB_GUARD_ENV_FILES": str(env_file)})
    return subprocess.run(
        [sys.executable, "-m", "pytest", *target.split(), "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
    )


FAKE_APP = "postgresql://appuser:s3cret-pw@db.invalid/appdb"


@pytest.mark.parametrize("target", ["backend/tests/test_health.py", "verification/test_v6_phase3.py"])
def test_conftest_aborts_session_when_test_db_is_app_db(target, tmp_path):
    res = _collect(target, FAKE_APP.replace("postgresql://", "postgresql+psycopg://") + "?sslmode=require", FAKE_APP, tmp_path)
    output = res.stdout + res.stderr
    assert res.returncode == pytest.ExitCode.USAGE_ERROR, output[-2000:]
    assert "Refusing to run tests: TEST_DATABASE_URL names the application database" in output
    assert "s3cret-pw" not in output


@pytest.mark.parametrize("target", [
    "backend/tests/test_health.py",
    "verification/test_v6_phase3.py",
    "backend/tests/test_health.py verification/test_v6_phase3.py",  # both conftests in one session
])
def test_conftest_allows_a_different_test_database(target, tmp_path):
    res = _collect(target, FAKE_APP.replace("/appdb", "/appdb_test"), FAKE_APP, tmp_path)
    assert res.returncode == pytest.ExitCode.OK, (res.stdout + res.stderr)[-2000:]
