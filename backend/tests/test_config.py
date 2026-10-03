"""Configuration loading, CORS safety, and credential hygiene."""

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings, get_settings


class TestSettingsLoading:
    def test_settings_load(self, settings):
        assert settings.PROJECT_NAME
        assert settings.API_V1_PREFIX == "/api/v1"
        assert settings.DATABASE_URL

    def test_settings_are_cached(self):
        assert get_settings() is get_settings()

    def test_database_url_is_required(self, monkeypatch):
        """No default credentials may be baked in."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]


class TestDatabaseUrlNormalisation:
    @pytest.mark.parametrize(
        "given",
        [
            "postgresql://u:p@host/db",
            "postgres://u:p@host/db",
            "postgresql+psycopg://u:p@host/db",
        ],
    )
    def test_driver_is_pinned_to_psycopg3(self, given):
        """Neon hands out bare postgresql:// URLs; psycopg2 is not installed."""
        s = Settings(DATABASE_URL=given, _env_file=None)  # type: ignore[call-arg]
        assert s.DATABASE_URL.startswith("postgresql+psycopg://")

    def test_neon_sslmode_is_preserved(self):
        url = "postgresql://u:p@ep-x.aws.neon.tech/db?sslmode=require"
        s = Settings(DATABASE_URL=url, _env_file=None)  # type: ignore[call-arg]
        assert s.DATABASE_URL.endswith("?sslmode=require")
        assert s.is_neon is True


class TestCorsSafety:
    def test_origins_parsed_from_comma_separated_env(self):
        s = Settings(
            DATABASE_URL="postgresql://u:p@h/d",
            CORS_ORIGINS="http://localhost:5173, https://app.example.com",
            _env_file=None,
        )  # type: ignore[call-arg]
        assert s.CORS_ORIGINS == ["http://localhost:5173", "https://app.example.com"]

    def test_localhost_react_dev_origin_supported_by_default(self, settings):
        assert "http://localhost:5173" in settings.CORS_ORIGINS

    def test_wildcard_forces_credentials_off(self):
        """'*' with credentials is rejected by browsers and leaks auth. Must not be possible."""
        s = Settings(
            DATABASE_URL="postgresql://u:p@h/d",
            CORS_ORIGINS="*",
            CORS_ALLOW_CREDENTIALS=True,
            _env_file=None,
        )  # type: ignore[call-arg]
        assert s.CORS_ALLOW_CREDENTIALS is False


class TestCredentialHygiene:
    def test_summary_never_exposes_password(self):
        s = Settings(
            DATABASE_URL="postgresql://admin:sup3rs3cret@ep-x.neon.tech/dropoutguard",
            _env_file=None,
        )  # type: ignore[call-arg]
        summary = s.safe_database_summary()
        assert "sup3rs3cret" not in summary
        assert "admin" not in summary
        assert "ep-x.neon.tech" in summary


class TestMlConfigReuse:
    def test_thresholds_come_from_ml_core(self):
        """Backend must not fork the risk thresholds."""
        from backend.app.core import config as backend_config
        from ml import config as ml_config

        assert backend_config.RISK_THRESHOLD_LOW == ml_config.RISK_THRESHOLD_LOW
        assert backend_config.RISK_THRESHOLD_HIGH == ml_config.RISK_THRESHOLD_HIGH
        assert backend_config.get_risk_tier is ml_config.get_risk_tier


# ---------------------------------------------------- risk thresholds and seed: one source (C1)

import json
import os
import subprocess
import sys

from backend.tests.db_guard import PROJECT_ROOT

FAKE_DB = "postgresql+psycopg://u:p@db.invalid/x"
PRINT_BOTH = (
    "import json, ml.config as m; from backend.app.core.config import get_settings; s = get_settings(); "
    "print(json.dumps({'settings': [s.RISK_THRESHOLD_LOW, s.RISK_THRESHOLD_HIGH, s.RANDOM_SEED], "
    "'ml_config': [m.RISK_THRESHOLD_LOW, m.RISK_THRESHOLD_HIGH, m.RANDOM_SEED]}))"
)


def _run(code: str, **overrides) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("RISK_THRESHOLD_LOW", "RISK_THRESHOLD_HIGH", "RANDOM_SEED")}
    env.update({"DATABASE_URL": FAKE_DB, **overrides})
    return subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)


class TestRiskThresholdSettings:
    def test_server_and_ml_core_read_identical_values(self, settings):
        import ml.config

        assert (settings.RISK_THRESHOLD_LOW, settings.RISK_THRESHOLD_HIGH, settings.RANDOM_SEED) == (
            ml.config.RISK_THRESHOLD_LOW, ml.config.RISK_THRESHOLD_HIGH, ml.config.RANDOM_SEED
        )

    def test_environment_override_reaches_both(self):
        res = _run(PRINT_BOTH, RISK_THRESHOLD_LOW="0.25", RISK_THRESHOLD_HIGH="0.7", RANDOM_SEED="7")
        assert res.returncode == 0, res.stderr[-2000:]
        values = json.loads(res.stdout.strip().splitlines()[-1])
        assert values["settings"] == values["ml_config"] == [0.25, 0.7, 7]

    @pytest.mark.parametrize("low, high", [("0.7", "0.5"), ("0.5", "0.5"), ("0", "0.5"), ("0.3", "1"), ("-0.1", "0.5")])
    def test_invalid_thresholds_fail_at_startup(self, low, high):
        for code in ("import ml.config", "import backend.app.main"):
            res = _run(code, RISK_THRESHOLD_LOW=low, RISK_THRESHOLD_HIGH=high)
            assert res.returncode != 0, f"{code} started with low={low}, high={high}"
            assert "Invalid risk thresholds" in res.stderr, res.stderr[-1000:]

    @pytest.mark.parametrize("low, high", [(0.7, 0.5), (0.5, 0.5), (0.0, 0.5), (0.3, 1.0)])
    def test_settings_rejects_invalid_thresholds(self, low, high):
        with pytest.raises(ValidationError, match="Invalid risk thresholds"):
            Settings(_env_file=None, DATABASE_URL=FAKE_DB, RISK_THRESHOLD_LOW=low, RISK_THRESHOLD_HIGH=high)

    def test_settings_rejects_values_that_disagree_with_ml_config(self):
        import ml.config

        other_low = ml.config.RISK_THRESHOLD_LOW / 2
        with pytest.raises(ValidationError, match="disagree with ml.config"):
            Settings(_env_file=None, DATABASE_URL=FAKE_DB, RISK_THRESHOLD_LOW=other_low)
