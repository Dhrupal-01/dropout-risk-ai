"""Configuration loading, CORS safety, and credential hygiene."""

import pytest

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
        with pytest.raises(Exception):
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
