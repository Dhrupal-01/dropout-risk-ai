"""
Shared fixtures for ml/tests.

- Generated simulated-cohort artifacts are built once per session in a temp dir
  (`simulated_artifacts`); the repository's ml/artifacts/ and data/processed/ are never written.
- Tests marked `data` need the real UCI / OULAD files under data/raw/. When those files are absent
  they are skipped with an explicit reason instead of letting the loader try a network download.
"""

import pytest

from ml.tests.simulated_artifacts import build_simulated_artifacts, redirect_artifacts_to_tempdir

# Must run before any test module imports ml.config
redirect_artifacts_to_tempdir()


@pytest.fixture(scope="session")
def simulated_artifacts():
    """Temp dir holding features.csv, the trained/calibrated model and the SHAP explainer."""
    return build_simulated_artifacts()


def _missing_raw_files():
    from ml.sources.oulad import OULAD_RAW_DIR, REQUIRED_TABLES
    from ml.sources.uci import UCI_CSV_PATH

    expected = [UCI_CSV_PATH] + [OULAD_RAW_DIR / t for t in REQUIRED_TABLES]
    return [str(p) for p in expected if not p.exists()]


def pytest_collection_modifyitems(config, items):
    data_items = [item for item in items if item.get_closest_marker("data")]
    if not data_items:
        return
    missing = _missing_raw_files()
    if not missing:
        return
    skip = pytest.mark.skip(
        reason=f"UCI/OULAD raw files not found under data/raw/ (missing: {', '.join(missing)}); "
        "download them (see ml/sources/*.py) or run `pytest -m \"not data\"`"
    )
    for item in data_items:
        item.add_marker(skip)
