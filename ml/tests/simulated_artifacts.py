"""
Test-session support: build the simulated-cohort artifacts in a temporary directory.

The model artifacts (features.csv, XGBoost + calibrated model, SHAP explainer, feature_names.json,
intervention catalog) are generated outputs, so a fresh clone may not have them. Tests that need them
request the `simulated_artifacts` session fixture (defined in ml/tests/conftest.py and
backend/tests/conftest.py), which builds them once per session under a temp dir. The repository's
ml/artifacts/ and data/processed/ are never written by the test suite.

redirect_artifacts_to_tempdir() must run before ml.config is imported; both conftests call it at
import time.
"""

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

ARTIFACTS_ENV = "DROPOUTGUARD_ARTIFACTS_DIR"
PROCESSED_ENV = "DROPOUTGUARD_PROCESSED_DATA_PATH"

_built = False


def redirect_artifacts_to_tempdir() -> Path:
    """Points ml.config's ARTIFACTS_DIR and PROCESSED_DATA_PATH at a session temp dir."""
    if ARTIFACTS_ENV not in os.environ:
        root = Path(tempfile.mkdtemp(prefix="dropoutguard-test-artifacts-"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        os.environ[ARTIFACTS_ENV] = str(root / "artifacts")
        os.environ[PROCESSED_ENV] = str(root / "processed" / "features.csv")

    expected = Path(os.environ[ARTIFACTS_ENV])
    config = sys.modules.get("ml.config")
    if config is not None and Path(config.ARTIFACTS_DIR) != expected:
        raise RuntimeError(
            f"ml.config was imported before the test artifact redirect (ARTIFACTS_DIR={config.ARTIFACTS_DIR}); "
            "tests would write into the repository. Do not import ml.config before the test conftests load."
        )
    return expected


def build_simulated_artifacts() -> Path:
    """Generates features.csv, trains and calibrates the model and builds the SHAP explainer (once)."""
    global _built
    artifacts_dir = redirect_artifacts_to_tempdir()
    if _built:
        return artifacts_dir

    from ml.config import PROCESSED_DATA_PATH
    from ml.data_pipeline.feature_engineering import generate_processed_feature_dataset
    from ml.intervention.engine import save_intervention_catalog
    from ml.models.calibrate import run_calibration_pipeline
    from ml.models.explain_shap import SHAPExplainerService
    from ml.models.train import train_pipeline

    generate_processed_feature_dataset(output_path=PROCESSED_DATA_PATH)
    # Temp-dir artifacts only: allow a dirty tree (developers run tests on uncommitted changes);
    # the temp JSON records allow_dirty=true.
    train_pipeline(allow_dirty=True)
    run_calibration_pipeline(allow_dirty=True)
    SHAPExplainerService(force_rebuild=True)
    save_intervention_catalog()
    _built = True
    return artifacts_dir
