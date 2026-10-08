"""
StudentFeatureInput derives attendance_risk_flag from the configured attendance threshold
(ml/simulation/assumptions.yaml via ml.config), not from a literal.
"""

import yaml

from backend.app.schemas.student import StudentFeatureInput
from ml import config as ml_config


def test_attendance_risk_flag_follows_configured_threshold(sample_raw_features, tmp_path, monkeypatch):
    original = ml_config.ATTENDANCE_THRESHOLD
    raised = original + 10.0
    payload = dict(sample_raw_features)
    payload.pop("attendance_risk_flag")
    payload["attendance_percentage"] = (original + raised) / 2  # at/above original, below raised

    assert StudentFeatureInput(**payload).attendance_risk_flag == 0

    # Change the threshold in a copy of assumptions.yaml and reload it through ml.config's own loader
    assumptions = yaml.safe_load(ml_config.ASSUMPTIONS_PATH.read_text(encoding="utf-8"))
    assumptions["regulations_and_thresholds"]["mandatory_attendance_threshold"]["value"] = raised
    changed = tmp_path / "assumptions.yaml"
    changed.write_text(yaml.safe_dump(assumptions), encoding="utf-8")
    monkeypatch.setattr(ml_config, "ASSUMPTIONS_PATH", changed)
    monkeypatch.setattr(ml_config, "ATTENDANCE_THRESHOLD", ml_config._attendance_threshold())
    assert ml_config.ATTENDANCE_THRESHOLD == raised

    assert StudentFeatureInput(**payload).attendance_risk_flag == 1
