"""
target_base_rate sensitivity: rendered from its JSON only, refused without provenance; and the
feature pipeline never writes the raw cohort into the repository's data/ directory.
"""

import json

import pytest

import ml.simulation.render_simulation_doc as rs
from ml.data_pipeline.feature_engineering import generate_processed_feature_dataset
from ml.data_pipeline.generate_synthetic_indian import SYNTHETIC_DATA_DIR
from ml.provenance import ProvenanceError


def _metric(point, lo, hi):
    return {"point": point, "ci_lower": lo, "ci_upper": hi}


def _sensitivity(provenance=True):
    data = {
        "analysis": "target_base_rate_sensitivity",
        "rates": [0.15, 0.355],
        "results": [
            {"target_base_rate": 0.15, "observed_dropout_rate": 0.1234, "n_test": 300,
             "metrics": {"roc_auc": _metric(0.91, 0.88, 0.94), "pr_auc": _metric(0.61, 0.52, 0.70),
                         "brier_score": _metric(0.071, 0.060, 0.083)}},
            {"target_base_rate": 0.355, "observed_dropout_rate": 0.3456, "n_test": 300,
             "metrics": {"roc_auc": _metric(0.93, 0.91, 0.95), "pr_auc": _metric(0.88, 0.84, 0.91),
                         "brier_score": _metric(0.082, 0.070, 0.095)}},
        ],
    }
    if provenance:
        data["provenance"] = {"git_commit": "c" * 40, "input_files": {}}
    return data


def test_sensitivity_table_renders_every_rate_from_the_json():
    text = rs.render_sensitivity_section(_sensitivity())
    assert "| 0.15 | 0.1234 | 300 | 0.9100 [0.8800, 0.9400] | 0.6100 [0.5200, 0.7000] | 0.0710 [0.0600, 0.0830] |" in text
    assert "| 0.355 | 0.3456 | 300 | 0.9300 [0.9100, 0.9500] |" in text
    assert "Simulated data only" in text


def test_simulation_doc_refuses_sensitivity_json_without_provenance(tmp_path, monkeypatch):
    est = tmp_path / "estimated_parameters.json"
    est.write_text(json.dumps({"provenance": {"git_commit": "c" * 40, "input_files": {}}}))
    sens = tmp_path / "simulation_sensitivity.json"
    sens.write_text(json.dumps(_sensitivity(provenance=False)))
    monkeypatch.setattr(rs, "ESTIMATED_PARAMETERS_PATH", est)
    monkeypatch.setattr(rs, "SENSITIVITY_PATH", sens)
    monkeypatch.setattr(rs, "DOCS_SIM_PATH", tmp_path / "simulation.md")
    with pytest.raises(ProvenanceError, match="no provenance"):
        rs.generate_simulation_doc()
    assert not (tmp_path / "simulation.md").exists()


def test_feature_generation_never_writes_into_repository_data_dir(tmp_path):
    def snapshot():
        if not SYNTHETIC_DATA_DIR.exists():
            return {}
        return {p.name: p.stat().st_mtime_ns for p in SYNTHETIC_DATA_DIR.iterdir()}

    before = snapshot()
    generate_processed_feature_dataset(n_students=50, seed=7, output_path=tmp_path / "features.csv")
    assert (tmp_path / "features.csv").exists()
    assert snapshot() == before, f"{SYNTHETIC_DATA_DIR} changed during feature generation"
