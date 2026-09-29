"""
Unit and Integration Tests for Phase 3 Simulation Grounding
Tests:
- assumptions.yaml schema validity and source taxonomy
- docs/simulation.md synchronization check
- Numerical calibration and mean base rate convergence across 20 random seeds
- Bernoulli sampling variation (p < 0.5 with label 1 and p >= 0.5 with label 0)
- Logging warning for TODO(citation) placeholders
- Protected attributes strictly zero coefficients
- Raw feature column contract match
- Sim-to-Real benchmark artifact schema and evaluation results
"""

import json
import logging
from pathlib import Path
import pytest
import numpy as np
import yaml

from ml.data_pipeline.generate_synthetic_indian import (
    generate_indian_student_cohort,
    load_simulation_assumptions,
    solve_intercept_for_base_rate,
)
from ml.simulation.render_simulation_doc import (
    verify_assumptions_doc_sync,
    ALLOWED_SOURCES,
    extract_all_keys,
)
from backend.app.services.ml_service import RAW_FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parents[2]
ASSUMPTIONS_PATH = BASE_DIR / "ml" / "simulation" / "assumptions.yaml"
DOCS_SIMULATION_PATH = BASE_DIR / "docs" / "simulation.md"
DOCS_MAPPING_PATH = BASE_DIR / "docs" / "simulation_mapping.md"
ESTIMATED_PARAMS_PATH = BASE_DIR / "ml" / "simulation" / "estimated_parameters.json"
SIM_TO_REAL_JSON_PATH = BASE_DIR / "ml" / "artifacts" / "benchmarks" / "sim_to_real.json"


class TestSimulationAssumptions:
    """Tests for assumptions.yaml and documentation synchronization."""

    def test_assumptions_file_exists_and_loads(self):
        assert ASSUMPTIONS_PATH.exists(), f"Missing {ASSUMPTIONS_PATH}"
        data = load_simulation_assumptions(ASSUMPTIONS_PATH)
        assert isinstance(data, dict)
        assert "cohort_metadata" in data
        assert "risk_coefficients" in data
        assert "regulations_and_thresholds" in data
        assert "demographics_distribution" in data
        assert "socioeconomic_distribution" in data
        assert "attendance_distribution" in data
        assert "academic_distribution" in data
        assert "learning_behavior_distribution" in data

    def test_all_entries_have_valid_sources_and_descriptions(self):
        data = load_simulation_assumptions(ASSUMPTIONS_PATH)
        flat = extract_all_keys(data)
        assert len(flat) >= 80, f"Expected at least 80 assumptions, found {len(flat)}"

        for key, entry in flat.items():
            assert "value" in entry, f"Entry '{key}' missing 'value'"
            assert "source" in entry, f"Entry '{key}' missing 'source'"
            assert "notes" in entry, f"Entry '{key}' missing 'notes'"
            src = entry["source"]
            assert src in ALLOWED_SOURCES, f"Entry '{key}' has invalid source '{src}'"
            assert len(str(entry["notes"]).strip()) > 0, f"Entry '{key}' has empty notes"

    def test_simulation_doc_is_strictly_synced_with_yaml(self):
        """Pre-commit / CI consistency check between assumptions.yaml and docs/simulation.md."""
        is_synced, errors = verify_assumptions_doc_sync()
        assert is_synced, f"docs/simulation.md is out of sync with assumptions.yaml:\n" + "\n".join(errors)

    def test_protected_attributes_have_zero_coefficients(self):
        """Demographic protected attributes must have 0.0 coefficients to prevent algorithmic bias."""
        data = load_simulation_assumptions(ASSUMPTIONS_PATH)
        risk_coefs = data["risk_coefficients"]
        for protected_key in ["gender_male", "category_obc", "category_sc", "category_st", "category_ews"]:
            assert protected_key in risk_coefs, f"Missing protected coefficient '{protected_key}'"
            val = risk_coefs[protected_key]["value"]
            assert val == 0.0, f"Protected attribute '{protected_key}' coefficient must be 0.0, got {val}"

    def test_estimated_parameters_json_and_mapping_doc_exist(self):
        assert ESTIMATED_PARAMS_PATH.exists(), f"Missing {ESTIMATED_PARAMS_PATH}"
        with open(ESTIMATED_PARAMS_PATH, "r", encoding="utf-8") as f:
            params = json.load(f)

        expected_empirical = [
            "current_cgpa", "backlog_count", "has_scholarship", "fee_payment_delay_days",
            "is_first_generation", "is_hosteler", "lms_logins_per_week",
            "days_since_last_lms_activity", "assignment_submission_lag_days"
        ]
        for feat in expected_empirical:
            assert feat in params, f"Missing empirical parameter '{feat}' in estimated_parameters.json"
            assert "standardized_effect" in params[feat]
            assert "indian_unit_coefficient" in params[feat]
            assert "indian_unit_ci_95" in params[feat]

        assert DOCS_MAPPING_PATH.exists(), f"Missing {DOCS_MAPPING_PATH}"
        mapping_text = DOCS_MAPPING_PATH.read_text(encoding="utf-8")
        assert "Proxy Confidence" in mapping_text


class TestSyntheticIndianCohortGeneration:
    """Tests for synthetic cohort generator, numerical solver, and Bernoulli sampling."""

    def test_numerical_intercept_solver(self):
        rng = np.random.default_rng(42)
        z = rng.normal(0, 2.0, size=2000)
        target_rate = 0.355
        beta_0 = solve_intercept_for_base_rate(z, target_rate)
        calibrated_p = 1.0 / (1.0 + np.exp(-(beta_0 + z)))
        assert abs(float(np.mean(calibrated_p)) - target_rate) < 1e-4

    def test_target_base_rate_warning_logged(self, caplog):
        with caplog.at_level(logging.WARNING):
            generate_indian_student_cohort(n_students=100, seed=42, output_path=None)
        assert any("TODO(citation)" in record.message or "placeholder" in record.message for record in caplog.records)

    def test_mean_dropout_rate_across_20_seeds_within_bounds(self):
        """Mean dropout rate across 20 random seeds must be within +-2 pp of target base rate (33.5% to 37.5%)."""
        rates = []
        for s in range(20):
            df = generate_indian_student_cohort(n_students=2000, seed=s, output_path=None)
            rates.append(df["is_dropout"].mean())

        mean_rate = float(np.mean(rates))
        assert 0.335 <= mean_rate <= 0.375, f"Mean dropout rate {mean_rate*100:.2f}% outside [33.5%, 37.5%]"

    def test_bernoulli_sampling_variation(self):
        """Confirms probabilistic Bernoulli draws: some p < 0.5 have label 1 and some p >= 0.5 have label 0."""
        df = generate_indian_student_cohort(n_students=2000, seed=42, output_path=None)
        p = df["ground_truth_risk_prob"].values
        y = df["is_dropout"].values

        under_05_dropped = ((p < 0.5) & (y == 1)).sum()
        over_05_stayed = ((p >= 0.5) & (y == 0)).sum()

        assert under_05_dropped > 0, "No students with p < 0.5 had is_dropout == 1"
        assert over_05_stayed > 0, "No students with p >= 0.5 had is_dropout == 0"

    def test_output_schema_strictly_matches_raw_feature_contract(self):
        """Every column required by backend RAW_FEATURE_COLUMNS must be generated."""
        df = generate_indian_student_cohort(n_students=100, seed=42, output_path=None)
        df_cols = set(df.columns)

        for col in RAW_FEATURE_COLUMNS:
            assert col in df_cols, f"Missing required raw feature column: '{col}'"

        assert "student_id" in df_cols
        assert "gender" in df_cols
        assert "category" in df_cols
        assert "hostel_status" in df_cols
        assert "family_income_slab" in df_cols
        assert "ground_truth_risk_prob" in df_cols
        assert "is_dropout" in df_cols


class TestSimToRealBenchmarkArtifact:
    """Tests for Sim-to-Real benchmark results and confidence intervals."""

    def test_sim_to_real_json_artifact_schema_and_metrics(self):
        assert SIM_TO_REAL_JSON_PATH.exists(), f"Missing {SIM_TO_REAL_JSON_PATH}"
        with open(SIM_TO_REAL_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data.get("benchmark") == "sim_to_real_transfer"
        evals = data.get("transfer_evaluations", {})
        for mode in ["real_on_real", "sim_on_sim", "sim_to_real", "real_to_sim"]:
            assert mode in evals, f"Missing transfer evaluation mode '{mode}'"
            item = evals[mode]
            assert "roc_auc" in item["metrics"]
            assert "pr_auc" in item["metrics"]
            roc = item["metrics"]["roc_auc"]
            pr = item["metrics"]["pr_auc"]
            assert 0.5 <= roc["point"] <= 1.0
            assert roc["ci_lower"] <= roc["point"] <= roc["ci_upper"]
            assert 0.0 <= pr["point"] <= 1.0
            assert pr["ci_lower"] <= pr["point"] <= pr["ci_upper"]
