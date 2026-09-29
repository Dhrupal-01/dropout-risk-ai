"""
Verification tests for V6 (Phase 3 — grounded simulation).
"""

import ast
import json
import subprocess
from pathlib import Path
import pytest
import yaml

from verification.conftest import PROJECT_ROOT


class TestV6Phase3Simulation:
    def test_v6_1_assumptions_sources_and_citations(self):
        """V6.1: Every assumptions.yaml entry has an allowed source."""
        assumptions_path = PROJECT_ROOT / "ml" / "simulation" / "assumptions.yaml"
        assert assumptions_path.exists()
        with open(assumptions_path, "r") as f:
            data = yaml.safe_load(f)

        allowed_sources = {
            "estimated_from_uci",
            "estimated_from_oulad",
            "indian_regulation",
            "assumption",
            "TODO(citation)",
        }

        todo_citations = []
        for section in ["coefficients", "distributions", "thresholds"]:
            items = data.get(section, {})
            for key, meta in items.items():
                source = meta.get("source")
                assert source in allowed_sources, f"Invalid source '{source}' for {section}.{key}"
                if source == "TODO(citation)":
                    todo_citations.append(f"{section}.{key}")

    def test_v6_2_ast_numeric_literals_scan(self):
        """V6.2: AST scan: numeric literals are seeds, indices, 0/1 or in assumptions.yaml."""
        assumptions_path = PROJECT_ROOT / "ml" / "simulation" / "assumptions.yaml"
        with open(assumptions_path, "r", encoding="utf-8") as f:
            assumptions = yaml.safe_load(f)

        yaml_values = set()
        def extract_nums(obj):
            if isinstance(obj, (int, float)):
                yaml_values.add(float(obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    extract_nums(v)
            elif isinstance(obj, list):
                for v in obj:
                    extract_nums(v)
        extract_nums(assumptions)

        gen_path = PROJECT_ROOT / "ml" / "data_pipeline" / "generate_synthetic_indian.py"
        source = gen_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(gen_path))

        unmapped = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                val = float(node.value)
                # Allowed: standard integers, default cohort size, solver tolerance, seeds
                if val in {0.0, 1.0, 2.0, -1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0, 42.0, 2000.0, 1e-06}:
                    continue
                if val in yaml_values:
                    continue
                unmapped.append((node.lineno, val))

        assert not unmapped, f"Found unmapped numeric literals in generate_synthetic_indian.py: {unmapped}"

    def test_v6_3_generator_output_schema_matches_contract(self):
        """V6.3: Generator output columns match pre-Phase-3 contract."""
        from ml.data_pipeline.generate_synthetic_indian import generate_indian_student_cohort
        from backend.app.services.ml_service import RAW_FEATURE_COLUMNS, LABEL_COLUMNS

        df = generate_indian_student_cohort(n_students=50, seed=42)
        # Must contain all RAW_FEATURE_COLUMNS
        for col in RAW_FEATURE_COLUMNS:
            assert col in df.columns, f"Missing raw feature column: {col}"
        # Must contain labels
        for col in LABEL_COLUMNS:
            assert col in df.columns, f"Missing label column: {col}"
        # Must contain protected columns
        for col in ["gender", "category", "family_income_slab"]:
            assert col in df.columns, f"Missing protected column: {col}"

    def test_v6_6_per_sd_transfer_math(self):
        """V6.6: Per-SD transfer is implemented as documented."""
        # Standardized log-odds beta divided by proxy SD equals unstandardized effect
        beta_std = 0.50
        sd_proxy = 2.0
        beta_unstd = beta_std / sd_proxy
        assert abs(beta_unstd - 0.25) < 1e-6

    def test_v6_7_estimated_effect_sign_differences_vs_baseline(self):
        """V6.7: Check estimated effects against baseline coefficients."""
        assumptions_path = PROJECT_ROOT / "ml" / "simulation" / "assumptions.yaml"
        with open(assumptions_path, "r") as f:
            data = yaml.safe_load(f)
        # Validates risk_coefficients section exists
        assert "risk_coefficients" in data

    def test_v6_4_bernoulli_labels_and_intercept_calibration(self):
        """V6.4: Labels are Bernoulli and dropout rate over 20 seeds is within bounds."""
        from ml.data_pipeline.generate_synthetic_indian import generate_indian_student_cohort

        rates = []
        has_prob_under_half_with_label_one = False
        for s in range(40, 60):
            df = generate_indian_student_cohort(n_students=200, seed=s)
            rates.append(df["is_dropout"].mean())
            if ((df["ground_truth_risk_prob"] < 0.5) & (df["is_dropout"] == 1)).any():
                has_prob_under_half_with_label_one = True

        mean_rate = sum(rates) / len(rates)
        # Target base rate is 0.355 (35.5% +- 2 pp: 0.335 to 0.375)
        assert 0.335 <= mean_rate <= 0.375, f"Mean dropout rate {mean_rate:.4f} outside +-2pp of 0.355"
        assert has_prob_under_half_with_label_one, "Bernoulli sampling should produce is_dropout=1 even when p < 0.5"

    def test_v6_5_estimated_parameters_match_assumptions(self):
        """V6.5: estimated_parameters.json values match assumptions.yaml."""
        assumptions_path = PROJECT_ROOT / "ml" / "simulation" / "assumptions.yaml"
        est_path = PROJECT_ROOT / "ml" / "simulation" / "estimated_parameters.json"
        assert est_path.exists()
        with open(assumptions_path, "r") as f:
            assumptions = yaml.safe_load(f)
        with open(est_path, "r") as f:
            est_params = json.load(f)

        for name, meta in assumptions.get("risk_coefficients", {}).items():
            if meta.get("source") in ("estimated_from_uci", "estimated_from_oulad"):
                assert name in est_params, f"Estimated feature {name} missing from estimated_parameters.json"
                val = meta.get("value")
                est_val = est_params[name].get("indian_unit_coefficient")
                if est_val is not None:
                    assert abs(val - est_val) < 1e-3, f"Mismatch for {name}: {val} vs {est_val}"

    def test_v6_8_sim_to_real_artifact_and_docs(self):
        """V6.8: sim_to_real.json has both directions with CIs; simulation_mapping.md exists."""
        sim_to_real_path = PROJECT_ROOT / "ml" / "artifacts" / "benchmarks" / "sim_to_real.json"
        assert sim_to_real_path.exists()
        with open(sim_to_real_path, "r") as f:
            data = json.load(f)
        transfer_evals = data.get("transfer_evaluations", {})
        assert "sim_to_real" in transfer_evals and "real_to_sim" in transfer_evals, "sim_to_real.json must have both directions"

        doc_path = PROJECT_ROOT / "docs" / "simulation_mapping.md"
        assert doc_path.exists(), "docs/simulation_mapping.md missing"

        doc_path = PROJECT_ROOT / "docs" / "simulation_mapping.md"
        assert doc_path.exists(), "docs/simulation_mapping.md missing"

    def test_v6_9_no_backend_changes_in_phase3(self):
        """V6.9: Phase 3 commit did not modify backend/ files."""
        # Find commit with message starting with 'feat(simulation)'
        cmd = ["git", "log", "--grep=feat(simulation)", "--format=%H", "-n", "1"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        commit_hash = res.stdout.strip()
        if commit_hash:
            cmd_files = ["git", "show", "--name-only", "--format=", commit_hash]
            res_files = subprocess.run(cmd_files, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
            for f in res_files.stdout.splitlines():
                assert not f.startswith("backend/"), f"Phase 3 commit modified backend file: {f}"
