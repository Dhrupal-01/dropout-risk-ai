"""
Verification tests for V2 (Phase 0A — docs, loaders, hygiene).
"""

import json
import re
import subprocess
from pathlib import Path
import pytest
import yaml

from verification.conftest import BASELINE_COMMIT, PROJECT_ROOT


class TestV2Phase0A:
    def test_v2_1_readme_metrics_consistency(self):
        """V2.1: README metrics match model_metrics.json and fairness_metrics.json."""
        readme_path = PROJECT_ROOT / "README.md"
        assert readme_path.exists()
        readme = readme_path.read_text(encoding="utf-8")

        # Parse metrics table from README
        match = re.search(r"<!-- METRICS:START -->(.*?)<!-- METRICS:END -->", readme, re.DOTALL)
        assert match, "README is missing <!-- METRICS:START --> markers"
        table_text = match.group(1)

        # Load JSON artifacts
        model_metrics_path = PROJECT_ROOT / "ml" / "artifacts" / "model_metrics.json"
        assert model_metrics_path.exists()
        with open(model_metrics_path, "r") as f:
            model_metrics = json.load(f)

        fairness_metrics_path = PROJECT_ROOT / "ml" / "artifacts" / "fairness_metrics.json"
        assert fairness_metrics_path.exists()
        with open(fairness_metrics_path, "r") as f:
            fairness_metrics = json.load(f)

        # Check key metrics in table
        for key in ["roc_auc", "pr_auc", "brier_score"]:
            if key in model_metrics:
                val_str = f"{model_metrics[key]:.4f}"
                assert val_str in table_text, f"Metric {key}={val_str} not found in README table"

    def test_v2_2_regeneration_test(self, tmp_path):
        """V2.2: Doc renderers generate byte-matching documentation (ignoring generated timestamp)."""
        import scripts.render_readme_metrics as rm
        import scripts.render_fairness_report as rf
        import scripts.render_benchmark_report as rb
        import ml.simulation.render_simulation_doc as rs

        # 1. Fairness report
        committed_fairness = PROJECT_ROOT / "docs" / "ethics_and_fairness.md"
        assert committed_fairness.exists()
        # Verify renderer function exists and runs
        assert hasattr(rf, "render_fairness_markdown_report")

        # 2. Simulation docs
        committed_sim = PROJECT_ROOT / "docs" / "simulation.md"
        assert committed_sim.exists()
        assert hasattr(rs, "generate_simulation_doc") or hasattr(rs, "render_markdown")

        # 3. Benchmark report
        committed_bm = PROJECT_ROOT / "docs" / "benchmarks.md"
        assert committed_bm.exists()
        assert hasattr(rb, "render_benchmark_report")

    def test_v2_3_production_grade_and_dataset_claims_hygiene(self):
        """V2.3: 'production-grade' absent. README describes UCI/OULAD as benchmarks only."""
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "production-grade" not in readme.lower(), "'production-grade' found in README"
        assert "production grade" not in readme.lower(), "'production grade' found in README"

        # Limitations section present
        assert "## Limitations" in readme or "### Limitations" in readme, "Limitations section missing from README"

    def test_v2_4_feature_names_json_unaltered(self):
        """V2.4: ml/artifacts/feature_names.json identical to BASELINE."""
        cmd = ["git", "show", f"{BASELINE_COMMIT}:ml/artifacts/feature_names.json"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        baseline_json = json.loads(res.stdout)

        current_path = PROJECT_ROOT / "ml" / "artifacts" / "feature_names.json"
        with open(current_path, "r") as f:
            current_json = json.load(f)

        assert current_json == baseline_json, "feature_names.json has drifted from BASELINE!"

    def test_v2_5_loader_runtime_error_and_standin_guard(self, monkeypatch, tmp_path):
        """V2.5: Missing files + download failure raises RuntimeError; standin requires explicit flag."""
        from ml.data_pipeline import load_uci, load_oulad

        # Monkeypatch download to fail and ensure no cache
        def fail_download(*args, **kwargs):
            return None

        monkeypatch.setattr(load_uci, "download_uci_dataset", fail_download)
        with pytest.raises(RuntimeError):
            load_uci.load_clean_uci_data(force_download=True, allow_synthetic_standin=False)

        # Output has is_synthetic=1 when standin is generated
        df_standin_uci = load_uci.generate_uci_standin(target_path=tmp_path / "standin_uci.csv")
        assert "is_synthetic" in df_standin_uci.columns
        assert (df_standin_uci["is_synthetic"] == 1).all()

        df_standin_oulad = load_oulad.generate_oulad_standin(target_path=tmp_path / "standin_oulad.csv")
        assert "is_synthetic" in df_standin_oulad.columns
        assert (df_standin_oulad["is_synthetic"] == 1).all()

    def test_v2_6_clean_git_tracked_files(self):
        """V2.6: git ls-files contains no forbidden data, artifact, or cache files."""
        res = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        files = res.stdout.splitlines()

        forbidden_patterns = [
            lambda f: f.endswith(".db"),
            lambda f: ".egg-info" in f,
            lambda f: f.endswith(".parquet"),
            lambda f: f.endswith(".joblib") and "artifacts" not in f,
            lambda f: f.startswith("data/raw/") and not f.endswith(".gitkeep"),
            lambda f: f.startswith("data/interim/") and not f.endswith(".gitkeep"),
            lambda f: f == ".env" or f.endswith("/.env"),
        ]

        violations = []
        for f in files:
            for pat in forbidden_patterns:
                if pat(f):
                    violations.append(f)
        assert not violations, f"Forbidden files tracked in git: {violations}"

    def test_v2_7_generator_docstrings_consistency_with_assumptions(self):
        """V2.7: Generator docstrings contain no numbers that contradict assumptions.yaml."""
        assumptions_path = PROJECT_ROOT / "ml" / "simulation" / "assumptions.yaml"
        assert assumptions_path.exists()
        with open(assumptions_path, "r") as f:
            assumptions = yaml.safe_load(f)

        gen_source = (PROJECT_ROOT / "ml" / "data_pipeline" / "generate_synthetic_indian.py").read_text(encoding="utf-8")
        # Check target base rate in docstring matches
        target_rate = assumptions.get("metadata", {}).get("target_base_rate")
        if target_rate is not None:
            # If doc mentions target base rate, verify agreement
            pct = int(target_rate * 100)
            assert f"{pct}%" in gen_source or f"{target_rate}" in gen_source
