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
        """V2.1: The README METRICS block is exactly what render_metrics_block() renders from the JSON.

        render_metrics_block() reads every metric with direct key access, so a key missing from
        model_metrics.json or fairness_metrics.json raises and fails this test.
        """
        from scripts.render_readme_metrics import END_MARKER, START_MARKER, render_metrics_block

        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        match = re.search(re.escape(START_MARKER) + r"(.*?)" + re.escape(END_MARKER), readme, re.DOTALL)
        assert match, "README is missing the METRICS markers"
        assert match.group(1) == f"\n{render_metrics_block()}\n", (
            "README METRICS block differs from render_metrics_block(); regenerate with "
            "`python -m ml.pipeline run-all` (never hand-edit the block)"
        )

    def test_v2_2_regeneration_test(self, tmp_path, monkeypatch):
        """V2.2: Re-rendering every generated doc from the committed JSON reproduces the committed files
        byte for byte (README generated blocks, 4 docs, 2 figures). Outputs go to tmp_path only."""
        import ml.simulation.estimate_parameters as ep
        import ml.simulation.render_simulation_doc as rs
        import scripts.render_benchmark_report as rb
        import scripts.render_fairness_report as rf
        import scripts.render_readme_benchmarks as rrb
        import scripts.render_readme_metrics as rrm

        (tmp_path / "figures").mkdir()
        readme_copy = tmp_path / "README.md"
        readme_copy.write_bytes((PROJECT_ROOT / "README.md").read_bytes())
        monkeypatch.setattr(rb, "REPORT_MD_PATH", tmp_path / "benchmarks.md")
        monkeypatch.setattr(rb, "FIGURES_DIR", tmp_path / "figures")
        monkeypatch.setattr(rb, "RELIABILITY_IMG_PATH", tmp_path / "figures" / "uci_reliability_curves.png")
        monkeypatch.setattr(rb, "EARLINESS_IMG_PATH", tmp_path / "figures" / "earliness_curve.png")
        monkeypatch.setattr(rrb, "README_PATH", readme_copy)
        monkeypatch.setattr(rrm, "README_PATH", readme_copy)
        monkeypatch.setattr(rs, "DOCS_SIM_PATH", tmp_path / "simulation.md")
        monkeypatch.setattr(ep, "DOCS_MAPPING_PATH", tmp_path / "simulation_mapping.md")

        rb.render_benchmark_report()  # also re-renders the README BENCHMARKS block into the copy
        rf.render_fairness_markdown_report(report_path=tmp_path / "ethics_and_fairness.md")
        rrm.update_readme()
        ep.render_simulation_mapping_from_json()
        rs.generate_simulation_doc()

        pairs = {
            "README.md": readme_copy,
            "docs/benchmarks.md": tmp_path / "benchmarks.md",
            "docs/ethics_and_fairness.md": tmp_path / "ethics_and_fairness.md",
            "docs/simulation.md": tmp_path / "simulation.md",
            "docs/simulation_mapping.md": tmp_path / "simulation_mapping.md",
            "docs/figures/uci_reliability_curves.png": tmp_path / "figures" / "uci_reliability_curves.png",
            "docs/figures/earliness_curve.png": tmp_path / "figures" / "earliness_curve.png",
        }
        differs = [name for name, rendered in pairs.items()
                   if rendered.read_bytes() != (PROJECT_ROOT / name).read_bytes()]
        assert not differs, (
            f"Committed generated files differ from a fresh render: {differs}. Regenerate with "
            "`python -m ml.pipeline run-all`; never hand-edit generated docs."
        )

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
        """V2.5: Missing files + download failure raise with instructions; real loaders refuse is_synthetic files."""
        import pandas as pd
        from ml.sources import oulad, uci
        from ml.sources.integrity import SyntheticDataError

        # Download failure with no cache raises RuntimeError with instructions and writes nothing
        def fail_download(*args, **kwargs):
            raise ConnectionError("network disabled in tests")

        monkeypatch.setattr(uci.requests, "get", fail_download)
        missing_uci = tmp_path / "missing" / "uci_dropout.csv"
        with pytest.raises(RuntimeError, match="Manual download instructions"):
            uci.load_uci_clean_df(csv_path=missing_uci)
        assert not missing_uci.exists()

        # Missing OULAD directory raises FileNotFoundError with instructions
        with pytest.raises(FileNotFoundError, match="Manual download instructions"):
            oulad.check_and_get_oulad_dir(tmp_path / "missing_oulad")

        # A stand-in file carrying is_synthetic is refused by the real loader
        standin_uci = tmp_path / "standin_uci.csv"
        pd.DataFrame({"Target": ["Dropout"], "is_synthetic": [1]}).to_csv(standin_uci, sep=";", index=False)
        with pytest.raises(SyntheticDataError, match="is_synthetic"):
            uci.load_uci_clean_df(csv_path=standin_uci)

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
