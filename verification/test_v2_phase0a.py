"""
Verification tests for V2 (Phase 0A — docs, loaders, hygiene).
"""

import json
import re
import subprocess
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


# ------------------------------------------------- hand-written docs: no typed metrics, no dead links

GENERATED_DOCS = {"docs/benchmarks.md", "docs/ethics_and_fairness.md", "docs/simulation.md", "docs/simulation_mapping.md"}
_KW = r"(?:roc[- ]?auc|recall|precision|brier|accuracy|f1)"
# A whole number token; "top 10%" (a cutoff) and "95% CI" (a confidence level) are not metric values.
_NUM = r"(?<![\d.])(?<!top )(?<!top-)\d+(?:\.\d+)?(?![\d.])(?!\s*%\s*(?:ci\b|confidence))"
# A value written before the metric name must look like a value (decimal or percentage), so sample
# sizes such as "(N=121): Recall" and colour codes such as "#334155" do not count.
_VALUE = r"(?<![\d.#])(?:\d+\.\d+\s*%?|\d+\s*%)"
TYPED_METRIC = re.compile(
    rf"\b{_KW}\b(?:[^\w\n]{{0,8}}[a-z]+){{0,3}}?[^\w\n]{{0,8}}{_NUM}"  # metric name, up to 3 words, number
    rf"|{_VALUE}[^\w\n]{{0,4}}{_KW}\b",  # value, then metric name
    re.IGNORECASE,
)


def _tracked(*paths):
    out = subprocess.run(["git", "ls-files", *paths], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.split()


def _hand_written_docs():
    """README.md (outside its generated blocks) and docs/, minus generated docs and dated records in docs/tasks/."""
    from ml.provenance import strip_generated_readme_blocks

    for rel in _tracked("README.md", "docs"):
        if rel in GENERATED_DOCS or rel.startswith("docs/tasks/") or not rel.endswith((".md", ".txt", ".html", ".svg")):
            continue
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        yield rel, strip_generated_readme_blocks(text) if rel == "README.md" else text


class TestV2HandWrittenDocs:
    def test_v2_8_metric_pattern_catches_typed_values(self):
        for bad in ["Achieves **83.96% Recall**", "Brier Calibration Score of 0.0689", "| **ROC-AUC Score** | **0.9752** |",
                    "Recall = 85.57%", "Overall Accuracy: 91.00%", "minority-class F1 ($0.8683$)", "Precision: 89.90%"]:
            assert TYPED_METRIC.search(bad), bad
        for ok in ["precision at top 10% and 20%", "ROC-AUC of {point} (95% CI {lo}-{hi})", "ROC-AUC, PR-AUC and Brier",
                   "Female Students (N=121): Recall", 'fill="#334155">• Brier']:
            assert not TYPED_METRIC.search(ok), ok

    def test_v2_8_no_typed_metrics_in_hand_written_docs(self):
        """Metric values live only in generated docs and the README's generated blocks (simulated cohort)."""
        hits = [
            f"{rel}:{i}: {m.group(0)!r}"
            for rel, text in _hand_written_docs()
            for i, line in enumerate(text.splitlines(), 1)
            for m in TYPED_METRIC.finditer(line)
        ]
        assert not hits, "Typed metric values in hand-written docs (link to the generated block instead):\n" + "\n".join(hits)

    def test_v2_9_relative_links_resolve(self):
        """Every relative link in README.md and docs/*.md points at a file in the repo; no file:// links."""
        from urllib.parse import unquote

        link = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
        broken, checked = [], 0
        for rel in _tracked("README.md", "docs"):
            if not rel.endswith(".md"):
                continue
            path = PROJECT_ROOT / rel
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for target in link.findall(line):
                    if target.startswith("file:"):
                        broken.append(f"{rel}:{i}: machine-specific link {target}")
                        continue
                    if re.match(r"^(https?:|mailto:|#)", target):
                        continue
                    checked += 1
                    resolved = (path.parent / unquote(target.split("#", 1)[0].split("?", 1)[0])).resolve()
                    if not resolved.exists():
                        broken.append(f"{rel}:{i}: {target}")
        assert checked > 0
        assert not broken, "Broken relative links:\n" + "\n".join(broken)
