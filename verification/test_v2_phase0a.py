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
        """V2.4: ml/artifacts/feature_names.json identical to BASELINE, except for the contract changes
        listed in removed_since_baseline (each an owner decision). Order is otherwise unchanged."""
        # 2026-10-08, owner decision: age is a protected attribute (ml/fairness/attributes.py), not a model input.
        removed_since_baseline = ["age"]

        cmd = ["git", "show", f"{BASELINE_COMMIT}:ml/artifacts/feature_names.json"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        baseline_json = json.loads(res.stdout)
        assert all(name in baseline_json for name in removed_since_baseline)
        expected = [name for name in baseline_json if name not in removed_since_baseline]

        current_path = PROJECT_ROOT / "ml" / "artifacts" / "feature_names.json"
        with open(current_path, "r") as f:
            current_json = json.load(f)

        assert current_json == expected, "feature_names.json has drifted from BASELINE (minus the decided removals)!"

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
# Any number with a %, pp or ms unit (LaTeX `\%` and `\text{ pp}` included), and any decimal between
# 0 and 1 with 2+ digits. Catches values TYPED_METRIC misses because the metric name is elsewhere
# (a table header, a chart label).
UNIT_NUMBER = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s*(?:\\text\{\s*)?\\?(?:%|pp\b|ms\b)"
    r"|(?<![\w.])0\.\d{2,}(?![\d.])"
)
# Exact lines (stripped) that may carry such a number, with the reason. Unused entries fail the test.
_R_ATT = "the statutory 75% attendance rule (ml.config.ATTENDANCE_THRESHOLD)"
_R_FORMULA = "formula constant copied from ml/data_pipeline/"
_R_TIER = "configured risk-tier thresholds (ml/config.py DEFAULT_RISK_THRESHOLD_LOW/HIGH)"
_R_OUT = "outcome band (backend/app/services/intervention_service.py RISK_DELTA_THRESHOLD)"
_R_EX = "example API response value from the simulated cohort, not a metric"
ALLOWED_NUMBER_LINES = {
    ('README.md', '1. **Attendance & Discipline**: Overall 3-month attendance %, recent-month trajectory, consecutive absence streaks, and the mandatory 75% AICTE/UGC debarment rule.'): _R_ATT,
    ('docs/frontend_api_handover.md', '| `Low` | `p < 0.33` | No urgent action; routine monitoring |'): _R_TIER,
    ('docs/frontend_api_handover.md', '| `Medium` | `0.33 ≤ p ≤ 0.66` | Watch list; early supportive outreach |'): _R_TIER,
    ('docs/frontend_api_handover.md', '| `High` | `p > 0.66` | Priority mentor outreach |'): _R_TIER,
    ('docs/frontend_api_handover.md', '| `IMPROVED` | delta > +0.05 |'): _R_OUT,
    ('docs/frontend_api_handover.md', '| `NO_CHANGE` | −0.05 ≤ delta ≤ +0.05 |'): _R_OUT,
    ('docs/frontend_api_handover.md', '| `DETERIORATED` | delta < −0.05 |'): _R_OUT,
    ('docs/frontend_api_handover.md', '"calibrated_risk_probability": 0.9257,'): _R_EX,
    ('docs/frontend_api_handover.md', '"risk_probability": 0.9257,'): _R_EX,
    ('docs/frontend_api_handover.md', '"plain_language_explanation": "Low recent-month (Month 3) attendance (36.0%) is increasing risk by 26.4 percentage points."'): _R_EX,
    ('docs/frontend_api_handover.md', '"current_risk_probability": 0.9257,'): _R_EX,
    ('docs/frontend_api_handover.md', '"rationale": "Triggered by Current Month Attendance (+26.4% risk impact): ..."'): _R_EX,
    ('docs/frontend_api_handover.md', '"current_risk_prob": 0.9257, "current_risk_tier": "High",'): _R_EX,
    ('docs/frontend_api_handover.md', '"projected_risk_prob": 0.0436, "projected_risk_tier": "Low",'): _R_EX,
    ('docs/frontend_api_handover.md', '"plain_language_action": "Raise class attendance from 44.2% to 80.0% (+35.8% via regular attendance & lab make-up)."'): _R_EX,
    ('docs/frontend_api_handover.md', '"baseline_risk_probability": 0.9257,'): _R_EX,
    ('docs/frontend_api_handover.md', '"post_intervention_risk_probability": 0.31,'): _R_EX,
    ('docs/frontend_api_handover.md', '"risk_delta": 0.6157,'): _R_EX,
    ('docs/frontend_api_handover.md', '"notes": "Mentor call scheduled | Attendance recovered to 79%",'): _R_EX,
    ('docs/frontend_api_handover.md', '"risk_probability": 0.9258,'): _R_EX,
    ('docs/data_dictionary.md', '| `behavioral_disengagement_index` | engineered | `0.35 * (1 - clip(lms_logins_per_week / 10)) + 0.35 * clip(assignment_submission_lag_days / 7) + 0.30 * clip(days_since_last_lms_activity / 30)`, rounded to 3 dp | `feature_engineering.py:133-136` |'): _R_FORMULA + 'feature_engineering.py:136',
    ('docs/data_dictionary.md', '| `financial_stress_index` | engineered | `0.40 * (3 - income_slab_idx) / 3 + 0.40 * clip(fee_payment_delay_days / 60) + 0.20 * (1 - has_scholarship)`, rounded to 3 dp | `feature_engineering.py:152-155` |'): _R_FORMULA + 'feature_engineering.py:155',
}


def _tracked(*paths):
    out = subprocess.run(["git", "ls-files", *paths], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.split()


def _markup_text(text):
    """Text content of an .svg/.html file: tags, <style> and <script> are blanked, newlines kept."""
    blank = lambda m: "\n" * m.group(0).count("\n")  # noqa: E731
    text = re.sub(r"<(style|script)\b.*?</\1>", blank, text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<[^>]*>", blank, text)


def _hand_written_docs():
    """README.md (outside its generated blocks) and docs/, minus generated docs. Only files from
    `git ls-files` are scanned, so local-only (ignored) files on a developer's disk are never checked."""
    from ml.provenance import strip_generated_readme_blocks

    for rel in _tracked("README.md", "docs"):
        if rel in GENERATED_DOCS or not rel.endswith((".md", ".txt", ".html", ".svg")):
            continue
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        yield rel, strip_generated_readme_blocks(text) if rel == "README.md" else text


def _unit_number_hits():
    """(path, line number, stripped line, matches) for every hand-written line with a UNIT_NUMBER match."""
    for rel, text in _hand_written_docs():
        if rel.endswith((".svg", ".html")):
            text = _markup_text(text)
        for i, line in enumerate(text.splitlines(), 1):
            found = [m.group(0) for m in UNIT_NUMBER.finditer(line)]
            if found:
                yield rel, i, line.strip(), found


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

    def test_v2_8_unit_number_pattern(self):
        for bad in ["│ 845 F / 1155 M      │ 84.39% vs 85.57%  │ 1.18 percentage pts  │", "• Gender Gap: 1.18 pp",
                    "FNR Gap <= 1.36 pp Verified", "• Sub-50ms Inference", "risk to **$49.4\\%$ (Medium)**",
                    "gaps are $\\le 1.36\\text{ pp}$", '"calibrated_risk_probability": 0.9257,', "takes ~150 ms"]:
            assert UNIT_NUMBER.search(bad), bad
        for ok in ["Python 3.12.10", "fill #fab219", "Held-Out Test Set $N=300$", "`top_k` (default `5`, 1–37)",
                   "version 0.12.3", "probability 0.5", "INT_ATT_01", "pool_recycle=280s"]:
            assert not UNIT_NUMBER.search(ok), ok

    def test_v2_8_no_unit_numbers_in_hand_written_docs(self):
        """No number with a %, pp or ms unit and no 0.xx decimal in hand-written docs (.svg text included),
        except exact lines listed in ALLOWED_NUMBER_LINES with a reason. Allowlist entries that no longer
        match a line fail too, so the list cannot go stale."""
        hits, used = [], set()
        for rel, i, line, found in _unit_number_hits():
            if (rel, line) in ALLOWED_NUMBER_LINES:
                used.add((rel, line))
            else:
                hits.append(f"{rel}:{i}: {found} in {line[:160]!r}")
        stale = [f"{rel}: {line[:160]!r}" for rel, line in ALLOWED_NUMBER_LINES if (rel, line) not in used]
        assert not hits, "Typed numbers in hand-written docs (allowlist the exact line with a reason if legitimate):\n" + "\n".join(hits)
        assert not stale, "ALLOWED_NUMBER_LINES entries that match no line:\n" + "\n".join(stale)

    def test_v2_9_relative_links_resolve(self):
        """Every relative link in tracked README.md and docs/*.md points at a tracked file or directory (a link
        to a local-only file resolves on one disk but is broken on GitHub); no file:// links."""
        from urllib.parse import unquote

        link = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
        tracked = {(PROJECT_ROOT / f).resolve() for f in _tracked()}
        tracked_dirs = {d for f in tracked for d in f.parents}
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
                    if resolved not in tracked and resolved not in tracked_dirs:
                        broken.append(f"{rel}:{i}: {target}")
        assert checked > 0
        assert not broken, "Broken relative links:\n" + "\n".join(broken)

    def test_v2_10_api_handover_documents_every_route(self):
        """V2.10: Every route in app.openapi() appears in docs/frontend_api_handover.md as `METHOD /path`,
        and every query parameter of every route appears in it as `name`."""
        from backend.app.main import app

        doc = (PROJECT_ROOT / "docs" / "frontend_api_handover.md").read_text(encoding="utf-8")
        methods = {"get", "post", "put", "patch", "delete"}
        operations = [(m.upper(), p, op) for p, ops in app.openapi()["paths"].items() for m, op in ops.items() if m in methods]
        assert operations
        missing = [f"{m} {p}" for m, p, _ in operations if not re.search(rf"\b{m} {re.escape(p)}(?![\w/{{])", doc)]
        assert not missing, "Routes missing from docs/frontend_api_handover.md:\n" + "\n".join(missing)

        query_params = [(m, p, q["name"]) for m, p, op in operations for q in op.get("parameters", []) if q.get("in") == "query"]
        assert query_params
        missing = [f"{m} {p}: `{name}`" for m, p, name in query_params if f"`{name}`" not in doc]
        assert not missing, "Query parameters missing from docs/frontend_api_handover.md:\n" + "\n".join(missing)

    def test_v2_11_data_dictionary_matches_feature_contract(self):
        """V2.11: docs/data_dictionary.md rows with role `raw` are exactly RAW_FEATURE_COLUMNS, rows with role
        `engineered` exactly ENGINEERED_FEATURE_COLUMNS, and together exactly feature_names.json. Every column
        of a generated cohort has exactly one row, and `age` is `audit-only`."""
        from backend.app.services.ml_service import ENGINEERED_FEATURE_COLUMNS, RAW_FEATURE_COLUMNS
        from ml.data_pipeline.feature_engineering import build_engineered_features
        from ml.data_pipeline.generate_synthetic_indian import generate_indian_student_cohort

        doc = (PROJECT_ROOT / "docs" / "data_dictionary.md").read_text(encoding="utf-8")
        rows = re.findall(r"^\| `([a-z0-9_]+)` \| (raw|engineered|audit-only|excluded|input-only|label|identifier) \|", doc, re.M)
        names = [name for name, _ in rows]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        assert not duplicates, f"Columns listed more than once: {duplicates}"
        role = dict(rows)

        raw = [n for n, r in rows if r == "raw"]
        engineered = [n for n, r in rows if r == "engineered"]
        assert sorted(raw) == sorted(RAW_FEATURE_COLUMNS)
        assert sorted(engineered) == sorted(ENGINEERED_FEATURE_COLUMNS)
        feature_names = json.loads((PROJECT_ROOT / "ml" / "artifacts" / "feature_names.json").read_text())
        assert sorted(raw + engineered) == sorted(feature_names)
        assert role.get("age") == "audit-only"

        cohort = build_engineered_features(generate_indian_student_cohort(n_students=50, seed=7, output_path=None))
        assert sorted(role) == sorted(cohort.columns), (
            f"In cohort, not in dictionary: {sorted(set(cohort.columns) - set(role))}; "
            f"in dictionary, not in cohort: {sorted(set(role) - set(cohort.columns))}"
        )
