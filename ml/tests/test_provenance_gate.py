"""
Artifact writers refuse a dirty working tree unless --allow-dirty is passed (recorded as
allow_dirty=true), renderers warn about artifacts generated from a dirty tree, and the verification
report hashes the installed package list (not the commit-dependent full pip freeze).
"""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

import ml.provenance as provenance
from ml.tests.artifact_checks import require_artifact
from ml.provenance import DirtyTreeError, build_provenance, dirty_artifact_warning, require_clean_tree

BASE_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = BASE_DIR / "ml" / "artifacts" / "benchmarks"
BENCHMARK_COMMAND = "python -m ml.evaluation.run --source uci` and `--source oulad"


def _fake_git(dirty: bool):
    def fake(*args):
        if args[0] == "status":
            return " M ml/some_module.py" if dirty else ""
        if args[0] == "rev-parse":
            return "0" * 40
        return ""
    return fake


@pytest.fixture
def dirty_tree(monkeypatch):
    monkeypatch.setattr(provenance, "_git", _fake_git(dirty=True))


@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(provenance, "_git", _fake_git(dirty=False))


# ----------------------------------------------------------------------------- gate

def test_require_clean_tree_refuses_dirty_tree(dirty_tree):
    with pytest.raises(DirtyTreeError, match="ml/some_module.py"):
        require_clean_tree(allow_dirty=False)


def test_require_clean_tree_allows_dirty_tree_when_flagged(dirty_tree):
    assert require_clean_tree(allow_dirty=True) is True


def test_build_provenance_records_allow_dirty(dirty_tree):
    prov = build_provenance({}, allow_dirty=True)
    assert prov["git_dirty"] is True and prov["allow_dirty"] is True
    with pytest.raises(DirtyTreeError):
        build_provenance({}, allow_dirty=False)


def test_build_provenance_on_clean_tree(clean_tree):
    prov = build_provenance({})
    assert prov["git_dirty"] is False and prov["allow_dirty"] is False


def _git_with_status(status: str, readme_head: str = ""):
    def fake(*args):
        if args[0] == "status":
            return status
        if args[0] == "rev-parse":
            return "0" * 40
        if args[0] == "show":
            return readme_head
        return ""
    return fake


def test_uncommitted_generated_outputs_do_not_block_or_mark_dirty(monkeypatch):
    monkeypatch.setattr(provenance, "_git", _git_with_status(
        " M ml/artifacts/benchmarks/uci_full_primary.json\n M docs/benchmarks.md\n M ml/simulation/estimated_parameters.json"
    ))
    assert require_clean_tree(allow_dirty=False) is False
    prov = build_provenance({})
    assert prov["git_dirty"] is False and prov["allow_dirty"] is False
    assert prov["uncommitted_generated_outputs"] == [
        "docs/benchmarks.md", "ml/artifacts/benchmarks/uci_full_primary.json", "ml/simulation/estimated_parameters.json"
    ]


def test_code_change_still_blocks_alongside_generated_outputs(monkeypatch):
    monkeypatch.setattr(provenance, "_git", _git_with_status(" M ml/artifacts/model_metrics.json\n M ml/models/train.py"))
    with pytest.raises(DirtyTreeError) as excinfo:
        require_clean_tree(allow_dirty=False)
    assert "ml/models/train.py" in str(excinfo.value)
    assert "model_metrics.json" not in str(excinfo.value)


README_HEAD = "# Title\n<!-- METRICS:START -->\nold metrics\n<!-- METRICS:END -->\ntext\n<!-- BENCHMARKS:START -->\nold\n<!-- BENCHMARKS:END -->\n"


@pytest.mark.parametrize("working, generated", [
    (README_HEAD.replace("old metrics", "new metrics").replace("\nold\n", "\nnew\n"), True),
    (README_HEAD.replace("text", "hand-edited text"), False),
    (README_HEAD.replace("old metrics", "new metrics").replace("# Title", "# Renamed"), False),
])
def test_readme_counts_as_generated_only_when_changes_stay_inside_marker_blocks(tmp_path, monkeypatch, working, generated):
    (tmp_path / "README.md").write_text(working)
    monkeypatch.setattr(provenance, "BASE_DIR", tmp_path)
    monkeypatch.setattr(provenance, "_git", _git_with_status(" M README.md", readme_head=README_HEAD))
    assert provenance.is_generated_output("README.md") is generated
    if generated:
        assert require_clean_tree(allow_dirty=False) is False
    else:
        with pytest.raises(DirtyTreeError, match="README.md"):
            require_clean_tree(allow_dirty=False)


WRITERS = [
    ("ml.evaluation.run", "run_uci_benchmark_suite"),
    ("ml.evaluation.run", "run_oulad_benchmark_suite"),
    ("ml.simulation.sim_to_real", "run_sim_to_real_benchmark"),
    ("ml.fairness.run_all_audits", "run_all_fairness_audits"),
    ("ml.models.fairness_audit", "run_comprehensive_fairness_audit"),
    ("ml.models.train", "train_pipeline"),
    ("ml.models.calibrate", "run_calibration_pipeline"),
    ("ml.validate_pipeline", "run_pipeline_validation"),
    ("ml.simulation.estimate_parameters", "estimate_all_parameters"),
    ("ml.simulation.sensitivity", "run_sensitivity_analysis"),
]


@pytest.mark.parametrize("module_name, func_name", WRITERS)
def test_every_writer_refuses_dirty_tree_before_loading_data(dirty_tree, monkeypatch, module_name, func_name):
    def data_loaded(*args, **kwargs):
        raise AssertionError(f"{module_name}.{func_name} loaded data before the dirty-tree gate")

    # Every writer reads its inputs through pandas.read_csv or joblib.load
    monkeypatch.setattr(pd, "read_csv", data_loaded)
    monkeypatch.setattr(joblib, "load", data_loaded)

    writer = getattr(importlib.import_module(module_name), func_name)
    with pytest.raises(DirtyTreeError):
        writer(allow_dirty=False)


@pytest.mark.parametrize("module_name", sorted({m for m, _ in WRITERS}))
def test_every_writer_cli_accepts_allow_dirty(module_name):
    parser = importlib.import_module(module_name).build_arg_parser()
    assert parser.parse_args(["--allow-dirty"]).allow_dirty is True
    assert parser.parse_args([]).allow_dirty is False


# ----------------------------------------------------------------------------- renderer warnings

def test_dirty_artifact_warning_lists_only_dirty_artifacts():
    clean = {"provenance": {"git_dirty": False, "git_commit": "a" * 40, "allow_dirty": False}}
    dirty = {"provenance": {"git_dirty": True, "git_commit": "b" * 40, "allow_dirty": True}}
    legacy = {"provenance": {"git_dirty": True, "git_commit": "c" * 40}}

    assert dirty_artifact_warning({"clean.json": clean}) is None
    warning = dirty_artifact_warning({"clean.json": clean, "dirty.json": dirty, "legacy.json": legacy})
    assert warning.startswith("> **Warning")
    assert "`dirty.json` (commit `bbbbbbb`, allow_dirty=true)" in warning
    assert "`legacy.json` (commit `ccccccc`, allow_dirty=not recorded)" in warning
    assert "clean.json" not in warning


@pytest.mark.artifacts
def test_readme_benchmarks_block_shows_warning_only_for_dirty_artifacts(tmp_path, monkeypatch):
    require_artifact(any(BENCHMARK_DIR.glob("uci_*_primary.json")), "ml/artifacts/benchmarks/uci_*_primary.json", BENCHMARK_COMMAND)
    import scripts.render_readme_benchmarks as rb

    names = [p.name for p in BENCHMARK_DIR.glob("uci_*_primary.json")] + [p.name for p in BENCHMARK_DIR.glob("oulad_snapshot_t*_withdrawn.json")]

    def write(dirty_name):
        for name in names:
            art = json.loads((BENCHMARK_DIR / name).read_text())
            art["provenance"]["git_dirty"] = name == dirty_name
            art["provenance"]["allow_dirty"] = name == dirty_name
            (tmp_path / name).write_text(json.dumps(art))

    monkeypatch.setattr(rb, "BENCHMARK_DIR", tmp_path)
    write(dirty_name=None)
    assert "Warning" not in rb.render_benchmarks_block()

    write(dirty_name=names[0])
    block = rb.render_benchmarks_block()
    assert block.startswith("> **Warning") and f"`{names[0]}`" in block


# ----------------------------------------------------------------------------- freeze hash

FREEZE = "-e git+https://example.com/repo.git@{sha}#egg=dropoutguard\nnumpy==2.5.2\npandas==3.0.5\n"
LOCK = "# Pinned lock file\n# Generated with: pip freeze --exclude-editable\npandas==3.0.5\nnumpy==2.5.2\n"


def test_package_fingerprint_ignores_editable_line_and_lock_comments():
    from scripts.render_verification_report import package_fingerprint

    assert package_fingerprint(FREEZE.format(sha="1" * 40)) == package_fingerprint(FREEZE.format(sha="2" * 40))
    assert package_fingerprint(FREEZE.format(sha="1" * 40)) == package_fingerprint(LOCK)
    assert package_fingerprint(LOCK) != package_fingerprint(LOCK.replace("numpy==2.5.2", "numpy==2.5.3"))


@pytest.mark.parametrize("lock_text, expected", [(LOCK, "true"), (LOCK.replace("pandas==3.0.5", "pandas==3.0.4"), "false")])
def test_build_metadata_hashes_exclude_editable_freeze(tmp_path, monkeypatch, lock_text, expected):
    import scripts.render_verification_report as rv

    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        out = {"rev-parse": "0" * 40, "status": "", "freeze": FREEZE.format(sha="3" * 40)}
        key = next((k for k in out if k in cmd), None)
        return SimpleNamespace(stdout=out.get(key, ""), returncode=0)

    lock = tmp_path / "requirements.lock"
    lock.write_text(lock_text)
    monkeypatch.setattr(rv, "_run", fake_run)
    monkeypatch.setattr(rv, "LOCK_PATH", lock)

    meta = rv.build_metadata()
    assert any(cmd[-2:] == ["freeze", "--exclude-editable"] for cmd in calls)
    assert meta["pip freeze --exclude-editable sha256"] == rv.package_fingerprint(FREEZE.format(sha="9" * 40))
    assert meta["matches requirements.lock"] == expected


@pytest.mark.artifacts
def test_readme_benchmarks_block_fails_on_mixed_commits(tmp_path, monkeypatch):
    require_artifact(any(BENCHMARK_DIR.glob("uci_*_primary.json")), "ml/artifacts/benchmarks/uci_*_primary.json", BENCHMARK_COMMAND)
    import scripts.render_readme_benchmarks as rb

    names = [p.name for p in BENCHMARK_DIR.glob("uci_*_primary.json")] + [p.name for p in BENCHMARK_DIR.glob("oulad_snapshot_t*_withdrawn.json")]
    for i, name in enumerate(names):
        art = json.loads((BENCHMARK_DIR / name).read_text())
        art["provenance"]["git_commit"] = ("a" if i == 0 else "b") * 40
        (tmp_path / name).write_text(json.dumps(art))
    monkeypatch.setattr(rb, "BENCHMARK_DIR", tmp_path)
    with pytest.raises(provenance.ProvenanceError, match="different git commits"):
        rb.render_benchmarks_block()

