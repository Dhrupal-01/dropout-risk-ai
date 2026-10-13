"""
frontend/src/content/evidence.json is the only source of model results on the public site. These tests
check that it is exactly what scripts/export_site_evidence.py produces from the committed benchmark
artifacts, and, independently of the exporter, that every exported value matches its source artifact.
"""

import json
import shutil
from pathlib import Path

import pytest

from ml.provenance import ProvenanceError
from ml.tests.artifact_checks import require_artifact
from scripts.export_site_evidence import (
    BENCHMARK_DIR,
    EVIDENCE_PATH,
    OULAD_FILES,
    UCI_FILES,
    build_evidence,
    render,
)

EXPORT_COMMAND = "python -m scripts.export_site_evidence --include-oulad"


def _committed_evidence() -> dict:
    require_artifact(EVIDENCE_PATH.exists(), str(EVIDENCE_PATH), EXPORT_COMMAND)
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


@pytest.mark.artifacts
def test_evidence_json_matches_fresh_export():
    committed = _committed_evidence()
    include_oulad = committed["options"]["include_oulad"]
    fresh = render(build_evidence(BENCHMARK_DIR, include_oulad=include_oulad))
    assert EVIDENCE_PATH.read_text(encoding="utf-8") == fresh, (
        f"{EVIDENCE_PATH.name} is stale; regenerate it with `{EXPORT_COMMAND}`"
    )


@pytest.mark.artifacts
def test_ci_level_comes_from_the_harness_setting():
    from ml.evaluation.harness import CI_LEVEL_PCT

    assert _committed_evidence()["ci_level_pct"] == CI_LEVEL_PCT


@pytest.mark.artifacts
def test_every_exported_value_matches_its_source_artifact():
    committed = _committed_evidence()
    names = UCI_FILES + (OULAD_FILES if committed["options"]["include_oulad"] else [])
    sources = {name: json.loads((BENCHMARK_DIR / name).read_text(encoding="utf-8")) for name in names}

    # Every (file, split, model) in the selected artifacts, read straight from the artifacts.
    expected_keys = {
        (art["dataset"], art["feature_set"], art.get("snapshot_t"), split, model)
        for art in sources.values()
        for split, models in art["results"].items()
        for model in models
    }
    exported_keys = [
        (row["dataset"], row["feature_set"], row.get("snapshot_t"), row["split"], row["model"])
        for row in committed["results"]
    ]
    assert len(exported_keys) == len(set(exported_keys)), "duplicate result rows in evidence.json"
    assert set(exported_keys) == expected_keys

    by_feature_set = {(art["dataset"], art["feature_set"]): art for art in sources.values()}
    for row in committed["results"]:
        art = by_feature_set[(row["dataset"], row["feature_set"])]
        res = art["results"][row["split"]][row["model"]]
        assert row["metrics"] == res["metrics"]
        assert row["n_evaluated"] == res["n_evaluated"]
        assert row["n_samples"] == art["n_samples"]
        assert row["prevalence"] == art["prevalence"]
        assert row.get("per_fold_summary") == res.get("per_fold", {}).get("summary")

    provenance = {art["provenance"]["run_id"] for art in sources.values()}
    assert provenance == {committed["provenance"]["run_id"]}
    assert {art["provenance"]["git_commit"] for art in sources.values()} == {committed["provenance"]["git_commit"]}


def _copy_artifacts(tmp_path: Path, names) -> Path:
    for name in names:
        shutil.copy(BENCHMARK_DIR / name, tmp_path / name)
    return tmp_path


def _edit_provenance(path: Path, **changes) -> None:
    art = json.loads(path.read_text(encoding="utf-8"))
    art["provenance"].update(changes)
    path.write_text(json.dumps(art), encoding="utf-8")


@pytest.mark.artifacts
def test_export_refuses_artifacts_from_different_commits(tmp_path):
    bench = _copy_artifacts(tmp_path, UCI_FILES)
    _edit_provenance(bench / UCI_FILES[1], git_commit="0" * 40)
    with pytest.raises(ProvenanceError, match="different git commits"):
        build_evidence(bench, include_oulad=False)


@pytest.mark.artifacts
def test_export_refuses_artifacts_from_a_dirty_tree(tmp_path):
    bench = _copy_artifacts(tmp_path, UCI_FILES)
    _edit_provenance(bench / UCI_FILES[0], git_dirty=True)
    with pytest.raises(ProvenanceError, match="git_dirty=true"):
        build_evidence(bench, include_oulad=False)


@pytest.mark.artifacts
def test_export_fails_loudly_when_an_artifact_is_missing(tmp_path):
    bench = _copy_artifacts(tmp_path, UCI_FILES[:-1])
    with pytest.raises(FileNotFoundError, match=UCI_FILES[-1]):
        build_evidence(bench, include_oulad=False)


@pytest.mark.artifacts
def test_oulad_rows_only_with_include_oulad(tmp_path):
    bench = _copy_artifacts(tmp_path, UCI_FILES + OULAD_FILES)
    without = build_evidence(bench, include_oulad=False)
    with_oulad = build_evidence(bench, include_oulad=True)
    assert {row["dataset"] for row in without["results"]} == {"uci_697"}
    assert {row["dataset"] for row in with_oulad["results"]} == {"uci_697", "oulad_349"}
    assert set(without["provenance"]["source_files"]) == set(UCI_FILES)
