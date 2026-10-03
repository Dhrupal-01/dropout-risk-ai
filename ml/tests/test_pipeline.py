"""
`python -m ml.pipeline run-all`: one tree check, one run id, steps in CLAUDE.md order, stop at the
first failure, renders only after every compute step and the run-id gate succeeded.
Steps are mocked (no data, no subprocesses).
"""

import json

import pytest

import ml.pipeline as pipeline
import ml.provenance as provenance
from ml.provenance import ProvenanceError, assert_consistent_provenance, build_provenance

COMMIT = "c" * 40
RUN_ID = "20261002T000000Z-ccccccc-abcdef"

CLAUDE_MD_ORDER = [
    ("ml.evaluation.run", ("--source", "uci")),
    ("ml.evaluation.run", ("--source", "oulad")),
    ("ml.simulation.estimate_parameters", ()),
    ("ml.data_pipeline.feature_engineering", ()),
    ("ml.validate_pipeline", ("--regenerate",)),
    ("ml.simulation.sensitivity", ()),
    ("ml.simulation.sim_to_real", ()),
    ("ml.fairness.run_all_audits", ("--force-rerun",)),
    ("scripts.render_benchmark_report", ()),
    ("scripts.render_fairness_report", ()),
    ("scripts.render_readme_metrics", ()),
    ("ml.simulation.estimate_parameters", ("--render-mapping",)),
    ("ml.simulation.render_simulation_doc", ()),
]


def _fake_git(status: str = ""):
    def fake(*args):
        if args[0] == "status":
            return status
        if args[0] == "rev-parse":
            return COMMIT
        return ""
    return fake


@pytest.fixture
def repo(monkeypatch, tmp_path):
    """Clean tree at COMMIT, fixed run id, no artifact redirection, gate reads tmp JSON."""
    for var in pipeline.ARTIFACT_PATH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(provenance, "_git", _fake_git())
    monkeypatch.setattr(pipeline, "new_run_id", lambda commit: RUN_ID)
    artifact = tmp_path / "artifact.json"
    monkeypatch.setattr(pipeline, "computed_artifact_paths", lambda: [artifact])
    return artifact


def _write_artifact(path, run_id=RUN_ID, commit=COMMIT):
    path.write_text(json.dumps({"provenance": {"run_id": run_id, "git_commit": commit}}))


def _recording_runner(monkeypatch, fail_at=None, on_compute_done=None):
    calls = []

    def run_step(step, env):
        calls.append((step, env))
        if step.id == fail_at:
            return 1, ["Traceback (most recent call last):", "AssertionError: boom"]
        if on_compute_done and step.id == "6":
            on_compute_done()
        return 0, []

    monkeypatch.setattr(pipeline, "run_step", run_step)
    return calls


def test_steps_follow_claude_md_order_with_renders_last():
    assert [(s.module, s.args) for s in pipeline.STEPS] == CLAUDE_MD_ORDER
    phases = [s.phase for s in pipeline.STEPS]
    assert phases == sorted(phases, key=lambda p: p != "compute")  # every compute before every render
    assert {s.phase for s in pipeline.STEPS} == {"compute", "render"}


def test_full_run_stamps_one_run_id_and_renders_after_compute(repo, monkeypatch, capsys):
    calls = _recording_runner(monkeypatch, on_compute_done=lambda: _write_artifact(repo))
    assert pipeline.run_all() == 0
    assert [c[0].id for c in calls] == [s.id for s in pipeline.STEPS]
    for _, env in calls:
        assert env[provenance.RUN_ID_ENV] == RUN_ID and env[provenance.RUN_COMMIT_ENV] == COMMIT
    assert "run-all completed" in capsys.readouterr().out


def test_dirty_code_tree_refuses_before_any_step(repo, monkeypatch):
    monkeypatch.setattr(provenance, "_git", _fake_git(" M ml/models/train.py"))
    calls = _recording_runner(monkeypatch)
    assert pipeline.main(["run-all"]) == 1
    assert calls == []


def test_uncommitted_generated_outputs_are_allowed(repo, monkeypatch):
    monkeypatch.setattr(provenance, "_git", _fake_git(" M ml/artifacts/model_metrics.json"))
    calls = _recording_runner(monkeypatch, on_compute_done=lambda: _write_artifact(repo))
    assert pipeline.run_all() == 0
    assert len(calls) == len(pipeline.STEPS)


def test_preset_artifact_redirect_refuses(repo, monkeypatch, tmp_path):
    monkeypatch.setenv("DROPOUTGUARD_ARTIFACTS_DIR", str(tmp_path))
    calls = _recording_runner(monkeypatch)
    assert pipeline.main(["run-all"]) == 1
    assert calls == []


def test_stops_at_first_failing_step_with_its_error(repo, monkeypatch, capsys):
    calls = _recording_runner(monkeypatch, fail_at="4b")
    assert pipeline.run_all() == 1
    ids = [c[0].id for c in calls]
    assert ids == ["1", "2", "3", "4a", "4b"]
    err = capsys.readouterr().err
    assert "step 4b `python -m ml.validate_pipeline --regenerate` failed (exit 1)" in err
    assert "AssertionError: boom" in err


def test_render_failure_stops_remaining_renders(repo, monkeypatch, capsys):
    calls = _recording_runner(monkeypatch, fail_at="7b", on_compute_done=lambda: _write_artifact(repo))
    assert pipeline.run_all() == 1
    assert [c[0].id for c in calls][-1] == "7b"
    assert "step 7b" in capsys.readouterr().err


@pytest.mark.parametrize("run_id, commit", [("other-run", COMMIT), (RUN_ID, "d" * 40), (None, COMMIT)])
def test_artifact_from_another_run_blocks_all_renders(repo, monkeypatch, capsys, run_id, commit):
    calls = _recording_runner(monkeypatch, on_compute_done=lambda: _write_artifact(repo, run_id, commit))
    assert pipeline.run_all() == 1
    assert all(c[0].phase == "compute" for c in calls)
    assert "not produced by this run" in capsys.readouterr().err


def test_missing_artifact_blocks_all_renders(repo, monkeypatch):
    calls = _recording_runner(monkeypatch)
    assert pipeline.run_all() == 1
    assert all(c[0].phase == "compute" for c in calls)


def test_dry_run_lists_every_step_and_runs_nothing(repo, monkeypatch, capsys):
    calls = _recording_runner(monkeypatch)
    assert pipeline.main(["run-all", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert calls == []
    for step in pipeline.STEPS:
        assert step.display in out


# ----------------------------------------------------------------------------- provenance

def test_build_provenance_records_run_id(monkeypatch):
    monkeypatch.setattr(provenance, "_git", _fake_git())
    monkeypatch.setenv(provenance.RUN_ID_ENV, RUN_ID)
    monkeypatch.setenv(provenance.RUN_COMMIT_ENV, COMMIT)
    prov = build_provenance({})
    assert prov["run_id"] == RUN_ID and prov["git_commit"] == COMMIT


def test_build_provenance_run_id_is_null_outside_a_pipeline_run(monkeypatch):
    monkeypatch.setattr(provenance, "_git", _fake_git())
    monkeypatch.delenv(provenance.RUN_ID_ENV, raising=False)
    monkeypatch.delenv(provenance.RUN_COMMIT_ENV, raising=False)
    assert build_provenance({})["run_id"] is None


def test_build_provenance_refuses_when_head_moved_during_run(monkeypatch):
    monkeypatch.setattr(provenance, "_git", _fake_git())
    monkeypatch.setenv(provenance.RUN_COMMIT_ENV, "d" * 40)
    with pytest.raises(ProvenanceError, match="HEAD moved during pipeline run"):
        build_provenance({})


def _art(commit, sha="1" * 64):
    return {"provenance": {"git_commit": commit, "input_files": {"data.csv": {"sha256": sha}}}}


def test_renderer_check_fails_on_mixed_commits():
    assert_consistent_provenance({"a.json": _art(COMMIT), "b.json": _art(COMMIT)})
    with pytest.raises(ProvenanceError, match="different git commits"):
        assert_consistent_provenance({"a.json": _art(COMMIT), "b.json": _art("d" * 40)})


def test_renderer_check_fails_on_mixed_checksums():
    with pytest.raises(ProvenanceError, match="different input files"):
        assert_consistent_provenance({"a.json": _art(COMMIT, "1" * 64), "b.json": _art(COMMIT, "2" * 64)})


def test_readme_metrics_render_fails_on_mixed_commits(tmp_path, monkeypatch):
    import scripts.render_readme_metrics as rm

    model, fairness = tmp_path / "model_metrics.json", tmp_path / "fairness_metrics.json"
    model.write_text(json.dumps({"recall_at_risk_minority": 0.5, **_art(COMMIT)}))
    fairness.write_text(json.dumps({"generator_sanity_check": {}, **_art("d" * 40)}))
    monkeypatch.setattr(rm, "MODEL_METRICS_PATH", model)
    monkeypatch.setattr(rm, "FAIRNESS_METRICS_PATH", fairness)
    with pytest.raises(ProvenanceError, match="different git commits"):
        rm.render_metrics_block()
