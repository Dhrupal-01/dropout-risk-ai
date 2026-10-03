"""
Real-data pipeline runner.

    python -m ml.pipeline run-all            # run every step, then render README/docs
    python -m ml.pipeline run-all --dry-run  # list the steps and preflight result; run nothing

One run:
1. Preflight, once: tracked code/config files must be unmodified (uncommitted generated artifacts
   and README generated blocks are allowed). Records HEAD and a new run id.
2. Compute steps in order (each a `python -m` subprocess carrying the run id and commit in its
   environment, so every provenance block records them). Stops at the first failing step.
3. Gate: every JSON artifact the compute steps own must record this run id and commit.
4. Render steps (README blocks, docs), only after all of the above succeeded. Renderers fail on
   mixed commits or input checksums. Stops at the first failing step.
"""

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ml import provenance
from ml.provenance import BASE_DIR, RUN_COMMIT_ENV, RUN_ID_ENV, DirtyTreeError, oulad_inputs, uci_inputs

ARTIFACT_PATH_ENV_VARS = ("DROPOUTGUARD_ARTIFACTS_DIR", "DROPOUTGUARD_PROCESSED_DATA_PATH")
OUTPUT_TAIL_LINES = 40


@dataclass(frozen=True)
class Step:
    id: str
    phase: str  # "compute" or "render"
    module: str
    args: Tuple[str, ...]
    outputs: str

    @property
    def argv(self) -> List[str]:
        return [sys.executable, "-m", self.module, *self.args]

    @property
    def display(self) -> str:
        return " ".join(["python", "-m", self.module, *self.args])


# Order follows the real-data pipeline in CLAUDE.md. Renders come last and run only if every
# compute step succeeded.
STEPS: List[Step] = [
    Step("1", "compute", "ml.evaluation.run", ("--source", "uci"), "ml/artifacts/benchmarks/uci_*.json"),
    Step("2", "compute", "ml.evaluation.run", ("--source", "oulad"), "ml/artifacts/benchmarks/oulad_snapshot_t*_withdrawn.json"),
    Step("3", "compute", "ml.simulation.estimate_parameters", (), "ml/simulation/estimated_parameters.json"),
    Step("4a", "compute", "ml.data_pipeline.feature_engineering", (), "data/processed/features.csv, data/processed/feature_metadata.json"),
    Step("4b", "compute", "ml.validate_pipeline", ("--regenerate",), "model artifacts (gitignored), ml/artifacts/model_metrics.json, ml/artifacts/fairness_metrics.json"),
    Step("4c", "compute", "ml.simulation.sensitivity", (), "ml/artifacts/simulation_sensitivity.json"),
    Step("5", "compute", "ml.simulation.sim_to_real", (), "ml/artifacts/benchmarks/sim_to_real.json"),
    Step("6", "compute", "ml.fairness.run_all_audits", ("--force-rerun",), "ml/artifacts/fairness/*.json, ml/artifacts/fairness_metrics.json"),
    Step("7a", "render", "scripts.render_benchmark_report", (), "docs/benchmarks.md, docs/figures/*, README.md BENCHMARKS block"),
    Step("7b", "render", "scripts.render_fairness_report", (), "docs/ethics_and_fairness.md"),
    Step("7c", "render", "scripts.render_readme_metrics", (), "README.md METRICS block"),
    Step("8a", "render", "ml.simulation.estimate_parameters", ("--render-mapping",), "docs/simulation_mapping.md"),
    Step("8b", "render", "ml.simulation.render_simulation_doc", (), "docs/simulation.md"),
]


class PipelineError(RuntimeError):
    pass


def new_run_id(commit: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{commit[:7]}-{secrets.token_hex(3)}"


def computed_artifact_paths(base_dir: Path = BASE_DIR) -> List[Path]:
    """JSON artifacts owned by the compute steps; each must record the current run id and commit."""
    artifacts = base_dir / "ml" / "artifacts"
    return sorted(
        list((artifacts / "benchmarks").glob("*.json"))
        + list((artifacts / "fairness").glob("*.json"))
        + [artifacts / "model_metrics.json", artifacts / "fairness_metrics.json", artifacts / "simulation_sensitivity.json",
           base_dir / "ml" / "simulation" / "estimated_parameters.json"]
    )


def check_run_artifacts(run_id: str, commit: str, paths: Sequence[Path]) -> None:
    """Raises PipelineError unless every artifact exists and records this run id and commit."""
    problems = []
    for path in paths:
        label = os.path.relpath(path, BASE_DIR)
        if not path.exists():
            problems.append(f"{label}: missing")
            continue
        prov = (json.loads(path.read_text(encoding="utf-8")) or {}).get("provenance") or {}
        if prov.get("run_id") != run_id or prov.get("git_commit") != commit:
            problems.append(
                f"{label}: run_id={prov.get('run_id')!r}, commit={str(prov.get('git_commit'))[:7]}"
            )
    if problems:
        raise PipelineError(
            f"Artifacts not produced by this run (run_id={run_id}, commit={commit[:7]}); nothing rendered:\n  "
            + "\n  ".join(problems)
        )


def preflight() -> str:
    """Single tree check at the start of a run. Returns HEAD. Raises on any problem."""
    preset = [k for k in ARTIFACT_PATH_ENV_VARS if k in os.environ]
    if preset:
        raise PipelineError(
            f"Refusing to run: {preset} redirect artifact paths away from the repository; unset them."
        )
    provenance.require_clean_tree(allow_dirty=False)
    commit = provenance._git("rev-parse", "HEAD")
    if not commit:
        raise PipelineError("Could not read HEAD; refusing to run without a known commit.")
    return commit


def run_step(step: Step, env: Dict[str, str]) -> Tuple[int, List[str]]:
    """Runs one step, streaming its merged output. Returns (exit code, last output lines)."""
    tail: deque = deque(maxlen=OUTPUT_TAIL_LINES)
    proc = subprocess.Popen(
        step.argv, cwd=BASE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        tail.append(line.rstrip("\n"))
    return proc.wait(), list(tail)


def _run_phase(steps: Sequence[Step], env: Dict[str, str], summary: List[Tuple[str, str, float]]) -> None:
    for step in steps:
        print(f"\n{'=' * 80}\n[pipeline] step {step.id} ({step.phase}): {step.display}\n{'=' * 80}", flush=True)
        start = time.time()
        code, tail = run_step(step, env)
        elapsed = time.time() - start
        if code != 0:
            summary.append((step.id, f"FAILED (exit {code})", elapsed))
            raise PipelineError(
                f"[✗] step {step.id} `{step.display}` failed (exit {code}). Later steps and renders were not run.\n"
                f"Last {len(tail)} output lines:\n" + "\n".join(f"    {line}" for line in tail)
            )
        summary.append((step.id, "ok", elapsed))


def run_all() -> int:
    commit = preflight()
    run_id = new_run_id(commit)
    _, generated = provenance.split_modified_files()
    print(f"[pipeline] commit {commit}\n[pipeline] run id {run_id}")
    print(f"[pipeline] uncommitted generated outputs at start: {generated or 'none'}")

    env = {**os.environ, RUN_ID_ENV: run_id, RUN_COMMIT_ENV: commit}
    summary: List[Tuple[str, str, float]] = []
    try:
        _run_phase([s for s in STEPS if s.phase == "compute"], env, summary)
        check_run_artifacts(run_id, commit, computed_artifact_paths())
        print(f"\n[pipeline] all compute artifacts record run id {run_id}; rendering README/docs")
        _run_phase([s for s in STEPS if s.phase == "render"], env, summary)
    except PipelineError as exc:
        _print_summary(run_id, commit, summary)
        print(f"\n{exc}", file=sys.stderr)
        return 1
    _print_summary(run_id, commit, summary)
    print("\n[pipeline] run-all completed: every step succeeded and README/docs were rendered.")
    return 0


def _print_summary(run_id: str, commit: str, summary: List[Tuple[str, str, float]]) -> None:
    print(f"\n[pipeline] summary (run id {run_id}, commit {commit[:7]})")
    for step_id, status, elapsed in summary:
        print(f"  step {step_id:<3} {status:<18} {elapsed:8.1f}s")


def dry_run() -> int:
    print("[dry-run] python -m ml.pipeline run-all — nothing is executed\n")
    preset = [k for k in ARTIFACT_PATH_ENV_VARS if k in os.environ]
    try:
        code, generated = provenance.split_modified_files()
        tree = (
            f"would REFUSE (modified code/config: {code})" if code
            else f"clean (uncommitted generated outputs allowed: {generated or 'none'})"
        )
    except DirtyTreeError as exc:
        tree = f"would REFUSE ({exc})"
    print(f"Preflight (checked once at start):")
    print(f"  HEAD:             {provenance._git('rev-parse', 'HEAD')}")
    print(f"  tree:             {tree}")
    print(f"  artifact env:     {'would REFUSE, set: ' + str(preset) if preset else 'not redirected'}")
    print(f"  run id:           <UTC timestamp>-<commit[:7]>-<6 hex>, generated at start, in every artifact's provenance")
    print("\nRequired raw inputs:")
    for name, path in {**uci_inputs(), **oulad_inputs()}.items():
        print(f"  {'present' if Path(path).exists() else 'MISSING':<8} {os.path.relpath(path, BASE_DIR)}")
    for phase, title in [("compute", "Compute steps (stop at first failure)"), ("render", "Render steps (only after all compute steps and the run-id gate succeed)")]:
        print(f"\n{title}:")
        for step in STEPS:
            if step.phase == phase:
                print(f"  {step.id:<3} {step.display}")
                print(f"      -> {step.outputs}")
        if phase == "compute":
            print("\nGate: every JSON below must record this run id and commit, else nothing is rendered:")
            for path in computed_artifact_paths():
                print(f"  {os.path.relpath(path, BASE_DIR)}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DropoutGuard real-data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-all", help="Run every pipeline step in order, then render README/docs")
    run.add_argument("--dry-run", action="store_true", help="List the steps and the preflight result; run nothing")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "run-all":
        if args.dry_run:
            return dry_run()
        try:
            return run_all()
        except (PipelineError, DirtyTreeError) as exc:
            print(f"[pipeline] {exc}", file=sys.stderr)
            return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
