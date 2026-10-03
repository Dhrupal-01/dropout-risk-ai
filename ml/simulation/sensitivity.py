"""
Sensitivity of the simulated-cohort model to `cohort_metadata.target_base_rate`.

There is no published national higher-education dropout rate for India, so the simulated base
rate is an assumption. This regenerates the cohort and retrains the model at several base rates
and reports held-out ROC-AUC, PR-AUC and Brier score with 95% bootstrap CIs.

Each rate runs in a worker subprocess whose artifact and feature paths point at a temp dir, so the
deployed model in ml/artifacts/ is never touched. The parent writes
ml/artifacts/simulation_sensitivity.json (with provenance); docs/simulation.md renders it.

    python -m ml.simulation.sensitivity            # all rates, writes the JSON
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

BASE_DIR = Path(__file__).resolve().parents[2]
ASSUMPTIONS_PATH = BASE_DIR / "ml" / "simulation" / "assumptions.yaml"
ESTIMATED_PARAMETERS_PATH = BASE_DIR / "ml" / "simulation" / "estimated_parameters.json"
OUTPUT_PATH = BASE_DIR / "ml" / "artifacts" / "simulation_sensitivity.json"

# Owner decision: the deployed value and two lower alternatives.
RATES = (0.15, 0.25, 0.355)
# Same cohort size and held-out split as the deployed model (ml.validate_pipeline / ml.models.train).
N_STUDENTS = 2000
TEST_SIZE = 0.15
N_BOOTSTRAPS = 1000
REPORTED_METRICS = ("roc_auc", "pr_auc", "brier_score")


def run_worker(rate: float, result_path: Path, allow_dirty: bool = False) -> Dict[str, Any]:
    """Inside a subprocess whose DROPOUTGUARD_* paths point at a temp dir: generate, train, evaluate."""
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from ml.config import FEATURE_NAMES_PATH, MODEL_ARTIFACT_PATH, PROCESSED_DATA_PATH, RANDOM_SEED
    from ml.data_pipeline.feature_engineering import generate_processed_feature_dataset
    from ml.evaluation.harness import compute_bootstrap_cis
    from ml.models.calibrate import predict_student_risk, run_calibration_pipeline
    from ml.models.train import train_pipeline

    generate_processed_feature_dataset(
        n_students=N_STUDENTS, seed=RANDOM_SEED, output_path=PROCESSED_DATA_PATH, target_base_rate=rate
    )
    train_pipeline(allow_dirty=allow_dirty)
    run_calibration_pipeline(allow_dirty=allow_dirty)

    model = joblib.load(MODEL_ARTIFACT_PATH)
    feature_names = json.loads(Path(FEATURE_NAMES_PATH).read_text(encoding="utf-8"))
    df = pd.read_csv(PROCESSED_DATA_PATH)
    y = df["is_dropout"].to_numpy()
    _, X_test, _, y_test = train_test_split(
        df[feature_names], y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    probs, _ = predict_student_risk(model, X_test)
    cis = compute_bootstrap_cis(y_test, probs, n_bootstraps=N_BOOTSTRAPS, seed=RANDOM_SEED)
    result = {
        "target_base_rate": rate,
        "n_students": N_STUDENTS,
        "observed_dropout_rate": round(float(y.mean()), 4),
        "n_test": int(len(y_test)),
        "metrics": {m: cis[m] for m in REPORTED_METRICS},
    }
    Path(result_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_sensitivity_analysis(
    rates: Sequence[float] = RATES, output_path: Optional[Path] = None, allow_dirty: bool = False
) -> Dict[str, Any]:
    """Runs one worker per rate (stopping at the first failure) and writes the combined JSON."""
    from ml.provenance import build_provenance, require_clean_tree

    require_clean_tree(allow_dirty)
    output_path = Path(output_path) if output_path else OUTPUT_PATH
    results = []
    for rate in rates:
        with tempfile.TemporaryDirectory(prefix="dropoutguard-sensitivity-") as tmp:
            env = {
                **os.environ,
                "DROPOUTGUARD_ARTIFACTS_DIR": str(Path(tmp) / "artifacts"),
                "DROPOUTGUARD_PROCESSED_DATA_PATH": str(Path(tmp) / "processed" / "features.csv"),
            }
            result_path = Path(tmp) / "result.json"
            cmd = [sys.executable, "-m", "ml.simulation.sensitivity", "--worker", str(rate), "--worker-output", str(result_path)]
            if allow_dirty:
                cmd.append("--allow-dirty")
            proc = subprocess.run(cmd, cwd=BASE_DIR, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-30:])
                raise RuntimeError(f"Sensitivity worker for target_base_rate={rate} failed (exit {proc.returncode}):\n{tail}")
            results.append(json.loads(result_path.read_text(encoding="utf-8")))

    artifact = {
        "analysis": "target_base_rate_sensitivity",
        "description": "Simulated cohort regenerated and model retrained at each target_base_rate; "
                       "held-out test split metrics with 95% bootstrap CIs. Simulated data only.",
        "rates": list(rates),
        "results": results,
        "provenance": build_provenance(
            {ASSUMPTIONS_PATH.name: ASSUMPTIONS_PATH, ESTIMATED_PARAMETERS_PATH.name: ESTIMATED_PARAMETERS_PATH},
            allow_dirty=allow_dirty,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    return artifact


def build_arg_parser() -> argparse.ArgumentParser:
    from ml.provenance import add_allow_dirty_argument

    parser = argparse.ArgumentParser(description="target_base_rate sensitivity of the simulated-cohort model")
    parser.add_argument("--output", type=str, default=None, help="Write the JSON here instead of ml/artifacts/")
    parser.add_argument("--worker", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=str, default=None, help=argparse.SUPPRESS)
    return add_allow_dirty_argument(parser)


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.worker is not None:
        run_worker(args.worker, Path(args.worker_output), allow_dirty=args.allow_dirty)
    else:
        run_sensitivity_analysis(output_path=args.output, allow_dirty=args.allow_dirty)
