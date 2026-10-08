"""
Render README Real-Data Benchmarks Script
Reads the UCI (primary label) and OULAD benchmark JSON artifacts in ml/artifacts/benchmarks/
and rewrites only the text between <!-- BENCHMARKS:START --> and <!-- BENCHMARKS:END --> in README.md.
Every number comes from the JSON; artifacts with missing or mixed provenance are refused.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from ml.provenance import assert_consistent_provenance, dirty_artifact_warning

BASE_DIR = Path(__file__).resolve().parents[1]
README_PATH = BASE_DIR / "README.md"
BENCHMARK_DIR = BASE_DIR / "ml" / "artifacts" / "benchmarks"

START_MARKER = "<!-- BENCHMARKS:START -->"
END_MARKER = "<!-- BENCHMARKS:END -->"

UCI_FEATURE_SETS = [("enrolment_time", "ENROLMENT_TIME"), ("end_of_sem1", "END_OF_SEM1"), ("full", "FULL (not early warning)")]
MODELS = [
    ("majority_class", "Majority class"),
    ("logistic_regression", "Logistic regression"),
    ("xgboost", "XGBoost"),
    ("pytorch_gru", "GRU"),
]
NOT_VALIDATED = (
    "> **Not validated on Indian college records.** This result is from {dataset}. The deployed model is "
    "trained on a simulated Indian cohort and has never been evaluated on real Indian student data."
)


def _stat(m: Dict[str, float]) -> str:
    return f"{m['point']:.4f} [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]"


def _load(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    if not paths:
        raise FileNotFoundError(f"No benchmark artifacts found in {BENCHMARK_DIR}")
    loaded = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            loaded[p.name] = json.load(f)
    return loaded


def _single(artifacts: List[Dict[str, Any]], key: str) -> Any:
    values = {a[key] for a in artifacts}
    if len(values) != 1:
        raise ValueError(f"Expected one '{key}' across artifacts, found {values}")
    return values.pop()


def render_benchmarks_block() -> str:
    uci = _load(sorted(BENCHMARK_DIR.glob("uci_*_primary.json")))
    oulad = _load(sorted(BENCHMARK_DIR.glob("oulad_snapshot_t*_withdrawn.json")))
    # Fails (ProvenanceError) on missing provenance, mixed git commits or mixed input checksums.
    assert_consistent_provenance({**uci, **oulad})

    uci_by_fs = {a["feature_set"]: a for a in uci.values()}
    oulad_sorted = sorted(oulad.values(), key=lambda a: a["snapshot_t"])
    every = list(uci.values()) + oulad_sorted

    commit = every[0]["provenance"]["git_commit"]  # one commit across all, checked above
    inputs = {}
    for a in every:
        for name, info in a["provenance"]["input_files"].items():
            inputs[name] = info["sha256"]
    n_boot = _single(every, "n_bootstraps")

    uci_n = _single(list(uci.values()), "n_samples")
    uci_pos = _single(list(uci.values()), "total_positives")
    uci_prev = _single(list(uci.values()), "prevalence")

    dirty_warning = dirty_artifact_warning({**uci, **oulad})
    lines = [
        *([dirty_warning, ""] if dirty_warning else []),
        f"Generated from `ml/artifacts/benchmarks/*.json` (commit `{commit[:7]}`; inputs "
        + ", ".join(f"`{name}` `{sha[:12]}`" for name, sha in sorted(inputs.items()))
        + "). Full results, all split strategies and metrics: [docs/benchmarks.md](docs/benchmarks.md). "
        f"Values are point estimates with 95% bootstrap CIs ({n_boot:,} resamples).",
        "",
        "#### UCI 697: Portuguese higher education",
        "",
        f"Primary label, Dropout vs Graduate (Enrolled excluded): n = {uci_n:,}, {uci_pos:,} dropouts, "
        f"prevalence {uci_prev:.4f}. Repeated stratified 5-fold CV (3 repeats).",
        "",
        "| Feature set | Features | Model | ROC-AUC [95% CI] | PR-AUC [95% CI] |",
        "| :--- | ---: | :--- | :--- | :--- |",
    ]
    for fs_key, fs_label in UCI_FEATURE_SETS:
        art = uci_by_fs[fs_key]
        res = art["results"]["repeated_stratified_cv"]
        for m_key, m_label in MODELS:
            if m_key in res:
                met = res[m_key]["metrics"]
                lines.append(f"| {fs_label} | {art['n_features']} | {m_label} | {_stat(met['roc_auc'])} | {_stat(met['pr_auc'])} |")
    lines.extend([
        "",
        NOT_VALIDATED.format(dataset="a Portuguese polytechnic (UCI 697)"),
        "",
        "#### OULAD: UK online learning",
        "",
        "Label: Withdrawn. Temporal holdout: train 2013B + 2013J, test 2014B + 2014J. "
        "n = registrations still enrolled at day t.",
        "",
        "| t (days) | n | Test n | Prevalence | Model | ROC-AUC [95% CI] | PR-AUC [95% CI] |",
        "| ---: | ---: | ---: | ---: | :--- | :--- | :--- |",
    ])
    for art in oulad_sorted:
        res = art["results"]["predefined_split"]
        for m_key, m_label in MODELS:
            if m_key in res:
                met = res[m_key]["metrics"]
                lines.append(
                    f"| {art['snapshot_t']} | {art['n_samples']:,} | {res[m_key]['n_evaluated']:,} | {art['prevalence']:.4f} "
                    f"| {m_label} | {_stat(met['roc_auc'])} | {_stat(met['pr_auc'])} |"
                )
    lines.extend([
        "",
        NOT_VALIDATED.format(dataset="UK distance-learning students (OULAD)"),
    ])
    return "\n".join(lines)


def update_readme() -> None:
    content = README_PATH.read_text(encoding="utf-8")
    start_idx = content.find(START_MARKER)
    end_idx = content.find(END_MARKER)
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        raise ValueError(f"Could not find valid {START_MARKER} and {END_MARKER} block in {README_PATH}")

    block = render_benchmarks_block()
    updated = f"{content[: start_idx + len(START_MARKER)]}\n{block}\n{content[end_idx:]}"
    README_PATH.write_text(updated, encoding="utf-8")
    print(f"Successfully updated real-data benchmarks in {README_PATH} from JSON artifacts.")


if __name__ == "__main__":
    update_readme()
