"""
README generated blocks must equal what their renderers produce from the committed JSON artifacts,
so a hand edit or a stale block fails.
"""

import re
from pathlib import Path

import pytest

from ml.tests.artifact_checks import require_artifact
from scripts.render_readme_benchmarks import BENCHMARK_DIR, render_benchmarks_block
from scripts.render_readme_metrics import FAIRNESS_METRICS_PATH, MODEL_METRICS_PATH, render_metrics_block

BASE_DIR = Path(__file__).resolve().parents[2]
README = BASE_DIR / "README.md"


def _block(name: str) -> str:
    match = re.search(rf"<!-- {name}:START -->\n(.*?)\n<!-- {name}:END -->", README.read_text(encoding="utf-8"), re.DOTALL)
    assert match, f"README is missing the {name} markers"
    return match.group(1)


@pytest.mark.artifacts
def test_readme_benchmarks_block_matches_rendered_json():
    require_artifact(
        any(BENCHMARK_DIR.glob("uci_*_primary.json")),
        "ml/artifacts/benchmarks/uci_*_primary.json",
        "python -m ml.evaluation.run --source uci` and `--source oulad",
    )
    assert _block("BENCHMARKS") == render_benchmarks_block()


@pytest.mark.artifacts
def test_readme_metrics_block_matches_rendered_json():
    require_artifact(
        MODEL_METRICS_PATH.exists() and FAIRNESS_METRICS_PATH.exists(),
        "ml/artifacts/model_metrics.json or fairness_metrics.json",
        "python -m ml.validate_pipeline --regenerate",
    )
    assert _block("METRICS") == render_metrics_block()
