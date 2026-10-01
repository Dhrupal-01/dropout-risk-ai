"""
Render README Metrics Script
Reads ml/artifacts/model_metrics.json and ml/artifacts/fairness_metrics.json
and rewrites only the text between <!-- METRICS:START --> and <!-- METRICS:END --> in README.md.
"""

import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
README_PATH = BASE_DIR / "README.md"
MODEL_METRICS_PATH = BASE_DIR / "ml" / "artifacts" / "model_metrics.json"
FAIRNESS_METRICS_PATH = BASE_DIR / "ml" / "artifacts" / "fairness_metrics.json"

START_MARKER = "<!-- METRICS:START -->"
END_MARKER = "<!-- METRICS:END -->"


def render_metrics_block() -> str:
    if not MODEL_METRICS_PATH.exists():
        raise FileNotFoundError(f"Model metrics JSON not found at {MODEL_METRICS_PATH}")
    if not FAIRNESS_METRICS_PATH.exists():
        raise FileNotFoundError(f"Fairness metrics JSON not found at {FAIRNESS_METRICS_PATH}")

    with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
        model_metrics = json.load(f)

    with open(FAIRNESS_METRICS_PATH, "r", encoding="utf-8") as f:
        fairness_metrics = json.load(f)

    recall = model_metrics["recall_at_risk_minority"] * 100.0
    precision = model_metrics["precision_at_risk_minority"] * 100.0
    f1_min = model_metrics["f1_at_risk_minority"]
    f1_macro = model_metrics["f1_macro"]
    roc_auc = model_metrics["roc_auc"]
    accuracy = model_metrics["accuracy"] * 100.0

    brier = model_metrics.get("brier_score")
    brier_str = f"`{brier:.4f}`" if brier is not None else "TODO(citation)"

    # Demographic FNR gaps on the simulated held-out test split (generator sanity check).
    # Direct key access: a missing value must fail, never render as 0.0.
    gaps = fairness_metrics["generator_sanity_check"]["test_split_fnr_gaps"]
    gender_gap = gaps["gender"] * 100.0
    econ_gap = gaps["economic_proxy"] * 100.0
    fg_gap = gaps["first_generation"] * 100.0

    lines = [
        f"- **At-Risk Recall (Sensitivity)**: `{recall:.2f}%` (Minimizes missed vulnerable students)",
        f"- **At-Risk Precision**: `{precision:.2f}%` (Prevents mentor alert fatigue)",
        f"- **Minority Class F1 Score**: `{f1_min:.4f}`",
        f"- **Macro-Averaged F1 Score**: `{f1_macro:.4f}`",
        f"- **ROC-AUC**: `{roc_auc:.4f}`",
        f"- **Overall Accuracy**: `{accuracy:.2f}%`",
        f"- **Brier Calibration Score**: {brier_str}",
        f"- **Demographic Disparity**: Gender FNR gap ${gender_gap:.2f}\\text{{ pp}}$, Economic proxy gap ${econ_gap:.2f}\\text{{ pp}}$, First-Gen gap ${fg_gap:.2f}\\text{{ pp}}$"
    ]
    return "\n".join(lines)


def update_readme():
    if not README_PATH.exists():
        raise FileNotFoundError(f"README.md not found at {README_PATH}")

    content = README_PATH.read_text(encoding="utf-8")

    start_idx = content.find(START_MARKER)
    end_idx = content.find(END_MARKER)

    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        raise ValueError(
            f"Could not find valid {START_MARKER} and {END_MARKER} block in {README_PATH}"
        )

    new_metrics = render_metrics_block()
    before = content[: start_idx + len(START_MARKER)]
    after = content[end_idx:]
    updated_content = f"{before}\n{new_metrics}\n{after}"

    README_PATH.write_text(updated_content, encoding="utf-8")
    print(f"Successfully updated metrics in {README_PATH} from JSON artifacts.")


if __name__ == "__main__":
    update_readme()
