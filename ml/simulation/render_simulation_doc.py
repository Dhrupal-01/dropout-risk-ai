"""
Generates docs/simulation.md from ml/simulation/assumptions.yaml and provides
consistency verification between the YAML specification and the rendered documentation.
"""

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
YAML_PATH = BASE_DIR / "ml" / "simulation" / "assumptions.yaml"
DOCS_SIM_PATH = BASE_DIR / "docs" / "simulation.md"

ALLOWED_SOURCES = {
    "estimated_from_uci",
    "estimated_from_oulad",
    "indian_regulation",
    "assumption",
    "TODO(citation)",
}


def load_assumptions() -> Dict[str, Any]:
    """assumptions.yaml with `value_from` references resolved (see load_simulation_assumptions)."""
    from ml.data_pipeline.generate_synthetic_indian import load_simulation_assumptions

    return load_simulation_assumptions(YAML_PATH)


def extract_all_keys(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extracts a flat mapping of fully-qualified key names to their item dicts."""
    flat: Dict[str, Dict[str, Any]] = {}
    for section, items in data.items():
        if isinstance(items, dict):
            for k, v in items.items():
                full_key = f"{section}.{k}"
                flat[full_key] = v
    return flat


def validate_assumptions_structure(flat: Dict[str, Dict[str, Any]]) -> None:
    for key, item in flat.items():
        if not isinstance(item, dict):
            raise ValueError(f"Entry {key} must be a dictionary with 'value', 'source', and 'notes'.")
        if "value" not in item:
            raise KeyError(f"Entry {key} is missing required field 'value'.")
        if "source" not in item:
            raise KeyError(f"Entry {key} is missing required field 'source'.")
        src = item["source"]
        if src not in ALLOWED_SOURCES:
            raise ValueError(
                f"Invalid source '{src}' in {key}. Must be one of {sorted(ALLOWED_SOURCES)}"
            )


def render_markdown(data: Dict[str, Any]) -> str:
    lines = [
        "# Simulated Indian Cohort Specification and Grounding",
        "",
        "> Complete inventory of all structural coefficients, distribution parameters, and policy thresholds",
        "> governing the simulated Indian collegiate dataset generator (`ml/data_pipeline/generate_synthetic_indian.py`).",
        "> Generated automatically from [`ml/simulation/assumptions.yaml`](../ml/simulation/assumptions.yaml).",
        "",
        "---",
        "",
        "## Source Taxonomy",
        "",
        "- `estimated_from_uci`: Empirically estimated via standardized logistic regression on UCI Student Dropout (ID 697).",
        "- `estimated_from_oulad`: Empirically estimated via standardized logistic regression on OULAD Learning Analytics (UCI ID 349).",
        "- `indian_regulation`: Grounded in Indian higher education regulatory mandates (e.g. AICTE/UGC 75% attendance rule, statutory reservation percentages).",
        "- `assumption`: Explicit engineering or behavioral assumption documented for transparent audit.",
        "- `TODO(citation)`: Temporary parameter placeholder pending published empirical Indian empirical survey citation.",
        "",
        "---",
    ]

    for section_name, items in data.items():
        title = section_name.replace("_", " ").title()
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Parameter Key | Value | Source | Notes / Description |")
        lines.append("| :--- | :--- | :--- | :--- |")

        for key, details in items.items():
            val = details.get("value")
            # Format value nicely
            if isinstance(val, float):
                val_str = f"`{val:g}`" if abs(val) < 1e4 else f"`{val}`"
            elif isinstance(val, (dict, list)):
                val_str = f"`{yaml.dump(val, default_flow_style=True).strip()}`"
            else:
                val_str = f"`{val}`"
            if "value_from" in details:
                val_str += f" (from `{details['value_from']}`)"

            src = details.get("source", "")
            notes = details.get("notes", "").replace("|", "\\|")
            full_key = f"`{section_name}.{key}`"
            lines.append(f"| {full_key} | {val_str} | `{src}` | {notes} |")

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def generate_simulation_doc() -> None:
    data = load_assumptions()
    flat = extract_all_keys(data)
    validate_assumptions_structure(flat)

    content = render_markdown(data)
    DOCS_SIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_SIM_PATH.write_text(content, encoding="utf-8")
    print(f"Successfully generated {DOCS_SIM_PATH} with {len(flat)} parameters.")


def verify_assumptions_doc_sync() -> Tuple[bool, List[str]]:
    """
    Checks if docs/simulation.md is strictly in sync with ml/simulation/assumptions.yaml.
    Returns (is_synced, error_messages).
    """
    if not DOCS_SIM_PATH.exists():
        return False, [f"{DOCS_SIM_PATH} does not exist."]

    data = load_assumptions()
    flat = extract_all_keys(data)
    validate_assumptions_structure(flat)

    doc_text = DOCS_SIM_PATH.read_text(encoding="utf-8")
    errors: List[str] = []

    for full_key in flat:
        pattern = f"`{full_key}`"
        if pattern not in doc_text:
            errors.append(f"Key {pattern} from assumptions.yaml missing in {DOCS_SIM_PATH.name}")

    return (len(errors) == 0), errors


if __name__ == "__main__":
    generate_simulation_doc()
    synced, errs = verify_assumptions_doc_sync()
    if synced:
        print("Verification passed: docs/simulation.md is in sync with assumptions.yaml")
    else:
        print("Verification failed:", errs)
