"""
Provenance for benchmark and fairness artifacts.

Every benchmark/fairness JSON records:
- SHA-256 of each input file
- the git commit (and whether tracked files had uncommitted changes)
- Python and library versions

Renderers call assert_consistent_provenance() and refuse to combine artifacts whose
input checksums differ, or artifacts that carry no provenance at all.
"""

import argparse
import fnmatch
import importlib
import platform
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ml.sources.integrity import sha256_file

BASE_DIR = Path(__file__).resolve().parents[1]

LIBRARIES = ["numpy", "pandas", "sklearn", "xgboost", "shap", "torch", "fairlearn"]


class ProvenanceError(RuntimeError):
    """Raised when artifacts lack provenance or were computed from different inputs."""


class DirtyTreeError(RuntimeError):
    """Raised when an artifact writer runs on uncommitted tracked changes without --allow-dirty."""


@lru_cache(maxsize=None)
def _cached_sha256(path: str, size: int, mtime_ns: int) -> str:
    return sha256_file(Path(path))


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _cached_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path.resolve())


def _git(*args: str) -> Optional[str]:
    try:
        res = subprocess.run(["git", *args], cwd=BASE_DIR, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _library_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.import_module(name).__version__
        except ImportError:
            continue
    return versions


def modified_tracked_files() -> List[str]:
    """Tracked files with uncommitted changes (`git status --porcelain --untracked-files=no`)."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is None:
        raise DirtyTreeError("Could not read git status; refusing to write artifacts without a known commit.")
    # each line is "<XY status> <path>"; split on whitespace (stdout is stripped, so column offsets are unreliable)
    return [line.split(None, 1)[1] for line in status.splitlines() if line.strip()]


# Tracked files that pipeline steps regenerate. Uncommitted changes to these do not make the code
# differ from the recorded commit, so a multi-step pipeline run can proceed on one commit.
GENERATED_OUTPUT_PATTERNS = [
    "ml/artifacts/*",
    "ml/simulation/estimated_parameters.json",
    "docs/benchmarks.md",
    "docs/ethics_and_fairness.md",
    "docs/simulation.md",
    "docs/simulation_mapping.md",
    "docs/figures/*",
    "verification/REPORT.md",
    "data/processed/feature_metadata.json",
]
README_PATH = "README.md"
README_BLOCKS = ("METRICS", "BENCHMARKS")


def strip_generated_readme_blocks(text: str) -> str:
    """README text with the contents of every generated marker block removed."""
    for name in README_BLOCKS:
        text = re.sub(rf"(<!-- {name}:START -->).*?(<!-- {name}:END -->)", r"\1\2", text, flags=re.DOTALL)
    return text.strip()


def readme_changes_confined_to_generated_blocks() -> bool:
    head = _git("show", f"HEAD:{README_PATH}")
    if head is None:
        return False
    working = (BASE_DIR / README_PATH).read_text(encoding="utf-8")
    return strip_generated_readme_blocks(head) == strip_generated_readme_blocks(working)


def is_generated_output(path: str) -> bool:
    if path == README_PATH:
        return readme_changes_confined_to_generated_blocks()
    return any(fnmatch.fnmatch(path, pattern) for pattern in GENERATED_OUTPUT_PATTERNS)


def split_modified_files() -> Tuple[List[str], List[str]]:
    """(modified code/config files, modified generated outputs) among tracked files."""
    code, generated = [], []
    for path in modified_tracked_files():
        (generated if is_generated_output(path) else code).append(path)
    return code, generated


def require_clean_tree(allow_dirty: bool = False) -> bool:
    """
    Gate for every artifact writer. Raises DirtyTreeError if tracked code/config files have
    uncommitted changes, unless allow_dirty is True. Uncommitted generated outputs (earlier pipeline
    steps) are allowed. Returns True when the code is dirty and that was explicitly allowed.
    """
    code, _ = split_modified_files()
    if code and not allow_dirty:
        raise DirtyTreeError(
            "Refusing to write artifacts: tracked files have uncommitted changes, so the recorded git "
            f"commit would not identify the code that produced them: {code}. "
            "Commit or stash them, or rerun with --allow-dirty (recorded as allow_dirty=true in the JSON)."
        )
    return bool(code)


def add_allow_dirty_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Write artifacts even if tracked files have uncommitted changes (recorded as allow_dirty=true).",
    )
    return parser


def dirty_artifact_warning(artifacts: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Markdown warning line listing artifacts generated from a dirty tree, or None if there are none."""
    dirty = []
    for label, art in sorted(artifacts.items()):
        prov = (art or {}).get("provenance") or {}
        if prov.get("git_dirty"):
            allowed = prov.get("allow_dirty", "not recorded")
            dirty.append(f"`{label}` (commit `{str(prov.get('git_commit'))[:7]}`, allow_dirty={str(allowed).lower()})")
    if not dirty:
        return None
    return (
        "> **Warning: generated from a working tree with uncommitted changes (git_dirty=true); the recorded "
        "commit does not fully identify the code.** " + ", ".join(dirty)
    )


def build_provenance(input_files: Dict[str, Path], allow_dirty: bool = False) -> Dict[str, Any]:
    """
    Provenance block for an artifact computed from `input_files` (name -> path).
    Re-checks the tree at write time; raises DirtyTreeError on a dirty tree unless allow_dirty.
    """
    dirty = require_clean_tree(allow_dirty)
    return {
        "input_files": {
            name: {"path": _relative(Path(p)), "sha256": _sha256(Path(p))}
            for name, p in sorted(input_files.items())
        },
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "allow_dirty": bool(allow_dirty),
        "uncommitted_generated_outputs": sorted(split_modified_files()[1]),
        "python": platform.python_version(),
        "library_versions": _library_versions(),
    }


def merge_input_files(*input_sets: Dict[str, Path]) -> Dict[str, Path]:
    """Union of several input sets; a name mapped to two different paths is an error."""
    merged: Dict[str, Path] = {}
    for inputs in input_sets:
        for name, path in inputs.items():
            if name in merged and Path(merged[name]).resolve() != Path(path).resolve():
                raise ProvenanceError(f"Input name '{name}' maps to both {merged[name]} and {path}")
            merged[name] = path
    return merged


def uci_inputs() -> Dict[str, Path]:
    from ml.sources.uci import UCI_CSV_PATH

    return {UCI_CSV_PATH.name: UCI_CSV_PATH}


def oulad_inputs(data_dir: Optional[Path] = None) -> Dict[str, Path]:
    from ml.sources.oulad import OULAD_RAW_DIR, REQUIRED_TABLES

    raw_dir = Path(data_dir) if data_dir else OULAD_RAW_DIR
    return {table: raw_dir / table for table in REQUIRED_TABLES}


def simulated_inputs() -> Dict[str, Path]:
    from ml.config import FEATURE_NAMES_PATH, MODEL_ARTIFACT_PATH, PROCESSED_DATA_PATH

    inputs = {PROCESSED_DATA_PATH.name: PROCESSED_DATA_PATH, FEATURE_NAMES_PATH.name: FEATURE_NAMES_PATH}
    if MODEL_ARTIFACT_PATH.exists():
        inputs[MODEL_ARTIFACT_PATH.name] = MODEL_ARTIFACT_PATH
    return inputs


def assert_consistent_provenance(artifacts: Dict[str, Dict[str, Any]]) -> None:
    """
    `artifacts` maps an artifact label (e.g. its file name) to its loaded JSON.
    Raises ProvenanceError if any artifact has no provenance, or if the same input file
    name was recorded with different SHA-256 checksums by different artifacts.
    """
    missing = sorted(label for label, art in artifacts.items() if not (art or {}).get("provenance"))
    if missing:
        raise ProvenanceError(
            f"Refusing to render: these artifacts have no provenance (input checksums unknown): {missing}. "
            "Regenerate them with the current pipeline before rendering."
        )

    seen: Dict[str, Dict[str, list]] = {}
    for label, art in sorted(artifacts.items()):
        for name, info in art["provenance"]["input_files"].items():
            seen.setdefault(name, {}).setdefault(info["sha256"], []).append(label)

    conflicts = {name: hashes for name, hashes in seen.items() if len(hashes) > 1}
    if conflicts:
        details = "; ".join(
            f"{name}: " + ", ".join(f"{sha[:12]}… in {labels}" for sha, labels in hashes.items())
            for name, hashes in sorted(conflicts.items())
        )
        raise ProvenanceError(f"Refusing to render: artifacts were computed from different input files. {details}")


def load_labelled_json(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    """Loads JSON files keyed by their file name, for assert_consistent_provenance."""
    import json

    loaded: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            loaded[Path(p).name] = json.load(f)
    return loaded
