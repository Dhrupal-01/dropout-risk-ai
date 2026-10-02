"""
Secret scanning for V9.3.

Every match of SECRET_PATTERNS in a scanned file is a violation unless the allowlist
(verification/secret_allowlist.json) has an entry with exactly that file path and exactly that
matched string, plus a non-empty reason. Allowlist entries that match nothing are reported too.
The allowlist file itself necessarily contains the allowlisted strings; in that file only exact
copies of an entry's matched string are allowed.
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SECRET_PATTERNS = [
    re.compile(r"postgres(?:ql)?://(?!<user>:<password>|user:password|u:p|admin:sup3rs3cret|test:test)[^:\s]+:[^@\s]+@", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]

SKIPPED_SUFFIXES = (".png", ".jpg", ".joblib", ".parquet")


def load_allowlist(path: Path) -> List[Dict[str, str]]:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    for i, entry in enumerate(entries):
        for key in ("file", "match", "reason"):
            if not str(entry.get(key, "")).strip():
                raise ValueError(f"{path} entry {i} has no '{key}': {entry}")
    return entries


def find_secret_violations(
    root: Path, rel_paths: Iterable[str], allowlist: List[Dict[str, str]], allowlist_file: Optional[str] = None
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """(violations as (file, matched string), allowlist entries that matched nothing)."""
    allowed = {(entry["file"], entry["match"]) for entry in allowlist}
    allowlisted_strings = {entry["match"] for entry in allowlist}
    used = set()
    violations = []
    for rel_path in rel_paths:
        file_path = Path(root) / rel_path
        if not file_path.is_file() or file_path.name.endswith(SKIPPED_SUFFIXES):
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                key = (rel_path, match.group(0))
                if rel_path == allowlist_file and match.group(0) in allowlisted_strings:
                    continue
                if key in allowed:
                    used.add(key)
                else:
                    violations.append(key)
    unused = [entry for entry in allowlist if (entry["file"], entry["match"]) not in used]
    return violations, unused
