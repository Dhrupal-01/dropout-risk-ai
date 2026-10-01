"""
Integrity guards shared by the real-data loaders (UCI 697, OULAD).

- Any file carrying an `is_synthetic` column is refused outright.
- Every raw file must match the SHA-256 recorded in checksums.json for the official release.
"""

import hashlib
import json
from pathlib import Path
from typing import Iterable

CHECKSUMS_PATH = Path(__file__).with_name("checksums.json")


class SyntheticDataError(RuntimeError):
    """Raised when a real-data loader is handed a synthetic stand-in file."""


class ChecksumMismatchError(RuntimeError):
    """Raised when a raw file does not match the official release checksum."""


def sha256_file(path: Path) -> str:
    """Returns the hex SHA-256 digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_synthetic(columns: Iterable[str], path: Path) -> None:
    """Raises SyntheticDataError if the file at `path` has an `is_synthetic` column."""
    if "is_synthetic" in {str(c).strip() for c in columns}:
        raise SyntheticDataError(
            f"Refusing to load {path}: it contains an 'is_synthetic' column, so it is a synthetic "
            f"stand-in, not the official dataset. Delete it and download the real data."
        )


def verify_checksum(source: str, filename: str, path: Path, instructions: str) -> None:
    """
    Compares the SHA-256 of `path` with checksums.json[source]["files"][filename].
    Raises ChecksumMismatchError (with download instructions) on mismatch.
    """
    with open(CHECKSUMS_PATH, "r", encoding="utf-8") as f:
        expected = json.load(f)[source]["files"][filename]["sha256"]

    actual = sha256_file(path)
    if actual != expected:
        raise ChecksumMismatchError(
            f"Checksum mismatch for {path}.\n"
            f"Expected SHA-256 (official release): {expected}\n"
            f"Actual SHA-256:                      {actual}\n"
            f"The file is not the official dataset (modified, re-exported or synthetic).\n"
            f"{instructions}"
        )
