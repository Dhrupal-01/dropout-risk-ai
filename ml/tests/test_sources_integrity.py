"""
Integrity tests for the real-data loaders (ml/sources).

Each test builds a temporary project directory, points the loader's real cache path at it, and
places a synthetic stand-in there. The real loader is called with no path argument and must raise.
Stand-in files are only ever written under pytest's tmp_path.
"""

import numpy as np
import pandas as pd
import pytest

import ml.sources.oulad as oulad
import ml.sources.uci as uci
from ml.sources.integrity import ChecksumMismatchError, SyntheticDataError

# Class counts of the official UCI 697 file, so the stand-in passes the row/target asserts.
UCI_STANDIN_TARGETS = {"Dropout": 1421, "Graduate": 2209, "Enrolled": 794}


@pytest.fixture
def temp_project(tmp_path, monkeypatch):
    """Temporary project root with data/raw and data/interim; loaders point at it; no network."""
    raw_dir = tmp_path / "data" / "raw"
    interim_dir = tmp_path / "data" / "interim"
    raw_dir.mkdir(parents=True)

    monkeypatch.setattr(uci, "UCI_CSV_PATH", raw_dir / "uci_dropout.csv")
    monkeypatch.setattr(oulad, "OULAD_RAW_DIR", raw_dir / "oulad")
    monkeypatch.setattr(oulad, "INTERIM_DATA_DIR", interim_dir)
    monkeypatch.setattr(oulad, "STUDENT_VLE_PARQUET_PATH", interim_dir / "studentVle.parquet")

    def no_network(*args, **kwargs):
        raise ConnectionError("network disabled in tests")

    monkeypatch.setattr(uci.requests, "get", no_network)
    return raw_dir


def _uci_standin(with_flag: bool) -> pd.DataFrame:
    """Schema-matching UCI stand-in: official headers, 4,424 rows, official class counts."""
    headers = list(dict.fromkeys(k.strip() for k in uci.UCI_COLUMN_MAPPING if k != "Target"))
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(0, 10, size=(uci.EXPECTED_ROWS, len(headers))), columns=headers)
    df["Target"] = np.repeat(list(UCI_STANDIN_TARGETS), list(UCI_STANDIN_TARGETS.values()))
    if with_flag:
        df["is_synthetic"] = 1
    return df


def _write_oulad_standin(oulad_dir, with_flag: bool) -> None:
    """All 7 OULAD tables; studentInfo has 32,593 rows and the 4 official outcomes."""
    oulad_dir.mkdir(parents=True)
    results = sorted(oulad.EXPECTED_RESULTS)
    info = pd.DataFrame({
        "code_module": "AAA",
        "code_presentation": "2013J",
        "id_student": np.arange(oulad.EXPECTED_ROWS),
        "final_result": [results[i % len(results)] for i in range(oulad.EXPECTED_ROWS)],
    })
    if with_flag:
        info["is_synthetic"] = 1
    info.to_csv(oulad_dir / "studentInfo.csv", index=False)
    for table in oulad.REQUIRED_TABLES:
        if table != "studentInfo.csv":
            pd.DataFrame({"dummy": [1]}).to_csv(oulad_dir / table, index=False)


def test_uci_loader_refuses_standin_with_is_synthetic_at_real_cache_path(temp_project):
    _uci_standin(with_flag=True).to_csv(uci.UCI_CSV_PATH, sep=";", index=False)

    with pytest.raises(SyntheticDataError, match="is_synthetic"):
        uci.load_uci_clean_df()


def test_uci_loader_refuses_unflagged_standin_via_checksum(temp_project):
    # Without the flag, the stand-in passes the row-count and class asserts; only the checksum stops it.
    _uci_standin(with_flag=False).to_csv(uci.UCI_CSV_PATH, sep=";", index=False)

    with pytest.raises(ChecksumMismatchError, match="Checksum mismatch") as excinfo:
        uci.load_uci_clean_df()
    assert uci.UCI_ZIP_URL in str(excinfo.value)


def test_uci_loader_missing_file_and_failed_download_raises_without_writing(temp_project):
    with pytest.raises(RuntimeError, match="Manual download instructions"):
        uci.load_uci_clean_df()
    assert not uci.UCI_CSV_PATH.exists()


def test_oulad_loader_refuses_standin_with_is_synthetic_at_real_cache_path(temp_project):
    _write_oulad_standin(oulad.OULAD_RAW_DIR, with_flag=True)

    with pytest.raises(SyntheticDataError, match="is_synthetic"):
        oulad.load_raw_tables()


def test_oulad_loader_refuses_unflagged_standin_via_checksum(temp_project):
    _write_oulad_standin(oulad.OULAD_RAW_DIR, with_flag=False)

    with pytest.raises(ChecksumMismatchError, match="Checksum mismatch") as excinfo:
        oulad.load_raw_tables()
    assert "analyse.kmi.open.ac.uk" in str(excinfo.value)
    assert not oulad.STUDENT_VLE_PARQUET_PATH.exists()
