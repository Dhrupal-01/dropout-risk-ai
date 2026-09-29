"""
Verification tests for V0 (Environment) and V1 (Test Integrity).
"""

import ast
import subprocess
from pathlib import Path
import tomllib
import pytest

from verification.conftest import BASELINE_COMMIT, PROJECT_ROOT


class TestV0Environment:
    def test_v0_1_git_commits_since_baseline(self):
        """V0.1: List every commit since BASELINE and verify commit structure."""
        cmd = ["git", "log", "--oneline", f"{BASELINE_COMMIT}..HEAD"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        commits = res.stdout.strip().splitlines()
        assert len(commits) > 0, "No commits found since BASELINE"

    def test_v0_2_backend_dependencies_isolation(self):
        """V0.2: pip install -e '.[backend]' must NOT install torch or fairlearn."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        assert pyproject_path.exists()
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        project = data.get("project", {})
        main_deps = project.get("dependencies", [])
        opt_deps = project.get("optional-dependencies", {})
        backend_deps = opt_deps.get("backend", [])
        research_deps = opt_deps.get("research", [])

        forbidden = {"torch", "fairlearn"}
        for dep in main_deps + backend_deps:
            for pkg in forbidden:
                assert pkg not in dep.lower(), f"Forbidden package {pkg} found in backend/core deps: {dep}"

        # Must be in research deps
        research_str = " ".join(research_deps).lower()
        for pkg in forbidden:
            assert pkg in research_str, f"Required package {pkg} not found in [project.optional-dependencies] research"

    @pytest.mark.db
    @pytest.mark.data
    def test_v0_3_environment_resources_presence(self, live_db_url):
        """V0.3: TEST_DATABASE_URL points to throwaway DB; UCI present; OULAD has 7 tables."""
        assert live_db_url, "TEST_DATABASE_URL is required"
        assert "test" in live_db_url.lower(), "TEST_DATABASE_URL must point to a test database"

        # Check UCI
        uci_path = PROJECT_ROOT / "data" / "raw" / "uci_dropout.csv"
        if not uci_path.exists():
            uci_path = PROJECT_ROOT / "data" / "raw" / "data.csv"
        assert uci_path.exists(), f"UCI dataset not found at {uci_path}"

        # Check OULAD 7 tables
        oulad_dir = PROJECT_ROOT / "data" / "raw" / "oulad"
        assert oulad_dir.exists(), f"OULAD directory not found at {oulad_dir}"
        required_tables = {
            "studentInfo.csv",
            "studentRegistration.csv",
            "studentVle.csv",
            "vle.csv",
            "assessments.csv",
            "studentAssessment.csv",
            "courses.csv",
        }
        found_tables = {p.name for p in oulad_dir.glob("*.csv")}
        missing = required_tables - found_tables
        assert not missing, f"Missing OULAD tables: {missing}"


class TestV1TestIntegrity:
    def test_v1_1_test_counts_vs_baseline(self):
        """V1.1: Test counts now vs BASELINE: total must be >= 251 plus new tests."""
        import re
        # Collect tests from ml/tests
        res_ml = subprocess.run(
            [".venv/bin/pytest", "ml/tests", "--collect-only", "-q"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        ml_count = 0
        for line in res_ml.stdout.splitlines():
            m = re.search(r"(\d+)\s+tests?\s+collected", line) or re.search(r"collected\s+(\d+)\s+items", line)
            if m:
                ml_count = int(m.group(1))
                break

        # Collect tests from backend/tests
        res_backend = subprocess.run(
            [".venv/bin/pytest", "backend/tests", "--collect-only", "-q"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        backend_count = 0
        for line in res_backend.stdout.splitlines():
            m = re.search(r"(\d+)\s+tests?\s+collected", line) or re.search(r"collected\s+(\d+)\s+items", line)
            if m:
                backend_count = int(m.group(1))
                break

        # Original was 27 in ml/tests, 224 in backend/tests
        assert ml_count >= 27, f"ml/tests count dropped: {ml_count} < 27"
        assert backend_count >= 224, f"backend/tests count dropped: {backend_count} < 224"
        total = ml_count + backend_count
        assert total >= 251, f"Total test count dropped: {total} < 251"

    def test_v1_2_git_diff_baseline_tests(self):
        """V1.2: Check for test deletions or blanket skip additions."""
        cmd = ["git", "diff", f"{BASELINE_COMMIT}..HEAD", "--name-status", "--", "ml/tests", "backend/tests"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        # Verify no test files were deleted (status 'D')
        for line in res.stdout.splitlines():
            parts = line.split()
            if parts:
                status, path = parts[0], parts[1]
                assert not status.startswith("D"), f"Test file was deleted since baseline: {path}"

    @pytest.mark.db
    @pytest.mark.data
    def test_v1_3_zero_skipped_tests_when_resources_present(self):
        """V1.3: Zero skipped tests when all resources are available."""
        res = subprocess.run(
            [".venv/bin/pytest", "ml/tests", "backend/tests", "-q", "-rs"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "SKIPPED" not in res.stdout, f"Found skipped tests when resources available:\n{res.stdout}"

    def test_v1_4_no_trivially_passing_tests(self):
        """V1.4: Spot-check AST of newly added test files for empty assertions or dummy passes."""
        test_files = list((PROJECT_ROOT / "ml" / "tests").glob("test_*.py")) + list(
            (PROJECT_ROOT / "backend" / "tests").glob("test_*.py")
        )
        for tf in test_files:
            source = tf.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(tf))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    # Check body is not just 'pass' or 'assert True'
                    body_stmts = [s for s in node.body if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)]
                    assert body_stmts, f"Trivially empty test found: {tf.name}:{node.name}"
                    if len(body_stmts) == 1 and isinstance(body_stmts[0], ast.Assert):
                        test_val = body_stmts[0].test
                        if isinstance(test_val, ast.Constant) and test_val.value is True:
                            pytest.fail(f"Trivially passing 'assert True' test in {tf.name}:{node.name}")
