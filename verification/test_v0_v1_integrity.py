"""
Verification tests for V0 (Environment) and V1 (Test Integrity).
"""

import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
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
            [sys.executable, "-m", "pytest", "ml/tests", "--collect-only", "-q"],
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
            [sys.executable, "-m", "pytest", "backend/tests", "--collect-only", "-q"],
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
        """
        V1.2: Since BASELINE, no test file was deleted, and every removed assert / numeric threshold /
        tolerance line and every added skip / skipif / xfail line in ml/tests and backend/tests is listed
        and must be reviewed in verification/reviewed_test_changes.json (file, sign, line, reason).
        """
        status = subprocess.run(
            ["git", "diff", f"{BASELINE_COMMIT}..HEAD", "--name-status", "--", "ml/tests", "backend/tests"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        deleted = [line.split()[1] for line in status.stdout.splitlines() if line.split() and line.split()[0].startswith("D")]
        assert not deleted, f"Test files deleted since baseline: {deleted}"

        diff = subprocess.run(
            ["git", "diff", "-U0", f"{BASELINE_COMMIT}..HEAD", "--", "ml/tests", "backend/tests"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout

        removed_check = re.compile(r"\bassert\b|pytest\.approx|\b(abs|rel|atol|rtol)\s*=|[<>]=?\s*-?\d")
        added_skip = re.compile(r"\b(skip|skipif|xfail)\b")

        # hunks: (file, [removed lines], [added lines])
        hunks, current_file, hunk = [], None, None
        for line in diff.splitlines():
            if line.startswith("+++ "):
                current_file = line[6:] if line.startswith("+++ b/") else None
            elif line.startswith("--- "):
                continue
            elif line.startswith("@@"):
                hunk = (current_file, [], [])
                hunks.append(hunk)
            elif hunk is not None and line.startswith("-"):
                hunk[1].append(line[1:].strip())
            elif hunk is not None and line.startswith("+"):
                hunk[2].append(line[1:].strip())

        flagged = []  # (file, sign, line, counterpart lines in the same hunk)
        for path, removed, added in hunks:
            if path is None:
                continue
            for text in removed:
                if removed_check.search(text):
                    counterpart = [a for a in added if removed_check.search(a)]
                    flagged.append((path, "-", text, counterpart))
            for text in added:
                if added_skip.search(text):
                    flagged.append((path, "+", text, []))

        reviewed_path = PROJECT_ROOT / "verification" / "reviewed_test_changes.json"
        reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
        reviewed_keys = {(r["file"], r["sign"], r["line"].strip()) for r in reviewed if r.get("reason", "").strip()}

        unreviewed = [f for f in flagged if (f[0], f[1], f[2]) not in reviewed_keys]
        if unreviewed:
            by_file = defaultdict(list)
            for path, sign, text, counterpart in unreviewed:
                entry = f"  {sign} {text}"
                if counterpart:
                    entry += "".join(f"\n      (+ in same hunk) {c}" for c in counterpart)
                by_file[path].append(entry)
            listing = "\n".join(f"{path}\n" + "\n".join(entries) for path, entries in sorted(by_file.items()))
            pytest.fail(
                f"{len(unreviewed)} unreviewed assert/threshold removals or skip additions since {BASELINE_COMMIT} "
                f"(add each to verification/reviewed_test_changes.json with a reason after review):\n{listing}"
            )

    @pytest.mark.db
    @pytest.mark.data
    def test_v1_3_zero_skipped_tests_when_resources_present(self):
        """V1.3: Zero skipped tests when all resources are available."""
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "ml/tests", "backend/tests", "-q", "-rs"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        tail = "\n".join((res.stdout + res.stderr).splitlines()[-30:])
        assert res.returncode == 0, f"Product suite failed (exit {res.returncode}). Last 30 lines:\n{tail}"
        assert "SKIPPED" not in res.stdout, f"Found skipped tests when resources available. Last 30 lines:\n{tail}"

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
