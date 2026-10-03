"""
Verification tests for V9 (End to end & security hygiene).
"""

import json
import subprocess
import pytest

from verification.conftest import PROJECT_ROOT


class TestV9EndToEnd:
    @pytest.mark.db
    def test_v9_1_api_contract_and_status_codes(self, client):
        """V9.1: Exercise API endpoints; valid input returns 2xx, invalid returns 4xx, never 500."""
        # 1. Health
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] in ("healthy", "degraded")

        # 2. Stats summary
        res = client.get("/api/v1/stats/summary")
        assert res.status_code == 200

        # 3. Filters
        res = client.get("/api/v1/mentors/filters")
        assert res.status_code == 200

        # 4. Queue valid
        res = client.get("/api/v1/mentors/queue?limit=10")
        assert res.status_code == 200

        # 5. Queue invalid (limit = 0 or negative offset)
        res_inv = client.get("/api/v1/mentors/queue?limit=0")
        assert res_inv.status_code == 422

        res_inv2 = client.get("/api/v1/mentors/queue?offset=-5")
        assert res_inv2.status_code == 422

        # 6. Predict invalid body
        res_pred = client.post("/api/v1/predict", json={"student_id": "TEST", "features": {}})
        assert res_pred.status_code == 422

    def test_v9_3_no_secrets_in_tracked_files(self):
        """V9.3: No secrets in tracked files (passwords in URLs, API keys).

        Every match is a violation unless verification/secret_allowlist.json lists that exact file
        and exact matched string with a reason; allowlist entries that match nothing also fail.
        """
        from verification.secret_scan import find_secret_violations, load_allowlist

        res = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        tracked_files = res.stdout.splitlines()
        allowlist_rel = "verification/secret_allowlist.json"
        allowlist = load_allowlist(PROJECT_ROOT / allowlist_rel)

        violations, unused = find_secret_violations(PROJECT_ROOT, tracked_files, allowlist, allowlist_file=allowlist_rel)

        assert not violations, f"Potential real secrets detected in tracked files: {violations}"
        assert not unused, f"Stale secret allowlist entries (match nothing): {unused}"

    def test_v9_3a_allowlist_honours_only_exact_matches(self, tmp_path):
        """V9.3a: The allowlist exempts only its exact (file, matched string) pairs; new fake secrets still fail."""
        from verification.secret_scan import find_secret_violations, load_allowlist

        # Built at runtime so this source file never matches the secret pattern itself.
        scheme = "postgresql" + "://"
        allowed = scheme + "fake-user:fake-pass" + "@"
        new_secret = scheme + "someone:hunter2" + "@"
        variant = scheme + "fake-user:fake-pasS" + "@"
        allowlist = [{"file": "a.py", "match": allowed, "reason": "fixture"}]

        def scan(files):
            for name, text in files.items():
                (tmp_path / name).write_text(text, encoding="utf-8")
            result = find_secret_violations(tmp_path, list(files), allowlist)
            for name in files:
                (tmp_path / name).unlink()
            return result

        assert scan({"a.py": f"URL = '{allowed}db.invalid/x'\n"}) == ([], [])
        violations, _ = scan({"a.py": f"URL = '{allowed}db.invalid/x'\n", "b.py": f"URL = '{new_secret}db.invalid/y'\n"})
        assert violations == [("b.py", new_secret)]
        violations, unused = scan({"b.py": f"URL = '{allowed}db.invalid/x'\n"})
        assert violations == [("b.py", allowed)] and unused == allowlist
        violations, _ = scan({"a.py": f"URL = '{variant}db.invalid/x'\n"})
        assert violations == [("a.py", variant)]
        violations, _ = scan({"a.py": f"A = '{allowed}db.invalid/x'\nB = '{new_secret}db.invalid/y'\n"})
        assert violations == [("a.py", new_secret)]

        bad = tmp_path / "allowlist.json"
        bad.write_text(json.dumps([{"file": "a.py", "match": allowed, "reason": " "}]), encoding="utf-8")
        with pytest.raises(ValueError):
            load_allowlist(bad)
