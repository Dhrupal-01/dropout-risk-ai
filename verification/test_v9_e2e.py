"""
Verification tests for V9 (End to end & security hygiene).
"""

import re
import subprocess
from pathlib import Path
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
        """V9.3: No secrets in tracked files (passwords in URLs, API keys)."""
        res = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        tracked_files = res.stdout.splitlines()

        secret_patterns = [
            re.compile(r"postgres(?:ql)?://(?!<user>:<password>|user:password|u:p|admin:sup3rs3cret|test:test)[^:\s]+:[^@\s]+@", re.IGNORECASE),
            re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----"),
            re.compile(r"AKIA[0-9A-Z]{16}"),
        ]

        violations = []
        for rel_path in tracked_files:
            file_path = PROJECT_ROOT / rel_path
            if file_path.is_file() and not file_path.name.endswith((".png", ".jpg", ".joblib", ".parquet")):
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    for pat in secret_patterns:
                        if pat.search(text):
                            violations.append(rel_path)
                            break
                except Exception:
                    pass

        assert not violations, f"Potential real secrets detected in tracked files: {violations}"
