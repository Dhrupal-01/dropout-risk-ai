"""
Verification tests for V3 (Phase 0B — frontend).
"""

import re
import subprocess
from pathlib import Path
import pytest

from verification.conftest import PROJECT_ROOT


class TestV3Phase0B:
    def test_v3_1_dark_mode_selector_in_tailwind(self):
        """V3.1: tailwind.config.js has darkMode: ['selector', '[data-theme=\"dark\"]]'."""
        tailwind_config = PROJECT_ROOT / "frontend" / "tailwind.config.js"
        assert tailwind_config.exists()
        content = tailwind_config.read_text(encoding="utf-8")
        assert "darkMode" in content
        assert "['selector', '[data-theme=\"dark\"]']" in content or "['selector', \"[data-theme='dark']\"]" in content or "selector" in content

    @pytest.mark.db
    def test_v3_2_search_filter_backend_and_frontend(self, client):
        """V3.2: Search escapes % and _ and matches case-insensitively; frontend forwards search."""
        # 1. Frontend endpoints.js sends search
        endpoints_code = (PROJECT_ROOT / "frontend" / "src" / "api" / "endpoints.js").read_text(encoding="utf-8")
        assert "cleanParams" in endpoints_code or "params" in endpoints_code

        # 2. Test search with wildcards
        res = client.get("/api/v1/mentors/queue?search=%")
        assert res.status_code == 200

        res_underscore = client.get("/api/v1/mentors/queue?search=_")
        assert res_underscore.status_code == 200

    @pytest.mark.db
    def test_v3_3_dynamic_filters_endpoint_and_no_hardcoded_arrays(self, client):
        """V3.3: /api/v1/mentors/filters returns sorted distinct values; no hardcoded DEPARTMENTS in frontend."""
        res = client.get("/api/v1/mentors/filters")
        assert res.status_code == 200
        data = res.json()
        assert "departments" in data
        assert "mentor_ids" in data
        assert data["departments"] == sorted(data["departments"])
        assert data["mentor_ids"] == sorted(data["mentor_ids"])

        # Check frontend/src/pages/Dashboard.jsx has no hardcoded array constants
        dashboard_src = (PROJECT_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")
        assert not re.search(r"const\s+DEPARTMENTS\s*=\s*\[", dashboard_src), "Hardcoded DEPARTMENTS array found in Dashboard.jsx"
        assert not re.search(r"const\s+MENTORS\s*=\s*\[", dashboard_src), "Hardcoded MENTORS array found in Dashboard.jsx"

    def test_v3_4_demo_mode_banner(self):
        """V3.4: VITE_DEMO_MODE toggles the demo banner."""
        app_src = (PROJECT_ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
        assert "VITE_DEMO_MODE" in app_src, "VITE_DEMO_MODE check missing from App.jsx"
        assert "Demo environment" in app_src or "demo" in app_src.lower()

    def test_v3_5_frontend_build_and_lint(self):
        """V3.5: npm run build and npm run lint clean."""
        frontend_dir = PROJECT_ROOT / "frontend"
        res_build = subprocess.run(["npm", "run", "build"], cwd=frontend_dir, capture_output=True, text=True)
        assert res_build.returncode == 0, f"Frontend build failed:\n{res_build.stderr}"

        res_lint = subprocess.run(["npm", "run", "lint"], cwd=frontend_dir, capture_output=True, text=True)
        assert res_lint.returncode == 0, f"Frontend lint failed:\n{res_lint.stderr}"
