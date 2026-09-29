"""
Locust load test for DropoutGuard Phase 5 PostgreSQL performance validation.
Simulates concurrent user load against an active 100,000 student database.
Target:
  /api/v1/mentors/queue p95 < 300 ms
  /api/v1/stats/summary p95 < 150 ms
"""

import random
from locust import HttpUser, between, task

DEPARTMENTS = [
    "Computer Science & Engineering",
    "Information Technology",
    "Electronics & Communication",
    "Mechanical Engineering",
    "Civil Engineering",
    "Electrical Engineering",
]
RISK_TIERS = ["High", "Medium", "Low"]
SEARCH_TERMS = ["Aarav", "Sharma", "Patel", "Singh", "BENCH_001", "BENCH_050"]


class DropoutGuardUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(4)
    def queue_default_page(self):
        """Standard mentor triage queue (page 1, limit 25)."""
        self.client.get("/api/v1/mentors/queue?limit=25", name="/api/v1/mentors/queue [default]")

    @task(3)
    def queue_filtered_dept(self):
        """Triage queue filtered by department."""
        dept = random.choice(DEPARTMENTS)
        self.client.get(
            f"/api/v1/mentors/queue?department={dept}&limit=25",
            name="/api/v1/mentors/queue [dept filter]",
        )

    @task(3)
    def queue_filtered_tier(self):
        """Triage queue filtered by risk tier."""
        tier = random.choice(RISK_TIERS)
        self.client.get(
            f"/api/v1/mentors/queue?risk_tier={tier}&limit=25",
            name="/api/v1/mentors/queue [tier filter]",
        )

    @task(2)
    def queue_search(self):
        """Trigram-accelerated search across 100k students."""
        term = random.choice(SEARCH_TERMS)
        self.client.get(
            f"/api/v1/mentors/queue?search={term}&limit=25",
            name="/api/v1/mentors/queue [trgm search]",
        )

    @task(2)
    def queue_cursor_paging(self):
        """Keyset cursor pagination across 100k students."""
        res = self.client.get("/api/v1/mentors/queue?limit=25", name="/api/v1/mentors/queue [cursor-p1]")
        if res.status_code == 200:
            data = res.json()
            cursor = data.get("next_cursor")
            if cursor:
                self.client.get(
                    f"/api/v1/mentors/queue?limit=25&cursor={cursor}",
                    name="/api/v1/mentors/queue [cursor-p2]",
                )

    @task(3)
    def stats_summary(self):
        """SQL GROUP BY summary aggregation over 100k students."""
        self.client.get("/api/v1/stats/summary", name="/api/v1/stats/summary")

    @task(1)
    def mentor_filters(self):
        """Distinct department and mentor filter lookup."""
        self.client.get("/api/v1/mentors/filters", name="/api/v1/mentors/filters")
