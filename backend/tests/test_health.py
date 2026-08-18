"""Health endpoint, CORS wiring, and router registration."""


class TestHealthEndpoint:
    def test_health_returns_the_handover_shape(self, client):
        response = client.get("/health")
        assert response.status_code == 200

        body = response.json()
        assert set(("status", "database", "model_loaded")).issubset(body)
        assert body["status"] in ("healthy", "degraded")
        assert body["database"] in ("connected", "disconnected")
        assert isinstance(body["model_loaded"], bool)

    def test_model_loaded_by_lifespan(self, client):
        """The lifespan handler must have loaded artifacts before the first request."""
        body = client.get("/health").json()
        assert body["model_loaded"] is True
        assert body["feature_count"] == 37
        assert body["model_version"].startswith("calibrated-")

    def test_no_credentials_leak(self, client):
        """Health output must never contain connection strings or secrets."""
        raw = client.get("/health").text.lower()
        for marker in ("password", "postgresql://", "postgres://", "@ep-", "sslmode"):
            assert marker not in raw

    def test_root_endpoint(self, client):
        body = client.get("/").json()
        assert body["health"] == "/health"


class TestApplicationWiring:
    def test_openapi_schema_builds(self, client):
        """A broken schema means broken response models somewhere."""
        assert client.get("/openapi.json").status_code == 200

    def test_v1_router_mounted(self, client):
        """Stage 2 inference endpoints are mounted under the configured v1 prefix."""
        from backend.app.core.config import get_settings

        paths = client.get("/openapi.json").json()["paths"]
        prefix = get_settings().API_V1_PREFIX

        assert "/health" in paths
        for route in (
            f"{prefix}/predict",
            f"{prefix}/predict/batch",
            f"{prefix}/predict/batch/csv",
            f"{prefix}/students/{{student_id}}/explanation",
        ):
            assert route in paths, f"missing route: {route}"

    def test_predict_routes_are_post_only(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert set(paths["/api/v1/predict"]) == {"post"}
        assert set(paths["/api/v1/students/{student_id}/explanation"]) == {"get"}

    def test_cors_preflight_allows_react_dev_origin(self, client):
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

    def test_cors_rejects_unknown_origin(self, client):
        response = client.options(
            "/health",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in response.headers

    def test_wildcard_origin_is_never_echoed(self, client):
        response = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert response.headers.get("access-control-allow-origin") != "*"
