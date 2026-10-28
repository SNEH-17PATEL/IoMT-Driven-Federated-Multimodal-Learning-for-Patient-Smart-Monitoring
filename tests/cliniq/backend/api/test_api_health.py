"""Tests for GET /health."""
import pytest


@pytest.mark.api
def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


@pytest.mark.api
def test_health_status_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


@pytest.mark.api
def test_health_service_cliniq(client):
    body = client.get("/health").json()
    assert body["service"] == "ClinIQ"


@pytest.mark.api
def test_health_json_content_type(client):
    r = client.get("/health")
    assert "application/json" in r.headers["content-type"]
