import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    # Entering TestClient runs the FastAPI lifespan, which creates the
    # SQLAlchemy tables before endpoint tests execute.
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_reports_api_and_database(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["database"] == "connected"


def test_model_schema_reports_missing_artifacts_cleanly(client):
    response = client.get("/api/v1/model/schema")
    assert response.status_code in (200, 503)


def test_demo_sample_reports_available_dataset_or_clear_setup_error(client):
    response = client.get("/api/v1/demo/sample")
    assert response.status_code in (200, 503)


def test_transaction_list_is_empty_or_paginated(client):
    response = client.get("/api/v1/transactions?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert "items" in body and "total" in body
    assert body["limit"] == 5


def test_investigation_endpoint_reports_missing_provider_key(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.post(
        "/api/v1/investigation/chat",
        json={"message": "Explain common account takeover indicators."},
    )
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]
