from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint_reports_api_and_database():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["database"] == "connected"

def test_model_schema_reports_missing_artifacts_cleanly():
    response = client.get("/api/v1/model/schema")
    assert response.status_code in (200, 503)

def test_transaction_list_is_empty_or_paginated():
    response = client.get("/api/v1/transactions?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert "items" in body and "total" in body
    assert body["limit"] == 5
