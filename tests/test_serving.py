"""Tests for FastAPI serving layer (Phase 7).

Verifies:
- /health endpoint status, backend, device, uptime
- /generate endpoint Pydantic validation, SQL generation, latency tracking
- 5 sequential requests returning non-empty SQL
- /metrics endpoint counters, percentiles, throughput
"""

import os
import pytest

# Configure mock model environment before loading app
os.environ["MOCK_MODEL"] = "1"
os.environ["BACKEND"] = "hf"

from fastapi.testclient import TestClient
from serving.app import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "sql-forge-1.5b"
    assert data["backend"] in ("hf", "dynamic")
    assert "uptime_seconds" in data

def test_generate_endpoint_validation(client):
    # Empty schema should fail validation
    resp_empty = client.post("/generate", json={"schema": "a", "question": "valid question?"})
    assert resp_empty.status_code == 422

    # Valid schema and question
    req_body = {
        "schema": "CREATE TABLE students (id INTEGER, name TEXT);",
        "question": "List all student names"
    }
    resp = client.post("/generate", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert "sql" in data
    assert len(data["sql"]) > 0
    assert data["model"] == "sql-forge-1.5b"
    assert data["output_tokens"] > 0
    assert data["latency_ms"] > 0

def test_five_sequential_requests(client):
    """Phase 7 requirement: 5 sequential requests checking SQL is non-empty."""
    samples = [
        ("CREATE TABLE dept (id INT, dname TEXT);", "Get all departments"),
        ("CREATE TABLE emp (id INT, sal REAL);", "Find highest salary"),
        ("CREATE TABLE orders (id INT, total REAL);", "Sum of all order totals"),
        ("CREATE TABLE products (pid INT, price REAL);", "Count items with price over 100"),
        ("CREATE TABLE accounts (acc_id INT, bal REAL);", "List accounts with positive balance")
    ]

    for schema, question in samples:
        resp = client.post("/generate", json={"schema": schema, "question": question})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sql"] is not None
        assert len(data["sql"].strip()) > 0
        assert data["sql"].strip().endswith(";")

def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_count"] >= 6
    assert data["error_count"] == 0
    assert data["latency_p50_ms"] >= 0
    assert data["mean_tokens_per_sec"] >= 0
