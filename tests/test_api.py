"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from main import app
from src.database import init_db

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_haberler_empty():
    resp = client.get("/api/haberler")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "articles" in data


def test_analiz_empty():
    resp = client.get("/api/analiz")
    assert resp.status_code == 200


def test_trends():
    resp = client.get("/api/trends?gun=7")
    assert resp.status_code == 200
    data = resp.json()
    assert "days" in data


def test_dashboard_serves_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "HaberNLP" in resp.text
