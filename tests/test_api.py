from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api import service


client = TestClient(app)
DATA_READY = (service.DATA_DIR / "INDEX_JKSE.csv").exists()


def test_health_and_default_strategy() -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    response = client.get("/api/strategies/default")
    assert response.status_code == 200
    assert response.json()["top_k"] == 3
    assert sum(item["weight"] for item in response.json()["factors"]) == pytest.approx(1.0)


@pytest.mark.skipif(not DATA_READY, reason="download the local market snapshot first")
def test_rankings_and_stock_features_use_cached_data() -> None:
    rankings = client.get("/api/rankings")
    assert rankings.status_code == 200
    assert len(rankings.json()["rankings"]) == 30
    assert sum(item["selected"] for item in rankings.json()["rankings"]) == 3

    features = client.get("/api/stocks/BBCA/features")
    assert features.status_code == 200
    assert features.json()["ticker"] == "BBCA"
    assert "momentum_12_1" in features.json()["factors"]


@pytest.mark.skipif(not DATA_READY, reason="download the local market snapshot first")
def test_backtest_exposes_timing_and_warnings() -> None:
    response = client.post("/api/backtests", json={"start": "2024-01-01"})

    assert response.status_code == 200
    payload = response.json()
    assert "following month" in payload["signal_timing"]
    assert payload["metrics"]["requested"]["months"] > 0
    assert payload["warnings"]


def test_unknown_ticker_is_rejected() -> None:
    response = client.get("/api/stocks/NOTREAL/features")
    assert response.status_code == 404
