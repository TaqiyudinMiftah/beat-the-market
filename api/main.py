"""Public FastAPI application for the stock research dashboard."""

from __future__ import annotations

import hmac
import os
from datetime import date

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import service
from .models import BacktestRequest, DataStatus, StrategyConfig


def _origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:4173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="Beat the Market API",
    version="1.0.0",
    description="Explainable Indonesian equity rankings and monthly backtests.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Refresh-Token"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ok", "service": "beat-the-market-api"}


@app.get("/api/data-status", response_model=DataStatus)
def get_data_status() -> DataStatus:
    return service.data_status()


@app.get("/api/strategies/default", response_model=StrategyConfig)
def get_default_strategy() -> StrategyConfig:
    return StrategyConfig()


@app.get("/api/universe")
def get_universe() -> dict[str, object]:
    stocks = service.catalog_frame().to_dict(orient="records")
    return {
        "benchmark": "^JKSE",
        "research_universe": "IDX30",
        "stocks": stocks,
    }


@app.get("/api/stocks/{ticker}/ohlcv")
def get_ohlcv(
    ticker: str,
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
) -> dict[str, object]:
    try:
        return {"ticker": ticker.upper(), "interval": "1d", "data": service.ohlcv(ticker, start, end)}
    except (KeyError, service.DataUnavailable) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/stocks/{ticker}/features")
def get_features(ticker: str) -> dict[str, object]:
    try:
        return service.stock_features(ticker)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except service.DataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/rankings")
def get_rankings(
    top_k: int = Query(default=3, ge=1, le=30),
    cost_bps: float = Query(default=25.0, ge=0, le=500),
    price_field: str = Query(default="adjclose", pattern="^(adjclose|close)$"),
) -> dict[str, object]:
    config = StrategyConfig(top_k=top_k, cost_bps=cost_bps, price_field=price_field)
    try:
        return {"as_of": service._latest_signal_date(service.market_bundle(price_field)[3]).date().isoformat(), "rankings": service.rankings(config)}
    except service.DataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/backtests")
def post_backtest(request: BacktestRequest) -> dict[str, object]:
    try:
        return service.backtest(request)
    except (ValueError, service.DataUnavailable) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _run_refresh() -> None:
    try:
        service.refresh_data()
    except Exception:  # pragma: no cover - surfaced through freshness status/logging
        import logging

        logging.getLogger(__name__).exception("Market data refresh failed")


@app.post("/api/admin/refresh", status_code=202)
def post_refresh(
    background_tasks: BackgroundTasks,
    x_refresh_token: str | None = Header(default=None),
) -> dict[str, str]:
    expected = os.getenv("REFRESH_TOKEN")
    if not expected or not x_refresh_token or not hmac.compare_digest(x_refresh_token, expected):
        raise HTTPException(status_code=403, detail="A valid refresh token is required")
    background_tasks.add_task(_run_refresh)
    return {"status": "refresh_started"}
