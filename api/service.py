"""Data access and response shaping for the FastAPI application."""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src import download_data, research
from src.strategy_engine import simulate_strategy, validate_config

from .models import BacktestRequest, DataStatus, StrategyConfig


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "yahoo"
METADATA_PATH = ROOT / "data" / "raw" / "yahoo_metadata.json"
UNIVERSE_PATH = ROOT / "data" / "universe_idx30_2026-08.csv"


class DataUnavailable(RuntimeError):
    """Raised when the local daily snapshot has not been downloaded."""


def _file_for(ticker: str) -> Path:
    return DATA_DIR / download_data.filename_for(ticker)


def normalize_ticker(ticker: str) -> str:
    value = ticker.strip().upper()
    if value.endswith(".JK") and value != "^JKSE":
        value = value[:-3]
    universe = set(pd.read_csv(UNIVERSE_PATH)["ticker"].tolist())
    if value != "^JKSE" and value not in universe:
        raise KeyError(f"Ticker {ticker} is not in the configured IDX30 universe")
    if not _file_for(value).exists():
        raise DataUnavailable("Market data is not available. Run the data refresh first.")
    return value


def _read_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def data_status() -> DataStatus:
    universe = pd.read_csv(UNIVERSE_PATH)
    tickers = ["^JKSE", *universe["ticker"].tolist()]
    files = {ticker: _file_for(ticker).exists() for ticker in tickers}
    missing = [ticker for ticker, exists in files.items() if not exists]
    metadata = _read_metadata()
    last_dates = [item.get("last_date") for item in metadata.get("files", []) if item.get("last_date")]
    last_trading_date = max(last_dates) if last_dates else None
    stale = bool(last_trading_date and last_trading_date < date.today().isoformat())
    return DataStatus(
        available=not missing,
        source=str(metadata.get("source", "Yahoo Finance chart API")),
        benchmark="^JKSE",
        universe_size=len(universe),
        fetched_at_utc=metadata.get("downloaded_at_utc"),
        requested_start=metadata.get("requested_start"),
        requested_end_exclusive=metadata.get("requested_end_exclusive"),
        last_trading_date=last_trading_date,
        missing_tickers=missing,
        stale=stale,
    )


def _ensure_data() -> None:
    status = data_status()
    if not status.available:
        raise DataUnavailable(
            "Market data is unavailable. Run `python3 src/download_data.py` or use the protected refresh endpoint."
        )


@lru_cache(maxsize=2)
def market_bundle(price_field: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    _ensure_data()
    prices, volumes, tickers = research.load_prices(price_field)
    features = research.make_features(prices, volumes)
    return prices, volumes, tickers, features


def clear_caches() -> None:
    market_bundle.cache_clear()


def ohlcv(ticker: str, start: date | None = None, end: date | None = None) -> list[dict[str, Any]]:
    normalized = normalize_ticker(ticker)
    frame = pd.read_csv(_file_for(normalized), parse_dates=["date"]).sort_values("date")
    if start:
        frame = frame[frame["date"].dt.date >= start]
    if end:
        frame = frame[frame["date"].dt.date <= end]
    frame = frame.tail(5000)
    frame = frame.replace({np.nan: None})
    result: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        row["date"] = pd.Timestamp(row["date"]).date().isoformat()
        result.append(_clean(row))
    return result


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _config_dict(config: StrategyConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, StrategyConfig):
        value = config.model_dump(exclude_none=True)
    else:
        value = dict(config)
    validate_config(value)
    return value


def _latest_signal_date(features: Mapping[str, Any]) -> pd.Timestamp:
    prices = features["monthly_prices"]
    assert isinstance(prices, pd.DataFrame)
    if prices.empty:
        raise DataUnavailable("The data snapshot has no completed monthly observations")
    return prices.index[-1]


def _factor_rows(score_date: pd.Timestamp, score: pd.DataFrame, factors: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scores = score.loc[score_date].dropna().sort_values(ascending=False)
    for rank, (ticker, value) in enumerate(scores.items(), start=1):
        contributions = {}
        for factor in factors:
            contribution = factor.contribution.loc[score_date].get(ticker, np.nan)
            raw = factor.raw.loc[score_date].get(ticker, np.nan)
            factor_rank = factor.rank.loc[score_date].get(ticker, np.nan)
            contributions[factor.factor_id] = {
                "label": factor.label,
                "raw": _clean(raw),
                "rank": _clean(factor_rank),
                "contribution": _clean(contribution),
            }
        rows.append({"ticker": ticker, "rank": rank, "score": _clean(value), "factors": contributions})
    return rows


def rankings(config: StrategyConfig | Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    config = config or StrategyConfig()
    config_dict = _config_dict(config)
    _, _, _, features = market_bundle(config_dict["price_field"])
    simulation, weights, factors = simulate_strategy(features, config_dict)
    del simulation
    from src.strategy_engine import score_strategy

    score, _ = score_strategy(features, config_dict)
    score_date = _latest_signal_date(features)
    rows = _factor_rows(score_date, score, factors)
    selected = set(weights.loc[score_date][weights.loc[score_date] > 0].index)
    for row in rows:
        row["selected"] = row["ticker"] in selected
    return rows


def stock_features(ticker: str, config: StrategyConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_ticker(ticker)
    config = config or StrategyConfig()
    config_dict = _config_dict(config)
    prices, _, _, features = market_bundle(config_dict["price_field"])
    simulation, weights, factors = simulate_strategy(features, config_dict)
    from src.strategy_engine import score_strategy

    score, _ = score_strategy(features, config_dict)
    score_date = _latest_signal_date(features)
    ranking_rows = _factor_rows(score_date, score, factors)
    row = next((item for item in ranking_rows if item["ticker"] == normalized), None)
    if row is None:
        raise KeyError(f"No completed signal is available for {ticker}")
    daily = prices[normalized].dropna()
    latest = {"date": daily.index[-1].date().isoformat(), "price": _clean(daily.iloc[-1])}
    monthly_returns = features["monthly_returns"]
    assert isinstance(monthly_returns, pd.DataFrame)
    latest_month_return = monthly_returns.loc[score_date].get(normalized, np.nan)
    signal_weight = _clean(weights.loc[score_date].get(normalized, 0.0))
    return {
        "ticker": normalized,
        "signal_date": score_date.date().isoformat(),
        "latest": latest,
        "rank": row["rank"],
        "score": row["score"],
        "selected": bool(signal_weight and signal_weight > 0),
        "weight": signal_weight,
        "factors": row["factors"],
        "monthly_return": _clean(latest_month_return),
    }


def _period_metrics(simulation: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    return _clean(research.metric_row(simulation, start, end, "period"))


def backtest(request: BacktestRequest) -> dict[str, Any]:
    config = _config_dict(request.strategy)
    _, _, _, features = market_bundle(config["price_field"])
    simulation, weights, factors = simulate_strategy(features, config)
    end = request.end.isoformat() if request.end else date.today().isoformat()
    start = request.start.isoformat()
    frame = simulation.loc[(simulation.index >= start) & (simulation.index <= end)].copy()
    frame = frame.dropna(subset=["strategy_return"])
    if frame.empty:
        raise ValueError("The requested period has no completed monthly returns")
    strategy_equity = (1 + frame["strategy_return"].fillna(0)).cumprod()
    benchmark_equity = (1 + frame["benchmark_return"].fillna(0)).cumprod()
    drawdown = strategy_equity / strategy_equity.cummax() - 1
    equity = []
    for timestamp in frame.index:
        equity.append(
            {
                "date": timestamp.date().isoformat(),
                "strategy": _clean(strategy_equity.loc[timestamp]),
                "benchmark": _clean(benchmark_equity.loc[timestamp]),
                "drawdown": _clean(drawdown.loc[timestamp]),
                "turnover": _clean(frame.loc[timestamp, "turnover"]),
            }
        )
    holdings = []
    for timestamp in weights.index[(weights.index >= start) & (weights.index <= end)]:
        positions = weights.loc[timestamp]
        holdings.append(
            {
                "signal_date": timestamp.date().isoformat(),
                "positions": [
                    {"ticker": ticker, "weight": _clean(weight)}
                    for ticker, weight in positions[positions > 0].items()
                ],
            }
        )
    from src.strategy_engine import score_strategy

    score, _ = score_strategy(features, config)
    score_date = _latest_signal_date(features)
    warnings = [
        "Historical current-universe results contain survivorship and index-membership look-ahead bias.",
        "Execution slippage, taxes, price limits, suspensions, and market impact are not modeled.",
    ]
    return {
        "strategy": request.strategy,
        "signal_timing": "Signal at completed month-end t; positions are held during month t+1 (the following month).",
        "data_status": data_status(),
        "metrics": {
            "requested": _period_metrics(simulation, start, end),
            "full": _period_metrics(simulation, "2015-01-01", end),
            "validation": _period_metrics(simulation, "2022-01-01", "2023-12-31"),
            "holdout": _period_metrics(simulation, "2024-01-01", end),
        },
        "equity_curve": equity,
        "holdings": holdings[-36:],
        "latest_factors": _factor_rows(score_date, score, factors),
        "warnings": warnings,
    }


def refresh_data() -> dict[str, object]:
    start = os.getenv("DATA_START", "2015-01-01")
    tomorrow = date.today() + timedelta(days=1)
    manifest = download_data.download_universe(start=start, end=tomorrow.isoformat())
    clear_caches()
    return manifest
