"""Typed request and response models for the public research API."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.strategy_engine import clone_default


FactorType = Literal["momentum", "volatility", "trend", "liquidity"]
Direction = Literal["high", "low"]


class FactorBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    type: FactorType
    lookback_months: int | None = Field(default=None, ge=1, le=36)
    skip_months: int = Field(default=0, ge=0, le=12)
    window_days: int | None = Field(default=None, ge=20, le=400)
    weight: float = Field(gt=0, le=1)
    direction: Direction = "high"
    label: str | None = None


def default_factor_blocks() -> list[FactorBlock]:
    return [FactorBlock.model_validate(item) for item in clone_default()["factors"]]


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    name: str = "Composite momentum"
    factors: list[FactorBlock] = Field(default_factory=default_factor_blocks, min_length=1, max_length=8)
    top_k: int = Field(default=3, ge=1, le=30)
    rebalance: Literal["monthly"] = "monthly"
    cost_bps: float = Field(default=25.0, ge=0, le=500)
    price_field: Literal["adjclose", "close"] = "adjclose"
    benchmark: Literal["^JKSE"] = "^JKSE"


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    start: date = date(2015, 1, 1)
    end: date | None = None


class ErrorResponse(BaseModel):
    detail: str


class DataStatus(BaseModel):
    available: bool
    source: str
    benchmark: str
    universe_size: int
    fetched_at_utc: str | None = None
    requested_start: str | None = None
    requested_end_exclusive: str | None = None
    last_trading_date: str | None = None
    missing_tickers: list[str] = Field(default_factory=list)
    stale: bool = False


class BacktestResponse(BaseModel):
    strategy: StrategyConfig
    signal_timing: str
    data_status: DataStatus
    metrics: dict[str, dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    holdings: list[dict[str, Any]]
    latest_factors: list[dict[str, Any]]
    warnings: list[str]
