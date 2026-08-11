"""Shared factor-block strategy calculations for the CLI and web API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import research


DEFAULT_STRATEGY: dict[str, Any] = {
    "version": 1,
    "name": "Composite momentum",
    "top_k": 3,
    "rebalance": "monthly",
    "cost_bps": 25.0,
    "price_field": "adjclose",
    "benchmark": "^JKSE",
    "factors": [
        {
            "id": "momentum_12_1",
            "type": "momentum",
            "lookback_months": 12,
            "skip_months": 1,
            "weight": 0.40,
            "direction": "high",
            "label": "12–1 month momentum",
        },
        {
            "id": "momentum_6_1",
            "type": "momentum",
            "lookback_months": 6,
            "skip_months": 1,
            "weight": 0.30,
            "direction": "high",
            "label": "6–1 month momentum",
        },
        {
            "id": "momentum_3",
            "type": "momentum",
            "lookback_months": 3,
            "skip_months": 0,
            "weight": 0.20,
            "direction": "high",
            "label": "3-month momentum",
        },
        {
            "id": "volatility_60",
            "type": "volatility",
            "window_days": 60,
            "weight": 0.10,
            "direction": "low",
            "label": "Low 60-day volatility",
        },
    ],
}

SUPPORTED_TYPES = {"momentum", "volatility", "trend", "liquidity"}


@dataclass(frozen=True)
class FactorResult:
    factor_id: str
    label: str
    raw: pd.DataFrame
    rank: pd.DataFrame
    contribution: pd.DataFrame


def clone_default() -> dict[str, Any]:
    """Return a detached default strategy configuration."""
    import copy

    return copy.deepcopy(DEFAULT_STRATEGY)


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate a strategy configuration independent of the web framework."""
    factors = config.get("factors")
    if not isinstance(factors, Sequence) or isinstance(factors, (str, bytes)) or not factors:
        raise ValueError("At least one factor block is required")
    if len(factors) > 8:
        raise ValueError("A strategy can contain at most eight factor blocks")
    top_k = int(config.get("top_k", 3))
    if not 1 <= top_k <= 30:
        raise ValueError("top_k must be between 1 and 30")
    cost_bps = float(config.get("cost_bps", 25.0))
    if not 0 <= cost_bps <= 500:
        raise ValueError("cost_bps must be between 0 and 500")
    if config.get("rebalance", "monthly") != "monthly":
        raise ValueError("Only monthly rebalancing is supported in v1")
    weights = []
    for factor in factors:
        if not isinstance(factor, Mapping):
            raise ValueError("Each factor block must be an object")
        factor_type = factor.get("type")
        if factor_type not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported factor type: {factor_type}")
        weight = float(factor.get("weight", 0))
        if not 0 < weight <= 1:
            raise ValueError("Factor weights must be greater than zero and at most one")
        if factor.get("direction", "high") not in {"high", "low"}:
            raise ValueError("Factor direction must be high or low")
        if factor_type == "momentum":
            lookback = int(factor.get("lookback_months", 0))
            skip = int(factor.get("skip_months", 0))
            if not 1 <= lookback <= 36 or not 0 <= skip <= 12:
                raise ValueError("Momentum lookback must be 1–36 months and skip 0–12 months")
        else:
            window = int(factor.get("window_days", 0))
            if not 20 <= window <= 400:
                raise ValueError("Daily factor windows must be between 20 and 400 days")
        weights.append(weight)
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError("Factor weights must sum to 1.0")


def _rank(values: pd.DataFrame, direction: str) -> pd.DataFrame:
    prepared = values if direction == "high" else -values
    return prepared.rank(axis=1, pct=True, method="average")


def _monthly_factor(values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return research.completed_monthly_last(values)


def factor_values(features: Mapping[str, pd.DataFrame | pd.Series], factor: Mapping[str, Any]) -> pd.DataFrame:
    """Calculate one raw factor panel, labelled at completed month ends."""
    factor_type = str(factor["type"])
    if factor_type == "momentum":
        prices = features["monthly_prices"]
        assert isinstance(prices, pd.DataFrame)
        lookback = int(factor.get("lookback_months", 12))
        skip = int(factor.get("skip_months", 0))
        end = prices.shift(skip) if skip else prices
        return end / prices.shift(lookback) - 1.0

    daily_prices = features["daily_prices"]
    daily_returns = features["daily_returns"]
    dollar_volume = features["dollar_volume"]
    assert isinstance(daily_prices, pd.DataFrame)
    assert isinstance(daily_returns, pd.DataFrame)
    assert isinstance(dollar_volume, pd.DataFrame)
    window = int(factor.get("window_days", 60))
    min_periods = max(20, int(window * 0.66))
    if factor_type == "volatility":
        raw = daily_returns.rolling(window, min_periods=min_periods).std() * np.sqrt(252)
    elif factor_type == "trend":
        raw = daily_prices / daily_prices.rolling(window, min_periods=min_periods).mean() - 1.0
    elif factor_type == "liquidity":
        raw = dollar_volume.rolling(window, min_periods=min_periods).median()
    else:  # pragma: no cover - validate_config guards this path
        raise ValueError(f"Unsupported factor type: {factor_type}")
    result = _monthly_factor(raw)
    assert isinstance(result, pd.DataFrame)
    return result


def score_strategy(
    features: Mapping[str, pd.DataFrame | pd.Series],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[FactorResult]]:
    """Return the composite score and factor-level explanation panels."""
    validate_config(config)
    contributions: list[pd.DataFrame] = []
    factor_results: list[FactorResult] = []
    for index, factor in enumerate(config["factors"]):
        factor_id = str(factor.get("id") or f"factor_{index + 1}")
        label = str(factor.get("label") or factor_id)
        raw = factor_values(features, factor)
        rank = _rank(raw, str(factor.get("direction", "high")))
        contribution = rank * float(factor["weight"])
        contributions.append(contribution)
        factor_results.append(FactorResult(factor_id, label, raw, rank, contribution))
    score = pd.concat(contributions).groupby(level=0).sum(min_count=len(contributions))
    return score, factor_results


def weights_from_scores(scores: pd.DataFrame, monthly_prices: pd.DataFrame, top_k: int) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for timestamp, row in scores.iterrows():
        eligible = row.dropna().index.intersection(monthly_prices.loc[timestamp].dropna().index)
        if len(eligible) == 0:
            continue
        chosen = row.loc[eligible].nlargest(min(top_k, len(eligible))).index
        weights.loc[timestamp, chosen] = 1.0 / len(chosen)
    return weights


def simulate_strategy(
    features: Mapping[str, pd.DataFrame | pd.Series],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[FactorResult]]:
    """Simulate a supported factor-block strategy with one-month signal lag."""
    validate_config(config)
    scores, factors = score_strategy(features, config)
    monthly_prices = features["monthly_prices"]
    monthly_returns = features["monthly_returns"]
    benchmark_returns = features["benchmark_returns"]
    assert isinstance(monthly_prices, pd.DataFrame)
    assert isinstance(monthly_returns, pd.DataFrame)
    assert isinstance(benchmark_returns, pd.Series)
    weights = weights_from_scores(scores, monthly_prices, int(config.get("top_k", 3)))
    next_returns = monthly_returns.shift(-1)
    next_benchmark = benchmark_returns.shift(-1)
    gross = (weights * next_returns).sum(axis=1, min_count=1)
    trade_turnover = research.turnover(weights)
    net = gross - trade_turnover * float(config.get("cost_bps", 25.0)) / 10_000.0
    simulation = pd.DataFrame(
        {
            "strategy_return": net,
            "gross_return": gross,
            "benchmark_return": next_benchmark,
            "turnover": trade_turnover,
        },
        index=weights.index,
    )
    simulation["active_return"] = simulation["strategy_return"] - simulation["benchmark_return"]
    simulation["strategy_equity"] = (1.0 + simulation["strategy_return"].fillna(0.0)).cumprod()
    simulation["benchmark_equity"] = (1.0 + simulation["benchmark_return"].fillna(0.0)).cumprod()
    simulation["active_equity"] = (1.0 + simulation["active_return"].fillna(0.0)).cumprod()
    return simulation, weights, factors


def default_parity_score(features: Mapping[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
    """Score the default strategy; useful for parity tests against research.py."""
    scores, _ = score_strategy(features, DEFAULT_STRATEGY)
    return scores
