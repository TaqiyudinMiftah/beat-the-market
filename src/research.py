"""Run monthly cross-sectional backtests on the cached IDX30 data.

Signals are measured at month-end and positions are held during the following
calendar month. This one-period lag is the key guard against using future prices.
The default price field is Yahoo's adjusted close, which handles splits and
distributions; use --price-field close for a price-only sensitivity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "data" / "universe_idx30_2026-08.csv"
DATA_DIR = ROOT / "data" / "raw" / "yahoo"
REPORT_DIR = ROOT / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-field", choices=("adjclose", "close"), default="adjclose")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, default=25.0, help="One-way cost per unit turnover")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-08-11")
    return parser.parse_args()


def load_prices(price_field: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    universe = pd.read_csv(UNIVERSE_PATH)
    tickers = universe["ticker"].tolist()
    all_tickers = ["^JKSE", *tickers]
    price_series: dict[str, pd.Series] = {}
    volume_series: dict[str, pd.Series] = {}
    for ticker in all_tickers:
        file_name = ticker.replace("^", "INDEX_").replace("/", "_") + ".csv"
        path = DATA_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run src/download_data.py first")
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        if price_field not in frame or frame[price_field].dropna().empty:
            raise ValueError(f"No usable {price_field} for {ticker}")
        price_series[ticker] = frame[price_field].astype(float)
        volume_series[ticker] = frame["volume"].astype(float)
    prices = pd.concat(price_series, axis=1).sort_index()
    volumes = pd.concat(volume_series, axis=1).sort_index()
    return prices, volumes, tickers


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True, method="average")


def completed_monthly_last(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Return month-end observations, excluding a partially observed last month."""
    monthly = frame.resample("ME").last()
    observed = frame.dropna(how="all") if isinstance(frame, pd.DataFrame) else frame.dropna()
    if not observed.empty and not monthly.empty and monthly.index[-1] > observed.index[-1]:
        monthly = monthly.iloc[:-1]
    return monthly


def make_features(prices: pd.DataFrame, volumes: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    market = prices["^JKSE"]
    stocks = prices.drop(columns="^JKSE")
    dollar_volume = stocks * volumes.drop(columns="^JKSE")

    month_end_prices = completed_monthly_last(stocks)
    month_end_market = completed_monthly_last(market)
    monthly_returns = month_end_prices.pct_change(fill_method=None)
    benchmark_returns = month_end_market.pct_change(fill_method=None)

    # t-12 through t-1; the most recent month is excluded from the signal.
    mom12_1 = month_end_prices.shift(1) / month_end_prices.shift(12) - 1.0
    mom6_1 = month_end_prices.shift(1) / month_end_prices.shift(6) - 1.0
    mom3 = month_end_prices / month_end_prices.shift(3) - 1.0

    daily_returns = market.to_frame().join(stocks).pct_change(fill_method=None)
    volatility = daily_returns[stocks.columns].rolling(60, min_periods=40).std() * np.sqrt(252)
    trend_sma200 = stocks / stocks.rolling(200, min_periods=150).mean() - 1.0
    avg_dollar_volume = dollar_volume.rolling(60, min_periods=40).median()

    month_end_volatility = completed_monthly_last(volatility)
    month_end_trend = completed_monthly_last(trend_sma200)
    month_end_liquidity = completed_monthly_last(avg_dollar_volume)
    market_sma200 = market / market.rolling(200, min_periods=150).mean() - 1.0
    market_trend = completed_monthly_last(market_sma200)
    market_mom12_1 = month_end_market.shift(1) / month_end_market.shift(12) - 1.0
    risk_on = (market_trend > 0.0) & (market_mom12_1 > 0.0)

    return {
        "monthly_prices": month_end_prices,
        "daily_prices": stocks,
        "daily_returns": stocks.pct_change(fill_method=None),
        "dollar_volume": dollar_volume,
        "monthly_returns": monthly_returns,
        "benchmark_returns": benchmark_returns,
        "mom12_1": mom12_1,
        "mom6_1": mom6_1,
        "mom3": mom3,
        "volatility": month_end_volatility,
        "trend200": month_end_trend,
        "liquidity": month_end_liquidity,
        "market_trend": market_trend,
        "market_mom12_1": market_mom12_1,
        "risk_on": risk_on,
    }


def score_frame(name: str, f: dict[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
    mom12 = f["mom12_1"]
    mom6 = f["mom6_1"]
    mom3 = f["mom3"]
    vol = f["volatility"]
    trend = f["trend200"]
    assert isinstance(mom12, pd.DataFrame)
    assert isinstance(mom6, pd.DataFrame)
    assert isinstance(mom3, pd.DataFrame)
    assert isinstance(vol, pd.DataFrame)
    assert isinstance(trend, pd.DataFrame)
    ranks = {
        "mom12_1": cross_sectional_rank(mom12),
        "mom6_1": cross_sectional_rank(mom6),
        "mom3": cross_sectional_rank(mom3),
        "lowvol": cross_sectional_rank(-vol),
        "trend200": cross_sectional_rank(trend),
    }
    if name == "mom12_1":
        return ranks["mom12_1"]
    if name == "mom6_1":
        return ranks["mom6_1"]
    if name == "mom3":
        return ranks["mom3"]
    if name == "lowvol":
        return ranks["lowvol"]
    if name == "trend_mom":
        return 0.50 * ranks["mom12_1"] + 0.30 * ranks["trend200"] + 0.20 * ranks["lowvol"]
    if name in {"composite", "composite_regime"}:
        return (
            0.40 * ranks["mom12_1"]
            + 0.30 * ranks["mom6_1"]
            + 0.20 * ranks["mom3"]
            + 0.10 * ranks["lowvol"]
        )
    raise KeyError(name)


def weights_for(
    strategy: str,
    features: dict[str, pd.DataFrame | pd.Series],
    top_k: int,
) -> pd.DataFrame:
    monthly_prices = features["monthly_prices"]
    regime_filter = strategy.endswith("_regime")
    base_strategy = strategy.removesuffix("_regime") if regime_filter else strategy
    scores = score_frame(base_strategy, features) if base_strategy != "equal_weight" else monthly_prices.copy()
    assert isinstance(monthly_prices, pd.DataFrame)
    risk_on = features["risk_on"]
    assert isinstance(risk_on, pd.Series)
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for timestamp, row in scores.iterrows():
        eligible = row.dropna().index.intersection(monthly_prices.loc[timestamp].dropna().index)
        if regime_filter and not bool(risk_on.get(timestamp, False)):
            continue
        if len(eligible) == 0:
            continue
        if base_strategy == "equal_weight":
            chosen = eligible
        else:
            chosen = row.loc[eligible].nlargest(min(top_k, len(eligible))).index
        weights.loc[timestamp, chosen] = 1.0 / len(chosen)
    return weights


def turnover(weights: pd.DataFrame) -> pd.Series:
    previous = weights.shift(1).fillna(0.0)
    return 0.5 * (weights - previous).abs().sum(axis=1)


def simulate(
    strategy: str,
    features: dict[str, pd.DataFrame | pd.Series],
    top_k: int,
    cost_bps: float,
) -> pd.DataFrame:
    monthly_returns = features["monthly_returns"]
    benchmark_returns = features["benchmark_returns"]
    assert isinstance(monthly_returns, pd.DataFrame)
    assert isinstance(benchmark_returns, pd.Series)
    weights = weights_for(strategy, features, top_k)

    # Signals and turnover at t are applied to returns in t+1.
    next_returns = monthly_returns.shift(-1)
    next_benchmark = benchmark_returns.shift(-1)
    gross = (weights * next_returns).sum(axis=1, min_count=1)
    trade_turnover = turnover(weights)
    net = gross - trade_turnover * cost_bps / 10_000.0
    out = pd.DataFrame(
        {
            "strategy_return": net,
            "gross_return": gross,
            "benchmark_return": next_benchmark,
            "turnover": trade_turnover,
            "risk_on": features["risk_on"].reindex(weights.index),
        },
        index=weights.index,
    )
    out["active_return"] = out["strategy_return"] - out["benchmark_return"]
    out["strategy_equity"] = (1.0 + out["strategy_return"].fillna(0.0)).cumprod()
    out["benchmark_equity"] = (1.0 + out["benchmark_return"].fillna(0.0)).cumprod()
    out["active_equity"] = (1.0 + out["active_return"].fillna(0.0)).cumprod()
    return out


def max_drawdown(equity: pd.Series) -> float:
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def metric_row(returns: pd.DataFrame, start: str, end: str, label: str) -> dict[str, float | str | int]:
    frame = returns.loc[(returns.index >= start) & (returns.index <= end)].dropna(subset=["strategy_return"])
    benchmark = frame["benchmark_return"].dropna()
    strategy = frame["strategy_return"]
    if frame.empty:
        return {"period": label, "months": 0}
    years = len(frame) / 12.0
    strategy_equity = (1.0 + strategy).cumprod()
    benchmark_equity = (1.0 + benchmark).cumprod()
    active = frame["active_return"].dropna()
    volatility = float(strategy.std(ddof=1) * np.sqrt(12)) if len(strategy) > 1 else np.nan
    sharpe = float(strategy.mean() / strategy.std(ddof=1) * np.sqrt(12)) if strategy.std(ddof=1) > 0 else np.nan
    beta = np.nan
    alpha = np.nan
    if len(active) > 2 and benchmark.var(ddof=1) > 0:
        aligned = frame[["strategy_return", "benchmark_return"]].dropna()
        beta = float(aligned["strategy_return"].cov(aligned["benchmark_return"]) / aligned["benchmark_return"].var())
        alpha_monthly = float(aligned["strategy_return"].mean() - beta * aligned["benchmark_return"].mean())
        alpha = (1.0 + alpha_monthly) ** 12 - 1.0
    tracking_error = float(active.std(ddof=1) * np.sqrt(12)) if len(active) > 1 else np.nan
    return {
        "period": label,
        "months": int(len(frame)),
        "strategy_total_return": float(strategy_equity.iloc[-1] - 1.0),
        "benchmark_total_return": float(benchmark_equity.iloc[-1] - 1.0),
        "strategy_cagr": float(strategy_equity.iloc[-1] ** (1.0 / years) - 1.0),
        "benchmark_cagr": float(benchmark_equity.iloc[-1] ** (1.0 / years) - 1.0),
        "excess_cagr": float(strategy_equity.iloc[-1] ** (1.0 / years) - benchmark_equity.iloc[-1] ** (1.0 / years)),
        "strategy_volatility": volatility,
        "strategy_sharpe_rf0": sharpe,
        "strategy_max_drawdown": max_drawdown(strategy_equity),
        "benchmark_max_drawdown": max_drawdown(benchmark_equity),
        "active_hit_rate": float((active > 0).mean()) if len(active) else np.nan,
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(12)) if len(active) > 1 and active.std(ddof=1) > 0 else np.nan,
        "beta_to_benchmark": beta,
        "annualized_alpha_rf0": alpha,
        "average_monthly_turnover": float(frame["turnover"].mean()),
    }


def main() -> None:
    args = parse_args()
    prices, volumes, tickers = load_prices(args.price_field)
    prices = prices.loc[(prices.index >= args.start) & (prices.index <= args.end)]
    volumes = volumes.reindex(prices.index)
    features = make_features(prices, volumes)

    strategies = ["equal_weight", "mom12_1", "mom6_1", "mom3", "lowvol", "trend_mom", "composite", "composite_regime"]
    all_rows: list[dict[str, object]] = []
    simulations: dict[str, pd.DataFrame] = {}
    splits = {
        "full": ("2015-01-01", args.end),
        "train": ("2015-01-01", "2021-12-31"),
        "validation": ("2022-01-01", "2023-12-31"),
        "test": ("2024-01-01", args.end),
    }
    for strategy in strategies:
        simulation = simulate(strategy, features, args.top_k, args.cost_bps)
        simulations[strategy] = simulation
        for split_name, (start, end) in splits.items():
            row = metric_row(simulation, start, end, split_name)
            row.update({"strategy": strategy, "price_field": args.price_field, "top_k": args.top_k, "cost_bps": args.cost_bps})
            all_rows.append(row)

    metrics = pd.DataFrame(all_rows)
    validation = metrics[metrics["period"] == "validation"].copy()
    selected = validation.sort_values(["strategy_sharpe_rf0", "excess_cagr"], ascending=False).iloc[0]["strategy"]
    selected_simulation = simulations[str(selected)]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / f"metrics_{args.price_field}.csv"
    metrics.to_csv(metrics_path, index=False)
    selected_simulation.to_csv(REPORT_DIR / f"monthly_{args.price_field}_{selected}.csv", index_label="signal_month")
    summary = {
        "research_universe": str(UNIVERSE_PATH.relative_to(ROOT)),
        "tickers": tickers,
        "benchmark": "^JKSE",
        "price_field": args.price_field,
        "top_k": args.top_k,
        "one_way_cost_bps": args.cost_bps,
        "signal_timing": "signal at month-end t; hold during month t+1",
        "strategies": strategies,
        "validation_selection_rule": "highest validation Sharpe, then excess CAGR",
        "selected_strategy": str(selected),
        "metrics_file": str(metrics_path.relative_to(ROOT)),
    }
    (REPORT_DIR / f"summary_{args.price_field}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    print(f"\nSelected on validation: {selected}")
    print(f"Wrote {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
