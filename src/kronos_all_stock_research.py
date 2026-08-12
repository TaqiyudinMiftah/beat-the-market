"""Leakage-safe daily Kronos forecasting on the all-listed IDX panel.

Kronos was previously evaluated only on the 30-stock IDX30 snapshot. This
runner applies the official Kronos OHLCV forecaster to the current catalog,
using a trailing liquidity screen and the same next-month portfolio protocol
used by the all-listed Chronos-2, TimesFM, and factor-MLP audits.

The model is zero-shot and uses one deterministic sample per stock-month. The
current catalog is not a point-in-time universe, so any result remains an
exploratory research result rather than a live recommendation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from . import (
        all_stock_ml_research,
        foundation_robustness,
        foundation_research,
        liquid_rank_ml_research,
        ml_research,
        research,
    )
except ImportError:  # pragma: no cover - direct CLI compatibility
    from src import (
        all_stock_ml_research,
        foundation_robustness,
        foundation_research,
        liquid_rank_ml_research,
        ml_research,
        research,
    )


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "yahoo_all"
DEFAULT_UNIVERSE_PATH = ROOT / "data" / "universe_idx_all_2026-08.csv"
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2026-08-11"
DEFAULT_CAP = 300
DEFAULT_CONTEXT_DAYS = 256
DEFAULT_HORIZON_DAYS = 21
DEFAULT_TOP_K = 3
DEFAULT_COST_BPS = 25.0
DEFAULT_MIN_TRAIN_ROWS = 1_000
DEFAULT_ALPHA = 10.0
DEFAULT_MODEL_ID = "NeoQuasar/Kronos-mini"
DEFAULT_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-2k"
DEFAULT_SAMPLE_COUNT = 1
DEFAULT_OUTPUT_PREFIX = "kronos_all"
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_BLOCK_LENGTH = 3
BOOTSTRAP_SEED = 20260812
TOP_K_SENSITIVITY = (1, 3, 5, 10)
COST_SENSITIVITY_BPS = (25.0, 100.0)

KRONOS_SOURCE = "https://github.com/shiyu-coder/Kronos"
KRONOS_PAPER = "https://arxiv.org/abs/2508.02739"
IDX_METHOD_SOURCE = (
    "https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf"
)
MODEL_NAME = "kronos_all_daily"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-field", choices=("adjclose", "close"), default="adjclose")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--context-days", type=int, default=DEFAULT_CONTEXT_DAYS)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--max-signals", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--kronos-repo", default=os.getenv("KRONOS_REPO"))
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_status(kronos_repo: str | None = None) -> dict[str, Any]:
    repo = kronos_repo or os.getenv("KRONOS_REPO")
    return {
        "kronos": {
            "available": bool(
                importlib.util.find_spec("model") is not None
                or (repo and (Path(repo) / "model").exists())
            ),
            "package": "official Kronos repository",
            "model_id": DEFAULT_MODEL_ID,
            "tokenizer_id": DEFAULT_TOKENIZER_ID,
            "repo": repo,
        },
        "universe": str(DEFAULT_UNIVERSE_PATH.relative_to(ROOT)),
        "data_dir": str(DEFAULT_DATA_DIR.relative_to(ROOT)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _validate_args(args: argparse.Namespace) -> None:
    if args.cap < 30:
        raise ValueError("cap must be at least 30")
    if args.context_days < 64 or args.horizon_days < 1:
        raise ValueError("context_days must be at least 64 and horizon_days must be positive")
    if not 1 <= args.top_k <= args.cap:
        raise ValueError("top_k must be between one and cap")
    if args.cost_bps < 0 or args.min_train_rows < 60 or args.alpha <= 0:
        raise ValueError("cost, minimum training rows, and alpha are invalid")
    if args.max_signals < 0 or args.sample_count < 1:
        raise ValueError("max_signals and sample_count are invalid")
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix:
        raise ValueError("output_prefix must be a non-empty filename prefix")
    if not args.kronos_repo:
        raise ValueError("--kronos-repo or KRONOS_REPO is required")


def _read_daily_frames(data_dir: Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        filename = ticker.replace("^", "INDEX_").replace("/", "_") + ".csv"
        path = data_dir / filename
        frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)
        frame["date"] = dates
        frames[ticker] = frame.reset_index(drop=True)
    return frames


def _make_context(
    ticker: str,
    frame: pd.DataFrame,
    signal_date: pd.Timestamp,
    price_field: str,
    context_days: int,
) -> foundation_research.DailyContext | None:
    """Build an OHLCV context using only rows dated on or before the signal."""
    available = frame.loc[frame["date"] <= signal_date].copy()
    price_column = price_field if price_field in available.columns else "close"
    required = [price_column, "open", "high", "low", "close"]
    available = available.dropna(subset=required).tail(context_days)
    if len(available) < context_days:
        return None
    prices = pd.to_numeric(available[price_column], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
        return None
    ohlcv = available[["open", "high", "low", "close"]].astype(float).copy()
    volume = available["volume"] if "volume" in available.columns else pd.Series(0.0, index=available.index)
    ohlcv["volume"] = pd.to_numeric(volume, errors="coerce").fillna(0.0).to_numpy()
    ohlcv["amount"] = ohlcv["close"] * ohlcv["volume"]
    return foundation_research.DailyContext(
        ticker=ticker,
        dates=pd.DatetimeIndex(available["date"]),
        log_prices=np.log(prices),
        ohlcv=ohlcv.reset_index(drop=True),
    )


def _metric_periods(start: str, end: str) -> tuple[tuple[str, str, str], ...]:
    return (
        ("train", start, "2021-12-31"),
        ("validation", "2022-01-01", "2023-12-31"),
        ("holdout_2024", "2024-01-01", "2024-12-31"),
        ("holdout_2025_2026", "2025-01-01", end),
        ("full", start, end),
    )


def _metric_map(simulation: pd.DataFrame, start: str, end: str) -> dict[str, dict[str, Any]]:
    return {
        label: ml_research.period_metrics(simulation, period_start, period_end, label)
        for label, period_start, period_end in _metric_periods(start, end)
    }


def _format_metric(value: Any) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _summary_table(
    metrics: dict[str, dict[str, dict[str, Any]]],
    rolling_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    gates: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "| model | validation excess | 2024 excess | 2025–2026 excess | rolling positive | bootstrap lower | holdout beta | holdout drawdown | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, values in metrics.items():
        rolling_row = rolling_summary.loc[
            rolling_summary["model"].eq(model)
            & np.isclose(rolling_summary["cost_bps"], 25.0)
        ]
        bootstrap_row = bootstrap.loc[bootstrap["model"].eq(model)]
        validation = values.get("validation", {})
        holdout_2024 = values.get("holdout_2024", {})
        holdout = values.get("holdout_2025_2026", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    _format_metric(validation.get("excess_cagr")),
                    _format_metric(holdout_2024.get("excess_cagr")),
                    _format_metric(holdout.get("excess_cagr")),
                    _format_metric(
                        rolling_row.iloc[0]["positive_excess_fraction"]
                        if not rolling_row.empty
                        else np.nan
                    ),
                    _format_metric(
                        bootstrap_row.iloc[0]["market_ci_lower"]
                        if not bootstrap_row.empty
                        else np.nan
                    ),
                    _format_metric(holdout.get("beta_to_benchmark")),
                    _format_metric(holdout.get("strategy_max_drawdown")),
                    "passes" if gates.get(model, {}).get("passed") else "rejects",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _bootstrap_rows(
    simulations: dict[str, pd.DataFrame],
    end: str,
    control_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position, (name, simulation) in enumerate(simulations.items()):
        if name == control_name:
            continue
        market = simulation[["benchmark_return"]].rename(
            columns={"benchmark_return": "strategy_return"}
        )
        market_result = liquid_rank_ml_research._block_bootstrap_active(
            simulation,
            market,
            "2024-01-01",
            end,
            BOOTSTRAP_SEED + position,
        )
        control_result = liquid_rank_ml_research._block_bootstrap_active(
            simulation,
            simulations[control_name],
            "2024-01-01",
            end,
            BOOTSTRAP_SEED + 100 + position,
        )
        rows.append(
            {
                "model": name,
                "months": market_result["months"],
                "market_active_annualized_mean": market_result["mean"],
                "market_ci_lower": market_result["lower"],
                "market_ci_median": market_result["median"],
                "market_ci_upper": market_result["upper"],
                "market_positive_probability": market_result["positive_probability"],
                "control_active_annualized_mean": control_result["mean"],
                "control_ci_lower": control_result["lower"],
                "control_ci_median": control_result["median"],
                "control_ci_upper": control_result["upper"],
                "control_positive_probability": control_result["positive_probability"],
                "samples": BOOTSTRAP_SAMPLES,
                "block_length": BOOTSTRAP_BLOCK_LENGTH,
            }
        )
    return pd.DataFrame(rows)


def _write_outputs(
    *,
    config: argparse.Namespace,
    metrics: dict[str, dict[str, dict[str, Any]]],
    diagnostics: pd.DataFrame,
    rolling: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    sensitivity: pd.DataFrame,
    context_rows: pd.DataFrame,
    training_rows: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    gates: dict[str, dict[str, Any]],
    validation_winner: str | None,
    kronos_validation_winner: str | None,
    preferred: str | None,
    loaded_tickers: list[str],
    skipped: list[str],
    errors: list[str],
    signal_count: int,
    forecast_count: int,
) -> None:
    config.report_dir.mkdir(parents=True, exist_ok=True)

    def output_path(suffix: str) -> Path:
        return config.report_dir / f"{config.output_prefix}_{suffix}"

    metrics_rows = [
        {"model": model, **period_values}
        for model, model_values in metrics.items()
        for _, period_values in model_values.items()
    ]
    pd.DataFrame(metrics_rows).to_csv(output_path("metrics.csv"), index=False)
    diagnostics.to_csv(output_path("robustness.csv"), index=False)
    rolling.to_csv(output_path("rolling.csv"), index=False)
    rolling_summary.to_csv(output_path("rolling_summary.csv"), index=False)
    bootstrap.to_csv(output_path("bootstrap.csv"), index=False)
    sensitivity.to_csv(output_path("sensitivity.csv"), index=False)
    context_rows.to_csv(output_path("context.csv"), index=False)
    training_rows.to_csv(output_path("training.csv"), index=False)

    forecast_rows: list[pd.DataFrame] = []
    for model, panel in panels.items():
        values = panel.to_numpy(dtype=float)
        row_indices, column_indices = np.where(np.isfinite(values))
        if len(row_indices):
            forecast_rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "signal_date": panel.index.to_numpy()[row_indices],
                        "ticker": panel.columns.to_numpy()[column_indices],
                        "forecast": values[row_indices, column_indices],
                    }
                )
            )
    if forecast_rows:
        pd.concat(forecast_rows, ignore_index=True).to_csv(
            output_path("forecasts.csv"), index=False
        )

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: (_relative_path(value) if isinstance(value, Path) else value)
            for key, value in vars(config).items()
        },
        "loaded_ticker_count": len(loaded_tickers),
        "skipped": skipped,
        "errors": errors,
        "signal_count": signal_count,
        "forecast_count": forecast_count,
        "metrics": metrics,
        "validation_winner": validation_winner,
        "kronos_validation_winner": kronos_validation_winner,
        "preferred_research_candidate": preferred,
        "gates": gates,
        "bootstrap": bootstrap.to_dict(orient="records"),
        "sources": {
            "kronos": KRONOS_SOURCE,
            "kronos_paper": KRONOS_PAPER,
            "idx_methodology": IDX_METHOD_SOURCE,
        },
    }
    output_path("summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    report = f"""# Kronos all-listed IDX research

Run date: {date.today().isoformat()}<br>
Model: {config.model_id}; tokenizer: {config.tokenizer_id}; sample count: {config.sample_count}<br>
Universe catalog: {_relative_path(config.universe)}; loaded tickers: {len(loaded_tickers)}; skipped: {len(skipped)}<br>
Eligibility: top {config.cap} by trailing 60-day median dollar volume at each completed month-end<br>
Context: {config.context_days} daily OHLCV observations; horizon: {config.horizon_days} business days<br>
Portfolio: top {config.top_k} equal-weight names; monthly signal; {config.cost_bps:.1f} bps one-way default cost<br>
Signal months attempted: {signal_count}; Kronos forecast values: {forecast_count}; recorded errors: {len(errors)}<br>

## Research basis

The official Kronos repository ({KRONOS_SOURCE}) and Kronos paper
({KRONOS_PAPER}) motivate testing an OHLCV-aware pretrained financial
forecaster. This audit tests transfer to Indonesian equities and evaluates
portfolio returns instead of assuming that token-level forecast accuracy is an
economic edge.

The liquidity proxy follows the official IDX80/LQ45/IDX30 methodology
({IDX_METHOD_SOURCE}). Historical membership and free-float data are not
available here, so the fixed top-{config.cap} dollar-volume screen is only a
robustness proxy.

## Fixed design and leakage controls

At each completed month-end t, only daily OHLCV observations dated on or
before t are supplied. Kronos predicts {config.horizon_days} business days and
the terminal close return becomes the next-month cross-sectional score. One
deterministic sample is used per stock-month (`sample_count={config.sample_count}`);
no holdout-selected sampling or temperature is used.

- The rank-Ridge comparator is fit expanding-window with signal dates strictly earlier than each forecast date.
- Only stocks eligible at t with a complete trailing daily OHLCV context are forecast. Missing history is dropped, not forward-filled.
- Costs, top K, rolling windows, bootstrap settings, model IDs, and context/horizon are fixed before holdout inspection.
- The current catalog omits historical delistings, suspensions, and membership changes. This cannot establish a live edge.
- Yahoo Finance data do not model spreads, taxes, price limits, market impact, failed fills, or capacity.

## Audit summary

The fixed gate requires beating the matching cap{config.cap}_composite control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025–2026.

{_summary_table(metrics, rolling_summary, bootstrap, gates)}

Overall validation winner (highest validation Sharpe, then excess CAGR): {validation_winner or "none"}. Kronos-only validation winner: {kronos_validation_winner or "none"}. Preferred after the fixed gate: {preferred or "none"}. A candidate that only looks strong in a holdout is not promoted.

## Reproduction

~~~bash
KRONOS_REPO=/tmp/kronos-src.xOaDC8 PYTHONPATH=$PWD \\
  /tmp/beat-market-ml-venv/bin/python -m src.kronos_all_stock_research
~~~

For a smoke run, add `--max-signals 1`. Raw forecasts remain ignored in
`reports/{config.output_prefix}_forecasts.csv`; committed tables use the same
`{config.output_prefix}_` prefix.

This is research, not investment advice, and no backtest guarantees future performance.
"""
    output_path("findings.md").write_text(report, encoding="utf-8")


def run(config: argparse.Namespace) -> dict[str, Any]:
    if not hasattr(config, "report_dir"):
        config.report_dir = REPORT_DIR
    repo_path = str(Path(config.kronos_repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    prices, volumes, loaded_tickers, skipped = all_stock_ml_research.load_prices(
        config.data_dir,
        config.universe,
        config.price_field,
    )
    prices = prices.loc[prices.index <= config.end]
    features = all_stock_ml_research._slice_features(
        research.make_features(prices, volumes.reindex(prices.index)),
        config.start,
        config.end,
    )
    monthly_prices = features["monthly_prices"]
    monthly_returns = features["monthly_returns"]
    assert isinstance(monthly_prices, pd.DataFrame)
    assert isinstance(monthly_returns, pd.DataFrame)
    eligibility = liquid_rank_ml_research.eligibility_mask(features, config.cap)

    control_name = f"cap{config.cap}_composite"
    rank_name = f"cap{config.cap}_rank_ridge"
    panels: dict[str, pd.DataFrame] = {
        control_name: research.score_frame("composite", features).where(eligibility),
    }
    base_panel = ml_research.make_panel(features)
    rank_panel = liquid_rank_ml_research.rank_target_panel(
        liquid_rank_ml_research.filter_panel(base_panel, eligibility)
    )
    rank_predictions, rank_training = ml_research.walk_forward_predictions(
        rank_panel,
        "ridge",
        config.min_train_rows,
        config.alpha,
    )
    panels[rank_name] = rank_predictions

    target = monthly_returns.shift(-1)
    signal_dates = pd.DatetimeIndex(
        [
            timestamp
            for timestamp in monthly_prices.index
            if pd.Timestamp(config.start) <= timestamp <= pd.Timestamp(config.end)
            and target.loc[timestamp].notna().any()
        ]
    )
    if config.max_signals:
        signal_dates = signal_dates[: config.max_signals]
    panels[MODEL_NAME] = pd.DataFrame(
        np.nan,
        index=signal_dates,
        columns=sorted(monthly_prices.columns),
        dtype=float,
    )

    frames = _read_daily_frames(config.data_dir, loaded_tickers)
    forecaster = foundation_research.KronosForecaster(
        config.model_id,
        config.tokenizer_id,
        config.device,
        config.kronos_repo,
        config.sample_count,
    )
    context_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for position, signal_date in enumerate(signal_dates, start=1):
        eligible_tickers = eligibility.columns[
            eligibility.loc[signal_date].fillna(False).to_numpy()
        ].astype(str).tolist()
        contexts: list[foundation_research.DailyContext] = []
        for ticker in eligible_tickers:
            context = _make_context(
                ticker,
                frames[ticker],
                pd.Timestamp(signal_date),
                config.price_field,
                config.context_days,
            )
            if context is not None:
                contexts.append(context)
        if len(contexts) < config.top_k:
            context_rows.append(
                {
                    "signal_date": signal_date,
                    "eligible_tickers": len(eligible_tickers),
                    "context_tickers": len(contexts),
                    "status": "skipped_too_few_tickers",
                }
            )
            continue
        try:
            values = forecaster.predict(contexts, config.horizon_days)
            if len(values) != len(contexts):
                raise ValueError(f"returned {len(values)} forecasts for {len(contexts)} contexts")
            predictions = pd.Series(values, index=[context.ticker for context in contexts])
            panels[MODEL_NAME].loc[signal_date, predictions.index] = predictions.to_numpy()
            context_rows.append(
                {
                    "signal_date": signal_date,
                    "eligible_tickers": len(eligible_tickers),
                    "context_tickers": len(contexts),
                    "forecast_values": int(predictions.notna().sum()),
                    "status": "ok",
                }
            )
            if position == 1 or position % 6 == 0 or position == len(signal_dates):
                print(
                    f"[kronos-all] signal {position}/{len(signal_dates)} "
                    f"({signal_date.date()})",
                    flush=True,
                )
        except Exception as exc:
            errors.append(f"{signal_date.date()}: {type(exc).__name__}: {exc}")
            context_rows.append(
                {
                    "signal_date": signal_date,
                    "eligible_tickers": len(eligible_tickers),
                    "context_tickers": len(contexts),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    simulations = {
        name: ml_research.simulate_predictions(panel, features, config.top_k, config.cost_bps)
        for name, panel in panels.items()
    }
    metrics = {
        name: _metric_map(simulation, config.start, config.end)
        for name, simulation in simulations.items()
    }
    diagnostics = foundation_robustness.build_cost_and_block_diagnostics(
        panels,
        features,
        config.top_k,
    )
    rolling = foundation_robustness.build_rolling_diagnostics(
        panels,
        features,
        config.top_k,
        cost_bps=config.cost_bps,
    )
    rolling_summary = foundation_robustness.summarize_rolling_diagnostics(rolling)
    bootstrap = _bootstrap_rows(simulations, config.end, control_name)

    candidate_names = [name for name in panels if name != control_name]
    gates: dict[str, dict[str, Any]] = {}
    for name in candidate_names:
        passed, reasons = liquid_rank_ml_research._gate_candidate(
            name,
            metrics[name],
            metrics[control_name],
            diagnostics,
            rolling_summary,
            bootstrap,
        )
        gates[name] = {"passed": passed, "reasons": reasons}
    validation_winner = max(
        candidate_names,
        key=lambda name: (
            metrics[name].get("validation", {}).get("strategy_sharpe_rf0", -np.inf),
            metrics[name].get("validation", {}).get("excess_cagr", -np.inf),
        ),
        default=None,
    )
    kronos_validation_winner = MODEL_NAME if MODEL_NAME in candidate_names else None
    preferred = validation_winner if validation_winner and gates[validation_winner]["passed"] else None

    sensitivity_rows: list[dict[str, Any]] = []
    for top_k in TOP_K_SENSITIVITY:
        for cost_bps in COST_SENSITIVITY_BPS:
            replay = {
                name: ml_research.simulate_predictions(panel, features, top_k, cost_bps)
                for name, panel in panels.items()
            }
            for model_name, simulation in replay.items():
                for period, period_start, period_end in _metric_periods(config.start, config.end):
                    if period in {"validation", "holdout_2024", "holdout_2025_2026"}:
                        sensitivity_rows.append(
                            {
                                "model": model_name,
                                "top_k": top_k,
                                "cost_bps": cost_bps,
                                **ml_research.period_metrics(
                                    simulation,
                                    period_start,
                                    period_end,
                                    period,
                                ),
                            }
                        )
    training = pd.DataFrame(
        [
            {
                "model": rank_name,
                "signals": int(rank_predictions.notna().any(axis=1).sum()),
                "forecast_rows": int(rank_predictions.notna().sum().sum()),
                "median_training_rows": float(rank_training.dropna().median())
                if not rank_training.dropna().empty
                else np.nan,
            }
        ]
    )
    _write_outputs(
        config=config,
        metrics=metrics,
        diagnostics=diagnostics,
        rolling=rolling,
        rolling_summary=rolling_summary,
        bootstrap=bootstrap,
        sensitivity=pd.DataFrame(sensitivity_rows),
        context_rows=pd.DataFrame(context_rows),
        training_rows=training,
        panels=panels,
        gates=gates,
        validation_winner=validation_winner,
        kronos_validation_winner=kronos_validation_winner,
        preferred=preferred,
        loaded_tickers=loaded_tickers,
        skipped=skipped,
        errors=errors,
        signal_count=len(signal_dates),
        forecast_count=int(panels[MODEL_NAME].notna().sum().sum()),
    )
    return {
        "metrics": metrics,
        "gates": gates,
        "validation_winner": validation_winner,
        "kronos_validation_winner": kronos_validation_winner,
        "preferred": preferred,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    if args.status:
        print(json.dumps(package_status(args.kronos_repo), indent=2))
        return
    _validate_args(args)
    args.data_dir = args.data_dir.resolve()
    args.universe = args.universe.resolve()
    args.report_dir = REPORT_DIR
    result = run(args)
    summary = pd.DataFrame(
        [
            {
                "model": model,
                "validation_excess_cagr": values.get("validation", {}).get("excess_cagr"),
                "validation_sharpe": values.get("validation", {}).get("strategy_sharpe_rf0"),
                "holdout_2024_excess_cagr": values.get("holdout_2024", {}).get("excess_cagr"),
                "holdout_2025_2026_excess_cagr": values.get("holdout_2025_2026", {}).get(
                    "excess_cagr"
                ),
            }
            for model, values in result["metrics"].items()
        ]
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    print(
        f"\nValidation winner: {result['validation_winner']}; "
        f"Kronos: {result['kronos_validation_winner']}; "
        f"preferred: {result['preferred']}"
    )
    print(f"Wrote reports/{args.output_prefix}_findings.md and reports/{args.output_prefix}_metrics.csv")


if __name__ == "__main__":
    main()
