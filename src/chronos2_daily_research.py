"""Leakage-safe daily Chronos-2 forecasting on the all-listed IDX panel.

This is a follow-up to the monthly Chronos-2 audit. It uses the official
Chronos-2 long-format DataFrame API with a trailing daily log-price context
and a fixed 21-business-day horizon, then ranks stocks by the terminal
forecasted return. Absolute log-price and per-series normalized log-price
contexts are compared with individual and cross-learning inference.

The model is zero-shot and optional. This runner is a research audit, not a
live trading signal. The all-listed catalog is a current snapshot and still
does not reconstruct historical delistings or index membership.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from . import (
        all_stock_ml_research,
        foundation_robustness,
        liquid_rank_ml_research,
        ml_research,
        research,
    )
except ImportError:  # pragma: no cover - direct CLI compatibility
    from src import (
        all_stock_ml_research,
        foundation_robustness,
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
DEFAULT_BATCH_SIZE = 100
DEFAULT_MODEL_ID = "amazon/chronos-2"
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_BLOCK_LENGTH = 3
BOOTSTRAP_SEED = 20260812
TOP_K_SENSITIVITY = (1, 3, 5, 10)
COST_SENSITIVITY_BPS = (25.0, 100.0)

MODEL_NAMES = (
    "chronos2_daily_abs_individual_median",
    "chronos2_daily_abs_cross_median",
    "chronos2_daily_abs_cross_lower10",
    "chronos2_daily_norm_cross_median",
    "chronos2_daily_norm_cross_lower10",
)

CHRONOS_SOURCE = "https://github.com/amazon-science/chronos-forecasting"
CHRONOS2_STUDY = "https://arxiv.org/abs/2605.21504"
IDX_METHOD_SOURCE = "https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf"


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
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-signals", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_status() -> dict[str, Any]:
    return {
        "chronos": {
            "available": importlib.util.find_spec("chronos") is not None,
            "package": "chronos-forecasting",
            "version": _version("chronos-forecasting"),
            "model_id": DEFAULT_MODEL_ID,
        },
        "universe": str(DEFAULT_UNIVERSE_PATH.relative_to(ROOT)),
        "data_dir": str(DEFAULT_DATA_DIR.relative_to(ROOT)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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
    if args.cap < 1:
        raise ValueError("cap must be at least one")
    if args.context_days < 32 or args.horizon_days < 1:
        raise ValueError("context_days must be at least 32 and horizon_days must be positive")
    if not 1 <= args.top_k <= args.cap:
        raise ValueError("top_k must be between one and cap")
    if args.cost_bps < 0 or args.min_train_rows < 60 or args.alpha <= 0:
        raise ValueError("cost, minimum training rows, and alpha are invalid")
    if args.batch_size < 1 or args.max_signals < 0:
        raise ValueError("batch_size and max_signals are invalid")


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


def _daily_context(
    frames: dict[str, pd.DataFrame],
    tickers: list[str],
    signal_date: pd.Timestamp,
    price_field: str,
    context_days: int,
    normalized: bool,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build a long daily context with no observations after signal_date."""
    rows: list[pd.DataFrame] = []
    last_logs: dict[str, float] = {}
    included: list[str] = []
    for ticker in sorted(tickers):
        frame = frames[ticker]
        price_column = price_field if price_field in frame.columns else "close"
        available = frame.loc[frame["date"] <= signal_date].dropna(
            subset=[price_column]
        ).tail(context_days)
        if len(available) < context_days:
            continue
        prices = pd.to_numeric(available[price_column], errors="coerce").to_numpy()
        if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
            continue
        log_prices = np.log(prices)
        last_log = float(log_prices[-1])
        target = log_prices - last_log if normalized else log_prices
        rows.append(
            pd.DataFrame(
                {
                    "item_id": ticker,
                    "timestamp": pd.DatetimeIndex(available["date"]),
                    "target": target,
                }
            )
        )
        last_logs[ticker] = last_log
        included.append(ticker)
    if not rows:
        return (
            pd.DataFrame(columns=["item_id", "timestamp", "target"]),
            pd.Series(dtype=float),
            [],
        )
    context = pd.concat(rows, ignore_index=True)
    context = context.sort_values(["item_id", "timestamp"], kind="stable").reset_index(drop=True)
    if not np.isfinite(context["target"].to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite target in context ending {signal_date.date()}")
    return context, pd.Series(last_logs, dtype=float), included


def _load_pipeline(model_id: str, device: str) -> Any:
    from chronos import Chronos2Pipeline

    device_map = "cuda:0" if device == "cuda" else device
    return Chronos2Pipeline.from_pretrained(model_id, device_map=device_map)


def _predict_daily(
    pipeline: Any,
    context: pd.DataFrame,
    last_logs: pd.Series,
    horizon_days: int,
    context_days: int,
    batch_size: int,
    device: str,
    cross_learning: bool,
    normalized: bool,
) -> pd.DataFrame:
    """Return terminal lower/median/upper return forecasts by ticker."""
    del device  # reserved for parity with the CLI and future pipeline options
    output = pipeline.predict_df(
        context,
        prediction_length=horizon_days,
        context_length=context_days,
        batch_size=batch_size,
        cross_learning=cross_learning,
        validate_inputs=True,
        freq="B",
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        quantile_levels=[0.1, 0.5, 0.9],
    )
    output = output.sort_values(["item_id", "timestamp"], kind="stable")
    terminal = output.groupby("item_id", sort=False).tail(1).set_index("item_id")
    forecasts: dict[str, pd.Series] = {}
    for quantile, name in (("0.1", "lower10"), ("0.5", "median"), ("0.9", "upper90")):
        if quantile not in terminal:
            raise ValueError(f"Chronos-2 output is missing quantile {quantile}")
        delta = pd.to_numeric(terminal[quantile], errors="coerce")
        if not normalized:
            delta = delta.subtract(last_logs, fill_value=np.nan)
        forecasts[name] = np.expm1(delta).clip(-0.99, 10.0)
    return pd.DataFrame(forecasts).sort_index()


def _metric_periods(start: str, end: str) -> tuple[tuple[str, str, str], ...]:
    return (
        ("train", start, "2021-12-31"),
        ("validation", "2022-01-01", "2023-12-31"),
        ("holdout_2024", "2024-01-01", "2024-12-31"),
        ("holdout_2025_2026", "2025-01-01", end),
        ("full", start, end),
    )


def _format_metric(value: Any) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _metric_map(
    simulation: pd.DataFrame,
    start: str,
    end: str,
) -> dict[str, dict[str, Any]]:
    return {
        label: ml_research.period_metrics(simulation, period_start, period_end, label)
        for label, period_start, period_end in _metric_periods(start, end)
    }


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
    chronos_validation_winner: str | None,
    preferred: str | None,
    loaded_tickers: list[str],
    skipped: list[str],
    errors: list[str],
    signal_count: int,
    forecast_count: int,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows = [
        {"model": model, **values}
        for model, model_metrics in metrics.items()
        for _, values in model_metrics.items()
    ]
    pd.DataFrame(metrics_rows).to_csv(REPORT_DIR / "chronos2_daily_metrics.csv", index=False)
    diagnostics.to_csv(REPORT_DIR / "chronos2_daily_robustness.csv", index=False)
    rolling.to_csv(REPORT_DIR / "chronos2_daily_rolling.csv", index=False)
    rolling_summary.to_csv(REPORT_DIR / "chronos2_daily_rolling_summary.csv", index=False)
    bootstrap.to_csv(REPORT_DIR / "chronos2_daily_bootstrap.csv", index=False)
    sensitivity.to_csv(REPORT_DIR / "chronos2_daily_sensitivity.csv", index=False)
    context_rows.to_csv(REPORT_DIR / "chronos2_daily_context.csv", index=False)
    training_rows.to_csv(REPORT_DIR / "chronos2_daily_training.csv", index=False)

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
            REPORT_DIR / "chronos2_daily_forecasts.csv",
            index=False,
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
        "chronos_validation_winner": chronos_validation_winner,
        "preferred_research_candidate": preferred,
        "gates": gates,
        "bootstrap": bootstrap.to_dict(orient="records"),
        "chronos_package_version": _version("chronos-forecasting"),
        "formula_variants": {
            "absolute_individual_median": "absolute daily log-price, individual inference, terminal median",
            "absolute_cross_median": "absolute daily log-price, cross-learning, terminal median",
            "absolute_cross_lower10": "absolute daily log-price, cross-learning, terminal tenth percentile",
            "normalized_cross_median": "last-price normalized log-price, cross-learning, terminal median",
            "normalized_cross_lower10": "last-price normalized log-price, cross-learning, terminal tenth percentile",
        },
        "sources": {
            "chronos": CHRONOS_SOURCE,
            "chronos2_study": CHRONOS2_STUDY,
            "idx_methodology": IDX_METHOD_SOURCE,
        },
    }
    (REPORT_DIR / "chronos2_daily_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    candidate_count = len(metrics) - 1
    report = f"""# Chronos-2 daily IDX research

Run date: {date.today().isoformat()}<br>
Model: {config.model_id}; chronos-forecasting: {_version("chronos-forecasting") or "unavailable"}<br>
Universe catalog: {_relative_path(config.universe)}; loaded tickers: {len(loaded_tickers)}; skipped: {len(skipped)}<br>
Eligibility: top {config.cap} by trailing 60-day median dollar volume at each completed month-end<br>
Context: {config.context_days} daily observations; horizon: {config.horizon_days} business days; batch size: {config.batch_size}<br>
Portfolio: top {config.top_k} equal-weight names; monthly signal; {config.cost_bps:.1f} bps one-way default cost<br>
Signal months attempted: {signal_count}; model forecast values: {forecast_count}; recorded errors: {len(errors)}<br>

## Research basis

The official Chronos repository ({CHRONOS_SOURCE}) documents the long-format predict_df interface, quantile forecasts, cross-learning controls, and batch sizing used here. The Chronos-2 multivariate study ({CHRONOS2_STUDY}) motivates testing cross-series learning while warning that unrelated series can add noise. This audit tests that claim on Indonesian equities.

The liquidity proxy follows the official IDX80/LQ45/IDX30 methodology ({IDX_METHOD_SOURCE}). Historical membership and free-float data are not available here, so the fixed top-{config.cap} dollar-volume screen is only a robustness proxy.

## Fixed variants

Every signal uses only daily observations dated on or before completed month-end t. Chronos-2 forecasts the terminal value after {config.horizon_days} business-day steps, which becomes a cross-sectional expected-return score for the following month.

| variant | context | inference | selected forecast |
|---|---|---|---|
| chronos2_daily_abs_individual_median | absolute log price | individual | terminal median |
| chronos2_daily_abs_cross_median | absolute log price | cross-learning | terminal median |
| chronos2_daily_abs_cross_lower10 | absolute log price | cross-learning | terminal lower 10% |
| chronos2_daily_norm_cross_median | last-price normalized log price | cross-learning | terminal median |
| chronos2_daily_norm_cross_lower10 | last-price normalized log price | cross-learning | terminal lower 10% |

Normalized variants remove stock price-level scale before inference; their forecast is already a log-return estimate. This transformation is fixed before holdout inspection, and the two representations are not independent evidence because they use the same stocks, dates, and model. No blend weight or holdout-selected threshold is used.

## Leakage controls and limitations

- Context rows are filtered with date <= t; the next-month target is never supplied to Chronos-2.
- Only stocks eligible at t with a complete trailing daily context are forecast. Missing history is dropped; it is not forward-filled.
- The rank-Ridge comparator is fit expanding-window with signal dates strictly earlier than each forecast.
- Top K, costs, rolling windows, bootstrap settings, model variants, and quantile choices are fixed before inspecting holdout results.
- The current catalog omits historical delistings, suspensions, and membership changes. This cannot establish a live edge.
- Yahoo Finance data do not model spreads, taxes, price limits, market impact, failed fills, or capacity.

## Audit summary

The fixed gate requires beating the matching cap{config.cap}_composite control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025-2026.

{_summary_table(metrics, rolling_summary, bootstrap, gates)}

Overall validation winner (highest validation Sharpe, then excess CAGR): {validation_winner or "none"}. Chronos-2-only validation winner: {chronos_validation_winner or "none"}. Preferred after the fixed gate: {preferred or "none"}. {candidate_count} candidate panels were evaluated. The lower-tail Chronos-2 panels can pass their individual gate rows while the overall preferred result remains none: the predeclared selection rule chooses the overall validation winner first, and that winner must also pass every gate condition.

## Reproduction

~~~bash
HF_HOME=/tmp/beat-market-hf PYTHONPATH=$PWD \\
  /tmp/beat-market-ml-venv/bin/python -m src.chronos2_daily_research
~~~

For a smoke run, add --max-signals 1. Raw forecasts remain ignored in reports/chronos2_daily_forecasts.csv. Committed tables are chronos2_daily_metrics.csv, chronos2_daily_robustness.csv, chronos2_daily_rolling_summary.csv, chronos2_daily_bootstrap.csv, chronos2_daily_sensitivity.csv, chronos2_daily_context.csv, and chronos2_daily_training.csv.

This is research, not investment advice, and no backtest guarantees future performance.
"""
    (REPORT_DIR / "chronos2_daily_findings.md").write_text(report, encoding="utf-8")


def _simulate_panels(
    panels: dict[str, pd.DataFrame],
    features: dict[str, pd.DataFrame | pd.Series],
    top_k: int,
    cost_bps: float,
) -> dict[str, pd.DataFrame]:
    return {
        name: ml_research.simulate_predictions(panel, features, top_k, cost_bps)
        for name, panel in panels.items()
    }


def run(config: argparse.Namespace) -> dict[str, Any]:
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
    for model in MODEL_NAMES:
        panels[model] = pd.DataFrame(
            np.nan,
            index=signal_dates,
            columns=sorted(monthly_prices.columns),
            dtype=float,
        )

    frames = _read_daily_frames(config.data_dir, loaded_tickers)
    pipeline = _load_pipeline(config.model_id, config.device)
    context_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for position, signal_date in enumerate(signal_dates, start=1):
        eligible_tickers = eligibility.columns[
            eligibility.loc[signal_date].fillna(False).to_numpy()
        ].astype(str).tolist()
        try:
            absolute, absolute_last, absolute_tickers = _daily_context(
                frames,
                eligible_tickers,
                pd.Timestamp(signal_date),
                config.price_field,
                config.context_days,
                normalized=False,
            )
            normalized, normalized_last, normalized_tickers = _daily_context(
                frames,
                eligible_tickers,
                pd.Timestamp(signal_date),
                config.price_field,
                config.context_days,
                normalized=True,
            )
            if len(absolute_tickers) < config.top_k or len(normalized_tickers) < config.top_k:
                context_rows.append(
                    {
                        "signal_date": signal_date,
                        "eligible_tickers": len(eligible_tickers),
                        "absolute_context_tickers": len(absolute_tickers),
                        "normalized_context_tickers": len(normalized_tickers),
                        "status": "skipped_too_few_tickers",
                    }
                )
                continue
            abs_individual = _predict_daily(
                pipeline,
                absolute,
                absolute_last,
                config.horizon_days,
                config.context_days,
                config.batch_size,
                config.device,
                cross_learning=False,
                normalized=False,
            )
            abs_cross = _predict_daily(
                pipeline,
                absolute,
                absolute_last,
                config.horizon_days,
                config.context_days,
                config.batch_size,
                config.device,
                cross_learning=True,
                normalized=False,
            )
            norm_cross = _predict_daily(
                pipeline,
                normalized,
                normalized_last,
                config.horizon_days,
                config.context_days,
                config.batch_size,
                config.device,
                cross_learning=True,
                normalized=True,
            )
            assignments = {
                "chronos2_daily_abs_individual_median": abs_individual["median"],
                "chronos2_daily_abs_cross_median": abs_cross["median"],
                "chronos2_daily_abs_cross_lower10": abs_cross["lower10"],
                "chronos2_daily_norm_cross_median": norm_cross["median"],
                "chronos2_daily_norm_cross_lower10": norm_cross["lower10"],
            }
            for model, forecast in assignments.items():
                panels[model].loc[signal_date, forecast.index] = forecast.to_numpy()
            context_rows.append(
                {
                    "signal_date": signal_date,
                    "eligible_tickers": len(eligible_tickers),
                    "absolute_context_tickers": len(absolute_tickers),
                    "normalized_context_tickers": len(normalized_tickers),
                    "absolute_forecasts": int(abs_individual["median"].notna().sum()),
                    "normalized_forecasts": int(norm_cross["median"].notna().sum()),
                    "status": "ok",
                }
            )
            if position == 1 or position % 6 == 0 or position == len(signal_dates):
                print(
                    f"[chronos2-daily] signal {position}/{len(signal_dates)} "
                    f"({signal_date.date()})",
                    flush=True,
                )
        except Exception as exc:
            errors.append(f"{signal_date.date()}: {type(exc).__name__}: {exc}")
            context_rows.append(
                {
                    "signal_date": signal_date,
                    "eligible_tickers": len(eligible_tickers),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    simulations = _simulate_panels(panels, features, config.top_k, config.cost_bps)
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
    chronos_candidates = [name for name in candidate_names if name.startswith("chronos2_daily_")]
    chronos_validation_winner = max(
        chronos_candidates,
        key=lambda name: (
            metrics[name].get("validation", {}).get("strategy_sharpe_rf0", -np.inf),
            metrics[name].get("validation", {}).get("excess_cagr", -np.inf),
        ),
        default=None,
    )
    preferred = validation_winner if validation_winner and gates[validation_winner]["passed"] else None

    sensitivity_rows: list[dict[str, Any]] = []
    for top_k in TOP_K_SENSITIVITY:
        for cost_bps in COST_SENSITIVITY_BPS:
            replay = _simulate_panels(panels, features, top_k, cost_bps)
            for model, simulation in replay.items():
                for period, period_start, period_end in _metric_periods(config.start, config.end):
                    if period in {"validation", "holdout_2024", "holdout_2025_2026"}:
                        sensitivity_rows.append(
                            {
                                "model": model,
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
    sensitivity = pd.DataFrame(sensitivity_rows)
    training = pd.DataFrame(
        [
            {
                "model": rank_name,
                "signals": int(rank_predictions.notna().any(axis=1).sum()),
                "forecast_rows": int(rank_predictions.notna().sum().sum()),
                "median_training_rows": float(rank_training.dropna().median()),
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
        sensitivity=sensitivity,
        context_rows=pd.DataFrame(context_rows),
        training_rows=training,
        panels=panels,
        gates=gates,
        validation_winner=validation_winner,
        chronos_validation_winner=chronos_validation_winner,
        preferred=preferred,
        loaded_tickers=loaded_tickers,
        skipped=skipped,
        errors=errors,
        signal_count=len(signal_dates),
        forecast_count=int(
            sum(
                panel.notna().sum().sum()
                for name, panel in panels.items()
                if name.startswith("chronos2_daily_")
            )
        ),
    )
    return {
        "metrics": metrics,
        "gates": gates,
        "validation_winner": validation_winner,
        "chronos_validation_winner": chronos_validation_winner,
        "preferred": preferred,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    if args.status:
        print(json.dumps(package_status(), indent=2))
        return
    _validate_args(args)
    args.data_dir = args.data_dir.resolve()
    args.universe = args.universe.resolve()
    result = run(args)
    summary = pd.DataFrame(
        [
            {
                "model": model,
                "validation_excess_cagr": values.get("validation", {}).get("excess_cagr"),
                "validation_sharpe": values.get("validation", {}).get("strategy_sharpe_rf0"),
                "holdout_2024_excess_cagr": values.get("holdout_2024", {}).get("excess_cagr"),
                "holdout_2025_2026_excess_cagr": values.get("holdout_2025_2026", {}).get("excess_cagr"),
            }
            for model, values in result["metrics"].items()
        ]
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    print(
        f"\nValidation winner: {result['validation_winner']}; "
        f"Chronos-2 daily winner: {result['chronos_validation_winner']}; "
        f"preferred: {result['preferred']}"
    )
    print("Wrote reports/chronos2_daily_findings.md and reports/chronos2_daily_metrics.csv")


if __name__ == "__main__":
    main()
