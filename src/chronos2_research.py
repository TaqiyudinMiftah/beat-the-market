"""Leakage-safe multivariate Chronos-2 research on the all-listed IDX panel.

Chronos-Bolt, TimesFM, and Kronos were previously evaluated one stock at a
time.  This runner tests the newer Chronos-2 cross-learning interface: at each
completed month-end it supplies only the trailing monthly log-price histories
of the currently eligible liquid stocks, then converts the one-month forecast
back into a return score.  ``cross_learning=True`` and ``False`` are replayed
as a predeclared comparison, together with a conservative lower-quantile
score.

The experiment is deliberately separate from the ordinary app and research
commands because Chronos-2 is an optional heavyweight dependency.  It is an
audit harness, not a claim that a foundation model can reliably beat IHSG.
The all-listed catalog is still a current snapshot, so delisted stocks and
historical index membership are not reconstructed here.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Support both ``python -m src...`` and direct execution.
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
DEFAULT_CONTEXT_MONTHS = 48
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
    "chronos2_individual_median",
    "chronos2_cross_median",
    "chronos2_cross_lower10",
)


@dataclass(frozen=True)
class Chronos2Config:
    price_field: str
    start: str
    end: str
    cap: int
    context_months: int
    top_k: int
    cost_bps: float
    min_train_rows: int
    alpha: float
    batch_size: int
    max_signals: int
    device: str
    model_id: str
    data_dir: Path
    universe_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-field", choices=("adjclose", "close"), default="adjclose")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--context-months", type=int, default=DEFAULT_CONTEXT_MONTHS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-signals",
        type=int,
        default=0,
        help="Optional smoke-run limit; zero evaluates every requested signal",
    )
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
    """Report optional package availability without loading model weights."""
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


def _parse_config(args: argparse.Namespace) -> Chronos2Config:
    if args.cap < 1:
        raise ValueError("cap must be at least one stock")
    if args.context_months < 24:
        raise ValueError("context_months must be at least 24")
    if not 1 <= args.top_k <= args.cap:
        raise ValueError("top_k must be between one and cap")
    if args.cost_bps < 0.0 or args.min_train_rows < 60 or args.alpha <= 0.0:
        raise ValueError("cost, minimum training rows, and alpha are invalid")
    if args.batch_size < 1 or args.max_signals < 0:
        raise ValueError("batch_size and max_signals are invalid")
    return Chronos2Config(
        price_field=args.price_field,
        start=args.start,
        end=args.end,
        cap=args.cap,
        context_months=args.context_months,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        min_train_rows=args.min_train_rows,
        alpha=args.alpha,
        batch_size=args.batch_size,
        max_signals=args.max_signals,
        device=args.device,
        model_id=args.model_id,
        data_dir=args.data_dir.resolve(),
        universe_path=args.universe.resolve(),
    )


def build_context_frame(
    monthly_prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    signal_date: pd.Timestamp,
    context_months: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a regular long panel ending at ``signal_date`` only.

    A stock is included only when it is eligible at the signal month and has a
    complete trailing context.  Requiring one shared monthly date index avoids
    silently filling listing gaps or using a value from after the signal.
    """
    if signal_date not in monthly_prices.index or signal_date not in eligibility.index:
        return pd.DataFrame(columns=["item_id", "timestamp", "target"]), []
    context = monthly_prices.loc[:signal_date].tail(context_months)
    if len(context) != context_months:
        return pd.DataFrame(columns=["item_id", "timestamp", "target"]), []
    eligible_names = eligibility.columns[eligibility.loc[signal_date].fillna(False).to_numpy()]
    complete = context.loc[:, eligible_names].notna().all(axis=0)
    tickers = sorted(complete.index[complete].astype(str).tolist())
    if not tickers:
        return pd.DataFrame(columns=["item_id", "timestamp", "target"]), []
    long = (
        context.loc[:, tickers]
        .rename_axis("timestamp")
        .reset_index()
        .melt(id_vars="timestamp", var_name="item_id", value_name="target")
    )
    long["target"] = np.log(pd.to_numeric(long["target"], errors="coerce"))
    long = long.sort_values(["item_id", "timestamp"], kind="stable").reset_index(drop=True)
    if long["target"].isna().any() or not np.isfinite(long["target"]).all():
        raise ValueError(f"non-finite log-price in context ending {signal_date.date()}")
    return long, tickers


def log_forecast_to_return(
    forecast_log_prices: pd.Series,
    last_log_prices: pd.Series,
) -> pd.Series:
    """Convert Chronos-2 log-level forecasts to bounded next-month returns."""
    aligned = pd.concat([forecast_log_prices, last_log_prices], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    returns = np.expm1(aligned.iloc[:, 0] - aligned.iloc[:, 1]).clip(-0.99, 10.0)
    returns.name = "forecast"
    return returns


def _load_pipeline(config: Chronos2Config) -> Any:
    from chronos import Chronos2Pipeline

    device_map = "cuda:0" if config.device == "cuda" else config.device
    return Chronos2Pipeline.from_pretrained(config.model_id, device_map=device_map)


def _predict_context(
    pipeline: Any,
    context: pd.DataFrame,
    config: Chronos2Config,
    cross_learning: bool,
) -> pd.DataFrame:
    """Return median, lower-tail, and upper-tail return forecasts."""
    output = pipeline.predict_df(
        context,
        prediction_length=1,
        context_length=config.context_months,
        batch_size=config.batch_size,
        cross_learning=cross_learning,
        validate_inputs=True,
        freq="ME",
        target="target",
        quantile_levels=[0.1, 0.5, 0.9],
    )
    output = output.set_index("item_id")
    last_log = context.groupby("item_id", sort=False)["target"].last()
    columns = {}
    for quantile, name in (("0.1", "lower10"), ("0.5", "median"), ("0.9", "upper90")):
        if quantile not in output:
            raise ValueError(f"Chronos-2 output is missing quantile {quantile}")
        columns[name] = log_forecast_to_return(output[quantile], last_log)
    return pd.DataFrame(columns).sort_index()


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
        holdout_2025 = values.get("holdout_2025_2026", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    _format_metric(validation.get("excess_cagr")),
                    _format_metric(holdout_2024.get("excess_cagr")),
                    _format_metric(holdout_2025.get("excess_cagr")),
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
                    _format_metric(holdout_2025.get("beta_to_benchmark")),
                    _format_metric(holdout_2025.get("strategy_max_drawdown")),
                    "passes" if gates.get(model, {}).get("passed") else "rejects",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _block_bootstrap_active(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
    start: str,
    end: str,
    seed: int,
) -> dict[str, Any]:
    return liquid_rank_ml_research._block_bootstrap_active(
        candidate,
        comparator,
        start,
        end,
        seed,
    )


def _write_outputs(
    *,
    config: Chronos2Config,
    metrics: dict[str, dict[str, dict[str, Any]]],
    diagnostics: pd.DataFrame,
    rolling: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    sensitivity: pd.DataFrame,
    context_rows: pd.DataFrame,
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
    pd.DataFrame(metrics_rows).to_csv(REPORT_DIR / "chronos2_metrics.csv", index=False)
    diagnostics.to_csv(REPORT_DIR / "chronos2_robustness.csv", index=False)
    rolling.to_csv(REPORT_DIR / "chronos2_rolling.csv", index=False)
    rolling_summary.to_csv(REPORT_DIR / "chronos2_rolling_summary.csv", index=False)
    bootstrap.to_csv(REPORT_DIR / "chronos2_bootstrap.csv", index=False)
    sensitivity.to_csv(REPORT_DIR / "chronos2_sensitivity.csv", index=False)
    context_rows.to_csv(REPORT_DIR / "chronos2_context.csv", index=False)

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
            REPORT_DIR / "chronos2_forecasts.csv",
            index=False,
        )

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: (
                _relative_path(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in config.__dict__.items()
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
    }
    (REPORT_DIR / "chronos2_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    candidate_count = len(metrics) - 1
    report = f"""# Chronos-2 multivariate IDX research

Run date: {date.today().isoformat()}<br>
Model: `{config.model_id}`; chronos-forecasting: `{_version("chronos-forecasting") or "unavailable"}`<br>
Universe catalog: `{_relative_path(config.universe_path)}`; loaded tickers: `{len(loaded_tickers)}`; skipped: `{len(skipped)}`<br>
Eligibility: top `{config.cap}` by trailing 60-day median dollar volume at each completed month-end<br>
Context: `{config.context_months}` monthly log-price observations; batch size `{config.batch_size}`; default top K `{config.top_k}`; cost `{config.cost_bps:.1f}` bps one-way<br>
Signal months attempted: `{signal_count}`; model forecast values: `{forecast_count}`; recorded errors: `{len(errors)}`<br>

## Research basis

The [Chronos-2 multivariate finance study](https://arxiv.org/abs/2605.21504) reports that cross-series inputs can improve forecasts when the series are related, while warning that noisy cross-series context can hurt. This runner tests that claim on Indonesian equities rather than assuming transfer from another market. It compares Chronos-2's `cross_learning=False` individual forecasts with `cross_learning=True` forecasts in the same monthly, one-step-ahead protocol.

The liquidity proxy follows the [official IDX80/LQ45/IDX30 methodology](https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf), which considers transaction value/frequency, free-float market capitalization, fundamentals, a six-month listing minimum, and consistent trading. Historical membership and free-float data are not available in this repository; therefore the fixed top-`{config.cap}` dollar-volume cap is only a robustness proxy.

## Leakage controls

- Each signal uses monthly prices through completed month-end `t`; the next-month price is never included in the model context.
- A stock must be eligible at `t` and have a complete shared trailing context; missing history is dropped rather than forward-filled.
- The lower-tail score uses the model's 10th-percentile log-price forecast and is fixed before inspecting holdout results.
- The `cap{config.cap}_rank_ridge` comparator is refit expanding-window with signal dates strictly earlier than each forecast date.
- Top K, costs, rolling windows, and block bootstrap settings are fixed. Holdout months are descriptive and do not select the preferred model.

## Audit summary

The fixed gate requires beating the `cap{config.cap}_composite` control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta ≤ `{liquid_rank_ml_research.RISK_BETA_LIMIT:.2f}`, drawdown ≥ `{liquid_rank_ml_research.RISK_DRAWDOWN_LIMIT:.2f}`, and turnover ≤ `{liquid_rank_ml_research.RISK_TURNOVER_LIMIT:.2f}` in 2025–2026.

{_summary_table(metrics, rolling_summary, bootstrap, gates)}

Overall validation winner (highest validation Sharpe, then excess CAGR): **{validation_winner or "none"}**. Chronos-2-only validation winner: **{chronos_validation_winner or "none"}**. Preferred research candidate after the fixed gate: **{preferred or "none"}**. `{candidate_count}` candidate panels were evaluated.

## Interpretation

Chronos-2 is being evaluated as a forecast/ranking input, not as an automatic trading oracle. A passing row would still be exploratory because the current catalog excludes delisted names and historical membership changes. The report must be rerun on a point-in-time universe and future unseen months before any live use.

## Reproduction

```bash
HF_HOME=/tmp/beat-market-hf PYTHONPATH=$PWD \\
  /tmp/beat-market-ml-venv/bin/python -m src.chronos2_research
```

For a smoke run without replaying every signal:

```bash
HF_HOME=/tmp/beat-market-hf PYTHONPATH=$PWD \\
  /tmp/beat-market-ml-venv/bin/python -m src.chronos2_research --max-signals 2
```

Raw forecasts remain ignored in `reports/chronos2_forecasts.csv`. Committed audit tables are `chronos2_metrics.csv`, `chronos2_robustness.csv`, `chronos2_rolling_summary.csv`, `chronos2_bootstrap.csv`, `chronos2_sensitivity.csv`, and `chronos2_context.csv`.

This is research, not investment advice, and no backtest guarantees future performance.
"""
    (REPORT_DIR / "chronos2_findings.md").write_text(report, encoding="utf-8")


def run(config: Chronos2Config) -> dict[str, Any]:
    prices, volumes, loaded_tickers, skipped = all_stock_ml_research.load_prices(
        config.data_dir,
        config.universe_path,
        config.price_field,
    )
    prices = prices.loc[prices.index <= config.end]
    all_features = research.make_features(prices, volumes.reindex(prices.index))
    warmup_start = pd.Timestamp(config.start) - pd.DateOffset(months=config.context_months + 12)
    features = all_stock_ml_research._slice_features(
        all_features,
        str(warmup_start.date()),
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
    signal_dates = monthly_prices.index[
        (monthly_prices.index >= pd.Timestamp(config.start))
        & (monthly_prices.index <= pd.Timestamp(config.end))
    ]
    signal_dates = pd.DatetimeIndex(
        [timestamp for timestamp in signal_dates if target.loc[timestamp].notna().any()]
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

    pipeline = _load_pipeline(config)
    context_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    forecast_count = 0
    for position, signal_date in enumerate(signal_dates, start=1):
        try:
            context, tickers = build_context_frame(
                monthly_prices,
                eligibility,
                pd.Timestamp(signal_date),
                config.context_months,
            )
            if len(tickers) < config.top_k:
                context_rows.append(
                    {
                        "signal_date": signal_date,
                        "eligible_context_tickers": len(tickers),
                        "context_months": len(context) // max(len(tickers), 1),
                        "status": "skipped_too_few_tickers",
                    }
                )
                continue
            individual = _predict_context(pipeline, context, config, cross_learning=False)
            cross = _predict_context(pipeline, context, config, cross_learning=True)
            individual_median = individual["median"]
            cross_median = cross["median"]
            cross_lower = cross["lower10"]
            panels["chronos2_individual_median"].loc[signal_date, individual_median.index] = (
                individual_median.to_numpy()
            )
            panels["chronos2_cross_median"].loc[signal_date, cross_median.index] = (
                cross_median.to_numpy()
            )
            panels["chronos2_cross_lower10"].loc[signal_date, cross_lower.index] = (
                cross_lower.to_numpy()
            )
            count = int(individual_median.notna().sum() + cross_median.notna().sum() + cross_lower.notna().sum())
            forecast_count += count
            context_rows.append(
                {
                    "signal_date": signal_date,
                    "eligible_context_tickers": len(tickers),
                    "context_months": len(context) // len(tickers),
                    "individual_forecasts": int(individual_median.notna().sum()),
                    "cross_forecasts": int(cross_median.notna().sum()),
                    "status": "ok",
                }
            )
            if position == 1 or position % 12 == 0 or position == len(signal_dates):
                print(f"[chronos2] signal {position}/{len(signal_dates)} ({signal_date.date()})", flush=True)
        except Exception as exc:  # retain partial diagnostics while exposing every error
            errors.append(f"{signal_date.date()}: {type(exc).__name__}: {exc}")
            context_rows.append(
                {
                    "signal_date": signal_date,
                    "eligible_context_tickers": np.nan,
                    "context_months": config.context_months,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    simulations = {
        name: ml_research.simulate_predictions(
            panel,
            features,
            config.top_k,
            config.cost_bps,
        )
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

    candidate_names = [name for name in panels if name != control_name]
    bootstrap_rows: list[dict[str, Any]] = []
    for position, name in enumerate(candidate_names):
        simulation = simulations[name]
        control_simulation = simulations[control_name]
        market = simulation[["benchmark_return"]].rename(
            columns={"benchmark_return": "strategy_return"}
        )
        market_bootstrap = _block_bootstrap_active(
            simulation,
            market,
            "2024-01-01",
            config.end,
            BOOTSTRAP_SEED + position,
        )
        control_bootstrap = _block_bootstrap_active(
            simulation,
            control_simulation,
            "2024-01-01",
            config.end,
            BOOTSTRAP_SEED + 100 + position,
        )
        bootstrap_rows.append(
            {
                "model": name,
                "months": market_bootstrap["months"],
                "market_active_annualized_mean": market_bootstrap["mean"],
                "market_ci_lower": market_bootstrap["lower"],
                "market_ci_median": market_bootstrap["median"],
                "market_ci_upper": market_bootstrap["upper"],
                "market_positive_probability": market_bootstrap["positive_probability"],
                "control_active_annualized_mean": control_bootstrap["mean"],
                "control_ci_lower": control_bootstrap["lower"],
                "control_ci_median": control_bootstrap["median"],
                "control_ci_upper": control_bootstrap["upper"],
                "control_positive_probability": control_bootstrap["positive_probability"],
                "samples": BOOTSTRAP_SAMPLES,
                "block_length": BOOTSTRAP_BLOCK_LENGTH,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)

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
    chronos_candidates = [name for name in candidate_names if name.startswith("chronos2_")]
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
            for name, panel in panels.items():
                simulation = ml_research.simulate_predictions(panel, features, top_k, cost_bps)
                for period, period_start, period_end in _metric_periods(config.start, config.end):
                    if period not in {"validation", "holdout_2024", "holdout_2025_2026"}:
                        continue
                    row = ml_research.period_metrics(
                        simulation,
                        period_start,
                        period_end,
                        period,
                    )
                    sensitivity_rows.append(
                        {
                            "model": name,
                            "top_k": top_k,
                            "cost_bps": cost_bps,
                            **row,
                        }
                    )
    sensitivity = pd.DataFrame(sensitivity_rows)
    context_frame = pd.DataFrame(context_rows)
    _write_outputs(
        config=config,
        metrics=metrics,
        diagnostics=diagnostics,
        rolling=rolling,
        rolling_summary=rolling_summary,
        bootstrap=bootstrap,
        sensitivity=sensitivity,
        context_rows=context_frame,
        panels=panels,
        gates=gates,
        validation_winner=validation_winner,
        chronos_validation_winner=chronos_validation_winner,
        preferred=preferred,
        loaded_tickers=loaded_tickers,
        skipped=skipped,
        errors=errors,
        signal_count=len(signal_dates),
        forecast_count=forecast_count,
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
    config = _parse_config(args)
    result = run(config)
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
        f"Chronos-2 winner: {result['chronos_validation_winner']}; "
        f"preferred: {result['preferred']}"
    )
    print("Wrote reports/chronos2_findings.md and reports/chronos2_metrics.csv")


if __name__ == "__main__":
    main()
