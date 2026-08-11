"""Historical-cutoff Chronos-2 fine-tuning audit on the all-listed IDX panel.

The zero-shot Chronos-2 audit is useful, but its pretrained distribution is not
specialized to Indonesian equities.  This runner fine-tunes a copy of
Chronos-2 once using only daily adjusted-close histories ending before 2022,
then freezes that model while producing monthly forecasts for 2022 onward.
Terminal median and lower-tail forecasts are both recorded as predeclared
variants.  No holdout data are used during fine-tuning or variant selection.

This remains a research audit: the stock catalog is a current snapshot and
does not reconstruct delistings or historical index membership.
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

try:
    from . import (
        all_stock_ml_research,
        chronos2_daily_research,
        foundation_robustness,
        liquid_rank_ml_research,
        ml_research,
        research,
    )
except ImportError:  # pragma: no cover - direct CLI compatibility
    from src import (
        all_stock_ml_research,
        chronos2_daily_research,
        foundation_robustness,
        liquid_rank_ml_research,
        ml_research,
        research,
    )


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "yahoo_all"
DEFAULT_UNIVERSE_PATH = ROOT / "data" / "universe_idx_all_2026-08.csv"
DEFAULT_MODEL_ID = "amazon/chronos-2"
DEFAULT_PRICE_FIELD = "adjclose"
DEFAULT_TRAIN_END = "2021-12-31"
DEFAULT_START = "2022-01-01"
DEFAULT_END = "2026-08-11"
DEFAULT_CAP = 300
DEFAULT_CONTEXT_DAYS = 256
DEFAULT_HORIZON_DAYS = 21
DEFAULT_TOP_K = 3
DEFAULT_COST_BPS = 25.0
DEFAULT_MIN_PAST = 256
DEFAULT_FINE_TUNE_STEPS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-6
DEFAULT_SEED = 20260812
DEFAULT_OUTPUT_PREFIX = "chronos2_finetune"
DEFAULT_CHECKPOINT_DIR = Path("/tmp/beat-market-chronos2-finetuned")
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_BLOCK_LENGTH = 3
BOOTSTRAP_SEED = 20260812
TOP_K_SENSITIVITY = (1, 3, 5, 10)
COST_SENSITIVITY_BPS = (25.0, 100.0)
CONTROL_SUFFIX = "_composite"
MODEL_NAMES = (
    "chronos2_ft_cross_median",
    "chronos2_ft_cross_lower10",
)


@dataclass(frozen=True)
class FineTuneConfig:
    price_field: str
    train_end: str
    start: str
    end: str
    cap: int
    context_days: int
    horizon_days: int
    top_k: int
    cost_bps: float
    min_past: int
    fine_tune_steps: int
    batch_size: int
    learning_rate: float
    seed: int
    device: str
    model_id: str
    max_signals: int
    data_dir: Path
    universe_path: Path
    checkpoint_dir: Path
    output_prefix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-field", choices=("adjclose", "close"), default=DEFAULT_PRICE_FIELD)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--context-days", type=int, default=DEFAULT_CONTEXT_DAYS)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--min-past", type=int, default=DEFAULT_MIN_PAST)
    parser.add_argument("--fine-tune-steps", type=int, default=DEFAULT_FINE_TUNE_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-signals", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
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


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _validate_config(config: FineTuneConfig) -> None:
    if config.cap < config.top_k or config.cap < 30:
        raise ValueError("cap must be at least top_k and at least 30")
    if config.context_days < 32 or config.horizon_days < 1:
        raise ValueError("context_days must be at least 32 and horizon_days must be positive")
    if config.min_past < 1:
        raise ValueError("min_past must be positive")
    if config.cost_bps < 0.0 or config.fine_tune_steps < 1 or config.batch_size < 1:
        raise ValueError("cost, fine-tune steps, and batch size are invalid")
    if config.learning_rate <= 0.0 or config.max_signals < 0:
        raise ValueError("learning rate or max_signals is invalid")
    if not config.output_prefix or Path(config.output_prefix).name != config.output_prefix:
        raise ValueError("output_prefix must be a non-empty filename prefix")
    if pd.Timestamp(config.start) <= pd.Timestamp(config.train_end):
        raise ValueError("start must be after train_end so the frozen model cannot see evaluation data")


def _output_path(config: FineTuneConfig, suffix: str) -> Path:
    return REPORT_DIR / f"{config.output_prefix}_{suffix}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def normalized_log_price(values: pd.Series) -> np.ndarray:
    """Return a finite, last-value-normalized log-price training series."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    numeric = numeric.loc[numeric > 0.0]
    if numeric.empty:
        return np.array([], dtype=np.float32)
    logs = np.log(numeric.to_numpy(dtype=float))
    if not np.isfinite(logs).all():
        return np.array([], dtype=np.float32)
    return (logs - logs[-1]).astype(np.float32)


def build_training_inputs(
    frames: dict[str, pd.DataFrame],
    tickers: list[str],
    price_field: str,
    train_end: str,
    minimum_length: int,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Build only pre-cutoff series and record their lengths for the audit."""
    cutoff = pd.Timestamp(train_end)
    inputs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for ticker in sorted(tickers):
        frame = frames[ticker]
        available = frame.loc[frame["date"] <= cutoff]
        column = price_field if price_field in available.columns else "close"
        values = normalized_log_price(available[column])
        if len(values) < minimum_length:
            continue
        inputs.append(values)
        rows.append(
            {
                "ticker": ticker,
                "observations": len(values),
                "last_observation": str(pd.Timestamp(available["date"].iloc[-1]).date()),
            }
        )
    return inputs, pd.DataFrame(rows)


def _seed_runtime(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def fine_tune_pipeline(
    config: FineTuneConfig,
    inputs: list[np.ndarray],
) -> Any:
    """Fine-tune once, before the evaluation period, using Chronos-2's API."""
    if not inputs:
        raise ValueError("no sufficiently long pre-cutoff series are available")
    _seed_runtime(config.seed)
    pipeline = chronos2_daily_research._load_pipeline(config.model_id, config.device)
    return pipeline.fit(
        inputs,
        prediction_length=config.horizon_days,
        finetune_mode="full",
        context_length=config.context_days,
        learning_rate=config.learning_rate,
        num_steps=config.fine_tune_steps,
        batch_size=config.batch_size,
        min_past=config.min_past,
        output_dir=config.checkpoint_dir,
        remove_printer_callback=True,
    )


def _metric_periods(start: str, end: str) -> tuple[tuple[str, str, str], ...]:
    return (
        ("validation", start, "2023-12-31"),
        ("holdout_2024", "2024-01-01", "2024-12-31"),
        ("holdout_2025_2026", "2025-01-01", end),
        ("full", start, end),
    )


def _metrics(simulation: pd.DataFrame, start: str, end: str) -> dict[str, dict[str, Any]]:
    return {
        label: ml_research.period_metrics(simulation, period_start, period_end, label)
        for label, period_start, period_end in _metric_periods(start, end)
    }


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

        def fmt(value: Any) -> str:
            if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
                return "—"
            return f"{float(value):.4f}" if isinstance(value, (float, np.floating)) else str(value)

        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    fmt(validation.get("excess_cagr")),
                    fmt(holdout_2024.get("excess_cagr")),
                    fmt(holdout.get("excess_cagr")),
                    fmt(
                        rolling_row.iloc[0]["positive_excess_fraction"]
                        if not rolling_row.empty
                        else np.nan
                    ),
                    fmt(
                        bootstrap_row.iloc[0]["market_ci_lower"]
                        if not bootstrap_row.empty
                        else np.nan
                    ),
                    fmt(holdout.get("beta_to_benchmark")),
                    fmt(holdout.get("strategy_max_drawdown")),
                    "passes" if gates.get(model, {}).get("passed") else "rejects",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_outputs(
    *,
    config: FineTuneConfig,
    metrics: dict[str, dict[str, dict[str, Any]]],
    diagnostics: pd.DataFrame,
    rolling: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    sensitivity: pd.DataFrame,
    context: pd.DataFrame,
    training: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    gates: dict[str, dict[str, Any]],
    validation_winner: str | None,
    preferred: str | None,
    loaded_tickers: list[str],
    skipped: list[str],
    training_series: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows = [
        {"model": model, **values}
        for model, model_metrics in metrics.items()
        for _, values in model_metrics.items()
    ]
    pd.DataFrame(metrics_rows).to_csv(_output_path(config, "metrics.csv"), index=False)
    diagnostics.to_csv(_output_path(config, "robustness.csv"), index=False)
    rolling.to_csv(_output_path(config, "rolling.csv"), index=False)
    rolling_summary.to_csv(_output_path(config, "rolling_summary.csv"), index=False)
    bootstrap.to_csv(_output_path(config, "bootstrap.csv"), index=False)
    sensitivity.to_csv(_output_path(config, "sensitivity.csv"), index=False)
    context.to_csv(_output_path(config, "context.csv"), index=False)
    training.to_csv(_output_path(config, "training.csv"), index=False)
    training_series.to_csv(_output_path(config, "training_series.csv"), index=False)

    rows: list[pd.DataFrame] = []
    for model, panel in panels.items():
        values = panel.to_numpy(dtype=float)
        row_indices, column_indices = np.where(np.isfinite(values))
        if len(row_indices):
            rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "signal_date": panel.index.to_numpy()[row_indices],
                        "ticker": panel.columns.to_numpy()[column_indices],
                        "forecast": values[row_indices, column_indices],
                    }
                )
            )
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(_output_path(config, "forecasts.csv"), index=False)

    control_name = f"cap{config.cap}{CONTROL_SUFFIX}"
    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: (_relative_path(value) if isinstance(value, Path) else value)
            for key, value in vars(config).items()
        },
        "loaded_ticker_count": len(loaded_tickers),
        "skipped": skipped,
        "training_series": training.to_dict(orient="records"),
        "metrics": metrics,
        "validation_winner": validation_winner,
        "preferred_research_candidate": preferred,
        "gates": gates,
        "bootstrap": bootstrap.to_dict(orient="records"),
        "package": {
            "chronos_forecasting": _version("chronos-forecasting"),
            "torch": _version("torch"),
        },
        "formula": "frozen Chronos-2 model trained before 2022; top-300 eligible stocks ranked by terminal cross-learning median or lower10 forecast",
        "sources": {
            "chronos": "https://github.com/amazon-science/chronos-forecasting",
            "chronos2_study": "https://arxiv.org/abs/2605.21504",
            "idx_methodology": chronos2_daily_research.IDX_METHOD_SOURCE,
        },
    }
    _output_path(config, "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    report = f"""# Chronos-2 historical-cutoff fine-tuning research

Run date: {date.today().isoformat()}<br>
Model: {config.model_id}; chronos-forecasting: {_version("chronos-forecasting") or "unavailable"}<br>
Universe catalog: {_relative_path(config.universe_path)}; loaded tickers: {len(loaded_tickers)}; skipped: {len(skipped)}<br>
Fine-tuning cutoff: {config.train_end}; evaluation: {config.start} through {config.end}<br>
Eligibility: top {config.cap} by trailing 60-day median dollar volume; context: {config.context_days} days; horizon: {config.horizon_days} business days<br>
Portfolio: top {config.top_k} equal-weight names; {config.cost_bps:.1f} bps one-way default cost<br>

## Research basis

This audit uses the official Chronos-2 `fit` and `predict_df` interfaces documented by [Amazon Science](https://github.com/amazon-science/chronos-forecasting), with the multivariate financial-forecasting motivation from the [Chronos-2 study](https://arxiv.org/abs/2605.21504). It adapts the pretrained model to Indonesian daily price histories before the evaluation period, rather than selecting a holdout-tuned model.

## Fixed design and leakage controls

- The model is fine-tuned exactly once on normalized daily log-price series with observations no later than {config.train_end}. The frozen checkpoint is then used for every evaluation month.
- Training uses {len(training_series)} current-catalog series with at least {config.min_past + config.horizon_days} pre-cutoff observations, {config.fine_tune_steps} full-model steps, learning rate {config.learning_rate:g}, batch size {config.batch_size}, and seed {config.seed}.
- At each completed month-end t, only the top {config.cap} names by trailing 60-day median dollar volume are forecast. Context rows are filtered to dates <= t.
- The two recorded variants are terminal cross-learning median and terminal cross-learning 10th-percentile forecasts. Their ranks and top-{config.top_k} portfolio rule are fixed before inspecting validation or holdout results.
- Validation is 2022–2023. 2024 and 2025–2026 are holdouts. No holdout return is used for fine-tuning, model choice, or tail selection.

## Audit summary

The fixed gate requires beating the matching {control_name} control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025–2026.

{_summary_table(metrics, rolling_summary, bootstrap, gates)}

Validation winner: {validation_winner or "none"}. Preferred after the fixed gate: {preferred or "none"}. Any passing row remains exploratory because the catalog is a current snapshot with survivorship and historical-membership bias.

## Limitations

- The current catalog omits historical delistings, suspensions, and membership changes.
- Full-model CPU fine-tuning is reproducible but expensive; it is not a substitute for a point-in-time, multi-market training panel.
- Yahoo Finance data do not model spreads, taxes, price limits, market impact, failed fills, or capacity.
- This is research, not investment advice, and no backtest guarantees future performance.

## Reproduction

~~~bash
HF_HOME=/tmp/beat-market-hf PYTHONPATH=$PWD \\
  /tmp/beat-market-ml-venv/bin/python -m src.chronos2_finetune_research
~~~

The generated checkpoint remains outside Git at `{config.checkpoint_dir}`. Raw forecasts remain ignored in reports/{config.output_prefix}_forecasts.csv; committed tables use the `{config.output_prefix}_` prefix.
"""
    _output_path(config, "findings.md").write_text(report, encoding="utf-8")


def run(config: FineTuneConfig) -> dict[str, Any]:
    _validate_config(config)
    prices, volumes, loaded_tickers, skipped = all_stock_ml_research.load_prices(
        config.data_dir,
        config.universe_path,
        config.price_field,
    )
    prices = prices.loc[prices.index <= pd.Timestamp(config.end)]
    features = all_stock_ml_research._slice_features(
        research.make_features(prices, volumes.reindex(prices.index)),
        config.start,
        config.end,
    )
    eligibility = liquid_rank_ml_research.eligibility_mask(features, config.cap)
    monthly_prices = features["monthly_prices"]
    monthly_returns = features["monthly_returns"]
    assert isinstance(monthly_prices, pd.DataFrame)
    assert isinstance(monthly_returns, pd.DataFrame)
    signal_dates = pd.DatetimeIndex(
        [timestamp for timestamp in monthly_prices.index if monthly_returns.loc[timestamp].notna().any()]
    )
    if config.max_signals:
        signal_dates = signal_dates[: config.max_signals]

    frames = chronos2_daily_research._read_daily_frames(config.data_dir, loaded_tickers)
    training_inputs, training_series = build_training_inputs(
        frames,
        loaded_tickers,
        config.price_field,
        config.train_end,
        config.min_past + config.horizon_days,
    )
    print(
        f"[chronos2-finetune] training {len(training_inputs)} series through {config.train_end}",
        flush=True,
    )
    pipeline = fine_tune_pipeline(config, training_inputs)

    columns = sorted(monthly_prices.columns)
    panels: dict[str, pd.DataFrame] = {
        f"cap{config.cap}{CONTROL_SUFFIX}": research.score_frame("composite", features).where(eligibility),
        **{
            name: pd.DataFrame(np.nan, index=signal_dates, columns=columns, dtype=float)
            for name in MODEL_NAMES
        },
    }
    context_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for position, signal_date in enumerate(signal_dates, start=1):
        eligible_tickers = eligibility.columns[
            eligibility.loc[signal_date].fillna(False).to_numpy()
        ].astype(str).tolist()
        try:
            context, last_logs, included = chronos2_daily_research._daily_context(
                frames,
                eligible_tickers,
                pd.Timestamp(signal_date),
                config.price_field,
                config.context_days,
                normalized=True,
            )
            if len(included) < config.top_k:
                raise ValueError(f"only {len(included)} eligible contexts available")
            forecasts = chronos2_daily_research._predict_daily(
                pipeline,
                context,
                last_logs,
                config.horizon_days,
                config.context_days,
                config.batch_size,
                config.device,
                cross_learning=True,
                normalized=True,
            )
            panels["chronos2_ft_cross_median"].loc[signal_date, forecasts.index] = forecasts[
                "median"
            ].to_numpy()
            panels["chronos2_ft_cross_lower10"].loc[signal_date, forecasts.index] = forecasts[
                "lower10"
            ].to_numpy()
            context_rows.append(
                {
                    "signal_date": signal_date,
                    "eligible_tickers": len(eligible_tickers),
                    "context_tickers": len(included),
                    "forecast_tickers": int(forecasts["median"].notna().sum()),
                    "status": "ok",
                }
            )
            if position == 1 or position % 6 == 0 or position == len(signal_dates):
                print(
                    f"[chronos2-finetune] signal {position}/{len(signal_dates)} ({signal_date.date()})",
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

    simulations = {
        name: ml_research.simulate_predictions(panel, features, config.top_k, config.cost_bps)
        for name, panel in panels.items()
    }
    metrics = {
        name: _metrics(simulation, config.start, config.end)
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
    control_name = f"cap{config.cap}{CONTROL_SUFFIX}"
    bootstrap = _bootstrap_rows(simulations, config.end, control_name)
    candidates = [name for name in panels if name != control_name]
    gates: dict[str, dict[str, Any]] = {}
    for name in candidates:
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
        candidates,
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
                    if period in {"validation", "holdout_2024", "holdout_2025_2026"}:
                        sensitivity_rows.append(
                            {
                                "model": name,
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
                "model": "chronos2_finetuned",
                "series": len(training_inputs),
                "median_training_observations": float(training_series["observations"].median()),
                "min_training_observations": int(training_series["observations"].min()),
                "max_training_observations": int(training_series["observations"].max()),
                "train_end": config.train_end,
                "fine_tune_steps": config.fine_tune_steps,
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
        context=pd.DataFrame(context_rows),
        training=training,
        panels=panels,
        gates=gates,
        validation_winner=validation_winner,
        preferred=preferred,
        loaded_tickers=loaded_tickers,
        skipped=skipped,
        training_series=training_series,
    )
    return {
        "metrics": metrics,
        "gates": gates,
        "validation_winner": validation_winner,
        "preferred": preferred,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    if args.status:
        print(json.dumps(package_status(), indent=2))
        return
    seeds = FineTuneConfig(
        price_field=args.price_field,
        train_end=args.train_end,
        start=args.start,
        end=args.end,
        cap=args.cap,
        context_days=args.context_days,
        horizon_days=args.horizon_days,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        min_past=args.min_past,
        fine_tune_steps=args.fine_tune_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        model_id=args.model_id,
        max_signals=args.max_signals,
        data_dir=args.data_dir.resolve(),
        universe_path=args.universe.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        output_prefix=args.output_prefix,
    )
    result = run(seeds)
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
        f"preferred: {result['preferred']}"
    )
    print(f"Wrote reports/{args.output_prefix}_findings.md")


if __name__ == "__main__":
    main()
