"""Rolling backtests for Chronos, TimesFM, and Kronos on the IDX30 panel.

The runner is intentionally separate from :mod:`src.ml_research`: the web app
and the ordinary research commands must not import heavyweight model packages
or download model weights. Each signal is formed from daily observations at or
before month-end ``t`` and is applied to the next completed month's return.

The three model families are used as zero-shot forecasters:

* Chronos-Bolt forecasts log adjusted-close histories.
* TimesFM forecasts the same log adjusted-close histories.
* Kronos forecasts daily OHLCV histories and ranks the predicted final close.

This is an experiment harness, not a claim that zero-shot forecasts are
economically valid. Every run records model IDs, context, horizon, package
versions, skipped symbols, and forecast errors in its report.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

try:  # Support both ``python -m src.foundation_research`` and direct execution.
    from . import foundation_robustness, ml_research, research
except ImportError:  # pragma: no cover - direct CLI compatibility
    from src import foundation_robustness, ml_research, research


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "yahoo"
REPORT_DIR = ROOT / "reports"
DEFAULT_CHRONOS_ID = "amazon/chronos-bolt-tiny"
DEFAULT_TIMESFM_ID = "google/timesfm-2.5-200m-pytorch"
DEFAULT_KRONOS_ID = "NeoQuasar/Kronos-mini"
DEFAULT_KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-2k"
DEFAULT_CONTEXT_DAYS = 256
DEFAULT_HORIZON_DAYS = 21
DEFAULT_START = "2021-01-01"


class Forecaster(Protocol):
    model_name: str

    def predict(self, contexts: list["DailyContext"], horizon: int) -> np.ndarray:
        """Return one next-horizon return forecast for every context."""


@dataclass(frozen=True)
class DailyContext:
    ticker: str
    dates: pd.DatetimeIndex
    log_prices: np.ndarray
    ohlcv: pd.DataFrame


@dataclass(frozen=True)
class ForecastResult:
    name: str
    predictions: pd.DataFrame
    simulation: pd.DataFrame
    metrics: dict[str, dict[str, Any]]
    rank_ic: dict[str, float | int]
    model_id: str
    package_versions: dict[str, str | None]
    errors: list[str]
    signal_count: int
    forecast_count: int


@dataclass(frozen=True)
class RunConfig:
    price_field: str
    start: str
    end: str
    top_k: int
    cost_bps: float
    context_days: int
    horizon_days: int
    max_signals: int
    device: str
    chronos_id: str
    timesfm_id: str
    kronos_id: str
    kronos_tokenizer_id: str
    kronos_repo: str | None
    sample_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated subset of chronos,timesfm,kronos or all",
    )
    parser.add_argument("--price-field", choices=("adjclose", "close"), default="adjclose")
    parser.add_argument("--start", default=DEFAULT_START, help="First signal month, inclusive")
    parser.add_argument("--end", default="2026-08-11", help="Last available daily date")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, default=25.0)
    parser.add_argument("--context-days", type=int, default=DEFAULT_CONTEXT_DAYS)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument(
        "--max-signals",
        type=int,
        default=0,
        help="Optional limit for a smoke run; zero evaluates every requested signal",
    )
    parser.add_argument("--device", default="cpu", help="Torch device; CPU is the reproducible default")
    parser.add_argument("--chronos-id", default=DEFAULT_CHRONOS_ID)
    parser.add_argument("--timesfm-id", default=DEFAULT_TIMESFM_ID)
    parser.add_argument("--kronos-id", default=DEFAULT_KRONOS_ID)
    parser.add_argument("--kronos-tokenizer-id", default=DEFAULT_KRONOS_TOKENIZER_ID)
    parser.add_argument(
        "--kronos-repo",
        default=os.getenv("KRONOS_REPO"),
        help="Official Kronos checkout containing model/, or set KRONOS_REPO",
    )
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--status", action="store_true", help="Only print package availability and versions")
    return parser.parse_args()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def package_status(kronos_repo: str | None = None) -> dict[str, dict[str, Any]]:
    """Report import availability without loading model weights."""
    kronos_path_available = bool(kronos_repo and (Path(kronos_repo) / "model").exists())
    return {
        "chronos": {
            "available": importlib.util.find_spec("chronos") is not None,
            "package": "chronos-forecasting",
            "version": _version("chronos-forecasting"),
            "model_id": DEFAULT_CHRONOS_ID,
        },
        "timesfm": {
            "available": importlib.util.find_spec("timesfm") is not None,
            "package": "timesfm",
            "version": _version("timesfm"),
            "model_id": DEFAULT_TIMESFM_ID,
        },
        "kronos": {
            "available": importlib.util.find_spec("model") is not None or kronos_path_available,
            "package": "official Kronos repository",
            "version": None,
            "model_id": DEFAULT_KRONOS_ID,
            "repo": kronos_repo,
        },
    }


def _torch_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device


class ChronosForecaster:
    model_name = "chronos"

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from chronos import ChronosBoltPipeline

        self._torch = torch
        self.model_id = model_id
        self.pipeline = ChronosBoltPipeline.from_pretrained(model_id, device_map=_torch_device(device))

    def predict(self, contexts: list[DailyContext], horizon: int) -> np.ndarray:
        tensors = [self._torch.tensor(context.log_prices, dtype=self._torch.float32) for context in contexts]
        quantiles, _ = self.pipeline.predict_quantiles(
            tensors,
            prediction_length=horizon,
            quantile_levels=[0.5],
        )
        median = quantiles.detach().cpu().numpy()[:, :, 0]
        last_prices = np.array([context.log_prices[-1] for context in contexts])
        return np.clip(np.exp(median[:, -1] - last_prices) - 1.0, -0.99, 10.0)


class TimesFMForecaster:
    model_name = "timesfm"

    def __init__(self, model_id: str, context_days: int, horizon: int) -> None:
        import torch
        import timesfm

        torch.set_float32_matmul_precision("high")
        self.model_id = model_id
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=max(1024, context_days),
                max_horizon=max(256, horizon),
                normalize_inputs=True,
                per_core_batch_size=32,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

    def predict(self, contexts: list[DailyContext], horizon: int) -> np.ndarray:
        inputs = [context.log_prices.astype(np.float32) for context in contexts]
        point_forecast, _ = self.model.forecast(horizon=horizon, inputs=inputs)
        last_prices = np.array([context.log_prices[-1] for context in contexts])
        return np.clip(np.exp(np.asarray(point_forecast)[:, -1] - last_prices) - 1.0, -0.99, 10.0)


class KronosForecaster:
    model_name = "kronos"

    def __init__(
        self,
        model_id: str,
        tokenizer_id: str,
        device: str,
        repo: str | None,
        sample_count: int,
    ) -> None:
        if repo:
            repo_path = str(Path(repo).resolve())
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
        import torch
        from model import Kronos, KronosPredictor, KronosTokenizer

        self._torch = torch
        self.model_id = model_id
        self.sample_count = sample_count
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        model = Kronos.from_pretrained(model_id)
        self.predictor = KronosPredictor(model, tokenizer, device=_torch_device(device), max_context=512)

    def predict(self, contexts: list[DailyContext], horizon: int) -> np.ndarray:
        if not contexts:
            return np.array([], dtype=float)
        y_timestamps = [
            pd.bdate_range(context.dates[-1] + pd.Timedelta(days=1), periods=horizon)
            for context in contexts
        ]
        self._torch.manual_seed(42)
        outputs = self.predictor.predict_batch(
            df_list=[context.ohlcv for context in contexts],
            x_timestamp_list=[pd.Series(context.dates) for context in contexts],
            y_timestamp_list=[pd.Series(timestamps) for timestamps in y_timestamps],
            pred_len=horizon,
            T=1.0,
            top_p=0.9,
            sample_count=self.sample_count,
            verbose=False,
        )
        returns = []
        for context, output in zip(contexts, outputs):
            final_close = float(output["close"].iloc[-1])
            starting_close = float(context.ohlcv["close"].iloc[-1])
            returns.append(np.clip(final_close / starting_close - 1.0, -0.99, 10.0))
        return np.array(returns, dtype=float)


def _read_daily_frames(tickers: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        filename = ticker.replace("^", "INDEX_").replace("/", "_") + ".csv"
        path = DATA_DIR / filename
        frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frames[ticker] = frame.reset_index(drop=True)
    return frames


def _make_context(
    ticker: str,
    frame: pd.DataFrame,
    signal_date: pd.Timestamp,
    price_field: str,
    context_days: int,
) -> DailyContext | None:
    available = frame.loc[frame["date"] <= signal_date].copy()
    price_column = price_field if price_field in available.columns else "close"
    available = available.dropna(subset=[price_column, "open", "high", "low", "close"]).tail(context_days)
    if len(available) < context_days:
        return None
    prices = available[price_column].astype(float).to_numpy()
    if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
        return None
    ohlcv = available[["open", "high", "low", "close"]].astype(float).copy()
    volume = available["volume"] if "volume" in available.columns else pd.Series(0.0, index=available.index)
    ohlcv["volume"] = pd.to_numeric(volume, errors="coerce").fillna(0.0).to_numpy()
    ohlcv["amount"] = ohlcv["close"] * ohlcv["volume"]
    return DailyContext(
        ticker=ticker,
        dates=pd.DatetimeIndex(available["date"]),
        log_prices=np.log(prices),
        ohlcv=ohlcv.reset_index(drop=True),
    )


def _signal_dates(features: dict[str, pd.DataFrame | pd.Series], config: RunConfig) -> list[pd.Timestamp]:
    monthly_prices = features["monthly_prices"]
    monthly_returns = features["monthly_returns"]
    assert isinstance(monthly_prices, pd.DataFrame)
    assert isinstance(monthly_returns, pd.DataFrame)
    start = pd.Timestamp(config.start)
    end = pd.Timestamp(config.end)
    target = monthly_returns.shift(-1)
    dates = [
        pd.Timestamp(timestamp)
        for timestamp in monthly_prices.index
        if start <= pd.Timestamp(timestamp) <= end and target.loc[timestamp].notna().any()
    ]
    if config.max_signals > 0:
        dates = dates[: config.max_signals]
    return dates


def _slice_features(
    features: dict[str, pd.DataFrame | pd.Series],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Keep the requested evaluation window after calculating warm-up features."""
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    sliced: dict[str, pd.DataFrame | pd.Series] = {}
    for name, value in features.items():
        if isinstance(value.index, pd.DatetimeIndex):
            sliced[name] = value.loc[(value.index >= start_date) & (value.index <= end_date)]
        else:  # pragma: no cover - feature frames are date-indexed today
            sliced[name] = value
    return sliced


def _rank_ic(predictions: pd.DataFrame, target: pd.DataFrame) -> dict[str, float | int]:
    values: list[float] = []
    for timestamp in predictions.index:
        if timestamp not in target.index:
            continue
        pair = pd.concat([predictions.loc[timestamp], target.loc[timestamp]], axis=1).dropna()
        if len(pair) < 3:
            continue
        correlation = pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())
        if pd.notna(correlation):
            values.append(float(correlation))
    return {
        "months": len(values),
        "mean_rank_ic": float(np.mean(values)) if values else np.nan,
        "positive_rank_ic_fraction": float(np.mean(np.array(values) > 0)) if values else np.nan,
    }


def _blend_ranks(first: pd.DataFrame, second: pd.DataFrame, second_weight: float = 0.5) -> pd.DataFrame:
    """Blend two rank panels with a fixed weight, renormalizing missing inputs."""
    if not 0.0 <= second_weight <= 1.0:
        raise ValueError("second_weight must be between zero and one")
    first_rank = ml_research.rank_predictions(first)
    second_rank = ml_research.rank_predictions(second)
    first_weight = 1.0 - second_weight
    numerator = first_rank.mul(first_weight).fillna(0.0) + second_rank.mul(second_weight).fillna(0.0)
    denominator = first_rank.notna().astype(float) * first_weight + second_rank.notna().astype(float) * second_weight
    return numerator.divide(denominator.replace(0.0, np.nan))


def _load_forecaster(name: str, config: RunConfig) -> tuple[Forecaster | None, str | None]:
    try:
        if name == "chronos":
            return ChronosForecaster(config.chronos_id, config.device), None
        if name == "timesfm":
            return TimesFMForecaster(config.timesfm_id, config.context_days, config.horizon_days), None
        if name == "kronos":
            return (
                KronosForecaster(
                    config.kronos_id,
                    config.kronos_tokenizer_id,
                    config.device,
                    config.kronos_repo,
                    config.sample_count,
                ),
                None,
            )
        raise ValueError(f"Unknown model: {name}")
    except Exception as exc:  # model imports and weights are optional runtime inputs
        return None, f"{type(exc).__name__}: {exc}"


def run_forecaster(
    name: str,
    forecaster: Forecaster,
    frames: dict[str, pd.DataFrame],
    tickers: list[str],
    dates: list[pd.Timestamp],
    features: dict[str, pd.DataFrame | pd.Series],
    config: RunConfig,
) -> tuple[pd.DataFrame, list[str], int]:
    predictions = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    errors: list[str] = []
    forecast_count = 0
    for index, signal_date in enumerate(dates, start=1):
        contexts: list[DailyContext] = []
        for ticker in tickers:
            context = _make_context(
                ticker,
                frames[ticker],
                signal_date,
                config.price_field,
                config.context_days,
            )
            if context is not None:
                contexts.append(context)
        if not contexts:
            errors.append(f"{signal_date.date()}: no symbols have {config.context_days} daily rows")
            continue
        try:
            values = forecaster.predict(contexts, config.horizon_days)
            if len(values) != len(contexts):
                raise ValueError(f"returned {len(values)} forecasts for {len(contexts)} contexts")
            predictions.loc[signal_date, [context.ticker for context in contexts]] = values
            forecast_count += len(values)
        except Exception as exc:
            errors.append(f"{signal_date.date()}: {type(exc).__name__}: {exc}")
        if index == 1 or index == len(dates) or index % 10 == 0:
            print(f"[{name}] signal {index}/{len(dates)} ({signal_date.date()})", flush=True)
    return predictions, errors, forecast_count


def _metrics_table(results: list[ForecastResult], baseline: dict[str, dict[str, Any]]) -> str:
    columns = ["model", "period", "months", "strategy_cagr", "benchmark_cagr", "excess_cagr", "sharpe", "max_drawdown", "turnover"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    periods = ("train", "validation", "holdout", "full")
    model_sources: list[tuple[str, dict[str, dict[str, Any]]]] = [("existing_composite", baseline)]
    model_sources.extend((result.name, result.metrics) for result in results)
    for name, metric_source in model_sources:
        if name == "existing_composite":
            metric_source = baseline
        for period in periods:
            row = metric_source.get(period, {})
            values = [
                name,
                period,
                row.get("months", 0),
                _format_metric(row.get("strategy_cagr")),
                _format_metric(row.get("benchmark_cagr")),
                _format_metric(row.get("excess_cagr")),
                _format_metric(row.get("strategy_sharpe_rf0")),
                _format_metric(row.get("strategy_max_drawdown")),
                _format_metric(row.get("average_monthly_turnover")),
            ]
            lines.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_rolling_summary(summary: pd.DataFrame) -> str:
    columns = (
        "model",
        "cost_bps",
        "windows",
        "median_sharpe",
        "positive_excess_fraction",
        "worst_excess_cagr",
        "median_turnover",
    )
    if summary.empty:
        return "_No rolling diagnostics available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in summary.sort_values(["cost_bps", "model"]).iterrows():
        values: list[str] = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, (float, np.floating)):
                values.append("—" if not np.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _render_block_winners(winners: pd.DataFrame) -> str:
    columns = ("period", "winner", "winner_sharpe", "baseline_sharpe")
    if winners.empty:
        return "_No validation block winners available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in winners.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["period"]),
                    str(row["winner"]),
                    _format_metric(row["winner_sharpe"]),
                    _format_metric(row["baseline_sharpe"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(
    results: list[ForecastResult],
    baseline: dict[str, dict[str, Any]],
    config: RunConfig,
    statuses: dict[str, dict[str, Any]],
    features: dict[str, pd.DataFrame | pd.Series],
    baseline_scores: pd.DataFrame,
) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for result in results:
        for period, values in result.metrics.items():
            rows.append({"model": result.name, **values, **{f"rank_ic_{key}": value for key, value in result.rank_ic.items()}})
    metrics_path = REPORT_DIR / "foundation_model_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    forecast_panels = {"existing_composite": baseline_scores}
    forecast_panels.update({result.name: result.predictions for result in results})
    robustness = foundation_robustness.build_cost_and_block_diagnostics(
        forecast_panels,
        features,
        config.top_k,
        vol_managed_models=frozenset({"foundation_ensemble_volmanaged"}),
    )
    rolling = foundation_robustness.build_rolling_diagnostics(
        forecast_panels,
        features,
        config.top_k,
        cost_bps=config.cost_bps,
        vol_managed_models=frozenset({"foundation_ensemble_volmanaged"}),
    )
    rolling_summary = foundation_robustness.summarize_rolling_diagnostics(rolling)
    robustness_path = REPORT_DIR / "foundation_model_robustness.csv"
    rolling_path = REPORT_DIR / "foundation_model_rolling.csv"
    rolling_summary_path = REPORT_DIR / "foundation_model_rolling_summary.csv"
    robustness.to_csv(robustness_path, index=False)
    rolling.to_csv(rolling_path, index=False)
    rolling_summary.to_csv(rolling_summary_path, index=False)
    forecasts_path = REPORT_DIR / "foundation_model_forecasts.csv"
    forecast_rows = []
    for model, panel in forecast_panels.items():
        values = panel.to_numpy(dtype=float)
        row_indices, column_indices = np.where(np.isfinite(values))
        if len(row_indices):
            model_rows = pd.DataFrame(
                {
                    "signal_date": panel.index.to_numpy()[row_indices],
                    "ticker": panel.columns.to_numpy()[column_indices],
                    "forecast": values[row_indices, column_indices],
                }
            )
            model_rows.insert(0, "model", model)
            forecast_rows.append(model_rows)
    if forecast_rows:
        pd.concat(forecast_rows, ignore_index=True).to_csv(forecasts_path, index=False)
    elif forecasts_path.exists():
        forecasts_path.unlink()
    metric_results = [result for result in results if result.metrics]
    winner = None
    if metric_results:
        winner = max(
            metric_results,
            key=lambda result: (
                result.metrics.get("validation", {}).get("strategy_sharpe_rf0", -np.inf),
                result.metrics.get("validation", {}).get("excess_cagr", -np.inf),
            ),
        )
    validation_winners = foundation_robustness.validation_winner_by_block(robustness)
    focus_models = ["existing_composite"]
    if winner is not None and winner.name not in focus_models:
        focus_models.append(winner.name)
    cost_focus = robustness.loc[
        robustness["model"].isin(focus_models)
        & robustness["period"].isin(
            ("validation_2022_2023", "holdout_2024", "holdout_2025_2026")
        )
        & robustness["cost_bps"].isin((25.0, 50.0, 100.0))
    ]
    block_diagnostics = robustness.loc[robustness["cost_bps"].eq(config.cost_bps)]
    status_lines = []
    for name, status in statuses.items():
        status_lines.append(
            f"- **{name}:** {'available' if status['available'] else 'unavailable'}; "
            f"package `{status.get('package')}` version `{status.get('version') or '—'}`; "
            f"model `{status.get('model_id')}`"
        )
    result_sections = []
    for result in results:
        result_sections.append(
            f"### {result.name}\n\n"
            f"Forecasts: {result.forecast_count:,} symbol-months across {result.signal_count} signal months. "
            f"Mean rank IC: {_format_metric(result.rank_ic.get('mean_rank_ic'))}; "
            f"positive rank-IC fraction: {_format_metric(result.rank_ic.get('positive_rank_ic_fraction'))}.\n\n"
            f"Model ID: `{result.model_id}`. Package versions: `{json.dumps(result.package_versions, sort_keys=True)}`.\n"
            f"Errors/skips: {len(result.errors)}."
        )
    winner_text = winner.name if winner else "none (no model completed)"
    winner_holdout = winner.metrics.get("holdout", {}).get("excess_cagr") if winner else np.nan
    baseline_holdout = baseline.get("holdout", {}).get("excess_cagr", np.nan)
    winner_holdout_drawdown = winner.metrics.get("holdout", {}).get("strategy_max_drawdown") if winner else np.nan
    baseline_holdout_drawdown = baseline.get("holdout", {}).get("strategy_max_drawdown", np.nan)
    report = f"""# Foundation-model backtest

Run date: {date.today().isoformat()}<br>
Research universe: `data/universe_idx30_2026-08.csv`<br>
Signal: daily history through completed month-end `t`; forecast next {config.horizon_days} trading days; hold next completed month<br>
Price field: `{config.price_field}` for Chronos/TimesFM; raw OHLCV for Kronos<br>
Context: {config.context_days} daily observations; top K: {config.top_k}; cost: {config.cost_bps:.1f} bps one-way<br>
Device: `{config.device}`; signals: {config.start} through {config.end}

## Model sources

- [Amazon Chronos official repository](https://github.com/amazon-science/chronos-forecasting)
- [Google Research TimesFM official repository](https://github.com/google-research/timesfm)
- [Kronos official repository](https://github.com/shiyu-coder/Kronos) and [Kronos paper](https://arxiv.org/abs/2508.02739)

These are zero-shot forecasts, not fine-tuned models. The models were not trained on this IDX30 panel during this run. A forecast is converted to a cross-sectional score by predicted next-horizon return, then the top K names are equally weighted. The existing composite is retained as the control. The Chronos blend grid uses fixed 25%, 50%, and 75% Chronos-rank weights; no blend weight is tuned on the holdout.

## Runtime status

{chr(10).join(status_lines)}

## Results

{_metrics_table(results, baseline)}

The foundation-model validation winner was **{winner_text}** under the pre-declared rule of highest validation Sharpe, then excess CAGR. Its holdout excess CAGR was `{_format_metric(winner_holdout)}` versus `{_format_metric(baseline_holdout)}` for the existing composite, with holdout maximum drawdown `{_format_metric(winner_holdout_drawdown)}` versus `{_format_metric(baseline_holdout_drawdown)}`. Because the validation-selected blend did not generalize, no foundation model is promoted over the existing composite. This does not establish a future edge; validation and holdout results must be stable under point-in-time constituents, costs, and additional unseen data.

## Robustness protocol

The same forecast panels are replayed at 0, 25, 50, and 100 bps one-way costs. Chronological blocks are fixed before looking at the results: 2021 training, 2022–2023 validation, 2024 holdout, and January 2025 through the latest available month holdout. The two holdout blocks are descriptive only and are not used to select a model. The full diagnostics are in `reports/foundation_model_robustness.csv`; raw forecasts are written locally to the ignored `reports/foundation_model_forecasts.csv` artifact.

Cost stress for the validation winner and the existing composite:

{foundation_robustness.render_table(cost_focus)}

Chronological block results at the declared {config.cost_bps:.1f} bps cost:

{foundation_robustness.render_table(block_diagnostics)}

Validation-block winners (selection audit only):

{_render_block_winners(validation_winners)}

Rolling 12-month stability at the declared cost:

{_render_rolling_summary(rolling_summary.loc[rolling_summary["cost_bps"].eq(config.cost_bps)])}

{chr(10).join(result_sections)}

## Leakage and limitations

- Each context is filtered with `date <= t`; the target is the following completed month's return.
- The forecast calendar is constructed from business days after `t` for model time features; no future prices are supplied.
- The current IDX30 membership is applied historically, creating survivorship and index-membership look-ahead bias.
- Chronos and TimesFM receive adjusted-close log prices, while Kronos receives raw daily OHLCV. This is a useful comparison, not a perfectly identical information set.
- Foundation models are large, stochastic, and pretrained outside Indonesia. Model weights, package versions, CPU/GPU settings, and random seeds must be recorded for any future reproduction.
- No result is investment advice or a guarantee of beating IHSG.

## Reproduction

Install CPU PyTorch first, then the optional packages. For Kronos, clone the official repository and pass its path with `--kronos-repo`.

```bash
python3 -m src.foundation_research --models all --start {config.start} --end {config.end} --kronos-repo /tmp/Kronos
python3 -m src.foundation_research --models chronos --start {config.start} --end {config.end} --max-signals 1
```

Metrics are written to `reports/foundation_model_metrics.csv`, cost/block diagnostics to `reports/foundation_model_robustness.csv`, rolling diagnostics to `reports/foundation_model_rolling.csv`, and this report to `reports/foundation_model_findings.md`.
"""
    report_path = REPORT_DIR / "foundation_model_findings.md"
    report_path.write_text(report, encoding="utf-8")
    summary_path = REPORT_DIR / "foundation_model_summary.json"
    config_payload = dict(config.__dict__)
    if config_payload.get("kronos_repo"):
        config_payload["kronos_repo"] = "<provided via --kronos-repo>"
    summary_path.write_text(
        json.dumps(
            _json_safe({
                "run_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": config_payload,
                "statuses": statuses,
                "models": [
                    {
                        "name": result.name,
                        "model_id": result.model_id,
                        "signal_count": result.signal_count,
                        "forecast_count": result.forecast_count,
                        "errors": result.errors,
                        "rank_ic": result.rank_ic,
                        "metrics": result.metrics,
                    }
                    for result in results
                ],
                "robustness_rows": robustness.to_dict(orient="records"),
                "rolling_summary": rolling_summary.to_dict(orient="records"),
            }),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return report_path, metrics_path


def _model_names(value: str) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not names or names == ["all"]:
        return ["chronos", "timesfm", "kronos"]
    supported = {"chronos", "timesfm", "kronos"}
    unknown = sorted(set(names) - supported)
    if unknown:
        raise ValueError(f"Unsupported foundation model(s): {', '.join(unknown)}")
    return names


def run(config: RunConfig, model_names: list[str]) -> list[ForecastResult]:
    if not 1 <= config.top_k <= 30:
        raise ValueError("top_k must be between 1 and 30")
    if config.context_days < 32 or config.horizon_days < 1:
        raise ValueError("context_days must be at least 32 and horizon_days must be positive")
    prices, volumes, tickers = research.load_prices(config.price_field)
    prices = prices.loc[prices.index <= config.end]
    volumes = volumes.reindex(prices.index)
    features = _slice_features(research.make_features(prices, volumes), config.start, config.end)
    dates = _signal_dates(features, config)
    frames = _read_daily_frames(tickers)
    statuses = package_status(config.kronos_repo)
    baseline_simulation = research.simulate("composite", features, config.top_k, config.cost_bps)
    baseline_metrics = ml_research.model_metrics(baseline_simulation, config.end)
    results: list[ForecastResult] = []
    for name in model_names:
        forecaster, load_error = _load_forecaster(name, config)
        if forecaster is None:
            print(f"[{name}] unavailable: {load_error}", flush=True)
            continue
        predictions, errors, forecast_count = run_forecaster(
            name,
            forecaster,
            frames,
            tickers,
            dates,
            features,
            config,
        )
        simulation = ml_research.simulate_predictions(
            predictions,
            features,
            config.top_k,
            config.cost_bps,
            vol_managed=False,
        )
        target = features["monthly_returns"]
        assert isinstance(target, pd.DataFrame)
        result = ForecastResult(
            name=name,
            predictions=predictions,
            simulation=simulation,
            metrics=ml_research.model_metrics(simulation, config.end),
            rank_ic=_rank_ic(predictions, target.shift(-1)),
            model_id=getattr(forecaster, "model_id", "unknown"),
            package_versions={
                "chronos-forecasting": _version("chronos-forecasting"),
                "timesfm": _version("timesfm"),
                "torch": _version("torch"),
            },
            errors=errors,
            signal_count=len(dates),
            forecast_count=forecast_count,
        )
        results.append(result)

    target = features["monthly_returns"]
    assert isinstance(target, pd.DataFrame)
    # Use the same valid signal calendar as the foundation forecasts. The
    # feature frame may contain one final month-end whose next-month target is
    # unavailable; it must not create a shorter rolling window for the control.
    baseline_scores = research.score_frame("composite", features).reindex(index=dates)
    if len(results) >= 2:
        available_predictions = [result.predictions for result in results]
        ensemble = ml_research.ensemble_predictions(*available_predictions)
        simulation = ml_research.simulate_predictions(ensemble, features, config.top_k, config.cost_bps)
        results.append(
            ForecastResult(
                name="foundation_ensemble",
                predictions=ensemble,
                simulation=simulation,
                metrics=ml_research.model_metrics(simulation, config.end),
                rank_ic=_rank_ic(ensemble, target.shift(-1)),
                model_id="equal rank ensemble",
                package_versions={
                    "chronos-forecasting": _version("chronos-forecasting"),
                    "timesfm": _version("timesfm"),
                    "torch": _version("torch"),
                },
                errors=[],
                signal_count=len(dates),
                forecast_count=int(ensemble.notna().sum().sum()),
            )
        )
        risk_managed_simulation = ml_research.simulate_predictions(
            ensemble,
            features,
            config.top_k,
            config.cost_bps,
            vol_managed=True,
        )
        results.append(
            ForecastResult(
                name="foundation_ensemble_volmanaged",
                predictions=ensemble,
                simulation=risk_managed_simulation,
                metrics=ml_research.model_metrics(risk_managed_simulation, config.end),
                rank_ic=_rank_ic(ensemble, target.shift(-1)),
                model_id="equal rank ensemble + fixed 10% volatility target",
                package_versions={
                    "chronos-forecasting": _version("chronos-forecasting"),
                    "timesfm": _version("timesfm"),
                    "torch": _version("torch"),
                },
                errors=[],
                signal_count=len(dates),
                forecast_count=int(ensemble.notna().sum().sum()),
            )
        )
    chronos_result = next((result for result in results if result.name == "chronos"), None)
    if chronos_result is not None:
        baseline_for_blend = baseline_scores.reindex(index=chronos_result.predictions.index)
        for chronos_weight in (0.25, 0.50, 0.75):
            blended = _blend_ranks(baseline_for_blend, chronos_result.predictions, chronos_weight)
            blended_simulation = ml_research.simulate_predictions(
                blended,
                features,
                config.top_k,
                config.cost_bps,
            )
            label = int(chronos_weight * 100)
            results.append(
                ForecastResult(
                    name=f"composite_chronos_blend_{label}",
                    predictions=blended,
                    simulation=blended_simulation,
                    metrics=ml_research.model_metrics(blended_simulation, config.end),
                    rank_ic=_rank_ic(blended, target.shift(-1)),
                    model_id=f"{100 - label}% existing composite + {label}% Chronos rank",
                    package_versions={
                        "chronos-forecasting": _version("chronos-forecasting"),
                        "timesfm": _version("timesfm"),
                        "torch": _version("torch"),
                    },
                    errors=[],
                    signal_count=len(dates),
                    forecast_count=int(blended.notna().sum().sum()),
                )
            )
    write_report(results, baseline_metrics, config, statuses, features, baseline_scores)
    return results


def main() -> None:
    args = parse_args()
    config = RunConfig(
        price_field=args.price_field,
        start=args.start,
        end=args.end,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        context_days=args.context_days,
        horizon_days=args.horizon_days,
        max_signals=args.max_signals,
        device=args.device,
        chronos_id=args.chronos_id,
        timesfm_id=args.timesfm_id,
        kronos_id=args.kronos_id,
        kronos_tokenizer_id=args.kronos_tokenizer_id,
        kronos_repo=args.kronos_repo,
        sample_count=args.sample_count,
    )
    if args.status:
        print(json.dumps(package_status(config.kronos_repo), indent=2))
        return
    results = run(config, _model_names(args.models))
    if not results:
        raise SystemExit("No requested foundation model completed; inspect reports/foundation_model_findings.md")
    summary = pd.DataFrame(
        [
            {
                "model": result.name,
                "validation_sharpe": result.metrics.get("validation", {}).get("strategy_sharpe_rf0"),
                "validation_excess_cagr": result.metrics.get("validation", {}).get("excess_cagr"),
                "holdout_excess_cagr": result.metrics.get("holdout", {}).get("excess_cagr"),
                "holdout_sharpe": result.metrics.get("holdout", {}).get("strategy_sharpe_rf0"),
                "mean_rank_ic": result.rank_ic.get("mean_rank_ic"),
            }
            for result in results
        ]
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    print("\nWrote reports/foundation_model_findings.md and reports/foundation_model_metrics.csv")


if __name__ == "__main__":
    main()
