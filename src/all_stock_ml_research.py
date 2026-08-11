"""Paper-backed factor ML on the downloaded all-listed IDX catalog.

The foundation-model experiment remains on the IDX30 panel because running
three daily foundation models over nearly one thousand symbols is expensive
and would not remove constituent survivorship bias.  This module uses the
same leakage-safe Ridge, LightGBM, and rank-ensemble machinery on the
separately cached current IDX catalog to test whether a broader cross-section
changes the result.

The catalog is current-universe data, not a point-in-time membership history.
Consequently, this is a breadth and data-quality experiment, not a claim of a
deployable all-stock strategy.
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

try:  # Support both ``python -m src.all_stock_ml_research`` and direct execution.
    from . import foundation_robustness, ml_research, research
except ImportError:  # pragma: no cover - direct CLI compatibility
    from src import foundation_robustness, ml_research, research


ROOT = Path(__file__).resolve().parents[1]
ALL_UNIVERSE_PATH = ROOT / "data" / "universe_idx_all_2026-08.csv"
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "yahoo_all"
REPORT_DIR = ROOT / "reports"
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-08-11"
DEFAULT_TOP_K = 3
DEFAULT_COST_BPS = 25.0
DEFAULT_MIN_TRAIN_ROWS = 1_000
DEFAULT_ALPHA = 10.0
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_BLOCK_LENGTH = 3
BOOTSTRAP_SEED = 20260812
SENSITIVITY_TOP_K = (1, 3, 5, 10)
SENSITIVITY_COSTS_BPS = (25.0, 100.0)


@dataclass(frozen=True)
class AllStockConfig:
    price_field: str
    start: str
    end: str
    top_k: int
    cost_bps: float
    min_train_rows: int
    alpha: float
    data_dir: Path
    universe_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-field", choices=("adjclose", "close"), default="adjclose")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--universe", type=Path, default=ALL_UNIVERSE_PATH)
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def package_status() -> dict[str, Any]:
    return {
        "lightgbm": {
            "available": importlib.util.find_spec("lightgbm") is not None,
            "version": _version("lightgbm"),
        },
        "universe": str(ALL_UNIVERSE_PATH.relative_to(ROOT)),
        "data_dir": str(DEFAULT_DATA_DIR.relative_to(ROOT)),
    }


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


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


def _filename_for(ticker: str) -> str:
    return ticker.replace("^", "INDEX_").replace("/", "_") + ".csv"


def load_prices(
    data_dir: Path,
    universe_path: Path,
    price_field: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Load every usable current-catalog file and report skipped symbols."""
    if not universe_path.exists():
        raise FileNotFoundError(f"Missing universe catalog: {universe_path}")
    universe = pd.read_csv(universe_path)
    tickers = universe["ticker"].astype(str).str.upper().tolist()
    price_series: dict[str, pd.Series] = {}
    volume_series: dict[str, pd.Series] = {}
    skipped: list[str] = []
    benchmark_path = data_dir / _filename_for("^JKSE")
    requested = ["^JKSE", *tickers]
    for ticker in requested:
        path = data_dir / _filename_for(ticker)
        if not path.exists():
            skipped.append(ticker)
            continue
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        if price_field not in frame or frame[price_field].dropna().empty:
            skipped.append(ticker)
            continue
        price_series[ticker] = pd.to_numeric(frame[price_field], errors="coerce")
        volume = frame.get("volume")
        if volume is None:
            volume = pd.Series(np.nan, index=frame.index)
        volume_series[ticker] = pd.to_numeric(volume, errors="coerce")
    if "^JKSE" not in price_series:
        raise FileNotFoundError(f"Missing usable benchmark data: {benchmark_path}")
    loaded_tickers = sorted(ticker for ticker in tickers if ticker in price_series)
    if not loaded_tickers:
        raise ValueError("No usable current-catalog stock files were found")
    ordered = ["^JKSE", *loaded_tickers]
    prices = pd.concat({ticker: price_series[ticker] for ticker in ordered}, axis=1).sort_index()
    volumes = pd.concat({ticker: volume_series[ticker] for ticker in ordered}, axis=1).sort_index()
    return prices, volumes, loaded_tickers, skipped


def _slice_features(
    features: dict[str, pd.DataFrame | pd.Series],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Keep the requested dates after calculating indicators with warm-up data."""
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    sliced: dict[str, pd.DataFrame | pd.Series] = {}
    for name, value in features.items():
        if isinstance(value.index, pd.DatetimeIndex):
            sliced[name] = value.loc[
                (value.index >= start_date) & (value.index <= end_date)
            ]
        else:  # pragma: no cover - current research inputs are date-indexed
            sliced[name] = value
    return sliced


def _metric_periods(config: AllStockConfig) -> dict[str, tuple[str, str]]:
    return {
        "full": (config.start, config.end),
        "train": (config.start, "2021-12-31"),
        "validation": ("2022-01-01", "2023-12-31"),
        "holdout": ("2024-01-01", config.end),
    }


def _metrics(simulation: pd.DataFrame, config: AllStockConfig) -> dict[str, dict[str, Any]]:
    return {
        label: ml_research.period_metrics(simulation, start, end, label)
        for label, (start, end) in _metric_periods(config).items()
    }


def _format_metric(value: Any) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _metrics_table(metrics: dict[str, dict[str, dict[str, Any]]]) -> str:
    columns = (
        "model",
        "period",
        "months",
        "excess_cagr",
        "strategy_sharpe_rf0",
        "strategy_max_drawdown",
        "average_monthly_turnover",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for model, model_metrics in metrics.items():
        for period in ("train", "validation", "holdout", "full"):
            row = model_metrics.get(period, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        model,
                        period,
                        str(row.get("months", 0)),
                        _format_metric(row.get("excess_cagr")),
                        _format_metric(row.get("strategy_sharpe_rf0")),
                        _format_metric(row.get("strategy_max_drawdown")),
                        _format_metric(row.get("average_monthly_turnover")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def _build_top_k_sensitivity(
    prediction_panels: dict[str, pd.DataFrame],
    features: dict[str, pd.DataFrame | pd.Series],
    end: str,
) -> pd.DataFrame:
    """Replay a fixed top-K/cost grid without refitting or selecting on holdout."""
    rows: list[dict[str, Any]] = []
    periods = {
        "validation": ("2022-01-01", "2023-12-31"),
        "holdout": ("2024-01-01", end),
    }
    for top_k in SENSITIVITY_TOP_K:
        for cost_bps in SENSITIVITY_COSTS_BPS:
            for model, panel in prediction_panels.items():
                if model == "existing_composite":
                    simulation = research.simulate("composite", features, top_k, cost_bps)
                else:
                    simulation = ml_research.simulate_predictions(
                        panel,
                        features,
                        top_k,
                        cost_bps,
                        vol_managed=model == "ensemble_volmanaged",
                    )
                for period, (start, finish) in periods.items():
                    row = ml_research.period_metrics(simulation, start, finish, period)
                    rows.append(
                        {
                            "model": model,
                            "top_k": top_k,
                            "cost_bps": cost_bps,
                            **row,
                        }
                    )
    return pd.DataFrame(rows)


def _top_k_sensitivity_table(sensitivity: pd.DataFrame) -> str:
    frame = sensitivity.loc[
        (sensitivity["cost_bps"] == 25.0)
        & sensitivity["model"].isin(
            ["existing_composite", "lightgbm", "ensemble_volmanaged"]
        )
        & sensitivity["period"].isin(["validation", "holdout"])
    ].copy()
    if frame.empty:
        return "_No sensitivity results available._"
    pivot = frame.pivot(index=["model", "top_k"], columns="period", values="excess_cagr")
    lines = [
        "| model | top K | validation excess CAGR | holdout excess CAGR |",
        "|---|---:|---:|---:|",
    ]
    for (model, top_k), row in pivot.sort_index().iterrows():
        lines.append(
            f"| {model} | {int(top_k)} | {_format_metric(row.get('validation'))} | "
            f"{_format_metric(row.get('holdout'))} |"
        )
    return "\n".join(lines)


def _block_bootstrap_active(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
    start: str,
    end: str,
    seed: int,
) -> dict[str, Any]:
    frame = pd.concat(
        [candidate["strategy_return"], comparator["strategy_return"]],
        axis=1,
        keys=("candidate", "comparator"),
    ).loc[start:end].dropna()
    difference = (frame["candidate"] - frame["comparator"]).to_numpy(dtype=float)
    if len(difference) < BOOTSTRAP_BLOCK_LENGTH:
        return {"months": len(difference), "mean": np.nan, "lower": np.nan, "median": np.nan, "upper": np.nan, "positive_probability": np.nan}
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(len(difference) / BOOTSTRAP_BLOCK_LENGTH))
    starts = rng.integers(0, len(difference), size=(BOOTSTRAP_SAMPLES, block_count))
    offsets = np.arange(BOOTSTRAP_BLOCK_LENGTH)[None, None, :]
    indices = (starts[:, :, None] + offsets).reshape(BOOTSTRAP_SAMPLES, -1)[:, : len(difference)]
    indices %= len(difference)
    annualized = difference[indices].mean(axis=1) * 12.0
    quantiles = np.quantile(annualized, [0.025, 0.50, 0.975])
    return {
        "months": int(len(difference)),
        "mean": float(difference.mean() * 12.0),
        "lower": float(quantiles[0]),
        "median": float(quantiles[1]),
        "upper": float(quantiles[2]),
        "positive_probability": float(np.mean(annualized > 0.0)),
    }


def _gate_candidate(
    model: str,
    model_metrics: dict[str, dict[str, Any]],
    baseline_metrics: dict[str, dict[str, Any]],
    robustness: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    validation = model_metrics.get("validation", {}).get("excess_cagr", np.nan)
    baseline = baseline_metrics.get("validation", {}).get("excess_cagr", np.nan)
    if not np.isfinite(validation) or not np.isfinite(baseline) or validation <= baseline:
        reasons.append("validation excess CAGR does not beat the existing composite")
    for cost in (25.0, 100.0):
        for period in ("holdout_2024", "holdout_2025_2026"):
            rows = robustness.loc[
                robustness["model"].eq(model)
                & robustness["period"].eq(period)
                & np.isclose(robustness["cost_bps"], cost)
            ]
            excess = rows.iloc[0]["excess_cagr"] if not rows.empty else np.nan
            if not np.isfinite(excess) or excess <= 0.0:
                reasons.append(f"excess CAGR is not positive in {period} at {int(cost)} bps")
    rows = rolling_summary.loc[
        rolling_summary["model"].eq(model)
        & np.isclose(rolling_summary["cost_bps"], 25.0)
    ]
    positive = rows.iloc[0]["positive_excess_fraction"] if not rows.empty else np.nan
    if not np.isfinite(positive) or positive < 0.60:
        reasons.append("fewer than 60% of trailing 12-month windows have positive excess CAGR")
    rows = bootstrap.loc[bootstrap["model"].eq(model)]
    lower = rows.iloc[0]["market_ci_lower"] if not rows.empty else np.nan
    if not np.isfinite(lower) or lower <= 0.0:
        reasons.append("95% block-bootstrap lower CI for active return versus IHSG is not above zero")
    return not reasons, reasons


def write_report(
    config: AllStockConfig,
    metrics: dict[str, dict[str, dict[str, Any]]],
    prediction_panels: dict[str, pd.DataFrame],
    features: dict[str, pd.DataFrame | pd.Series],
    training_rows: pd.DataFrame,
    loaded_tickers: list[str],
    skipped: list[str],
) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    robustness = foundation_robustness.build_cost_and_block_diagnostics(
        prediction_panels, features, config.top_k
    )
    sensitivity = _build_top_k_sensitivity(prediction_panels, features, config.end)
    rolling = foundation_robustness.build_rolling_diagnostics(
        prediction_panels,
        features,
        config.top_k,
        cost_bps=config.cost_bps,
        vol_managed_models=frozenset({"ensemble_volmanaged"}),
    )
    rolling_summary = foundation_robustness.summarize_rolling_diagnostics(rolling)
    baseline_simulation = ml_research.simulate_predictions(
        prediction_panels["existing_composite"], features, config.top_k, config.cost_bps
    )
    market = baseline_simulation[["benchmark_return"]].rename(
        columns={"benchmark_return": "strategy_return"}
    )
    bootstrap_rows: list[dict[str, Any]] = []
    for model, panel in prediction_panels.items():
        if model == "existing_composite":
            continue
        simulation = ml_research.simulate_predictions(
            panel,
            features,
            config.top_k,
            config.cost_bps,
            vol_managed=model == "ensemble_volmanaged",
        )
        market_bootstrap = _block_bootstrap_active(
            simulation, market, "2024-01-01", config.end, BOOTSTRAP_SEED
        )
        control_bootstrap = _block_bootstrap_active(
            simulation, baseline_simulation, "2024-01-01", config.end, BOOTSTRAP_SEED
        )
        bootstrap_rows.append(
            {
                "model": model,
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
    for model in prediction_panels:
        if model == "existing_composite":
            continue
        passed, reasons = _gate_candidate(
            model,
            metrics[model],
            metrics["existing_composite"],
            robustness,
            rolling_summary,
            bootstrap,
        )
        gates[model] = {"passed": passed, "reasons": reasons}
    candidates = [model for model in prediction_panels if model != "existing_composite"]
    validation_winner = max(
        candidates,
        key=lambda model: (
            metrics[model].get("validation", {}).get("strategy_sharpe_rf0", -np.inf),
            metrics[model].get("validation", {}).get("excess_cagr", -np.inf),
        ),
    ) if candidates else None
    preferred = validation_winner if validation_winner and gates[validation_winner]["passed"] else None

    metrics_rows = [
        {"model": model, **values}
        for model, model_metrics in metrics.items()
        for _, values in model_metrics.items()
    ]
    metrics_path = REPORT_DIR / "all_stock_ml_metrics.csv"
    robustness_path = REPORT_DIR / "all_stock_ml_robustness.csv"
    rolling_path = REPORT_DIR / "all_stock_ml_rolling.csv"
    rolling_summary_path = REPORT_DIR / "all_stock_ml_rolling_summary.csv"
    bootstrap_path = REPORT_DIR / "all_stock_ml_bootstrap.csv"
    training_path = REPORT_DIR / "all_stock_ml_training.csv"
    sensitivity_path = REPORT_DIR / "all_stock_ml_sensitivity.csv"
    summary_path = REPORT_DIR / "all_stock_ml_summary.json"
    report_path = REPORT_DIR / "all_stock_ml_findings.md"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    robustness.to_csv(robustness_path, index=False)
    rolling.to_csv(rolling_path, index=False)
    rolling_summary.to_csv(rolling_summary_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    training_rows.to_csv(training_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)

    forecast_rows: list[pd.DataFrame] = []
    for model, panel in prediction_panels.items():
        values = panel.to_numpy(dtype=float)
        row_indices, column_indices = np.where(np.isfinite(values))
        if len(row_indices):
            rows = pd.DataFrame(
                {
                    "signal_date": panel.index.to_numpy()[row_indices],
                    "ticker": panel.columns.to_numpy()[column_indices],
                    "forecast": values[row_indices, column_indices],
                }
            )
            rows.insert(0, "model", model)
            forecast_rows.append(rows)
    if forecast_rows:
        pd.concat(forecast_rows, ignore_index=True).to_csv(
            REPORT_DIR / "all_stock_ml_forecasts.csv", index=False
        )

    gate_lines = []
    for model in candidates:
        status = "passes" if gates[model]["passed"] else "rejects"
        reasons = "; ".join(gates[model]["reasons"]) if gates[model]["reasons"] else "all fixed audit checks passed"
        gate_lines.append(f"- **{model}:** {status} the fixed audit gate ({reasons}).")
    bootstrap_lines = [
        "| model | months | market active annualized mean | market 95% CI | control active annualized mean | control 95% CI |",
        "|---|---:|---:|---|---:|---|",
    ]
    for _, row in bootstrap.iterrows():
        bootstrap_lines.append(
            f"| {row['model']} | {int(row['months'])} | {_format_metric(row['market_active_annualized_mean'])} | "
            f"[{row['market_ci_lower']:.4f}, {row['market_ci_upper']:.4f}] | "
            f"{_format_metric(row['control_active_annualized_mean'])} | "
            f"[{row['control_ci_lower']:.4f}, {row['control_ci_upper']:.4f}] |"
        )
    lightgbm_holdout = metrics.get("lightgbm", {}).get("holdout", {})
    report = f"""# All-listed Indonesian factor-ML research

Run date: {date.today().isoformat()}<br>
Universe catalog: `{config.universe_path.relative_to(ROOT)}`<br>
Downloaded data directory: `{config.data_dir.relative_to(ROOT)}`<br>
Price field: `{config.price_field}`; top K: `{config.top_k}`; cost: `{config.cost_bps:.1f}` bps one-way<br>
Loaded current-catalog tickers: `{len(loaded_tickers)}`; skipped/unavailable: `{len(skipped)}`<br>

## Research basis

[Gu, Kelly, and Xiu](https://www.nber.org/papers/w25398) compare linear, tree, and neural-network methods for cross-sectional return prediction and emphasize nonlinear interactions among momentum, liquidity, and volatility. This run applies their model-comparison idea to a broader Indonesian panel. It is not a literal replication and does not solve the current-universe survivorship problem.

## Design and leakage controls

- Daily OHLCV is loaded for the current IDX catalog and transformed into the repository's monthly momentum, volatility, trend, liquidity, and market-regime features.
- A signal at completed month-end `t` predicts and trades the next completed month's return.
- Ridge and LightGBM use expanding walk-forward training. Training rows are strictly earlier than the signal month; normalization and target winsorization are training-only.
- Ridge alpha is fixed at `{config.alpha:g}`. LightGBM uses the existing shallow fixed specification and is not tuned on the holdout.
- The fixed candidate set is the existing composite, Ridge, LightGBM, equal rank ensemble, and the equal ensemble with the fixed 10% inverse-volatility exposure rule.
- Validation is 2022–2023. 2024 and 2025–2026 are holdouts. Costs are replayed at 0, 25, 50, and 100 bps.

## Results

{_metrics_table(metrics)}

Validation winner: **{validation_winner or 'none'}** under highest validation Sharpe, then excess CAGR. A candidate is considered audit-passing only if it beats the control in validation, remains positive in both holdout blocks at 25 and 100 bps, is positive in at least 60% of rolling 12-month windows, and has a positive 95% block-bootstrap lower CI versus IHSG.

## Fixed audit gate

{chr(10).join(gate_lines) if gate_lines else '_No candidates completed._'}

Preferred research candidate: **{preferred or 'none'}**. No candidate is promoted automatically because the catalog is a current snapshot: it includes survivorship and index-membership look-ahead, and the historical panel does not contain delisted names or historical membership dates.

## Predeclared top-K sensitivity

The default top K is three. The same forecasts are replayed at top K = 1, 3, 5, and 10, with 25 and 100 bps costs, without refitting or using holdout results to change the model. The compact 25 bps view below compares the control, LightGBM, and its rank ensemble with the fixed volatility rule; the complete grid is in `reports/all_stock_ml_sensitivity.csv`.

{_top_k_sensitivity_table(sensitivity)}

## Paired block bootstrap

The 2024–2026 bootstrap resamples paired three-month circular blocks 5,000 times. The control comparison is against the all-listed existing composite, not the IDX30 composite.

{chr(10).join(bootstrap_lines)}

## Limitations

- A broad current catalog improves cross-sectional sample size but is not a point-in-time universe. A credible deployment study still needs historical IDX membership, delistings, suspensions, corporate actions, and liquidity/execution constraints.
- Yahoo Finance is a convenient research source rather than a licensed exchange feed. Some files have short histories or only one usable row; these symbols naturally drop from feature-ranked signals until enough history exists.
- Small and illiquid stocks can dominate a top-K portfolio. Results must be stress-tested with explicit liquidity, spread, price-limit, and capacity rules before any live use.
- At the default top-three setting, the LightGBM holdout beta is `{_format_metric(lightgbm_holdout.get('beta_to_benchmark'))}` and its maximum drawdown is `{_format_metric(lightgbm_holdout.get('strategy_max_drawdown'))}`. The apparent excess return therefore comes with substantial concentration and market exposure; it is not a low-risk alpha claim.
- No result is investment advice or a guarantee of beating IHSG.

## Reproduction

```bash
python3 src/download_data.py --universe all --output-dir data/raw/yahoo_all \
  --metadata-path data/raw/yahoo_all_metadata.json --continue-on-error
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.all_stock_ml_research
```

Raw all-stock forecasts are written to ignored `reports/all_stock_ml_forecasts.csv`. Committed tables are in `reports/all_stock_ml_metrics.csv`, `reports/all_stock_ml_robustness.csv`, `reports/all_stock_ml_rolling_summary.csv`, `reports/all_stock_ml_bootstrap.csv`, `reports/all_stock_ml_training.csv`, and `reports/all_stock_ml_sensitivity.csv`.
"""
    report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "run_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": {
                        key: (
                            str(value.relative_to(ROOT))
                            if isinstance(value, Path) and value.is_absolute()
                            else str(value) if isinstance(value, Path) else value
                        )
                        for key, value in config.__dict__.items()
                    },
                    "loaded_ticker_count": len(loaded_tickers),
                    "skipped": skipped,
                    "metrics": metrics,
                    "validation_winner": validation_winner,
                    "preferred_research_candidate": preferred,
                    "gates": gates,
                    "bootstrap": bootstrap.to_dict(orient="records"),
                    "top_k_sensitivity": sensitivity.to_dict(orient="records"),
                }
            ),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return report_path, metrics_path


def run(config: AllStockConfig) -> dict[str, dict[str, dict[str, Any]]]:
    if not 1 <= config.top_k <= 30:
        raise ValueError("top_k must be between 1 and 30")
    if config.cost_bps < 0.0 or config.min_train_rows < 60 or config.alpha <= 0.0:
        raise ValueError("cost, minimum training rows, and alpha are invalid")
    prices, volumes, loaded_tickers, skipped = load_prices(
        config.data_dir, config.universe_path, config.price_field
    )
    prices = prices.loc[prices.index <= config.end]
    volumes = volumes.reindex(prices.index)
    features = _slice_features(
        research.make_features(prices, volumes), config.start, config.end
    )
    panel = ml_research.make_panel(features)
    ridge_predictions, ridge_training = ml_research.walk_forward_predictions(
        panel, "ridge", config.min_train_rows, config.alpha
    )
    errors: dict[str, str] = {}
    try:
        lightgbm_predictions, lightgbm_training = ml_research.walk_forward_predictions(
            panel, "lightgbm", config.min_train_rows, config.alpha
        )
    except RuntimeError as exc:
        errors["lightgbm"] = str(exc)
        lightgbm_predictions = pd.DataFrame(
            index=ridge_predictions.index, columns=ridge_predictions.columns, dtype=float
        )
        lightgbm_training = pd.Series(index=ridge_predictions.index, dtype=float)
    ensemble = ml_research.ensemble_predictions(ridge_predictions, lightgbm_predictions)
    baseline = research.score_frame("composite", features)
    prediction_panels = {
        "existing_composite": baseline,
        "ridge": ridge_predictions,
        "lightgbm": lightgbm_predictions,
        "ensemble": ensemble,
        "ensemble_volmanaged": ensemble,
    }
    simulations = {
        "existing_composite": research.simulate("composite", features, config.top_k, config.cost_bps),
        "ridge": ml_research.simulate_predictions(
            ridge_predictions, features, config.top_k, config.cost_bps
        ),
        "lightgbm": ml_research.simulate_predictions(
            lightgbm_predictions, features, config.top_k, config.cost_bps
        ),
        "ensemble": ml_research.simulate_predictions(
            ensemble, features, config.top_k, config.cost_bps
        ),
        "ensemble_volmanaged": ml_research.simulate_predictions(
            ensemble, features, config.top_k, config.cost_bps, vol_managed=True
        ),
    }
    metrics = {name: _metrics(simulation, config) for name, simulation in simulations.items()}
    training_rows = pd.DataFrame(
        [
            {
                "model": "ridge",
                "signals": int(ridge_predictions.notna().any(axis=1).sum()),
                "forecast_rows": int(ridge_predictions.notna().sum().sum()),
                "median_training_rows": float(ridge_training.dropna().median()) if not ridge_training.dropna().empty else np.nan,
            },
            {
                "model": "lightgbm",
                "signals": int(lightgbm_predictions.notna().any(axis=1).sum()),
                "forecast_rows": int(lightgbm_predictions.notna().sum().sum()),
                "median_training_rows": float(lightgbm_training.dropna().median()) if not lightgbm_training.dropna().empty else np.nan,
            },
        ]
    )
    write_report(
        config,
        metrics,
        prediction_panels,
        features,
        training_rows,
        loaded_tickers,
        skipped,
    )
    return metrics


def main() -> None:
    args = parse_args()
    if args.status:
        print(json.dumps(package_status(), indent=2))
        return
    config = AllStockConfig(
        price_field=args.price_field,
        start=args.start,
        end=args.end,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        min_train_rows=args.min_train_rows,
        alpha=args.alpha,
        data_dir=args.data_dir.resolve(),
        universe_path=args.universe.resolve(),
    )
    metrics = run(config)
    summary = pd.DataFrame(
        [
            {
                "model": model,
                "validation_excess_cagr": values.get("validation", {}).get("excess_cagr"),
                "validation_sharpe": values.get("validation", {}).get("strategy_sharpe_rf0"),
                "holdout_excess_cagr": values.get("holdout", {}).get("excess_cagr"),
                "holdout_sharpe": values.get("holdout", {}).get("strategy_sharpe_rf0"),
            }
            for model, values in metrics.items()
        ]
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    print("\nWrote reports/all_stock_ml_findings.md and reports/all_stock_ml_metrics.csv")


if __name__ == "__main__":
    main()
