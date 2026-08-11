"""Paper-backed machine-learning experiments for the IDX30 research panel.

This module deliberately keeps the experiment small and auditable. It builds a
monthly cross-sectional panel from information available at each month-end,
trains expanding-window models, and applies predictions to the following month.
The default model set is fixed before a run: a regularized linear model, a
shallow gradient-boosted tree model, their equal-weight ensemble, and the same
ensemble with inverse-volatility exposure control.

Chronos, TimesFM, and Kronos are evaluated by the separate
``src.foundation_research`` runner. They are not imported at module load time
because their packages and model weights are intentionally not dependencies of
the reproducible baseline. Use that runner for actual zero-shot forecasts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:  # Support both ``python -m src.ml_research`` and direct script execution.
    from . import research
except ImportError:  # pragma: no cover - direct CLI compatibility
    from src import research


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
UNIVERSE_PATH = ROOT / "data" / "universe_idx30_2026-08.csv"
DATA_DIR = ROOT / "data" / "raw" / "yahoo"

FEATURE_NAMES = (
    "momentum_12_1",
    "momentum_6_1",
    "momentum_3",
    "low_volatility",
    "trend_200",
    "liquidity",
    "market_trend",
    "market_momentum",
)
DEFAULT_ALPHA = 10.0
DEFAULT_MIN_TRAIN_ROWS = 60
DEFAULT_TOP_K = 3
DEFAULT_COST_BPS = 25.0
VOL_TARGET = 0.10


@dataclass(frozen=True)
class ModelResult:
    name: str
    predictions: pd.DataFrame
    simulation: pd.DataFrame
    metrics: dict[str, dict[str, Any]]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-field", choices=("adjclose", "close"), default="adjclose")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-08-11")
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    parser.add_argument(
        "--foundation-status",
        action="store_true",
        help="Only print optional Chronos, TimesFM, and Kronos package availability",
    )
    return parser.parse_args()


def foundation_status() -> dict[str, dict[str, Any]]:
    """Return package availability without importing heavyweight model code."""
    packages = {
        "chronos": ("chronos", "chronos-forecasting"),
        "timesfm": ("timesfm", "timesfm"),
        # Kronos' official repository exposes its model code as ``model``.
        "kronos": ("model", "Kronos repository model package"),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (module, install_hint) in packages.items():
        result[name] = {
            "available": importlib.util.find_spec(module) is not None,
            "module": module,
            "install_hint": install_hint,
        }
    return result


def _rank(frame: pd.DataFrame, low_is_good: bool = False) -> pd.DataFrame:
    values = -frame if low_is_good else frame
    return values.rank(axis=1, pct=True, method="average")


def _as_frame(value: pd.DataFrame | pd.Series, columns: Iterable[str]) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        return value.to_frame().reindex(columns=list(columns))
    return value


def build_feature_frames(features: dict[str, pd.DataFrame | pd.Series]) -> dict[str, pd.DataFrame]:
    """Create fixed cross-sectional percentile features from the research panel."""
    prices = features["monthly_prices"]
    assert isinstance(prices, pd.DataFrame)
    columns = list(prices.columns)
    raw_frames = {
        "momentum_12_1": _as_frame(features["mom12_1"], columns),
        "momentum_6_1": _as_frame(features["mom6_1"], columns),
        "momentum_3": _as_frame(features["mom3"], columns),
        "low_volatility": _as_frame(features["volatility"], columns),
        "trend_200": _as_frame(features["trend200"], columns),
        "liquidity": _as_frame(features["liquidity"], columns),
    }
    ranked = {
        name: _rank(frame, low_is_good=name == "low_volatility")
        for name, frame in raw_frames.items()
    }
    market_trend = features["market_trend"]
    market_momentum = features["market_mom12_1"]
    assert isinstance(market_trend, pd.Series)
    assert isinstance(market_momentum, pd.Series)
    ranked["market_trend"] = pd.DataFrame(
        np.repeat(market_trend.to_numpy()[:, None], len(columns), axis=1),
        index=market_trend.index,
        columns=columns,
    )
    ranked["market_momentum"] = pd.DataFrame(
        np.repeat(market_momentum.to_numpy()[:, None], len(columns), axis=1),
        index=market_momentum.index,
        columns=columns,
    )
    return ranked


def make_panel(features: dict[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
    """Return one row per signal month and stock with a one-month-ahead target."""
    feature_frames = build_feature_frames(features)
    monthly_returns = features["monthly_returns"]
    assert isinstance(monthly_returns, pd.DataFrame)
    target = monthly_returns.shift(-1)
    rows: list[pd.DataFrame] = []
    for name in FEATURE_NAMES:
        if name not in feature_frames:
            raise KeyError(f"Missing feature frame: {name}")
    for timestamp in target.index:
        frame = pd.DataFrame(index=target.columns)
        for name in FEATURE_NAMES:
            frame[name] = feature_frames[name].reindex(index=[timestamp], columns=target.columns).iloc[0]
        frame["target"] = target.loc[timestamp]
        frame["signal_date"] = timestamp
        frame["ticker"] = frame.index
        rows.append(frame.reset_index(drop=True))
    if not rows:
        return pd.DataFrame(columns=[*FEATURE_NAMES, "target", "signal_date", "ticker"])
    panel = pd.concat(rows, ignore_index=True)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    return panel


def _training_and_current(
    panel: pd.DataFrame,
    signal_date: pd.Timestamp,
    min_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    train = panel.loc[panel["signal_date"] < signal_date].dropna(subset=[*FEATURE_NAMES, "target"])
    current = panel.loc[panel["signal_date"] == signal_date].dropna(subset=FEATURE_NAMES)
    if len(train) < min_train_rows or current.empty:
        return None
    return train, current


def _winsorize(values: np.ndarray) -> np.ndarray:
    """Use a fixed, training-only outlier guard for noisy one-month returns."""
    lower, upper = np.nanpercentile(values, [1.0, 99.0])
    return np.clip(values, lower, upper)


def _ridge_fit_predict(train: pd.DataFrame, current: pd.DataFrame, alpha: float) -> pd.Series:
    x_train = train.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    y_train = _winsorize(train["target"].to_numpy(dtype=float))
    x_current = current.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z_train = (x_train - mean) / scale
    z_current = (x_current - mean) / scale
    system = z_train.T @ z_train + alpha * np.eye(z_train.shape[1])
    coefficients = np.linalg.solve(system, z_train.T @ (y_train - y_train.mean()))
    prediction = y_train.mean() + z_current @ coefficients
    return pd.Series(prediction, index=current["ticker"].to_numpy())


def _lightgbm_fit_predict(train: pd.DataFrame, current: pd.DataFrame) -> pd.Series:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("LightGBM is not installed; install requirements-ml.txt") from exc
    x_train = train.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    x_current = current.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    dataset = lgb.Dataset(
        x_train,
        label=_winsorize(train["target"].to_numpy(dtype=float)),
        feature_name=list(FEATURE_NAMES),
        free_raw_data=True,
    )
    model = lgb.train(
        {
            "objective": "regression",
            "learning_rate": 0.03,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 30,
            "lambda_l2": 10.0,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": 42,
            "verbosity": -1,
        },
        dataset,
        num_boost_round=100,
    )
    prediction = model.predict(x_current)
    return pd.Series(prediction, index=current["ticker"].to_numpy())


def walk_forward_predictions(
    panel: pd.DataFrame,
    model: str,
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fit each model using only rows whose signal date precedes the forecast date."""
    tickers = sorted(panel["ticker"].dropna().unique())
    predictions = pd.DataFrame(index=sorted(panel["signal_date"].unique()), columns=tickers, dtype=float)
    training_counts = pd.Series(index=predictions.index, dtype=float)
    for signal_date in predictions.index:
        prepared = _training_and_current(panel, pd.Timestamp(signal_date), min_train_rows)
        if prepared is None:
            continue
        train, current = prepared
        if model == "ridge":
            predicted = _ridge_fit_predict(train, current, alpha)
        elif model == "lightgbm":
            predicted = _lightgbm_fit_predict(train, current)
        else:
            raise ValueError(f"Unsupported walk-forward model: {model}")
        predictions.loc[signal_date, predicted.index] = predicted.to_numpy()
        training_counts.loc[signal_date] = len(train)
    return predictions, training_counts


def rank_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Convert model forecasts into a comparable cross-sectional selection score."""
    return predictions.rank(axis=1, pct=True, method="average")


def ensemble_predictions(*predictions: pd.DataFrame) -> pd.DataFrame:
    """Average available model percentile ranks without filling missing forecasts."""
    ranked = [rank_predictions(prediction) for prediction in predictions]
    stacked = pd.concat(ranked, keys=range(len(ranked)), names=["model", "date"])
    return stacked.groupby(level="date").mean()


def inverse_volatility_exposure(features: dict[str, pd.DataFrame | pd.Series]) -> pd.Series:
    """Compute a fixed 10% target-volatility exposure from lagged market data."""
    prices = features["daily_prices"]
    assert isinstance(prices, pd.DataFrame)
    market = features.get("daily_prices")
    del market
    # The feature builder receives stock-only daily prices, so use the market
    # series retained in the monthly benchmark calculation when available.
    benchmark_returns = features["benchmark_returns"]
    assert isinstance(benchmark_returns, pd.Series)
    monthly_vol = benchmark_returns.rolling(6, min_periods=3).std() * np.sqrt(12)
    exposure = (VOL_TARGET / monthly_vol).clip(lower=0.0, upper=1.0)
    return exposure


def _cash_turnover(weights: pd.DataFrame) -> pd.Series:
    cash = 1.0 - weights.sum(axis=1)
    previous_weights = weights.shift(1).fillna(0.0)
    previous_cash = cash.shift(1).fillna(1.0)
    return 0.5 * ((weights - previous_weights).abs().sum(axis=1) + (cash - previous_cash).abs())


def weights_from_predictions(predictions: pd.DataFrame, features: dict[str, pd.DataFrame | pd.Series], top_k: int, vol_managed: bool) -> pd.DataFrame:
    prices = features["monthly_prices"]
    assert isinstance(prices, pd.DataFrame)
    weights = pd.DataFrame(0.0, index=predictions.index, columns=prices.columns)
    for timestamp, row in predictions.iterrows():
        eligible = row.dropna().index.intersection(prices.reindex(index=[timestamp]).dropna(axis=1).columns)
        if len(eligible) == 0:
            continue
        chosen = row.loc[eligible].nlargest(min(top_k, len(eligible))).index
        weights.loc[timestamp, chosen] = 1.0 / len(chosen)
    if vol_managed:
        exposure = inverse_volatility_exposure(features).reindex(weights.index).fillna(1.0)
        weights = weights.mul(exposure, axis=0)
    return weights


def simulate_predictions(
    predictions: pd.DataFrame,
    features: dict[str, pd.DataFrame | pd.Series],
    top_k: int,
    cost_bps: float,
    vol_managed: bool = False,
) -> pd.DataFrame:
    """Apply monthly predictions to the next month's returns with costs."""
    monthly_returns = features["monthly_returns"]
    benchmark_returns = features["benchmark_returns"]
    assert isinstance(monthly_returns, pd.DataFrame)
    assert isinstance(benchmark_returns, pd.Series)
    weights = weights_from_predictions(predictions, features, top_k, vol_managed)
    weights = weights.reindex(index=monthly_returns.index, columns=monthly_returns.columns).fillna(0.0)
    next_returns = monthly_returns.shift(-1)
    next_benchmark = benchmark_returns.shift(-1)
    gross = (weights * next_returns).sum(axis=1, min_count=1)
    trade_turnover = _cash_turnover(weights)
    net = gross - trade_turnover * cost_bps / 10_000.0
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
    simulation["exposure"] = weights.sum(axis=1)
    return simulation


def period_metrics(simulation: pd.DataFrame, start: str, end: str, label: str) -> dict[str, Any]:
    return research.metric_row(simulation, start, end, label)


def model_metrics(simulation: pd.DataFrame, end: str) -> dict[str, dict[str, Any]]:
    periods = {
        "full": ("2015-01-01", end),
        "train": ("2015-01-01", "2021-12-31"),
        "validation": ("2022-01-01", "2023-12-31"),
        "holdout": ("2024-01-01", end),
    }
    return {label: period_metrics(simulation, start, finish, label) for label, (start, finish) in periods.items()}


def _format_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metrics_table(results: list[ModelResult]) -> str:
    columns = ["model", "period", "months", "strategy_cagr", "benchmark_cagr", "excess_cagr", "sharpe", "max_drawdown", "turnover"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for result in results:
        for period in ("train", "validation", "holdout", "full"):
            row = result.metrics.get(period, {})
            values = [
                result.name,
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


def write_report(
    results: list[ModelResult],
    panel: pd.DataFrame,
    training_counts: dict[str, pd.Series],
    args: argparse.Namespace,
    foundation: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    for result in results:
        for period, values in result.metrics.items():
            metrics_rows.append({"model": result.name, **values})
    metrics_path = REPORT_DIR / "ml_model_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    selected = max(
        results,
        key=lambda result: (
            result.metrics.get("validation", {}).get("strategy_sharpe_rf0", -np.inf),
            result.metrics.get("validation", {}).get("excess_cagr", -np.inf),
        ),
    )
    ml_results = [result for result in results if result.name != "existing_composite"]
    selected_ml = max(
        ml_results,
        key=lambda result: (
            result.metrics.get("validation", {}).get("strategy_sharpe_rf0", -np.inf),
            result.metrics.get("validation", {}).get("excess_cagr", -np.inf),
        ),
    )
    latest_signal = panel["signal_date"].max().date().isoformat() if not panel.empty else "—"
    foundation_lines = "\n".join(
        f"- **{name}:** {'available' if status['available'] else 'not installed'} (`{status['install_hint']}`)"
        for name, status in foundation.items()
    )
    report = f"""# Paper-backed machine-learning research

Run date: {date.today().isoformat()}<br>
Price field: `{args.price_field}`<br>
Research universe: `data/universe_idx30_2026-08.csv`<br>
Signal convention: features are observed at completed month-end `t`; the target is the next month's return and is traded only after `t`.<br>
Latest signal row in the panel: {latest_signal}<br>
Cost: {args.cost_bps:.1f} bps per unit turnover; top K: {args.top_k}

## What the papers contributed

- Jegadeesh and Titman document intermediate-horizon momentum and motivate the existing 12–1 feature: [The Journal of Finance (1993)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb04702.x).
- Gu, Kelly, and Xiu compare regularized linear, tree, and neural-network methods for empirical asset pricing and identify momentum, liquidity, and volatility as important predictors: [NBER Working Paper 25398](https://www.nber.org/papers/w25398).
- Moreira and Muir motivate reducing exposure when realized volatility is high: [NBER Working Paper 22208](https://www.nber.org/papers/w22208).

These papers support testable design choices; they do not establish an Indonesian equity edge or guarantee future performance. This run keeps the features, model hyperparameters, top K, cost, and time splits fixed before comparing results.

## Models tested

- **Ridge:** expanding-window pooled cross-sectional regression with training-only standardization, 1%/99% training-target winsorization, and fixed alpha `{DEFAULT_ALPHA:g}`.
- **LightGBM:** shallow boosted trees with fixed depth/leaves, regularization, and 100 estimators. It is included as the nonlinear tree method suggested by the asset-pricing literature.
- **Equal ensemble:** average of the two models' cross-sectional percentile forecasts when both are available.
- **Volatility-managed ensemble:** the equal ensemble with long-only exposure scaled by `min(1, 10% / six-month realized IHSG volatility)`; uninvested exposure is cash with zero return.
- **Existing composite:** the repository's current 12–1/6–1/3-month/low-volatility rank formula, included as a non-ML control.

The models are retrained at every signal month using only earlier panel rows. Validation (2022–2023) is the only selection window; 2024 onward is a holdout and is not used to select weights or hyperparameters.

## Results

{_metrics_table(results)}

The all-candidate validation winner under the pre-declared rule (highest Sharpe, then excess CAGR) was **{selected.name}**. The ML-only winner was **{selected_ml.name}**. That is a selection result, not a claim that either method beats the market live. If the existing composite wins validation, the evidence does not justify replacing it with the ML candidate; compare holdout performance and rerun on a point-in-time universe first.

Panel rows: {len(panel):,}. Minimum training rows: {args.min_train_rows}.<br>
Training row counts are recorded in the run code; no future target is included in a forecast month.

## Foundation-model availability

The requested financial/time-series foundation models are optional because their packages and weights are large and their APIs change independently of this repository:

{foundation_lines}

Chronos, TimesFM, and Kronos should be added only through a separate run that records exact model IDs, package versions, context length, forecast horizon, and download date. A foundation-model forecast must pass the same expanding-window, next-month, transaction-cost, and holdout protocol before it can be compared with these results.

The completed zero-shot rolling comparison is in `reports/foundation_model_findings.md` and is reproduced with `python3 -m src.foundation_research` in the CPU foundation environment.

## Limitations

- The IDX30 snapshot is a current-universe panel and therefore has survivorship and index-membership look-ahead bias.
- Yahoo Finance is a convenient research source, not a licensed exchange feed; prices, corporate actions, suspensions, limits, taxes, spreads, and market impact are incomplete or unmodeled.
- A 30-stock panel is small for machine learning. The all-listed catalog is now available to the app, but a point-in-time all-stock history is still required for a stronger ML study.
- Out-of-sample backtest performance is not a promise of excess returns or investment advice.

## Reproduction

```bash
python3 -m src.ml_research
python3 -m src.ml_research --price-field close
python3 -m src.ml_research --foundation-status
```

Generated metrics are in `reports/ml_model_metrics.csv` and this report is in `reports/ml_research_findings.md`.
"""
    report_path = REPORT_DIR / "ml_research_findings.md"
    report_path.write_text(report, encoding="utf-8")
    summary_path = REPORT_DIR / "ml_research_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_at_utc": datetime.now(timezone.utc).isoformat(),
                "selected_validation_model": selected.name,
                "selected_ml_validation_model": selected_ml.name,
                "price_field": args.price_field,
                "top_k": args.top_k,
                "cost_bps": args.cost_bps,
                "minimum_training_rows": args.min_train_rows,
                "foundation_status": foundation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path, metrics_path


def run(args: argparse.Namespace) -> list[ModelResult]:
    if not 1 <= args.top_k <= 30:
        raise ValueError("top_k must be between 1 and 30")
    if args.cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    prices, volumes, _ = research.load_prices(args.price_field)
    prices = prices.loc[(prices.index >= args.start) & (prices.index <= args.end)]
    volumes = volumes.reindex(prices.index)
    features = research.make_features(prices, volumes)
    panel = make_panel(features)
    foundation = foundation_status()
    ridge_predictions, ridge_training = walk_forward_predictions(panel, "ridge", args.min_train_rows)
    lightgbm_error: str | None = None
    try:
        lightgbm_predictions, lightgbm_training = walk_forward_predictions(panel, "lightgbm", args.min_train_rows)
    except RuntimeError as exc:
        lightgbm_error = str(exc)
        lightgbm_predictions = pd.DataFrame(index=ridge_predictions.index, columns=ridge_predictions.columns, dtype=float)
        lightgbm_training = pd.Series(index=ridge_predictions.index, dtype=float)

    ensemble = ensemble_predictions(ridge_predictions, lightgbm_predictions)
    model_inputs = [
        ("ridge", ridge_predictions, False, "Regularized linear model; alpha fixed at 10."),
        ("lightgbm", lightgbm_predictions, False, lightgbm_error or "Fixed shallow gradient-boosted tree model."),
        ("ensemble", ensemble, False, "Equal average of Ridge and LightGBM cross-sectional percentile scores."),
        ("ensemble_volmanaged", ensemble, True, "Equal ensemble with fixed 10% inverse-volatility exposure control."),
    ]
    baseline_simulation = research.simulate("composite", features, args.top_k, args.cost_bps)
    results: list[ModelResult] = [
        ModelResult(
            name="existing_composite",
            predictions=pd.DataFrame(index=baseline_simulation.index),
            simulation=baseline_simulation,
            metrics=model_metrics(baseline_simulation, args.end),
            notes="Existing fixed rank-composite control from research.py.",
        )
    ]
    for name, predictions, vol_managed, notes in model_inputs:
        simulation = simulate_predictions(predictions, features, args.top_k, args.cost_bps, vol_managed)
        results.append(
            ModelResult(
                name=name,
                predictions=predictions,
                simulation=simulation,
                metrics=model_metrics(simulation, args.end),
                notes=notes,
            )
        )
    write_report(
        results,
        panel,
        {"ridge": ridge_training, "lightgbm": lightgbm_training},
        args,
        foundation,
    )
    return results


def main() -> None:
    args = parse_args()
    if args.foundation_status:
        print(json.dumps(foundation_status(), indent=2))
        return
    results = run(args)
    summary = pd.DataFrame(
        [
            {
                "model": result.name,
                "validation_sharpe": result.metrics.get("validation", {}).get("strategy_sharpe_rf0"),
                "validation_excess_cagr": result.metrics.get("validation", {}).get("excess_cagr"),
                "holdout_excess_cagr": result.metrics.get("holdout", {}).get("excess_cagr"),
                "holdout_sharpe": result.metrics.get("holdout", {}).get("strategy_sharpe_rf0"),
            }
            for result in results
        ]
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    print("\nWrote reports/ml_research_findings.md and reports/ml_model_metrics.csv")


if __name__ == "__main__":
    main()
