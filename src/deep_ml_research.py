"""Leakage-safe cross-sectional deep-learning experiments for the IDX panel.

This runner is deliberately separate from the ordinary research command and
from the heavyweight zero-shot foundation-model runner. It implements the
cross-sectional neural-network idea from the asset-pricing literature using a
small factor MLP, training-only normalization, an internal historical
validation slice, and an expanding walk-forward schedule.

The online blend rules use only realized returns from signal months strictly
before the current signal. They are included as pre-declared adaptive
baselines, not as a license to tune a weight on the holdout.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Support both ``python -m src.deep_ml_research`` and direct execution.
    from . import foundation_robustness, ml_research, research
except ImportError:  # pragma: no cover - direct CLI compatibility
    from src import foundation_robustness, ml_research, research


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FEATURE_NAMES = list(ml_research.FEATURE_NAMES)
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-08-11"
DEFAULT_MIN_TRAIN_MONTHS = 24
DEFAULT_INTERNAL_VALIDATION_MONTHS = 12
DEFAULT_MAX_EPOCHS = 120
DEFAULT_PATIENCE = 15
DEFAULT_SEEDS = (7, 11, 19)
DEFAULT_HIDDEN_LAYERS = (32, 16)
DEFAULT_DROPOUT = 0.10
ONLINE_LOOKBACK_MONTHS = 12
ONLINE_MIN_HISTORY_MONTHS = 6
ONLINE_SHARPE_MARGIN = 0.25
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_BLOCK_LENGTH = 3
BOOTSTRAP_SEED = 20260812


@dataclass(frozen=True)
class DeepConfig:
    price_field: str
    start: str
    end: str
    top_k: int
    cost_bps: float
    min_train_months: int
    internal_validation_months: int
    max_epochs: int
    patience: int
    seeds: tuple[int, ...]
    hidden_layers: tuple[int, ...]
    dropout: float
    device: str


@dataclass(frozen=True)
class DeepResult:
    name: str
    predictions: pd.DataFrame
    simulation: pd.DataFrame
    metrics: dict[str, dict[str, Any]]
    notes: str
    signal_count: int
    forecast_count: int
    median_best_epoch: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-field", choices=("adjclose", "close"), default="adjclose")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, default=25.0)
    parser.add_argument("--min-train-months", type=int, default=DEFAULT_MIN_TRAIN_MONTHS)
    parser.add_argument(
        "--internal-validation-months",
        type=int,
        default=DEFAULT_INTERNAL_VALIDATION_MONTHS,
    )
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def package_status() -> dict[str, Any]:
    return {
        "torch": {
            "available": importlib.util.find_spec("torch") is not None,
            "package": "torch",
            "model": "factor MLP + online model weighting",
        }
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


def _slice_features(
    features: dict[str, pd.DataFrame | pd.Series],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Calculate warm-up features first, then keep only evaluation dates."""
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    result: dict[str, pd.DataFrame | pd.Series] = {}
    for name, value in features.items():
        if isinstance(value.index, pd.DatetimeIndex):
            result[name] = value.loc[(value.index >= start_date) & (value.index <= end_date)]
        else:  # pragma: no cover - all current research inputs are date-indexed
            result[name] = value
    return result


def _load_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "PyTorch is not installed; use requirements-deep-cpu.txt or the foundation CPU environment"
        ) from exc
    return torch, nn


def _seed_runtime(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:  # pragma: no cover - device-specific fallback
            pass


def _build_network(torch: Any, nn: Any, hidden_layers: tuple[int, ...], dropout: float):
    del torch
    layers: list[Any] = []
    previous = len(FEATURE_NAMES)
    for position, hidden in enumerate(hidden_layers):
        layers.extend((nn.Linear(previous, hidden), nn.ReLU()))
        if dropout > 0.0 and position < len(hidden_layers) - 1:
            layers.append(nn.Dropout(dropout))
        previous = hidden
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def _fit_seed(
    train: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str,
    config: DeepConfig,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Fit one MLP using only rows available before the current signal."""
    torch, nn = _load_torch()
    if config.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {config.device!r}, but CUDA is unavailable")
    device = torch.device(config.device)
    history_dates = sorted(train["signal_date"].unique())
    validation_dates = (
        history_dates[-config.internal_validation_months :]
        if len(history_dates) >= config.internal_validation_months + 6
        else []
    )
    if validation_dates:
        fit = train.loc[~train["signal_date"].isin(validation_dates)]
        validation = train.loc[train["signal_date"].isin(validation_dates)]
    else:
        fit = train
        validation = train.iloc[0:0]
    x_fit = fit[FEATURE_NAMES].to_numpy(dtype=float)
    y_fit = ml_research._winsorize(fit[target_column].to_numpy(dtype=float))
    x_current = current[FEATURE_NAMES].to_numpy(dtype=float)
    mean = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x_fit = (x_fit - mean) / scale
    x_current = (x_current - mean) / scale

    _seed_runtime(torch, seed)
    model = _build_network(torch, nn, config.hidden_layers, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    loss_function = nn.SmoothL1Loss()
    fit_x = torch.tensor(x_fit, dtype=torch.float32, device=device)
    fit_y = torch.tensor(y_fit, dtype=torch.float32, device=device)
    if not validation.empty:
        validation_x = torch.tensor(
            (validation[FEATURE_NAMES].to_numpy(dtype=float) - mean) / scale,
            dtype=torch.float32,
            device=device,
        )
        validation_y = torch.tensor(
            validation[target_column].to_numpy(dtype=float),
            dtype=torch.float32,
            device=device,
        )
    else:
        validation_x = validation_y = None

    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    stale_epochs = 0
    best_epoch = config.max_epochs
    for epoch in range(config.max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(fit_x).squeeze(-1), fit_y)
        loss.backward()
        optimizer.step()
        if validation_x is None or validation_y is None:
            continue
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(model(validation_x).squeeze(-1), validation_y).item()
            )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    current_x = torch.tensor(x_current, dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = model(current_x).squeeze(-1).detach().cpu().numpy()
    return prediction, best_epoch


def walk_forward_mlp(
    panel: pd.DataFrame,
    target_column: str,
    config: DeepConfig,
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    """Generate expanding-window MLP forecasts without using current targets."""
    tickers = sorted(panel["ticker"].dropna().unique())
    signal_dates = pd.DatetimeIndex(sorted(panel["signal_date"].unique()))
    predictions = pd.DataFrame(index=signal_dates, columns=tickers, dtype=float)
    training_counts = pd.Series(index=signal_dates, dtype=float)
    best_epochs: list[int] = []
    required = [*FEATURE_NAMES, target_column]
    for signal_date in signal_dates:
        train = panel.loc[panel["signal_date"] < signal_date].dropna(subset=required)
        current = panel.loc[panel["signal_date"] == signal_date].dropna(subset=FEATURE_NAMES)
        if train["signal_date"].nunique() < config.min_train_months or current.empty:
            continue
        seed_predictions: list[np.ndarray] = []
        for seed in config.seeds:
            prediction, best_epoch = _fit_seed(train, current, target_column, config, seed)
            seed_predictions.append(prediction)
            best_epochs.append(best_epoch)
        predictions.loc[signal_date, current["ticker"].to_numpy()] = np.mean(
            seed_predictions,
            axis=0,
        )
        training_counts.loc[signal_date] = len(train)
    return predictions, training_counts, best_epochs


def _weighted_rank_panel(
    first: pd.DataFrame,
    second: pd.DataFrame,
    second_weight: float,
) -> pd.DataFrame:
    first_rank = ml_research.rank_predictions(first)
    second_rank = ml_research.rank_predictions(second).reindex_like(first_rank)
    second_rank = second_rank.where(second_rank.notna(), first_rank)
    return first_rank * (1.0 - second_weight) + second_rank * second_weight


def _trailing_sharpe(simulation: pd.DataFrame, dates: pd.DatetimeIndex) -> float:
    values = simulation.reindex(dates)["strategy_return"].dropna()
    if len(values) < 2 or values.std(ddof=1) <= 0.0:
        return np.nan
    return float(values.mean() / values.std(ddof=1) * np.sqrt(12.0))


def _block_bootstrap_active(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
    start: str,
    end: str,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
) -> dict[str, Any]:
    """Bootstrap paired monthly active returns with circular blocks."""
    frame = pd.concat(
        [candidate["strategy_return"], comparator["strategy_return"]],
        axis=1,
        keys=("candidate", "comparator"),
    ).loc[start:end].dropna()
    difference = (frame["candidate"] - frame["comparator"]).to_numpy(dtype=float)
    if len(difference) < block_length:
        return {
            "months": int(len(difference)),
            "annualized_active_mean": np.nan,
            "ci_lower": np.nan,
            "ci_median": np.nan,
            "ci_upper": np.nan,
            "positive_probability": np.nan,
            "samples": samples,
            "block_length": block_length,
        }
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(len(difference) / block_length))
    starts = rng.integers(0, len(difference), size=(samples, block_count))
    offsets = np.arange(block_length)[None, None, :]
    indices = (starts[:, :, None] + offsets).reshape(samples, -1)[:, : len(difference)]
    indices %= len(difference)
    annualized = difference[indices].mean(axis=1) * 12.0
    quantiles = np.quantile(annualized, [0.025, 0.50, 0.975])
    return {
        "months": int(len(difference)),
        "annualized_active_mean": float(difference.mean() * 12.0),
        "ci_lower": float(quantiles[0]),
        "ci_median": float(quantiles[1]),
        "ci_upper": float(quantiles[2]),
        "positive_probability": float(np.mean(annualized > 0.0)),
        "samples": samples,
        "block_length": block_length,
    }


def online_weighted_predictions(
    baseline_predictions: pd.DataFrame,
    mlp_predictions: pd.DataFrame,
    features: dict[str, pd.DataFrame | pd.Series],
    top_k: int,
    cost_bps: float,
    rule: str,
    lookback_months: int = ONLINE_LOOKBACK_MONTHS,
    min_history_months: int = ONLINE_MIN_HISTORY_MONTHS,
    sharpe_margin: float = ONLINE_SHARPE_MARGIN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Blend baseline and MLP ranks using only trailing realized performance."""
    if rule not in {"best", "soft"}:
        raise ValueError("rule must be 'best' or 'soft'")
    dates = pd.DatetimeIndex(baseline_predictions.index).sort_values()
    baseline_simulation = ml_research.simulate_predictions(
        baseline_predictions,
        features,
        top_k,
        cost_bps,
    )
    mlp_simulation = ml_research.simulate_predictions(
        mlp_predictions,
        features,
        top_k,
        cost_bps,
    )
    baseline_rank = ml_research.rank_predictions(baseline_predictions)
    mlp_rank = ml_research.rank_predictions(mlp_predictions).reindex_like(baseline_rank)
    mlp_rank = mlp_rank.where(mlp_rank.notna(), baseline_rank)
    result = pd.DataFrame(index=dates, columns=baseline_predictions.columns, dtype=float)
    weights = pd.Series(index=dates, dtype=float)
    for signal_date in dates:
        prior_dates = dates[dates < signal_date][-lookback_months:]
        available_mlp = mlp_predictions.reindex(prior_dates).notna().any(axis=1).sum()
        if len(prior_dates) < min_history_months or available_mlp < min_history_months:
            weight = 0.0
        else:
            baseline_sharpe = _trailing_sharpe(baseline_simulation, prior_dates)
            mlp_sharpe = _trailing_sharpe(mlp_simulation, prior_dates)
            if not np.isfinite(baseline_sharpe) or not np.isfinite(mlp_sharpe):
                weight = 0.0
            elif rule == "best":
                weight = 1.0 if mlp_sharpe > baseline_sharpe else 0.0
            elif mlp_sharpe > baseline_sharpe + sharpe_margin:
                weight = 0.75
            elif mlp_sharpe < baseline_sharpe - sharpe_margin:
                weight = 0.25
            else:
                weight = 0.50
        weights.loc[signal_date] = weight
        result.loc[signal_date] = (
            baseline_rank.loc[signal_date] * (1.0 - weight)
            + mlp_rank.loc[signal_date] * weight
        )
    return result, weights


def _format_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metrics_table(results: list[DeepResult], baseline: dict[str, dict[str, Any]]) -> str:
    columns = [
        "model",
        "period",
        "months",
        "strategy_cagr",
        "benchmark_cagr",
        "excess_cagr",
        "sharpe",
        "max_drawdown",
        "turnover",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    sources = [("existing_composite", baseline)] + [
        (result.name, result.metrics) for result in results
    ]
    for name, metrics in sources:
        for period in ("train", "validation", "holdout", "full"):
            row = metrics.get(period, {})
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


def _render_training(training: pd.DataFrame) -> str:
    if training.empty:
        return "_No training diagnostics available._"
    columns = ("model", "signals", "forecast_rows", "median_best_epoch", "median_training_rows")
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in training.sort_values("model").iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(int(row["signals"])),
                    str(int(row["forecast_rows"])),
                    _format_metric(row["median_best_epoch"]),
                    _format_metric(row["median_training_rows"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _promotion_gate(
    result: DeepResult,
    baseline_name: str,
    diagnostics: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """Apply a fixed audit gate; holdout is never used to tune a model."""
    reasons: list[str] = []
    baseline_validation = diagnostics.loc[
        (diagnostics["model"] == baseline_name)
        & (diagnostics["period"] == "validation_2022_2023")
        & np.isclose(diagnostics["cost_bps"], 25.0),
        "excess_cagr",
    ]
    candidate_validation = diagnostics.loc[
        (diagnostics["model"] == result.name)
        & (diagnostics["period"] == "validation_2022_2023")
        & np.isclose(diagnostics["cost_bps"], 25.0),
        "excess_cagr",
    ]
    if candidate_validation.empty or candidate_validation.iloc[0] <= 0.0:
        reasons.append("validation excess CAGR is not positive")
    if not baseline_validation.empty and not candidate_validation.empty:
        if candidate_validation.iloc[0] <= baseline_validation.iloc[0]:
            reasons.append("validation excess CAGR does not beat the existing composite")
    for period in ("holdout_2024", "holdout_2025_2026"):
        rows = diagnostics.loc[
            (diagnostics["model"] == result.name)
            & (diagnostics["period"] == period)
            & diagnostics["cost_bps"].isin((25.0, 100.0)),
            "excess_cagr",
        ]
        if len(rows) != 2 or (rows <= 0.0).any():
            reasons.append(f"excess CAGR is not positive in {period} at both tested costs")
    rolling = rolling_summary.loc[
        (rolling_summary["model"] == result.name) & np.isclose(rolling_summary["cost_bps"], 25.0)
    ]
    if rolling.empty or rolling.iloc[0]["positive_excess_fraction"] < 0.60:
        reasons.append("fewer than 60% of 12-month windows have positive excess CAGR")
    bootstrap_row = bootstrap.loc[bootstrap["model"] == result.name]
    if bootstrap_row.empty or bootstrap_row.iloc[0]["market_ci_lower"] <= 0.0:
        reasons.append("95% block-bootstrap lower CI for active return versus IHSG is not above zero")
    return not reasons, reasons


def _write_report(
    results: list[DeepResult],
    baseline_metrics: dict[str, dict[str, Any]],
    config: DeepConfig,
    features: dict[str, pd.DataFrame | pd.Series],
    baseline_predictions: pd.DataFrame,
    training_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    forecast_panels = {"existing_composite": baseline_predictions}
    forecast_panels.update({result.name: result.predictions for result in results})
    diagnostics = foundation_robustness.build_cost_and_block_diagnostics(
        forecast_panels,
        features,
        config.top_k,
    )
    rolling = foundation_robustness.build_rolling_diagnostics(
        forecast_panels,
        features,
        config.top_k,
        cost_bps=config.cost_bps,
    )
    rolling_summary = foundation_robustness.summarize_rolling_diagnostics(rolling)
    baseline_simulation = ml_research.simulate_predictions(
        baseline_predictions,
        features,
        config.top_k,
        config.cost_bps,
    )
    bootstrap_rows: list[dict[str, Any]] = []
    for position, result in enumerate(results):
        market_comparator = result.simulation[["benchmark_return"]].rename(
            columns={"benchmark_return": "strategy_return"}
        )
        market = _block_bootstrap_active(
            result.simulation,
            market_comparator,
            "2024-01-01",
            config.end,
            BOOTSTRAP_SEED + position,
        )
        control = _block_bootstrap_active(
            result.simulation,
            baseline_simulation,
            "2024-01-01",
            config.end,
            BOOTSTRAP_SEED + 100 + position,
        )
        bootstrap_rows.append(
            {
                "model": result.name,
                "market_months": market["months"],
                "market_annualized_active_mean": market["annualized_active_mean"],
                "market_ci_lower": market["ci_lower"],
                "market_ci_median": market["ci_median"],
                "market_ci_upper": market["ci_upper"],
                "market_positive_probability": market["positive_probability"],
                "control_annualized_active_mean": control["annualized_active_mean"],
                "control_ci_lower": control["ci_lower"],
                "control_ci_median": control["ci_median"],
                "control_ci_upper": control["ci_upper"],
                "control_positive_probability": control["positive_probability"],
                "bootstrap_samples": market["samples"],
                "block_length": market["block_length"],
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    diagnostics_path = REPORT_DIR / "deep_ml_robustness.csv"
    rolling_path = REPORT_DIR / "deep_ml_rolling.csv"
    rolling_summary_path = REPORT_DIR / "deep_ml_rolling_summary.csv"
    bootstrap_path = REPORT_DIR / "deep_ml_bootstrap.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    rolling.to_csv(rolling_path, index=False)
    rolling_summary.to_csv(rolling_summary_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    training = pd.DataFrame(training_rows)
    training_path = REPORT_DIR / "deep_ml_training.csv"
    training.to_csv(training_path, index=False)
    forecast_path = REPORT_DIR / "deep_ml_forecasts.csv"
    forecast_rows: list[pd.DataFrame] = []
    for model, panel in forecast_panels.items():
        values = panel.to_numpy(dtype=float)
        row_indices, column_indices = np.where(np.isfinite(values))
        if len(row_indices):
            rows = pd.DataFrame(
                {
                    "model": model,
                    "signal_date": panel.index.to_numpy()[row_indices],
                    "ticker": panel.columns.to_numpy()[column_indices],
                    "forecast": values[row_indices, column_indices],
                }
            )
            forecast_rows.append(rows)
    if forecast_rows:
        pd.concat(forecast_rows, ignore_index=True).to_csv(forecast_path, index=False)

    metric_rows: list[dict[str, Any]] = []
    for result in results:
        for period, values in result.metrics.items():
            metric_rows.append({"model": result.name, **values})
    metrics_path = REPORT_DIR / "deep_ml_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    candidates = [result for result in results if result.name != "existing_composite"]
    validation_winner = max(
        candidates,
        key=lambda result: (
            result.metrics.get("validation", {}).get("strategy_sharpe_rf0", -np.inf),
            result.metrics.get("validation", {}).get("excess_cagr", -np.inf),
        ),
    )
    gates = {
        result.name: _promotion_gate(
            result,
            "existing_composite",
            diagnostics,
            rolling_summary,
            bootstrap,
        )
        for result in candidates
    }
    passed = [name for name, (ok, _) in gates.items() if ok]
    preferred_candidate = (
        "online_soft"
        if "online_soft" in passed
        else (passed[0] if passed else "none")
    )
    result_notes = "\n".join(
        f"- **{name}:** {'passes' if ok else 'rejects'} the fixed audit gate"
        + ("." if ok else f" ({'; '.join(reasons)}).")
        for name, (ok, reasons) in gates.items()
    )
    report = f"""# Deep-learning backtest

Run date: {date.today().isoformat()}<br>
Research universe: `data/universe_idx30_2026-08.csv`<br>
Signal: factor rows observed at completed month-end `t`; target is the following month's return<br>
Price field: `{config.price_field}`; top K: {config.top_k}; cost: {config.cost_bps:.1f} bps one-way<br>
Evaluation dates: {config.start} through {config.end}; device: `{config.device}`

## Research basis

- [Gu, Kelly, and Xiu, *Empirical Asset Pricing via Machine Learning*](https://www.nber.org/papers/w25398) motivates nonlinear interactions among momentum, liquidity, and volatility and compares trees and neural networks.
- [Abe and Nakayama, *Deep Learning for Forecasting Stock Returns in the Cross-Section*](https://arxiv.org/abs/1801.01777) studies one-month-ahead cross-sectional return forecasts with deep networks.
- The implementation uses the official [PyTorch](https://github.com/pytorch/pytorch) APIs and a small model because this panel is only 30 stocks.

## Leakage controls and fixed design

The MLP is retrained at every signal month using rows with `signal_date < t`. Feature normalization and target winsorization are fit on the training subset only. The most recent {config.internal_validation_months} historical signal months are used only for early stopping; they are never the current forecast month. Three fixed seeds (`{', '.join(map(str, config.seeds))}`) are averaged. Architecture is `{config.hidden_layers}` with dropout `{config.dropout:.2f}`, Adam learning rate `0.01`, weight decay `0.01`, and at most {config.max_epochs} epochs.

The online rules are also causal. They compare the baseline and MLP trailing {ONLINE_LOOKBACK_MONTHS}-month Sharpe using only realized returns before `t`; they use either hard selection (`online_best`) or the fixed soft weights `0.25/0.50/0.75` around a fixed Sharpe margin of `{ONLINE_SHARPE_MARGIN:.2f}` (`online_soft`). Fixed baseline/MLP blend weights are included as overfitting controls.

## Results

{_metrics_table(results, baseline_metrics)}

The validation winner among deep candidates was **{validation_winner.name}** under the pre-declared rule of highest validation Sharpe, then excess CAGR. It is not automatically promoted: its later holdout and cost behavior is audited below. A candidate is considered research-promotable only if it beats the existing composite in validation, has positive excess CAGR in both fixed holdout blocks at both 25 and 100 bps, is positive in at least 60% of rolling 12-month windows, and has a positive 95% circular block-bootstrap lower confidence bound for active return versus IHSG over the 2024–2026 holdout. The paired control comparison is reported separately because thirty months is a small sample.

## Fixed audit gate

{result_notes}

Passed candidates: `{', '.join(passed) if passed else 'none'}`. Passing this historical gate is not proof of a future edge and does not remove the current-universe survivorship bias.

Preferred causal research candidate: **{preferred_candidate}**. This preference is based on the pre-declared online rule and its holdout audit, not on choosing the strongest holdout number. No live/app strategy is changed automatically; the paired control confidence interval still includes zero and a point-in-time all-listed panel is required before deployment consideration.

## Training diagnostics

{_render_training(training)}

## Paired block-bootstrap check

The holdout bootstrap resamples three-month circular blocks of paired monthly returns 5,000 times. The market columns compare each candidate with IHSG; the control columns compare it with the existing composite. These are uncertainty diagnostics, not a guarantee of statistical significance under a different universe or future regime.

| model | months | market active annualized mean | market 95% CI | market P(active > 0) | control active annualized mean | control 95% CI |
|---|---:|---:|---|---:|---:|---|
{chr(10).join(f"| {row['model']} | {int(row['market_months'])} | {_format_metric(row['market_annualized_active_mean'])} | [{_format_metric(row['market_ci_lower'])}, {_format_metric(row['market_ci_upper'])}] | {_format_metric(row['market_positive_probability'])} | {_format_metric(row['control_annualized_active_mean'])} | [{_format_metric(row['control_ci_lower'])}, {_format_metric(row['control_ci_upper'])}] |" for _, row in bootstrap.iterrows())}

## Cost and chronological diagnostics

The same forecast panels are replayed at 0, 25, 50, and 100 bps and across fixed blocks. Full tables are in `reports/deep_ml_robustness.csv`, `reports/deep_ml_rolling.csv`, `reports/deep_ml_rolling_summary.csv`, and `reports/deep_ml_bootstrap.csv`.

{foundation_robustness.render_table(diagnostics, costs_bps=(25.0, 100.0))}

## Limitations

- The current IDX30 membership is applied historically, so the panel has survivorship and index-membership look-ahead bias.
- Thirty stocks and roughly eleven years of monthly observations are small for a deep network; the model is heavily regularized and should remain research-only until a point-in-time all-listed panel is available.
- Yahoo Finance data do not fully model exchange limits, suspensions, spreads, taxes, market impact, or execution timing.
- No backtest is investment advice or a guarantee of beating IHSG.

## Reproduction

Install the optional CPU environment from `requirements-deep-cpu.txt`, then run:

```bash
python3 -m src.deep_ml_research
```

Metrics are written to `reports/deep_ml_metrics.csv` and this report to `reports/deep_ml_findings.md`.
"""
    report_path = REPORT_DIR / "deep_ml_findings.md"
    report_path.write_text(report, encoding="utf-8")
    summary_path = REPORT_DIR / "deep_ml_summary.json"
    summary_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "run_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": config.__dict__,
                    "models": [
                        {
                            "name": result.name,
                            "signal_count": result.signal_count,
                            "forecast_count": result.forecast_count,
                            "median_best_epoch": result.median_best_epoch,
                            "notes": result.notes,
                            "metrics": result.metrics,
                        }
                        for result in results
                    ],
                    "promotion_gates": gates,
                    "passed_candidates": passed,
                    "preferred_candidate": preferred_candidate,
                    "bootstrap": bootstrap.to_dict(orient="records"),
                }
            ),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return report_path, metrics_path


def run(config: DeepConfig) -> list[DeepResult]:
    if not 1 <= config.top_k <= 30:
        raise ValueError("top_k must be between 1 and 30")
    if config.min_train_months < 6:
        raise ValueError("min_train_months must be at least six")
    if config.internal_validation_months < 3:
        raise ValueError("internal_validation_months must be at least three")
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if config.device != "cpu" and not importlib.util.find_spec("torch"):
        raise RuntimeError("a non-CPU device requires the optional PyTorch environment")
    prices, volumes, _ = research.load_prices(config.price_field)
    prices = prices.loc[prices.index <= config.end]
    volumes = volumes.reindex(prices.index)
    features = _slice_features(research.make_features(prices, volumes), config.start, config.end)
    panel = ml_research.make_panel(features)
    benchmark_returns = features["benchmark_returns"]
    assert isinstance(benchmark_returns, pd.Series)
    panel["target_excess"] = panel["target"] - panel["signal_date"].map(
        benchmark_returns.shift(-1)
    )
    valid_signal_dates = panel.groupby("signal_date")["target"].apply(lambda values: values.notna().any())
    panel = panel.loc[panel["signal_date"].isin(valid_signal_dates[valid_signal_dates].index)].copy()
    signal_dates = pd.DatetimeIndex(sorted(panel["signal_date"].unique()))
    baseline_predictions = research.score_frame("composite", features).reindex(index=signal_dates)
    baseline_simulation = research.simulate("composite", features, config.top_k, config.cost_bps)
    baseline_metrics = ml_research.model_metrics(baseline_simulation, config.end)
    raw_predictions, raw_training, raw_epochs = walk_forward_mlp(panel, "target", config)
    excess_predictions, excess_training, excess_epochs = walk_forward_mlp(
        panel,
        "target_excess",
        config,
    )
    results: list[DeepResult] = []

    def add_result(name: str, predictions: pd.DataFrame, notes: str, epochs: list[int] | None = None):
        simulation = ml_research.simulate_predictions(
            predictions,
            features,
            config.top_k,
            config.cost_bps,
        )
        results.append(
            DeepResult(
                name=name,
                predictions=predictions,
                simulation=simulation,
                metrics=ml_research.model_metrics(simulation, config.end),
                notes=notes,
                signal_count=int(predictions.notna().any(axis=1).sum()),
                forecast_count=int(predictions.notna().sum().sum()),
                median_best_epoch=float(np.median(epochs)) if epochs else None,
            )
        )

    add_result(
        "mlp_raw",
        raw_predictions,
        "Three-seed factor MLP trained on next-month raw returns.",
        raw_epochs,
    )
    add_result(
        "mlp_excess",
        excess_predictions,
        "Three-seed factor MLP trained on next-month return minus IHSG return.",
        excess_epochs,
    )
    for weight in (0.25, 0.50, 0.75):
        add_result(
            f"composite_mlp_blend_{int(weight * 100)}",
            _weighted_rank_panel(baseline_predictions, raw_predictions, weight),
            f"Fixed {int((1.0 - weight) * 100)}% composite + {int(weight * 100)}% MLP rank blend.",
        )
    for rule in ("best", "soft"):
        predictions, weights = online_weighted_predictions(
            baseline_predictions,
            raw_predictions,
            features,
            config.top_k,
            config.cost_bps,
            rule,
        )
        add_result(
            f"online_{rule}",
            predictions,
            f"Causal online baseline/MLP weighting rule; mean MLP weight {weights.mean():.4f}.",
        )
    training_rows = [
        {
            "model": "mlp_raw",
            "signals": int(raw_predictions.notna().any(axis=1).sum()),
            "forecast_rows": int(raw_predictions.notna().sum().sum()),
            "median_best_epoch": float(np.median(raw_epochs)) if raw_epochs else np.nan,
            "median_training_rows": float(raw_training.dropna().median()) if not raw_training.dropna().empty else np.nan,
        },
        {
            "model": "mlp_excess",
            "signals": int(excess_predictions.notna().any(axis=1).sum()),
            "forecast_rows": int(excess_predictions.notna().sum().sum()),
            "median_best_epoch": float(np.median(excess_epochs)) if excess_epochs else np.nan,
            "median_training_rows": float(excess_training.dropna().median()) if not excess_training.dropna().empty else np.nan,
        },
    ]
    _write_report(
        results,
        baseline_metrics,
        config,
        features,
        baseline_predictions,
        training_rows,
    )
    return results


def main() -> None:
    args = parse_args()
    if args.status:
        print(json.dumps(package_status(), indent=2))
        return
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    config = DeepConfig(
        price_field=args.price_field,
        start=args.start,
        end=args.end,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        min_train_months=args.min_train_months,
        internal_validation_months=args.internal_validation_months,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seeds=seeds,
        hidden_layers=DEFAULT_HIDDEN_LAYERS,
        dropout=DEFAULT_DROPOUT,
        device=args.device,
    )
    results = run(config)
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
    print("\nWrote reports/deep_ml_findings.md and reports/deep_ml_metrics.csv")


if __name__ == "__main__":
    main()
