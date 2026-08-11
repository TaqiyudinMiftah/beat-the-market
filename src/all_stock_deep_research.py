"""Leakage-safe deep-learning audit on the all-listed IDX liquidity panel.

This runner applies the repository's small factor MLP to a broader current IDX
catalog than the original IDX30 experiment. It predicts cross-sectional
next-month return ranks inside a trailing-liquidity top-300 screen, using only
signal months strictly before each forecast. The MLP architecture, seeds,
early-stopping split, and causal blend rules are fixed before the holdout.

The current catalog is still not point-in-time, so this is a research audit and
not a live trading recommendation.
"""

from __future__ import annotations

import argparse
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
        deep_ml_research,
        foundation_robustness,
        liquid_rank_ml_research,
        ml_research,
        research,
    )
except ImportError:  # pragma: no cover - direct CLI compatibility
    from src import (
        all_stock_ml_research,
        deep_ml_research,
        foundation_robustness,
        liquid_rank_ml_research,
        ml_research,
        research,
    )


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "yahoo_all"
DEFAULT_UNIVERSE_PATH = ROOT / "data" / "universe_idx_all_2026-08.csv"
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-08-11"
DEFAULT_CAP = 300
DEFAULT_TOP_K = 3
DEFAULT_COST_BPS = 25.0
DEFAULT_MIN_TRAIN_MONTHS = 24
DEFAULT_INTERNAL_VALIDATION_MONTHS = 12
DEFAULT_MAX_EPOCHS = 60
DEFAULT_PATIENCE = 10
DEFAULT_SEEDS = (7, 11, 19)
DEFAULT_OUTPUT_PREFIX = "all_stock_deep"
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_BLOCK_LENGTH = 3
BOOTSTRAP_SEED = 20260812
TOP_K_SENSITIVITY = (1, 3, 5, 10)
COST_SENSITIVITY_BPS = (25.0, 100.0)


@dataclass(frozen=True)
class AllStockDeepConfig:
    price_field: str
    start: str
    end: str
    cap: int
    top_k: int
    cost_bps: float
    min_train_months: int
    internal_validation_months: int
    max_epochs: int
    patience: int
    seeds: tuple[int, ...]
    device: str
    max_signals: int
    data_dir: Path
    universe_path: Path
    output_prefix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-field", choices=("adjclose", "close"), default="adjclose")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
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
    parser.add_argument("--max-signals", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def _validate_config(config: AllStockDeepConfig) -> None:
    if config.cap < config.top_k or config.cap < 30:
        raise ValueError("cap must be at least top_k and at least 30")
    if not 1 <= config.top_k <= 30:
        raise ValueError("top_k must be between one and 30")
    if config.cost_bps < 0.0:
        raise ValueError("cost_bps must be non-negative")
    if config.min_train_months < 6 or config.internal_validation_months < 3:
        raise ValueError("training and internal validation windows are too short")
    if config.max_epochs < 1 or config.patience < 1 or not config.seeds:
        raise ValueError("MLP training settings are invalid")
    if config.max_signals < 0:
        raise ValueError("max_signals must be non-negative")
    if not config.output_prefix or Path(config.output_prefix).name != config.output_prefix:
        raise ValueError("output_prefix must be a non-empty filename prefix")


def _output_path(config: AllStockDeepConfig, suffix: str) -> Path:
    return REPORT_DIR / f"{config.output_prefix}_{suffix}"


def _metric_periods(start: str, end: str) -> tuple[tuple[str, str, str], ...]:
    return (
        ("train", start, "2021-12-31"),
        ("validation", "2022-01-01", "2023-12-31"),
        ("holdout_2024", "2024-01-01", "2024-12-31"),
        ("holdout_2025_2026", "2025-01-01", end),
        ("full", start, end),
    )


def _metrics(simulation: pd.DataFrame, start: str, end: str) -> dict[str, dict[str, Any]]:
    return {
        label: ml_research.period_metrics(simulation, period_start, period_end, label)
        for label, period_start, period_end in _metric_periods(start, end)
    }


def build_rank_target_panel(
    features: dict[str, pd.DataFrame | pd.Series],
    cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a cap-filtered panel and its point-in-time eligibility mask."""
    eligibility = liquid_rank_ml_research.eligibility_mask(features, cap)
    panel = liquid_rank_ml_research.filter_panel(
        ml_research.make_panel(features),
        eligibility,
    )
    panel["target_rank"] = panel.groupby("signal_date")["target"].rank(
        pct=True,
        method="average",
    )
    benchmark_returns = features["benchmark_returns"]
    assert isinstance(benchmark_returns, pd.Series)
    panel["target_excess"] = panel["target"] - panel["signal_date"].map(
        benchmark_returns.shift(-1)
    )
    return panel, eligibility


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
    config: AllStockDeepConfig,
    metrics: dict[str, dict[str, dict[str, Any]]],
    panels: dict[str, pd.DataFrame],
    diagnostics: pd.DataFrame,
    rolling: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    sensitivity: pd.DataFrame,
    training: pd.DataFrame,
    gates: dict[str, dict[str, Any]],
    validation_winner: str | None,
    preferred: str | None,
    loaded_tickers: list[str],
    skipped: list[str],
    eligible_counts: pd.Series,
    online_weights: dict[str, pd.Series],
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
    training.to_csv(_output_path(config, "training.csv"), index=False)

    context = pd.DataFrame(
        {
            "signal_date": eligible_counts.index,
            "eligible_tickers": eligible_counts.to_numpy(dtype=float),
        }
    )
    for name, weights in online_weights.items():
        context[f"{name}_mlp_weight"] = weights.reindex(context["signal_date"]).to_numpy()
    context.to_csv(_output_path(config, "context.csv"), index=False)

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
            _output_path(config, "forecasts.csv"),
            index=False,
        )

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: (str(value.relative_to(ROOT)) if isinstance(value, Path) else value)
            for key, value in vars(config).items()
        },
        "loaded_ticker_count": len(loaded_tickers),
        "skipped": skipped,
        "eligible_counts": context.to_dict(orient="records"),
        "metrics": metrics,
        "validation_winner": validation_winner,
        "preferred_research_candidate": preferred,
        "gates": gates,
        "bootstrap": bootstrap.to_dict(orient="records"),
        "online_mean_weights": {
            name: float(weights.mean()) for name, weights in online_weights.items()
        },
        "architecture": {
            "target": "cross-sectional next-month return percentile rank",
            "hidden_layers": [32, 16],
            "dropout": 0.10,
            "seeds": config.seeds,
            "internal_validation_months": config.internal_validation_months,
        },
    }
    _output_path(config, "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    report = f"""# All-listed deep-learning research

Run date: {date.today().isoformat()}<br>
Universe catalog: {config.universe_path.relative_to(ROOT)}; loaded tickers: {len(loaded_tickers)}; skipped: {len(skipped)}<br>
Price field: {config.price_field}; liquidity screen: top {config.cap}; top K: {config.top_k}; cost: {config.cost_bps:.1f} bps one-way<br>
Signal dates: {len(eligible_counts)}; median eligible stocks: {eligible_counts.median():.1f}<br>

## Research basis

This audit adapts the cross-sectional deep-learning setup in [Abe and Nakayama's stock-return forecasting study](https://arxiv.org/abs/1801.01777) and the model-comparison discipline of [Gu, Kelly, and Xiu](https://www.nber.org/papers/w25398). PyTorch is used only for a small factor MLP; the model is not pretrained on this IDX panel.

## Fixed design and leakage controls

- At each completed month-end t, eligibility is computed from trailing 60-day median dollar volume and only the top {config.cap} names are retained.
- The target is the next month's cross-sectional return percentile rank. The MLP sees only rows with signal dates strictly earlier than the current forecast date.
- Feature normalization, target winsorization, and early stopping use training history only. The latest {config.internal_validation_months} historical signal months form an internal validation slice.
- The network is fixed at hidden layers (32, 16), dropout 0.10, Adam learning rate 0.01, weight decay 0.01, seeds {config.seeds}, and at most {config.max_epochs} epochs.
- `baseline_mlp_blend_50` is a fixed 50/50 percentile-rank blend. `online_best` and `online_soft` use only realized strategy returns before t; `online_soft` uses the fixed 0.25/0.50/0.75 weights around a 0.25 trailing-Sharpe margin.
- Validation is 2022–2023. 2024 and 2025–2026 are holdouts; no holdout result chooses architecture, seed, or blend weight.

## Audit summary

The fixed gate requires beating the matching top-{config.cap} composite control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025–2026.

{_summary_table(metrics, rolling_summary, bootstrap, gates)}

Validation winner: {validation_winner or "none"}. Preferred after the fixed gate: {preferred or "none"}. Any passing row remains exploratory because the catalog is a current snapshot with survivorship and historical-membership bias.

## Limitations

- The current all-listed catalog does not reconstruct delisted names, historical suspensions, or point-in-time index membership.
- Yahoo Finance data omit or simplify spreads, taxes, price limits, market impact, failed fills, and capacity.
- Deep-learning results are sensitive to the universe, price field, and training window; this runner must be stress-tested before any deployment consideration.
- This is research, not investment advice, and no backtest guarantees future performance.

## Reproduction

~~~bash
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.all_stock_deep_research
~~~

Raw forecasts remain ignored in reports/{config.output_prefix}_forecasts.csv. Committed tables use the {config.output_prefix}_ prefix.
"""
    _output_path(config, "findings.md").write_text(report, encoding="utf-8")


def run(config: AllStockDeepConfig) -> dict[str, Any]:
    _validate_config(config)
    prices, volumes, loaded_tickers, skipped = all_stock_ml_research.load_prices(
        config.data_dir,
        config.universe_path,
        config.price_field,
    )
    prices = prices.loc[prices.index <= config.end]
    features = all_stock_ml_research._slice_features(
        research.make_features(prices, volumes.reindex(prices.index)),
        config.start,
        config.end,
    )
    panel, eligibility = build_rank_target_panel(features, config.cap)
    signal_dates = pd.DatetimeIndex(sorted(panel["signal_date"].unique()))
    if config.max_signals:
        signal_dates = signal_dates[: config.max_signals]
        panel = panel.loc[panel["signal_date"].isin(signal_dates)].copy()

    baseline = research.score_frame("composite", features).where(eligibility)
    baseline = baseline.reindex(index=signal_dates)
    deep_config = deep_ml_research.DeepConfig(
        price_field=config.price_field,
        start=config.start,
        end=config.end,
        top_k=config.top_k,
        cost_bps=config.cost_bps,
        min_train_months=config.min_train_months,
        internal_validation_months=config.internal_validation_months,
        max_epochs=config.max_epochs,
        patience=config.patience,
        seeds=config.seeds,
        hidden_layers=(32, 16),
        dropout=0.10,
        device=config.device,
    )
    rank_predictions, rank_training, epochs = deep_ml_research.walk_forward_mlp(
        panel,
        "target_rank",
        deep_config,
    )
    rank_predictions = rank_predictions.reindex(index=signal_dates)
    blend = deep_ml_research._weighted_rank_panel(baseline, rank_predictions, 0.50)
    online_panels: dict[str, pd.DataFrame] = {}
    online_weights: dict[str, pd.Series] = {}
    for rule in ("best", "soft"):
        predictions, weights = deep_ml_research.online_weighted_predictions(
            baseline,
            rank_predictions,
            features,
            config.top_k,
            config.cost_bps,
            rule,
        )
        online_panels[f"online_{rule}"] = predictions
        online_weights[f"online_{rule}"] = weights

    panels: dict[str, pd.DataFrame] = {
        "cap300_composite": baseline,
        "mlp_rank": rank_predictions,
        "baseline_mlp_blend_50": blend,
        **online_panels,
    }
    simulations = {
        name: ml_research.simulate_predictions(panel_value, features, config.top_k, config.cost_bps)
        for name, panel_value in panels.items()
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
    bootstrap = _bootstrap_rows(simulations, config.end, "cap300_composite")

    candidates = [name for name in panels if name != "cap300_composite"]
    gates: dict[str, dict[str, Any]] = {}
    for name in candidates:
        passed, reasons = liquid_rank_ml_research._gate_candidate(
            name,
            metrics[name],
            metrics["cap300_composite"],
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
            for name, panel_value in panels.items():
                simulation = ml_research.simulate_predictions(
                    panel_value,
                    features,
                    top_k,
                    cost_bps,
                )
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
                "model": "mlp_rank",
                "signals": int(rank_predictions.notna().any(axis=1).sum()),
                "forecast_rows": int(rank_predictions.notna().sum().sum()),
                "median_training_rows": float(rank_training.dropna().median())
                if not rank_training.dropna().empty
                else np.nan,
                "median_best_epoch": float(np.median(epochs)) if epochs else np.nan,
            }
        ]
    )
    eligible_counts = eligibility.reindex(index=signal_dates).sum(axis=1)
    _write_outputs(
        config=config,
        metrics=metrics,
        panels=panels,
        diagnostics=diagnostics,
        rolling=rolling,
        rolling_summary=rolling_summary,
        bootstrap=bootstrap,
        sensitivity=pd.DataFrame(sensitivity_rows),
        training=training,
        gates=gates,
        validation_winner=validation_winner,
        preferred=preferred,
        loaded_tickers=loaded_tickers,
        skipped=skipped,
        eligible_counts=eligible_counts,
        online_weights=online_weights,
    )
    return {
        "metrics": metrics,
        "gates": gates,
        "validation_winner": validation_winner,
        "preferred": preferred,
    }


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    config = AllStockDeepConfig(
        price_field=args.price_field,
        start=args.start,
        end=args.end,
        cap=args.cap,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        min_train_months=args.min_train_months,
        internal_validation_months=args.internal_validation_months,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seeds=seeds,
        device=args.device,
        max_signals=args.max_signals,
        data_dir=args.data_dir.resolve(),
        universe_path=args.universe.resolve(),
        output_prefix=args.output_prefix,
    )
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
        f"preferred: {result['preferred']}"
    )
    print(f"Wrote reports/{args.output_prefix}_findings.md")


if __name__ == "__main__":
    main()
