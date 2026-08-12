"""Liquidity-filtered, rank-target ML research on the all-listed IDX panel.

The current IDX stock catalog is not a point-in-time database. This runner
therefore tests a conservative proxy for the exchange's liquidity screen:
at each completed month-end, only stocks with usable trailing 60-day median
dollar volume are eligible, and fixed caps of 150, 300, 500, and all eligible
names are replayed. Models are trained on next-month cross-sectional return
ranks within the eligible universe as a robustness check against extreme
micro-cap return magnitudes.

This is an audit experiment, not a claim that the present-day catalog can
stand in for historical delistings, suspensions, or free-float membership.
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
    from . import all_stock_ml_research, foundation_robustness, ml_research, research
except ImportError:  # pragma: no cover - direct CLI compatibility
    from src import all_stock_ml_research, foundation_robustness, ml_research, research


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "yahoo_all"
DEFAULT_UNIVERSE_PATH = ROOT / "data" / "universe_idx_all_2026-08.csv"
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-08-11"
DEFAULT_TOP_K = 3
DEFAULT_COST_BPS = 25.0
DEFAULT_MIN_TRAIN_ROWS = 1_000
DEFAULT_ALPHA = 10.0
DEFAULT_CAPS = (150, 300, 500, 0)
TOP_K_SENSITIVITY = (1, 3, 5, 10)
COST_SENSITIVITY_BPS = (25.0, 100.0)
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_BLOCK_LENGTH = 3
BOOTSTRAP_SEED = 20260812
RISK_BETA_LIMIT = 1.50
RISK_DRAWDOWN_LIMIT = -0.60
RISK_TURNOVER_LIMIT = 0.90

@dataclass(frozen=True)
class LiquidConfig:
    price_field: str
    start: str
    end: str
    top_k: int
    cost_bps: float
    min_train_rows: int
    alpha: float
    caps: tuple[int, ...]
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
    parser.add_argument(
        "--caps",
        default=",".join(map(str, DEFAULT_CAPS)),
        help="Comma-separated liquidity caps; use 0 for all eligible names",
    )
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
        "lightgbm": {
            "available": importlib.util.find_spec("lightgbm") is not None,
            "version": _version("lightgbm"),
        },
        "universe": str(DEFAULT_UNIVERSE_PATH.relative_to(ROOT)),
        "data_dir": str(DEFAULT_DATA_DIR.relative_to(ROOT)),
    }


def parse_caps(value: str) -> tuple[int, ...]:
    try:
        caps = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as exc:
        raise ValueError("caps must be a comma-separated list of integers") from exc
    if not caps or any(cap < 0 for cap in caps):
        raise ValueError("caps must contain at least one non-negative integer")
    if any(cap not in (0,) and cap < 30 for cap in caps):
        raise ValueError("a non-zero cap must be at least 30 stocks")
    return caps


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


def cap_label(cap: int) -> str:
    return "all" if cap == 0 else str(cap)


def eligibility_mask(
    features: dict[str, pd.DataFrame | pd.Series],
    cap: int,
) -> pd.DataFrame:
    """Return month-end eligibility from trailing volume and observed prices."""
    liquidity = features["liquidity"]
    monthly_prices = features["monthly_prices"]
    assert isinstance(liquidity, pd.DataFrame)
    assert isinstance(monthly_prices, pd.DataFrame)
    valid = liquidity.notna() & monthly_prices.notna()
    if cap == 0:
        return valid
    liquidity_rank = liquidity.rank(axis=1, ascending=False, method="first")
    return valid & liquidity_rank.le(cap)


def filter_panel(panel: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Keep panel rows whose signal-month ticker is eligible without a future join."""
    signal_codes = mask.index.get_indexer(pd.to_datetime(panel["signal_date"]))
    ticker_codes = mask.columns.get_indexer(panel["ticker"].astype(str))
    eligible = np.zeros(len(panel), dtype=bool)
    valid = (signal_codes >= 0) & (ticker_codes >= 0)
    mask_values = mask.to_numpy(dtype=bool)
    eligible[valid] = mask_values[signal_codes[valid], ticker_codes[valid]]
    return panel.loc[eligible].copy()


def rank_target_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Replace raw next-month returns with within-universe percentile targets."""
    ranked = panel.copy()
    ranked["target"] = ranked.groupby("signal_date")["target"].rank(
        pct=True,
        method="average",
    )
    return ranked


def _metric_periods(end: str, start: str) -> tuple[tuple[str, str, str], ...]:
    return (
        ("train", start, "2021-12-31"),
        ("validation", "2022-01-01", "2023-12-31"),
        ("holdout_2024", "2024-01-01", "2024-12-31"),
        ("holdout_2025_2026", "2025-01-01", end),
        ("full", start, end),
    )


def metric_map(simulation: pd.DataFrame, start: str, end: str) -> dict[str, dict[str, Any]]:
    return {
        label: ml_research.period_metrics(simulation, period_start, period_end, label)
        for label, period_start, period_end in _metric_periods(end, start)
    }


def _format_metric(value: Any) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _simulate_panels(
    panels: dict[str, pd.DataFrame],
    features: dict[str, pd.DataFrame | pd.Series],
    top_k: int,
    cost_bps: float,
    vol_managed_models: frozenset[str],
) -> dict[str, pd.DataFrame]:
    return {
        name: ml_research.simulate_predictions(
            panel,
            features,
            top_k,
            cost_bps,
            vol_managed=name in vol_managed_models,
        )
        for name, panel in panels.items()
    }


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
        return {
            "months": int(len(difference)),
            "mean": np.nan,
            "lower": np.nan,
            "median": np.nan,
            "upper": np.nan,
            "positive_probability": np.nan,
        }
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(len(difference) / BOOTSTRAP_BLOCK_LENGTH))
    starts = rng.integers(0, len(difference), size=(BOOTSTRAP_SAMPLES, block_count))
    offsets = np.arange(BOOTSTRAP_BLOCK_LENGTH)[None, None, :]
    indices = (starts[:, :, None] + offsets).reshape(BOOTSTRAP_SAMPLES, -1)
    indices = indices[:, : len(difference)] % len(difference)
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
    metrics: dict[str, dict[str, Any]],
    control_metrics: dict[str, dict[str, Any]],
    diagnostics: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    validation = metrics.get("validation", {}).get("excess_cagr", np.nan)
    control_validation = control_metrics.get("validation", {}).get("excess_cagr", np.nan)
    if not np.isfinite(validation) or not np.isfinite(control_validation) or validation <= control_validation:
        reasons.append("validation excess CAGR does not beat the matching liquidity control")
    for cost in (25.0, 100.0):
        for period in ("holdout_2024", "holdout_2025_2026"):
            rows = diagnostics.loc[
                diagnostics["model"].eq(model)
                & diagnostics["period"].eq(period)
                & np.isclose(diagnostics["cost_bps"], cost)
            ]
            excess = rows.iloc[0]["excess_cagr"] if not rows.empty else np.nan
            if not np.isfinite(excess) or excess <= 0.0:
                reasons.append(f"excess CAGR is not positive in {period} at {int(cost)} bps")
    rolling = rolling_summary.loc[
        rolling_summary["model"].eq(model)
        & np.isclose(rolling_summary["cost_bps"], 25.0)
    ]
    positive_fraction = rolling.iloc[0]["positive_excess_fraction"] if not rolling.empty else np.nan
    if not np.isfinite(positive_fraction) or positive_fraction < 0.60:
        reasons.append("fewer than 60% of trailing 12-month windows have positive excess CAGR")
    bootstrap_row = bootstrap.loc[bootstrap["model"].eq(model)]
    lower = bootstrap_row.iloc[0]["market_ci_lower"] if not bootstrap_row.empty else np.nan
    if not np.isfinite(lower) or lower <= 0.0:
        reasons.append("95% block-bootstrap lower CI versus IHSG is not above zero")
    holdout = metrics.get("holdout_2025_2026", {})
    beta = holdout.get("beta_to_benchmark", np.nan)
    drawdown = holdout.get("strategy_max_drawdown", np.nan)
    turnover = holdout.get("average_monthly_turnover", np.nan)
    if not np.isfinite(beta) or beta > RISK_BETA_LIMIT:
        reasons.append(f"holdout beta exceeds the fixed {RISK_BETA_LIMIT:.2f} risk limit")
    if not np.isfinite(drawdown) or drawdown < RISK_DRAWDOWN_LIMIT:
        reasons.append(f"holdout drawdown is worse than the fixed {RISK_DRAWDOWN_LIMIT:.2f} risk limit")
    if not np.isfinite(turnover) or turnover > RISK_TURNOVER_LIMIT:
        reasons.append(f"holdout turnover exceeds the fixed {RISK_TURNOVER_LIMIT:.2f} risk limit")
    return not reasons, reasons


def _parse_config(args: argparse.Namespace) -> LiquidConfig:
    caps = parse_caps(args.caps)
    if not 1 <= args.top_k <= 30:
        raise ValueError("top_k must be between 1 and 30")
    if args.cost_bps < 0.0 or args.min_train_rows < 60 or args.alpha <= 0.0:
        raise ValueError("cost, minimum training rows, and alpha are invalid")
    return LiquidConfig(
        price_field=args.price_field,
        start=args.start,
        end=args.end,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        min_train_rows=args.min_train_rows,
        alpha=args.alpha,
        caps=caps,
        data_dir=args.data_dir.resolve(),
        universe_path=args.universe.resolve(),
    )


def run(config: LiquidConfig) -> dict[str, Any]:
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
    base_panel = ml_research.make_panel(features)
    panels: dict[str, pd.DataFrame] = {}
    simulations: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    control_names: dict[str, str] = {}
    vol_managed_models: set[str] = set()
    training_rows: list[dict[str, Any]] = []

    for cap in config.caps:
        label = cap_label(cap)
        mask = eligibility_mask(features, cap)
        control_name = f"cap{label}_composite"
        control_names[label] = control_name
        panels[control_name] = research.score_frame("composite", features).where(mask)

        filtered_panel = filter_panel(base_panel, mask)
        rank_panel = rank_target_panel(filtered_panel)
        model_inputs = {
            "raw": filtered_panel,
            "rank": rank_panel,
        }
        for target_name, model_input in model_inputs.items():
            for model_name in ("ridge", "lightgbm"):
                prediction_name = f"cap{label}_{target_name}_{model_name}"
                try:
                    predictions, training_counts = ml_research.walk_forward_predictions(
                        model_input,
                        model_name,
                        config.min_train_rows,
                        config.alpha,
                    )
                except RuntimeError:
                    predictions = pd.DataFrame(
                        index=sorted(model_input["signal_date"].unique()),
                        columns=sorted(model_input["ticker"].unique()),
                        dtype=float,
                    )
                    training_counts = pd.Series(index=predictions.index, dtype=float)
                panels[prediction_name] = predictions
                training_rows.append(
                    {
                        "model": prediction_name,
                        "cap": cap,
                        "target": target_name,
                        "signals": int(predictions.notna().any(axis=1).sum()),
                        "forecast_rows": int(predictions.notna().sum().sum()),
                        "median_training_rows": (
                            float(training_counts.dropna().median())
                            if not training_counts.dropna().empty
                            else np.nan
                        ),
                    }
                )

        raw_ensemble = f"cap{label}_raw_ensemble"
        rank_ensemble = f"cap{label}_rank_ensemble"
        panels[raw_ensemble] = ml_research.ensemble_predictions(
            panels[f"cap{label}_raw_ridge"],
            panels[f"cap{label}_raw_lightgbm"],
        )
        panels[rank_ensemble] = ml_research.ensemble_predictions(
            panels[f"cap{label}_rank_ridge"],
            panels[f"cap{label}_rank_lightgbm"],
        )
        rank_volmanaged = f"cap{label}_rank_ensemble_volmanaged"
        panels[rank_volmanaged] = panels[rank_ensemble]
        vol_managed_models.add(rank_volmanaged)

    simulations = _simulate_panels(
        panels,
        features,
        config.top_k,
        config.cost_bps,
        frozenset(vol_managed_models),
    )
    metrics = {
        name: metric_map(simulation, config.start, config.end)
        for name, simulation in simulations.items()
    }

    diagnostics = foundation_robustness.build_cost_and_block_diagnostics(
        panels,
        features,
        config.top_k,
        vol_managed_models=frozenset(vol_managed_models),
    )
    rolling = foundation_robustness.build_rolling_diagnostics(
        panels,
        features,
        config.top_k,
        cost_bps=config.cost_bps,
        vol_managed_models=frozenset(vol_managed_models),
    )
    rolling_summary = foundation_robustness.summarize_rolling_diagnostics(rolling)

    bootstrap_rows: list[dict[str, Any]] = []
    candidate_names = [name for name in panels if name not in control_names.values()]
    for position, name in enumerate(candidate_names):
        label = name.split("_", 2)[0].removeprefix("cap")
        simulation = simulations[name]
        control_simulation = simulations[control_names[label]]
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
                "cap": label,
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
        label = name.split("_", 2)[0].removeprefix("cap")
        passed, reasons = _gate_candidate(
            name,
            metrics[name],
            metrics[control_names[label]],
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
    ) if candidate_names else None
    preferred = validation_winner if validation_winner and gates[validation_winner]["passed"] else None

    sensitivity_rows: list[dict[str, Any]] = []
    for top_k in TOP_K_SENSITIVITY:
        for cost_bps in COST_SENSITIVITY_BPS:
            sensitivity_simulations = _simulate_panels(
                panels,
                features,
                top_k,
                cost_bps,
                frozenset(vol_managed_models),
            )
            for name, simulation in sensitivity_simulations.items():
                for period, period_start, period_end in _metric_periods(config.end, config.start):
                    if period not in ("validation", "holdout_2024", "holdout_2025_2026"):
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

    _write_outputs(
        config=config,
        features=features,
        panels=panels,
        simulations=simulations,
        metrics=metrics,
        diagnostics=diagnostics,
        rolling=rolling,
        rolling_summary=rolling_summary,
        bootstrap=bootstrap,
        sensitivity=sensitivity,
        training_rows=pd.DataFrame(training_rows),
        gates=gates,
        validation_winner=validation_winner,
        preferred=preferred,
        loaded_tickers=loaded_tickers,
        skipped=skipped,
    )
    return {
        "metrics": metrics,
        "gates": gates,
        "validation_winner": validation_winner,
        "preferred": preferred,
        "loaded_ticker_count": len(loaded_tickers),
        "skipped": skipped,
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
    for model in metrics:
        if model.endswith("_composite"):
            continue
        values = metrics[model]
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


def _write_outputs(
    *,
    config: LiquidConfig,
    features: dict[str, pd.DataFrame | pd.Series],
    panels: dict[str, pd.DataFrame],
    simulations: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, dict[str, Any]]],
    diagnostics: pd.DataFrame,
    rolling: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    sensitivity: pd.DataFrame,
    training_rows: pd.DataFrame,
    gates: dict[str, dict[str, Any]],
    validation_winner: str | None,
    preferred: str | None,
    loaded_tickers: list[str],
    skipped: list[str],
) -> None:
    del features
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows = [
        {"model": model, **values}
        for model, model_metrics in metrics.items()
        for _, values in model_metrics.items()
    ]
    pd.DataFrame(metrics_rows).to_csv(REPORT_DIR / "liquid_rank_ml_metrics.csv", index=False)
    diagnostics.to_csv(REPORT_DIR / "liquid_rank_ml_robustness.csv", index=False)
    rolling.to_csv(REPORT_DIR / "liquid_rank_ml_rolling.csv", index=False)
    rolling_summary.to_csv(REPORT_DIR / "liquid_rank_ml_rolling_summary.csv", index=False)
    bootstrap.to_csv(REPORT_DIR / "liquid_rank_ml_bootstrap.csv", index=False)
    sensitivity.to_csv(REPORT_DIR / "liquid_rank_ml_sensitivity.csv", index=False)
    training_rows.to_csv(REPORT_DIR / "liquid_rank_ml_training.csv", index=False)

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
            REPORT_DIR / "liquid_rank_ml_forecasts.csv",
            index=False,
        )

    summary_path = REPORT_DIR / "liquid_rank_ml_summary.json"
    summary_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "run_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": {
                        key: (
                            str(value.relative_to(ROOT))
                            if isinstance(value, Path) and value.is_absolute()
                            else list(value) if isinstance(value, tuple)
                            else value
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
                }
            ),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    control_names = [model for model in metrics if model.endswith("_composite")]
    candidate_counts = len(metrics) - len(control_names)
    report = f"""# Liquidity-filtered rank-target ML research

Run date: {date.today().isoformat()}<br>
Universe catalog: `{config.universe_path.relative_to(ROOT)}`<br>
Data directory: `{config.data_dir.relative_to(ROOT)}`<br>
Loaded catalog tickers: `{len(loaded_tickers)}`; skipped/unavailable: `{len(skipped)}`<br>
Caps tested: `{', '.join('all' if cap == 0 else str(cap) for cap in config.caps)}`; default top K: `{config.top_k}`; cost: `{config.cost_bps:.1f}` bps one-way<br>

## Why this experiment exists

The current catalog is not a point-in-time universe. The official [IDX80/LQ45/IDX30 methodology](https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf) describes liquidity, market capitalization, fundamentals, a six-month listing minimum, and consistent trading as part of index selection. Historical free-float and membership data are unavailable here, so this run uses only the observable trailing 60-day median dollar-volume feature to create fixed eligibility caps. It is a robustness proxy, not a reconstruction of IDX membership.

## Paper-backed design

[Gu, Kelly, and Xiu](https://www.nber.org/papers/w25398) motivate comparing regularized linear models and nonlinear trees with strict out-of-sample testing. Each completed month-end `t` uses only data through `t`:

- eligible names are the top fixed liquidity cap at `t`, with no forward volume information;
- raw-return models predict next-month returns; rank-target models predict each eligible stock's percentile rank among the next-month realized returns in historical training months;
- Ridge and shallow LightGBM are refit expanding-window, with training signal dates strictly earlier than the forecast date;
- forecasts are ranked cross-sectionally and the top K names are equally weighted for the following month;
- volatility-managed variants use the existing fixed 10% benchmark-volatility exposure rule;
- caps, top K values, costs, rolling windows, and bootstrap settings are fixed before inspecting holdout results.

Rank targets are a robustness transformation against a few extreme micro-cap returns dominating the raw-return loss. They do not create additional information and are not a guarantee of better prediction.

## Audit summary

The fixed audit gate requires beating the matching liquidity control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% block-bootstrap lower bound versus IHSG, and fixed risk limits of beta ≤ `{RISK_BETA_LIMIT:.2f}`, drawdown ≥ `{RISK_DRAWDOWN_LIMIT:.2f}`, and turnover ≤ `{RISK_TURNOVER_LIMIT:.2f}` in the 2025–2026 holdout. Holdout data are not used to choose the variant.

{_summary_table(metrics, rolling_summary, bootstrap, gates)}

Validation winner under the predeclared rule (highest validation Sharpe, then excess CAGR): **{validation_winner or 'none'}**. It is only considered preferred if it passes every audit check: **{preferred or 'none'}**. `{candidate_counts}` candidate panels were evaluated; a passing row remains exploratory because the current catalog omits delisted names and historical membership.

## Top-K and cost sensitivity

The complete fixed replay is in `reports/liquid_rank_ml_sensitivity.csv`. It covers top K = 1, 3, 5, and 10, costs = 25 and 100 bps, all tested caps, controls, raw targets, rank targets, ensembles, and volatility-managed variants. No top-K or cap was selected using the holdout.

## Limitations

- This is still a current-universe snapshot. It cannot correct survivorship, delisting, historical suspension, free-float, or corporate-action selection bias.
- Yahoo Finance is a research input, not a licensed exchange execution feed. Spreads, price limits, taxes, impact, capacity, and failed fills are not fully modeled.
- A fixed liquidity rank is not the same as the IDX's transaction-value, free-float, or fundamental screens.
- A backtest result is not investment advice and does not guarantee beating IHSG.

## Reproduction

```bash
python3 src/download_data.py --universe all --start 2015-01-01 --end 2026-08-12 \\
  --output-dir data/raw/yahoo_all --metadata-path data/raw/yahoo_all_metadata.json \\
  --continue-on-error
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.liquid_rank_ml_research
```

Raw forecasts are written to ignored `reports/liquid_rank_ml_forecasts.csv`. Committed audit tables are `reports/liquid_rank_ml_metrics.csv`, `reports/liquid_rank_ml_robustness.csv`, `reports/liquid_rank_ml_rolling_summary.csv`, `reports/liquid_rank_ml_bootstrap.csv`, `reports/liquid_rank_ml_sensitivity.csv`, and `reports/liquid_rank_ml_training.csv`.
"""
    (REPORT_DIR / "liquid_rank_ml_findings.md").write_text(report, encoding="utf-8")


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
    print("\nWrote reports/liquid_rank_ml_findings.md and reports/liquid_rank_ml_metrics.csv")


if __name__ == "__main__":
    main()
