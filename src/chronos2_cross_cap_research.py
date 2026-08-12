"""Replay a fixed cross-cap rank ensemble from completed Chronos-2 audits.

The daily Chronos-2 runner is expensive because it performs model inference at
every signal month. This module therefore does no new inference. It combines
the already-recorded terminal lower-tail forecasts from the top-150, top-300,
and top-500 liquidity screens using equal cross-sectional percentile ranks, then
replays that fixed score with the same next-month, cost, rolling, and bootstrap
protocol. The ensemble is an audit candidate, not a live signal.
"""

from __future__ import annotations

import argparse
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
    from . import chronos2_daily_research as daily_research
except ImportError:  # pragma: no cover - direct CLI compatibility
    from src import (
        all_stock_ml_research,
        foundation_robustness,
        liquid_rank_ml_research,
        ml_research,
        research,
    )
    from src import chronos2_daily_research as daily_research


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "yahoo_all"
DEFAULT_UNIVERSE_PATH = ROOT / "data" / "universe_idx_all_2026-08.csv"
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2026-08-11"
DEFAULT_TOP_K = 3
DEFAULT_COST_BPS = 25.0
DEFAULT_OUTPUT_PREFIX = "chronos2_cross_cap"
FORECAST_MODEL = "chronos2_daily_abs_cross_lower10"
SOURCE_CAPS = (150, 300, 500)
TOP_K_SENSITIVITY = (1, 3, 5, 10)
COST_SENSITIVITY_BPS = (25.0, 100.0)
CONTROL_NAME = "cap500_composite"
ENSEMBLE_NAME = "chronos2_cross_cap_rank_ensemble"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE_PATH)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.top_k <= 30:
        raise ValueError("top_k must be between one and 30")
    if args.cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix:
        raise ValueError("output_prefix must be a non-empty filename prefix")


def _load_forecast_panel(path: Path, model: str = FORECAST_MODEL) -> pd.DataFrame:
    """Load one recorded model panel and reject duplicate signal/ticker rows."""
    if not path.exists():
        raise FileNotFoundError(f"Missing forecast file {path}; run the daily audits first")
    frame = pd.read_csv(path, parse_dates=["signal_date"])
    required = {"model", "signal_date", "ticker", "forecast"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Forecast file {path} is missing columns: {sorted(missing)}")
    selected = frame.loc[frame["model"].eq(model), ["signal_date", "ticker", "forecast"]].copy()
    if selected.empty:
        raise ValueError(f"Forecast file {path} has no rows for model {model}")
    if selected.duplicated(["signal_date", "ticker"]).any():
        raise ValueError(f"Forecast file {path} has duplicate signal/ticker rows")
    selected["ticker"] = selected["ticker"].astype(str)
    selected["forecast"] = pd.to_numeric(selected["forecast"], errors="coerce")
    panel = selected.pivot(index="signal_date", columns="ticker", values="forecast")
    return panel.sort_index().sort_index(axis=1)


def cross_cap_rank_ensemble(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average available within-cap percentile ranks without imputation."""
    if not panels:
        raise ValueError("at least one forecast panel is required")
    ranked = [panel.rank(axis=1, pct=True, method="average") for panel in panels.values()]
    stacked = pd.concat(
        ranked,
        keys=list(panels),
        names=["source", "signal_date"],
    )
    return stacked.groupby(level="signal_date", sort=True).mean()


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


def _output_path(config: argparse.Namespace, suffix: str) -> Path:
    return REPORT_DIR / f"{config.output_prefix}_{suffix}"


def _coverage_rows(source_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, panel in source_panels.items():
        counts = panel.notna().sum(axis=1)
        rows.append(
            {
                "source": source,
                "signals": int(len(panel.index)),
                "forecast_rows": int(panel.notna().sum().sum()),
                "tickers_min": int(counts.min()),
                "tickers_median": float(counts.median()),
                "tickers_max": int(counts.max()),
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

        def format_value(value: Any) -> str:
            if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
                return "—"
            return f"{float(value):.4f}" if isinstance(value, (float, np.floating)) else str(value)

        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    format_value(validation.get("excess_cagr")),
                    format_value(holdout_2024.get("excess_cagr")),
                    format_value(holdout.get("excess_cagr")),
                    format_value(
                        rolling_row.iloc[0]["positive_excess_fraction"]
                        if not rolling_row.empty
                        else np.nan
                    ),
                    format_value(
                        bootstrap_row.iloc[0]["market_ci_lower"]
                        if not bootstrap_row.empty
                        else np.nan
                    ),
                    format_value(holdout.get("beta_to_benchmark")),
                    format_value(holdout.get("strategy_max_drawdown")),
                    "passes" if gates.get(model, {}).get("passed") else "rejects",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def run(config: argparse.Namespace) -> dict[str, Any]:
    prices, volumes, loaded_tickers, skipped = all_stock_ml_research.load_prices(
        config.data_dir,
        config.universe,
        "adjclose",
    )
    prices = prices.loc[prices.index <= config.end]
    features = all_stock_ml_research._slice_features(
        research.make_features(prices, volumes.reindex(prices.index)),
        config.start,
        config.end,
    )

    source_panels = {
        f"cap{cap}": _load_forecast_panel(
            REPORT_DIR / ("chronos2_daily_forecasts.csv" if cap == 300 else f"chronos2_daily_cap{cap}_forecasts.csv")
        )
        for cap in SOURCE_CAPS
    }
    ensemble = cross_cap_rank_ensemble(source_panels)
    control = research.score_frame("composite", features).where(
        liquid_rank_ml_research.eligibility_mask(features, cap=500)
    )
    panels: dict[str, pd.DataFrame] = {CONTROL_NAME: control}
    panels.update({f"{source}_lower10": panel for source, panel in source_panels.items()})
    panels[ENSEMBLE_NAME] = ensemble

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
    bootstrap = daily_research._bootstrap_rows(simulations, config.end, CONTROL_NAME)

    candidates = [name for name in panels if name != CONTROL_NAME]
    gates: dict[str, dict[str, Any]] = {}
    for name in candidates:
        passed, reasons = liquid_rank_ml_research._gate_candidate(
            name,
            metrics[name],
            metrics[CONTROL_NAME],
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
            replay = {
                name: ml_research.simulate_predictions(panel, features, top_k, cost_bps)
                for name, panel in panels.items()
            }
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
    coverage = _coverage_rows(source_panels)
    _write_outputs(
        config=config,
        metrics=metrics,
        diagnostics=diagnostics,
        rolling=rolling,
        rolling_summary=rolling_summary,
        bootstrap=bootstrap,
        sensitivity=sensitivity,
        coverage=coverage,
        gates=gates,
        validation_winner=validation_winner,
        preferred=preferred,
        loaded_tickers=loaded_tickers,
        skipped=skipped,
        source_panels=source_panels,
    )
    return {
        "metrics": metrics,
        "gates": gates,
        "validation_winner": validation_winner,
        "preferred": preferred,
    }


def _write_outputs(
    *,
    config: argparse.Namespace,
    metrics: dict[str, dict[str, dict[str, Any]]],
    diagnostics: pd.DataFrame,
    rolling: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    sensitivity: pd.DataFrame,
    coverage: pd.DataFrame,
    gates: dict[str, dict[str, Any]],
    validation_winner: str | None,
    preferred: str | None,
    loaded_tickers: list[str],
    skipped: list[str],
    source_panels: dict[str, pd.DataFrame],
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
    coverage.to_csv(_output_path(config, "coverage.csv"), index=False)

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: (daily_research._relative_path(value) if isinstance(value, Path) else value)
            for key, value in vars(config).items()
        },
        "loaded_ticker_count": len(loaded_tickers),
        "skipped": skipped,
        "forecast_model": FORECAST_MODEL,
        "source_caps": SOURCE_CAPS,
        "source_coverage": coverage.to_dict(orient="records"),
        "metrics": metrics,
        "validation_winner": validation_winner,
        "preferred_research_candidate": preferred,
        "gates": gates,
        "bootstrap": bootstrap.to_dict(orient="records"),
        "formula": "mean of available within-cap percentile ranks of terminal Chronos-2 lower10 forecasts from caps 150, 300, and 500",
        "sources": {
            "chronos": daily_research.CHRONOS_SOURCE,
            "chronos2_study": daily_research.CHRONOS2_STUDY,
            "idx_methodology": daily_research.IDX_METHOD_SOURCE,
        },
    }
    _output_path(config, "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    source_lines = "\n".join(
        f"- cap {source.removeprefix('cap')}: {len(panel.index)} signals; {int(panel.notna().sum().sum()):,} forecast values"
        for source, panel in source_panels.items()
    )
    report = f"""# Chronos-2 cross-cap rank ensemble research

Run date: {date.today().isoformat()}<br>
Forecast model: {FORECAST_MODEL}; source panels are completed daily Chronos-2 audits<br>
Universe catalog: {daily_research._relative_path(config.universe)}; loaded tickers: {len(loaded_tickers)}; skipped: {len(skipped)}<br>
Portfolio: top {config.top_k} equal-weight names; monthly signal; {config.cost_bps:.1f} bps one-way default cost<br>

## Research basis

Chronos-2 cross-learning is tested in the source audits using the official long-format forecasting interface and the multivariate financial-forecasting study. This replay adds a fixed rank-aggregation layer: it does not fit weights, inspect returns, or run new model inference. The liquidity caps follow the official IDX80/LQ45/IDX30 methodology as a proxy because historical membership and free-float data are unavailable.

## Fixed formula

For each signal month t and ticker i, let F be the terminal lower-10% forecast from each completed source panel. The score is:

~~~text
score(t, i) = mean(
    rank_pct(F_cap150(t, i)),
    rank_pct(F_cap300(t, i)),
    rank_pct(F_cap500(t, i)),
)
~~~

The mean uses only available forecasts; no missing value is imputed. Each cap is ranked within its own cross-section before averaging, so a larger universe cannot dominate by forecast scale. The three cap weights are equal and fixed. The fair comparator is the existing composite restricted to the top-500 liquidity proxy, because the ensemble's broadest source is cap 500.

Source coverage:

{source_lines}

## Leakage controls and limitations

- Every source forecast was produced using daily observations on or before the completed signal month; this replay only reads those recorded forecasts.
- The ensemble weights and percentile transformation are fixed; no holdout return is used in aggregation.
- The current catalog omits historical delistings, suspensions, and membership changes, so cap robustness does not remove survivorship bias.
- Forecast files are generated from adjusted-price daily Chronos-2 runs. Yahoo data do not model spreads, taxes, price limits, market impact, failed fills, or capacity.

## Audit summary

The fixed gate requires beating the cap-500 composite control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025-2026.

{_summary_table(metrics, rolling_summary, bootstrap, gates)}

Overall validation winner: {validation_winner or "none"}. Preferred after the fixed gate: {preferred or "none"}. The ensemble is retained as an exploratory rank-aggregation result; it is not promoted unless it survives future point-in-time and unseen-data validation.

## Reproduction

~~~bash
PYTHONPATH=$PWD python3 -m src.chronos2_cross_cap_research
~~~

The runner expects the three ignored source files: reports/chronos2_daily_cap150_forecasts.csv, reports/chronos2_daily_forecasts.csv, and reports/chronos2_daily_cap500_forecasts.csv. Committed outputs use the {config.output_prefix}_ prefix.

This is research, not investment advice, and no backtest guarantees future performance.
"""
    _output_path(config, "findings.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
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
        f"preferred: {result['preferred']}"
    )
    print(f"Wrote reports/{args.output_prefix}_findings.md")


if __name__ == "__main__":
    main()
