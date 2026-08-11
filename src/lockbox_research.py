"""Evaluate completed model forecasts on the newest unseen monthly lockbox.

This runner performs no model fitting and no new inference.  It compares the
already-generated causal forecast panels on 2026, the newest completed period
available in the local data, while retaining earlier validation and holdout
blocks for context.  The six-month lockbox is descriptive only and cannot
establish a durable future edge.
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
    from . import all_stock_ml_research, liquid_rank_ml_research, ml_research, research
except ImportError:  # pragma: no cover - direct CLI compatibility
    from src import all_stock_ml_research, liquid_rank_ml_research, ml_research, research


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "yahoo_all"
DEFAULT_UNIVERSE_PATH = ROOT / "data" / "universe_idx_all_2026-08.csv"
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2026-08-11"
DEFAULT_CAP = 300
DEFAULT_TOP_K = 3
DEFAULT_COST_BPS = 25.0
DEFAULT_OUTPUT_PREFIX = "lockbox"
LOCKBOX_START = "2026-01-01"

SOURCE_SPECS = {
    "mlp_rank": ("all_stock_deep_forecasts.csv", "mlp_rank"),
    "chronos2_zero_shot_lower10": (
        "chronos2_daily_forecasts.csv",
        "chronos2_daily_abs_cross_lower10",
    ),
    "chronos2_annual_ft_lower10": (
        "chronos2_finetune_walkforward_forecasts.csv",
        "chronos2_ft_cross_lower10",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--lockbox-start", default=LOCKBOX_START)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_forecast_panel(path: Path, model: str, end: str) -> pd.DataFrame:
    """Load one causal forecast panel and reject duplicate rows."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the source model audit first")
    frame = pd.read_csv(path, parse_dates=["signal_date"])
    required = {"model", "signal_date", "ticker", "forecast"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    selected = frame.loc[
        frame["model"].eq(model) & (frame["signal_date"] <= pd.Timestamp(end)),
        ["signal_date", "ticker", "forecast"],
    ].copy()
    if selected.empty:
        raise ValueError(f"{path} has no rows for model {model}")
    if selected.duplicated(["signal_date", "ticker"]).any():
        raise ValueError(f"{path} has duplicate signal/ticker rows for {model}")
    selected["ticker"] = selected["ticker"].astype(str)
    selected["forecast"] = pd.to_numeric(selected["forecast"], errors="coerce")
    return selected.pivot(index="signal_date", columns="ticker", values="forecast").sort_index()


def _periods(start: str, end: str, lockbox_start: str) -> tuple[tuple[str, str, str], ...]:
    return (
        ("validation_2022_2023", "2022-01-01", "2023-12-31"),
        ("holdout_2024", "2024-01-01", "2024-12-31"),
        ("holdout_2025", "2025-01-01", "2025-12-31"),
        ("lockbox_2026", lockbox_start, end),
        ("full", start, end),
    )


def _metrics(
    simulation: pd.DataFrame,
    start: str,
    end: str,
    lockbox_start: str,
) -> dict[str, dict[str, Any]]:
    return {
        label: ml_research.period_metrics(simulation, period_start, period_end, label)
        for label, period_start, period_end in _periods(start, end, lockbox_start)
    }


def _format(value: Any) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _metrics_table(metrics: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "| model | validation excess | 2024 excess | 2025 excess | 2026 lockbox excess | lockbox months | lockbox Sharpe | lockbox drawdown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, values in metrics.items():
        validation = values.get("validation_2022_2023", {})
        holdout_2024 = values.get("holdout_2024", {})
        holdout_2025 = values.get("holdout_2025", {})
        lockbox = values.get("lockbox_2026", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    _format(validation.get("excess_cagr")),
                    _format(holdout_2024.get("excess_cagr")),
                    _format(holdout_2025.get("excess_cagr")),
                    _format(lockbox.get("excess_cagr")),
                    _format(lockbox.get("months")),
                    _format(lockbox.get("strategy_sharpe_rf0")),
                    _format(lockbox.get("strategy_max_drawdown")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cap < args.top_k or args.cap < 30:
        raise ValueError("cap must be at least top_k and at least 30")
    if args.top_k < 1 or args.cost_bps < 0.0:
        raise ValueError("top_k or cost is invalid")
    if pd.Timestamp(args.lockbox_start) <= pd.Timestamp("2025-12-31"):
        raise ValueError("lockbox-start must be in 2026 or later for this audit")
    prices, volumes, loaded_tickers, skipped = all_stock_ml_research.load_prices(
        args.data_dir,
        args.universe,
        "adjclose",
    )
    prices = prices.loc[prices.index <= pd.Timestamp(args.end)]
    features = all_stock_ml_research._slice_features(
        research.make_features(prices, volumes.reindex(prices.index)),
        args.start,
        args.end,
    )
    eligibility = liquid_rank_ml_research.eligibility_mask(features, args.cap)
    control_name = f"cap{args.cap}_composite"
    panels: dict[str, pd.DataFrame] = {
        control_name: research.score_frame("composite", features).where(eligibility)
    }
    source_paths: dict[str, str] = {}
    for name, (filename, model) in SOURCE_SPECS.items():
        path = args.report_dir / filename
        panels[name] = load_forecast_panel(path, model, args.end)
        source_paths[name] = str(path)

    simulations = {
        name: ml_research.simulate_predictions(panel, features, args.top_k, args.cost_bps)
        for name, panel in panels.items()
    }
    metrics = {
        name: _metrics(simulation, args.start, args.end, args.lockbox_start)
        for name, simulation in simulations.items()
    }
    candidates = [name for name in panels if name != control_name]
    validation_winner = max(
        candidates,
        key=lambda name: (
            metrics[name]["validation_2022_2023"].get("strategy_sharpe_rf0", -np.inf),
            metrics[name]["validation_2022_2023"].get("excess_cagr", -np.inf),
        ),
        default=None,
    )
    lockbox_winner = max(
        candidates,
        key=lambda name: (
            metrics[name]["lockbox_2026"].get("strategy_sharpe_rf0", -np.inf),
            metrics[name]["lockbox_2026"].get("excess_cagr", -np.inf),
        ),
        default=None,
    )

    cost_rows: list[dict[str, Any]] = []
    for name, panel in panels.items():
        for cost in (25.0, 100.0):
            simulation = ml_research.simulate_predictions(panel, features, args.top_k, cost)
            for period, period_start, period_end in (
                ("holdout_2025", "2025-01-01", "2025-12-31"),
                ("lockbox_2026", args.lockbox_start, args.end),
            ):
                cost_rows.append(
                    {
                        "model": name,
                        "cost_bps": cost,
                        **ml_research.period_metrics(
                            simulation,
                            period_start,
                            period_end,
                            period,
                        ),
                    }
                )
    cost_frame = pd.DataFrame(cost_rows)

    monthly_rows: list[pd.DataFrame] = []
    for name, simulation in simulations.items():
        frame = simulation.loc[args.lockbox_start : args.end].reset_index(names="signal_date")
        frame.insert(0, "model", name)
        monthly_rows.append(frame)
    monthly = pd.concat(monthly_rows, ignore_index=True)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.report_dir / args.output_prefix
    metrics_frame = pd.DataFrame(
        [
            {"model": model, **period_values}
            for model, model_values in metrics.items()
            for _, period_values in model_values.items()
        ]
    )
    metrics_frame.to_csv(f"{prefix}_metrics.csv", index=False)
    cost_frame.to_csv(f"{prefix}_costs.csv", index=False)
    monthly.to_csv(f"{prefix}_monthly.csv", index=False)
    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "loaded_ticker_count": len(loaded_tickers),
        "skipped": skipped,
        "source_paths": source_paths,
        "source_models": SOURCE_SPECS,
        "metrics": metrics,
        "validation_winner": validation_winner,
        "lockbox_winner_descriptive_only": lockbox_winner,
        "lockbox_months": metrics[lockbox_winner]["lockbox_2026"].get("months")
        if lockbox_winner
        else 0,
        "warning": "2026 lockbox contains only six completed monthly observations and is not a promotion gate",
    }
    (Path(f"{prefix}_summary.json")).write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report = f"""# Newest-data model lockbox

Run date: {date.today().isoformat()}<br>
Universe: `{args.universe}`; loaded tickers: {len(loaded_tickers)}; skipped: {len(skipped)}<br>
Liquidity proxy: top {args.cap}; portfolio: top {args.top_k} equal-weight names; cost: {args.cost_bps:.1f} bps one-way<br>
Lockbox: {args.lockbox_start} through {args.end}; completed observations: {metrics[validation_winner]['lockbox_2026'].get('months') if validation_winner else 0}<br>

## Purpose

This report is a strict newest-data check, not another model-selection sweep.
It compares the previously generated causal panels from the paper-style MLP,
zero-shot Chronos-2, and annually refit Chronos-2. The 2026 period was not
used to choose their architecture, seeds, forecast tail, or portfolio rule.

## Results

{_metrics_table(metrics)}

Validation winner under the fixed Sharpe-then-excess rule: **{validation_winner or "none"}**. Descriptive 2026 lockbox winner: **{lockbox_winner or "none"}**. The lockbox winner is not promoted because it contains only six completed monthly observations.

At the declared cost, the newest lockbox is a useful stress point but not
statistical proof. A short favorable period can reflect market regime, sample
noise, or data-mining even when the forecast was generated causally. Promotion
still requires a longer future sample and a point-in-time universe.

## Reproduction inputs

{chr(10).join(f"- `{name}`: `{path}` (`{model}`)" for name, path in source_paths.items() for _, model in [SOURCE_SPECS[name]])}

The source runners must be executed before this replay. Committed outputs are
`{args.output_prefix}_metrics.csv`, `{args.output_prefix}_costs.csv`,
`{args.output_prefix}_monthly.csv`, and `{args.output_prefix}_summary.json`.

This is research, not investment advice, and no backtest guarantees future performance.
"""
    Path(f"{prefix}_findings.md").write_text(report, encoding="utf-8")
    return {
        "metrics": metrics,
        "validation_winner": validation_winner,
        "lockbox_winner": lockbox_winner,
    }


def main() -> None:
    args = parse_args()
    result = run(args)
    summary = pd.DataFrame(
        [
            {
                "model": model,
                "validation_excess": values["validation_2022_2023"].get("excess_cagr"),
                "lockbox_excess": values["lockbox_2026"].get("excess_cagr"),
                "lockbox_months": values["lockbox_2026"].get("months"),
            }
            for model, values in result["metrics"].items()
        ]
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    print(
        f"\nValidation winner: {result['validation_winner']}; "
        f"descriptive lockbox winner: {result['lockbox_winner']}"
    )
    print("Wrote lockbox report and tables")


if __name__ == "__main__":
    main()
