"""Robustness diagnostics for zero-shot foundation-model forecasts.

The forecasts are generated once by :mod:`src.foundation_research`. This
module deliberately contains no model imports: it replays the same forecast
panels through different cost assumptions and chronological windows. That
keeps robustness analysis cheap, deterministic, and separate from model
selection.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

try:  # Support both ``python -m src...`` and direct execution imports.
    from . import ml_research
except ImportError:  # pragma: no cover - direct CLI compatibility
    from src import ml_research


COST_GRID_BPS = (0.0, 25.0, 50.0, 100.0)
"""Pre-declared one-way cost assumptions used in the stress test."""

CHRONOLOGICAL_BLOCKS = (
    ("train_2021", "2021-01-01", "2021-12-31"),
    ("validation_2022_2023", "2022-01-01", "2023-12-31"),
    ("holdout_2024", "2024-01-01", "2024-12-31"),
    ("holdout_2025_2026", "2025-01-01", "2026-08-11"),
)
"""Fixed chronological blocks; the last two are never used for selection."""

ROLLING_WINDOW_MONTHS = 12


def _as_metric_row(
    model: str,
    cost_bps: float,
    period_name: str,
    simulation: pd.DataFrame,
    start: str,
    end: str,
) -> dict[str, Any]:
    row = ml_research.period_metrics(simulation, start, end, period_name)
    return {"model": model, "cost_bps": cost_bps, **row}


def build_cost_and_block_diagnostics(
    predictions: Mapping[str, pd.DataFrame],
    features: dict[str, pd.DataFrame | pd.Series],
    top_k: int,
    costs_bps: tuple[float, ...] = COST_GRID_BPS,
    blocks: tuple[tuple[str, str, str], ...] = CHRONOLOGICAL_BLOCKS,
    vol_managed_models: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Replay every forecast panel across fixed costs and time blocks.

    ``predictions`` may include the existing composite rank panel as a control
    alongside foundation-model forecasts. No values are fit or tuned here.
    """
    rows: list[dict[str, Any]] = []
    for model, panel in predictions.items():
        for cost_bps in costs_bps:
            simulation = ml_research.simulate_predictions(
                panel,
                features,
                top_k,
                cost_bps,
                vol_managed=model in vol_managed_models,
            )
            for period_name, start, end in blocks:
                rows.append(
                    _as_metric_row(model, float(cost_bps), period_name, simulation, start, end)
                )
    return pd.DataFrame(rows)


def build_rolling_diagnostics(
    predictions: Mapping[str, pd.DataFrame],
    features: dict[str, pd.DataFrame | pd.Series],
    top_k: int,
    cost_bps: float = 25.0,
    window_months: int = ROLLING_WINDOW_MONTHS,
    vol_managed_models: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Return fixed-length trailing-window metrics at one declared cost.

    A window ending at signal month ``t`` measures returns applied after the
    signal dates in that window. Windows are descriptive stability checks, not
    additional opportunities to choose a model.
    """
    if window_months < 3:
        raise ValueError("window_months must be at least three")
    rows: list[dict[str, Any]] = []
    for model, panel in predictions.items():
        simulation = ml_research.simulate_predictions(
            panel,
            features,
            top_k,
            cost_bps,
            vol_managed=model in vol_managed_models,
        )
        signal_dates = pd.DatetimeIndex(panel.index).sort_values().unique()
        for end_position in range(window_months - 1, len(signal_dates)):
            start_date = signal_dates[end_position - window_months + 1]
            end_date = signal_dates[end_position]
            period_name = f"{start_date.date()}_{end_date.date()}"
            rows.append(
                _as_metric_row(
                    model,
                    float(cost_bps),
                    period_name,
                    simulation,
                    str(start_date.date()),
                    str(end_date.date()),
                )
            )
    return pd.DataFrame(rows)


def summarize_rolling_diagnostics(rolling: pd.DataFrame) -> pd.DataFrame:
    """Summarize rolling windows without ranking models on holdout data."""
    if rolling.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "cost_bps",
                "windows",
                "median_sharpe",
                "positive_excess_fraction",
                "worst_excess_cagr",
                "median_turnover",
            ]
        )
    grouped = rolling.groupby(["model", "cost_bps"], dropna=False)
    return (
        grouped.agg(
            windows=("months", "count"),
            median_sharpe=("strategy_sharpe_rf0", "median"),
            positive_excess_fraction=("excess_cagr", lambda values: float((values > 0).mean())),
            worst_excess_cagr=("excess_cagr", "min"),
            median_turnover=("average_monthly_turnover", "median"),
        )
        .reset_index()
        .sort_values(["cost_bps", "model"])
    )


def validation_winner_by_block(
    diagnostics: pd.DataFrame,
    baseline_model: str = "existing_composite",
    validation_periods: tuple[str, ...] = ("train_2021", "validation_2022_2023"),
    cost_bps: float = 25.0,
) -> pd.DataFrame:
    """Show which candidate wins each pre-holdout block at the fixed cost.

    This table is an audit trail. It does not change the runner's selection
    rule and intentionally excludes the two holdout blocks.
    """
    if diagnostics.empty:
        return pd.DataFrame(columns=["period", "winner", "winner_sharpe", "baseline_sharpe"])
    eligible = diagnostics.loc[
        diagnostics["period"].isin(validation_periods)
        & np.isclose(diagnostics["cost_bps"], cost_bps)
    ].copy()
    rows: list[dict[str, Any]] = []
    for period, frame in eligible.groupby("period", sort=False):
        frame = frame.dropna(subset=["strategy_sharpe_rf0"])
        if frame.empty:
            continue
        winner = frame.sort_values(
            ["strategy_sharpe_rf0", "excess_cagr"], ascending=[False, False]
        ).iloc[0]
        baseline = frame.loc[frame["model"] == baseline_model]
        rows.append(
            {
                "period": period,
                "winner": winner["model"],
                "winner_sharpe": winner["strategy_sharpe_rf0"],
                "baseline_sharpe": baseline.iloc[0]["strategy_sharpe_rf0"]
                if not baseline.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def render_table(
    diagnostics: pd.DataFrame,
    *,
    periods: tuple[str, ...] | None = None,
    costs_bps: tuple[float, ...] | None = None,
) -> str:
    """Render a compact Markdown table for a report."""
    columns = (
        "model",
        "cost_bps",
        "period",
        "months",
        "excess_cagr",
        "strategy_sharpe_rf0",
        "strategy_max_drawdown",
        "average_monthly_turnover",
    )
    frame = diagnostics.copy()
    if periods is not None:
        frame = frame.loc[frame["period"].isin(periods)]
    if costs_bps is not None:
        frame = frame.loc[frame["cost_bps"].isin(costs_bps)]
    if frame.empty:
        return "_No diagnostics available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in frame.sort_values(["period", "cost_bps", "model"]).iterrows():
        values: list[str] = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, (float, np.floating)):
                values.append("—" if not np.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
