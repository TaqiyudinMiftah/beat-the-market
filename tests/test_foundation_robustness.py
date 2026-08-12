from __future__ import annotations

import numpy as np
import pandas as pd

from src.foundation_robustness import (
    build_cost_and_block_diagnostics,
    build_rolling_diagnostics,
    summarize_rolling_diagnostics,
    validation_winner_by_block,
)


def _synthetic_inputs() -> tuple[dict[str, pd.DataFrame | pd.Series], dict[str, pd.DataFrame]]:
    index = pd.date_range("2021-01-31", periods=15, freq="ME")
    columns = ["AAA", "BBB", "CCC", "DDD"]
    prices = pd.DataFrame(
        100.0
        * np.cumprod(
            1.0 + np.tile(np.array([[0.01, 0.02, -0.01, 0.015]]), (len(index), 1)),
            axis=0,
        ),
        index=index,
        columns=columns,
    )
    returns = prices.pct_change(fill_method=None)
    benchmark = pd.Series(
        [0.20, -0.18, 0.16, -0.14, 0.12, -0.10, 0.09, -0.08, 0.07, -0.06, 0.05, -0.04, 0.03, -0.02, 0.01],
        index=index,
        name="^JKSE",
    )
    predictions = pd.DataFrame(
        np.tile(np.array([[0.1, 0.2, 0.3, 0.4]]), (len(index), 1)),
        index=index,
        columns=columns,
    )
    inverse = predictions.iloc[:, ::-1].copy()
    features: dict[str, pd.DataFrame | pd.Series] = {
        "monthly_prices": prices,
        "monthly_returns": returns,
        "benchmark_returns": benchmark,
        "daily_prices": prices,
    }
    panels = {"existing_composite": predictions, "foundation": inverse}
    return features, panels


def test_cost_and_block_diagnostics_replay_all_declared_cases() -> None:
    features, panels = _synthetic_inputs()
    blocks = (("early", "2021-01-01", "2021-06-30"), ("late", "2021-07-01", "2022-12-31"))

    diagnostics = build_cost_and_block_diagnostics(
        panels,
        features,
        top_k=2,
        costs_bps=(0.0, 25.0),
        blocks=blocks,
    )

    assert len(diagnostics) == 2 * 2 * 2
    assert set(diagnostics["cost_bps"]) == {0.0, 25.0}
    assert set(diagnostics["period"]) == {"early", "late"}
    assert diagnostics["months"].min() > 0


def test_rolling_summary_and_validation_winner_are_deterministic() -> None:
    features, panels = _synthetic_inputs()
    rolling = build_rolling_diagnostics(panels, features, top_k=2, window_months=6)
    summary = summarize_rolling_diagnostics(rolling)

    assert len(rolling) == 2 * (15 - 6 + 1)
    assert set(summary["windows"]) == {10}

    blocks = (("train_2021", "2021-01-01", "2021-06-30"), ("validation_2022_2023", "2021-07-01", "2022-12-31"))
    diagnostics = build_cost_and_block_diagnostics(
        panels,
        features,
        top_k=2,
        costs_bps=(25.0,),
        blocks=blocks,
    )
    winners = validation_winner_by_block(diagnostics)

    assert list(winners["period"]) == ["train_2021", "validation_2022_2023"]
    assert set(winners["winner"]) <= {"existing_composite", "foundation"}


def test_volatility_management_is_carried_into_replay() -> None:
    features, panels = _synthetic_inputs()
    blocks = (("full", "2021-01-01", "2022-12-31"),)

    unmanaged = build_cost_and_block_diagnostics(
        {"foundation": panels["foundation"]},
        features,
        top_k=2,
        costs_bps=(25.0,),
        blocks=blocks,
    )
    managed = build_cost_and_block_diagnostics(
        {"foundation": panels["foundation"]},
        features,
        top_k=2,
        costs_bps=(25.0,),
        blocks=blocks,
        vol_managed_models=frozenset({"foundation"}),
    )

    assert not np.isclose(
        unmanaged.iloc[0]["strategy_total_return"],
        managed.iloc[0]["strategy_total_return"],
    )
