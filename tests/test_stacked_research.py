from __future__ import annotations

import numpy as np
import pandas as pd

from src.stacked_research import (
    BASE_MODELS,
    STACK_FEATURE_NAMES,
    _safe_nanmean,
    build_stack_panel,
    rank_blend,
    walk_forward_stacker,
)


def _synthetic_stack_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame | pd.Series]]:
    dates = pd.date_range("2021-01-31", periods=5, freq="ME")
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    panels = {
        model: pd.DataFrame(
            np.tile(np.array([[0.1, 0.2, 0.3, 0.4]]), (len(dates), 1))
            + position * 0.01,
            index=dates,
            columns=tickers,
        )
        for position, model in enumerate(BASE_MODELS)
    }
    returns = pd.DataFrame(
        np.tile(np.array([[0.01, 0.02, -0.01, 0.015]]), (len(dates) + 1, 1)),
        index=pd.date_range("2021-01-31", periods=6, freq="ME"),
        columns=tickers,
    )
    prices = 100.0 * (1.0 + returns).cumprod()
    benchmark = pd.Series(
        np.linspace(-0.02, 0.03, len(returns)),
        index=returns.index,
        name="^JKSE",
    )
    features: dict[str, pd.DataFrame | pd.Series] = {
        "monthly_prices": prices,
        "monthly_returns": returns,
        "benchmark_returns": benchmark,
    }
    return panels, features


def test_safe_nanmean_returns_nan_only_for_all_missing_slice() -> None:
    values = np.array([[1.0, np.nan], [np.nan, np.nan]])
    result = _safe_nanmean(values, axis=1)

    np.testing.assert_allclose(result, np.array([1.0, np.nan]), equal_nan=True)


def test_build_stack_panel_creates_rank_and_next_month_target() -> None:
    panels, features = _synthetic_stack_inputs()

    panel, ensemble, _ = build_stack_panel(panels, features, "2021-01-01", "2021-05-31")

    assert list(panel.columns) == ["ticker", *STACK_FEATURE_NAMES, "target", "target_excess", "signal_date"]
    assert ensemble.shape == (5, 4)
    first = panel.loc[panel["signal_date"].eq(pd.Timestamp("2021-01-31"))]
    np.testing.assert_allclose(first["target"].to_numpy(), np.array([0.01, 0.02, -0.01, 0.015]))


def test_walk_forward_stacker_does_not_use_current_or_future_targets() -> None:
    panels, features = _synthetic_stack_inputs()
    panel, _, _ = build_stack_panel(panels, features, "2021-01-01", "2021-05-31")
    first_run, _ = walk_forward_stacker(panel, "ridge", "target", min_train_months=2)
    altered = panel.copy()
    altered.loc[altered["signal_date"].eq(pd.Timestamp("2021-05-31")), "target"] = 99.0
    second_run, _ = walk_forward_stacker(altered, "ridge", "target", min_train_months=2)

    assert first_run.loc[pd.Timestamp("2021-01-31")].isna().all()
    np.testing.assert_allclose(
        first_run.loc[pd.Timestamp("2021-04-30")].to_numpy(dtype=float),
        second_run.loc[pd.Timestamp("2021-04-30")].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_rank_blend_uses_available_forecasts_without_imputation() -> None:
    index = pd.to_datetime(["2021-01-31"])
    first = pd.DataFrame([[1.0, 2.0, np.nan]], index=index, columns=["A", "B", "C"])
    second = pd.DataFrame([[np.nan, 1.0, 3.0]], index=index, columns=["A", "B", "C"])

    blended = rank_blend(first, second)

    assert np.isfinite(blended.loc[index[0], "A"])
    assert np.isfinite(blended.loc[index[0], "C"])
    assert np.isfinite(blended.loc[index[0], "B"])
