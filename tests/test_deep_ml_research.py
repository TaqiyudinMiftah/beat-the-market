from __future__ import annotations

import numpy as np
import pandas as pd

from src.deep_ml_research import _weighted_rank_panel, online_weighted_predictions


def _synthetic_features() -> dict[str, pd.DataFrame | pd.Series]:
    index = pd.date_range("2021-01-31", periods=18, freq="ME")
    columns = ["AAA", "BBB", "CCC", "DDD"]
    returns = pd.DataFrame(
        np.tile(np.array([[0.02, 0.01, -0.01, 0.015]]), (len(index), 1)),
        index=index,
        columns=columns,
    )
    prices = 100.0 * (1.0 + returns).cumprod()
    benchmark = pd.Series(np.linspace(-0.02, 0.03, len(index)), index=index, name="^JKSE")
    return {
        "monthly_prices": prices,
        "monthly_returns": returns,
        "benchmark_returns": benchmark,
    }


def test_weighted_rank_panel_handles_missing_mlp_forecasts() -> None:
    index = pd.to_datetime(["2021-01-31"])
    baseline = pd.DataFrame([[1.0, 2.0, 3.0]], index=index, columns=["A", "B", "C"])
    mlp = pd.DataFrame([[np.nan, 3.0, 1.0]], index=index, columns=["A", "B", "C"])

    blended = _weighted_rank_panel(baseline, mlp, 0.5)

    assert np.isclose(blended.loc[index[0], "A"], 1.0 / 3.0)
    assert np.isclose(blended.loc[index[0], "C"], 0.75)
    assert blended.loc[index[0], "B"] > blended.loc[index[0], "C"]


def test_online_weighting_has_no_model_weight_before_history_exists() -> None:
    features = _synthetic_features()
    index = features["monthly_prices"].index
    columns = features["monthly_prices"].columns
    baseline = pd.DataFrame(
        np.tile(np.array([[0.1, 0.2, 0.3, 0.4]]), (len(index), 1)),
        index=index,
        columns=columns,
    )
    mlp = baseline.iloc[:, ::-1].copy()

    _, weights = online_weighted_predictions(
        baseline,
        mlp,
        features,
        top_k=2,
        cost_bps=25.0,
        rule="soft",
    )

    assert (weights.iloc[:6] == 0.0).all()
    assert weights.index.equals(index)
