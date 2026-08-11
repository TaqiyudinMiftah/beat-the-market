from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import research
from src.strategy_engine import DEFAULT_STRATEGY, default_parity_score, simulate_strategy, validate_config


DATA_READY = (research.DATA_DIR / "INDEX_JKSE.csv").exists()


def test_completed_monthly_last_excludes_partial_month() -> None:
    dates = pd.to_datetime(["2026-06-29", "2026-06-30", "2026-07-31", "2026-08-10"])
    values = pd.Series([100.0, 101.0, 105.0, 106.0], index=dates)

    result = research.completed_monthly_last(values)

    assert result.index.tolist() == [pd.Timestamp("2026-06-30"), pd.Timestamp("2026-07-31")]


@pytest.mark.skipif(not DATA_READY, reason="download the local market snapshot first")
def test_default_strategy_matches_original_research_score() -> None:
    prices, volumes, _ = research.load_prices("adjclose")
    features = research.make_features(prices, volumes)

    original = research.score_frame("composite", features)
    reusable = default_parity_score(features)

    np.testing.assert_allclose(original.to_numpy(), reusable.to_numpy(), equal_nan=True, atol=1e-12)


@pytest.mark.skipif(not DATA_READY, reason="download the local market snapshot first")
def test_strategy_returns_are_lagged_one_completed_month() -> None:
    prices, volumes, _ = research.load_prices("adjclose")
    features = research.make_features(prices, volumes)
    simulation, _, _ = simulate_strategy(features, DEFAULT_STRATEGY)

    latest_signal_date = features["monthly_prices"].index[-1]
    assert pd.isna(simulation.loc[latest_signal_date, "strategy_return"])
    assert pd.isna(simulation.loc[latest_signal_date, "benchmark_return"])


def test_strategy_rejects_unbalanced_weights() -> None:
    invalid = {**DEFAULT_STRATEGY, "factors": [dict(DEFAULT_STRATEGY["factors"][0], weight=0.5)]}

    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_config(invalid)
