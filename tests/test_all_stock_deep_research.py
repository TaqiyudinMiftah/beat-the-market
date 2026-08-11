from __future__ import annotations

import numpy as np
import pandas as pd

from src.all_stock_deep_research import build_rank_target_panel


def test_build_rank_target_panel_ranks_only_same_signal_date_and_keeps_excess_target() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    columns = ["AAA", "BBB", "CCC"]
    features: dict[str, pd.DataFrame | pd.Series] = {
        "monthly_prices": pd.DataFrame(100.0, index=dates, columns=columns),
        "monthly_returns": pd.DataFrame(
            [[0.10, 0.20, 0.05], [0.01, 0.03, 0.02]],
            index=dates,
            columns=columns,
        ),
        "benchmark_returns": pd.Series([0.02, 0.01], index=dates, name="^JKSE"),
        "liquidity": pd.DataFrame(
            [[300.0, 200.0, 100.0], [300.0, 200.0, 100.0]],
            index=dates,
            columns=columns,
        ),
        "mom12_1": pd.DataFrame(0.1, index=dates, columns=columns),
        "mom6_1": pd.DataFrame(0.1, index=dates, columns=columns),
        "mom3": pd.DataFrame(0.1, index=dates, columns=columns),
        "volatility": pd.DataFrame(0.1, index=dates, columns=columns),
        "trend200": pd.DataFrame(0.1, index=dates, columns=columns),
        "market_trend": pd.Series(0.1, index=dates),
        "market_mom12_1": pd.Series(0.1, index=dates),
        "risk_on": pd.Series(True, index=dates),
    }

    panel, eligibility = build_rank_target_panel(features, cap=2)

    first = panel.loc[panel["signal_date"].eq(dates[0])].set_index("ticker")
    assert eligibility.loc[dates[0]].tolist() == [True, True, False]
    assert np.isclose(first.loc["AAA", "target_rank"], 0.5)
    assert np.isclose(first.loc["BBB", "target_rank"], 1.0)
    assert np.isclose(first.loc["AAA", "target_excess"], 0.0)
    assert "CCC" not in set(first.index)
