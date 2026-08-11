from __future__ import annotations

import numpy as np
import pandas as pd

from src.liquid_rank_ml_research import (
    eligibility_mask,
    filter_panel,
    parse_caps,
    rank_target_panel,
)


def _features() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2024-01-31", periods=2, freq="ME")
    columns = ["AAA", "BBB", "CCC"]
    return {
        "liquidity": pd.DataFrame(
            [[300.0, 200.0, 100.0], [100.0, 300.0, 200.0]],
            index=dates,
            columns=columns,
        ),
        "monthly_prices": pd.DataFrame(
            100.0,
            index=dates,
            columns=columns,
        ),
    }


def test_eligibility_mask_selects_top_liquidity_names_per_month() -> None:
    mask = eligibility_mask(_features(), cap=2)

    assert mask.loc[pd.Timestamp("2024-01-31")].tolist() == [True, True, False]
    assert mask.loc[pd.Timestamp("2024-02-29")].tolist() == [False, True, True]


def test_filter_panel_applies_month_specific_eligibility() -> None:
    features = _features()
    panel = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2024-01-31", "2024-01-31", "2024-02-29"]
            ),
            "ticker": ["AAA", "CCC", "AAA"],
            "target": [0.1, 0.2, 0.3],
        }
    )

    filtered = filter_panel(panel, eligibility_mask(features, cap=2))

    assert filtered["ticker"].tolist() == ["AAA"]
    assert filtered["signal_date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-31"]


def test_rank_target_panel_ranks_only_same_signal_month() -> None:
    panel = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2024-01-31", "2024-01-31", "2024-02-29"]
            ),
            "ticker": ["AAA", "BBB", "AAA"],
            "target": [0.10, 0.30, np.nan],
        }
    )

    ranked = rank_target_panel(panel)

    assert ranked.loc[0, "target"] == 0.5
    assert ranked.loc[1, "target"] == 1.0
    assert pd.isna(ranked.loc[2, "target"])


def test_parse_caps_accepts_all_names_as_zero_and_deduplicates() -> None:
    assert parse_caps("150,300,150,0") == (150, 300, 0)
