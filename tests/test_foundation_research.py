from __future__ import annotations

import numpy as np
import pandas as pd

from src.foundation_research import _blend_ranks, _make_context


def test_daily_context_stops_at_signal_date() -> None:
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": np.arange(10, dtype=float) + 100,
            "high": np.arange(10, dtype=float) + 101,
            "low": np.arange(10, dtype=float) + 99,
            "close": np.arange(10, dtype=float) + 100,
            "adjclose": np.arange(10, dtype=float) + 100,
            "volume": np.full(10, 1_000.0),
        }
    )
    context = _make_context("TEST", frame, pd.Timestamp("2026-01-08"), "adjclose", 4)

    assert context is not None
    assert context.dates.max() == pd.Timestamp("2026-01-08")
    assert context.dates.min() == pd.Timestamp("2026-01-05")
    assert len(context.ohlcv) == 4


def test_rank_blend_preserves_cross_sectional_order() -> None:
    index = pd.to_datetime(["2026-01-31"])
    first = pd.DataFrame([[1.0, 2.0, 3.0]], index=index, columns=["A", "B", "C"])
    second = pd.DataFrame([[3.0, 1.0, 2.0]], index=index, columns=["A", "B", "C"])

    blended = _blend_ranks(first, second)

    assert np.isclose(blended.loc[index[0], "A"], 2 / 3)
    assert np.isclose(blended.loc[index[0], "B"], 1 / 2)
    assert np.isclose(blended.loc[index[0], "C"], 5 / 6)
