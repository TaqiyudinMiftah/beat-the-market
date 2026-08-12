from __future__ import annotations

import numpy as np
import pandas as pd

from src.kronos_all_stock_research import _make_context


def _ohlcv_frame(values: list[float]) -> pd.DataFrame:
    close = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(values), freq="D"),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "adjclose": close,
            "volume": np.full(len(values), 1000.0),
        }
    )


def test_make_context_is_cut_at_signal_and_contains_kronos_columns() -> None:
    context = _make_context(
        "AAA",
        _ohlcv_frame([100.0 + value for value in range(10)]),
        pd.Timestamp("2026-01-10"),
        "adjclose",
        context_days=5,
    )

    assert context is not None
    assert context.ticker == "AAA"
    assert context.dates.max() == pd.Timestamp("2026-01-10")
    assert len(context.ohlcv) == 5
    assert list(context.ohlcv.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert np.isclose(context.ohlcv.iloc[-1]["amount"], 109.0 * 1000.0)


def test_make_context_rejects_short_or_invalid_price_history() -> None:
    short = _make_context(
        "AAA",
        _ohlcv_frame([100.0, 101.0]),
        pd.Timestamp("2026-01-02"),
        "close",
        context_days=5,
    )
    invalid_frame = _ohlcv_frame([100.0, 0.0, 102.0, 103.0, 104.0])
    invalid = _make_context(
        "BBB",
        invalid_frame,
        pd.Timestamp("2026-01-05"),
        "close",
        context_days=5,
    )

    assert short is None
    assert invalid is None
