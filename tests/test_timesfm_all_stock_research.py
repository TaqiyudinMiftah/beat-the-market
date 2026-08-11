from __future__ import annotations

import numpy as np
import pandas as pd

from src.timesfm_all_stock_research import _daily_context, _predict_timesfm


def _daily_frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(values), freq="D"),
            "close": values,
            "adjclose": values,
        }
    )


def test_daily_context_is_cut_at_signal_and_normalized_at_last_price() -> None:
    frames = {"AAA": _daily_frame([100.0 + value for value in range(10)])}

    tickers, inputs, last_logs = _daily_context(
        frames,
        ["AAA"],
        pd.Timestamp("2026-01-10"),
        "adjclose",
        context_days=5,
        normalized=True,
    )

    assert tickers == ["AAA"]
    assert len(inputs) == 1
    assert len(inputs[0]) == 5
    assert np.isclose(inputs[0][-1], 0.0)
    assert np.isclose(inputs[0][0], np.log(105.0) - np.log(109.0))
    assert np.isclose(last_logs["AAA"], np.log(109.0))


def test_daily_context_drops_tickers_without_a_full_trailing_window() -> None:
    frames = {"AAA": _daily_frame([100.0, 101.0]), "BBB": _daily_frame([50.0] * 5)}

    tickers, inputs, last_logs = _daily_context(
        frames,
        ["AAA", "BBB"],
        pd.Timestamp("2026-01-05"),
        "close",
        context_days=5,
        normalized=False,
    )

    assert tickers == ["BBB"]
    assert len(inputs) == 1
    assert list(last_logs.index) == ["BBB"]


class _FakeTimesFM:
    def forecast(self, *, horizon: int, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        assert horizon == 3
        assert len(inputs) == 2
        point = np.array([[np.log(109.0), np.log(109.0), np.log(110.0)], [0.0, 0.0, 0.0]])
        quantiles = np.zeros((2, 3, 10), dtype=float)
        quantiles[0, -1, 1] = np.log(105.0)
        quantiles[0, -1, 5] = np.log(110.0)
        quantiles[1, -1, 1] = -0.1
        return point, quantiles


def test_predict_timesfm_uses_terminal_point_and_lower10() -> None:
    forecasts = _predict_timesfm(
        _FakeTimesFM(),
        ["AAA", "BBB"],
        [np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)],
        pd.Series({"AAA": np.log(100.0), "BBB": np.log(50.0)}),
        horizon_days=3,
        normalized=False,
    )

    assert forecasts.index.tolist() == ["AAA", "BBB"]
    assert np.isclose(forecasts.loc["AAA", "point"], 0.10)
    assert np.isclose(forecasts.loc["AAA", "lower10"], 0.05)
    assert np.isclose(forecasts.loc["BBB", "lower10"], np.exp(-0.1 - np.log(50.0)) - 1.0)
