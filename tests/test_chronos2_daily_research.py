from __future__ import annotations

import numpy as np
import pandas as pd

from src.chronos2_daily_research import _daily_context, _predict_daily


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

    context, last_logs, included = _daily_context(
        frames,
        ["AAA"],
        pd.Timestamp("2026-01-10"),
        "adjclose",
        context_days=5,
        normalized=True,
    )

    assert included == ["AAA"]
    assert context["timestamp"].max() == pd.Timestamp("2026-01-10")
    assert len(context) == 5
    assert np.isclose(context.iloc[-1]["target"], 0.0)
    assert np.isclose(context.iloc[0]["target"], np.log(105.0) - np.log(109.0))
    assert np.isclose(last_logs["AAA"], np.log(109.0))


def test_daily_context_drops_tickers_without_a_full_trailing_window() -> None:
    frames = {"AAA": _daily_frame([100.0, 101.0]), "BBB": _daily_frame([50.0] * 5)}

    context, last_logs, included = _daily_context(
        frames,
        ["AAA", "BBB"],
        pd.Timestamp("2026-01-05"),
        "close",
        context_days=5,
        normalized=False,
    )

    assert included == ["BBB"]
    assert set(context["item_id"]) == {"BBB"}
    assert list(last_logs.index) == ["BBB"]


class _FakeChronos:
    def predict_df(self, context: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        assert len(context) == 4
        assert kwargs["prediction_length"] == 3
        assert kwargs["context_length"] == 2
        assert kwargs["cross_learning"] is True
        assert kwargs["freq"] == "B"
        assert kwargs["quantile_levels"] == [0.1, 0.5, 0.9]
        return pd.DataFrame(
            {
                "item_id": ["AAA", "AAA", "BBB", "BBB"],
                "timestamp": pd.to_datetime(
                    ["2026-01-06", "2026-01-08", "2026-01-06", "2026-01-08"]
                ),
                "0.1": [np.log(104.0), np.log(105.0), np.log(46.0), np.log(45.0)],
                "0.5": [np.log(109.0), np.log(110.0), np.log(49.0), np.log(50.0)],
                "0.9": [np.log(114.0), np.log(115.0), np.log(54.0), np.log(55.0)],
            }
        )


def test_predict_daily_uses_terminal_quantiles_and_converts_log_prices() -> None:
    context = pd.DataFrame(
        {
            "item_id": ["AAA", "AAA", "BBB", "BBB"],
            "timestamp": pd.to_datetime(
                ["2026-01-02", "2026-01-05", "2026-01-02", "2026-01-05"]
            ),
            "target": [0.0, 0.01, 0.0, -0.01],
        }
    )
    last_logs = pd.Series({"AAA": np.log(100.0), "BBB": np.log(50.0)})

    forecasts = _predict_daily(
        _FakeChronos(),
        context,
        last_logs,
        horizon_days=3,
        context_days=2,
        batch_size=10,
        device="cpu",
        cross_learning=True,
        normalized=False,
    )

    assert forecasts.index.tolist() == ["AAA", "BBB"]
    assert np.isclose(forecasts.loc["AAA", "lower10"], 0.05)
    assert np.isclose(forecasts.loc["AAA", "median"], 0.10)
    assert np.isclose(forecasts.loc["BBB", "median"], 0.0)
    assert np.isclose(forecasts.loc["BBB", "upper90"], 0.10)
