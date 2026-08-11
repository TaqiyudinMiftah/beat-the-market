import numpy as np
import pandas as pd

from src.chronos2_research import build_context_frame, log_forecast_to_return


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-31", periods=5, freq="ME")
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 103.0, 104.0],
            "BBB": [50.0, 51.0, 52.0, 53.0, 54.0],
            "CCC": [20.0, 21.0, np.nan, 23.0, 24.0],
        },
        index=dates,
    )
    eligibility = pd.DataFrame(
        {
            "AAA": [True] * 5,
            "BBB": [True] * 5,
            "CCC": [True, True, True, False, False],
        },
        index=dates,
    )
    return prices, eligibility


def test_context_uses_only_complete_eligible_histories_through_signal() -> None:
    prices, eligibility = _fixture()

    frame, tickers = build_context_frame(
        prices,
        eligibility,
        pd.Timestamp("2024-05-31"),
        context_months=3,
    )

    assert tickers == ["AAA", "BBB"]
    assert frame["item_id"].unique().tolist() == ["AAA", "BBB"]
    assert frame["timestamp"].max() == pd.Timestamp("2024-05-31")
    assert frame.groupby("item_id").size().tolist() == [3, 3]


def test_context_drops_stock_with_missing_value_inside_trailing_window() -> None:
    prices, eligibility = _fixture()

    frame, tickers = build_context_frame(
        prices,
        eligibility,
        pd.Timestamp("2024-04-30"),
        context_months=3,
    )

    assert tickers == ["AAA", "BBB"]
    assert "CCC" not in set(frame["item_id"])


def test_log_forecast_conversion_is_bounded_and_aligned() -> None:
    forecast = pd.Series({"AAA": np.log(110.0), "BBB": np.log(45.0)})
    last = pd.Series({"AAA": np.log(100.0), "BBB": np.log(50.0), "CCC": np.log(1.0)})

    returns = log_forecast_to_return(forecast, last)

    assert returns.index.tolist() == ["AAA", "BBB"]
    assert np.isclose(returns["AAA"], 0.10)
    assert np.isclose(returns["BBB"], -0.10)
