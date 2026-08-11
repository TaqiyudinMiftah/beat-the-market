from __future__ import annotations

import numpy as np
import pandas as pd

from src.all_stock_ml_research import _slice_features, load_prices


def _write_price_file(path, ticker: str, include_volume: bool = True) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "adjclose": [100.0, 101.0],
        }
    )
    if include_volume:
        frame["volume"] = [1_000.0, 1_100.0]
    frame.to_csv(path / f"{ticker}.csv", index=False)


def test_load_prices_handles_missing_volume_and_reports_unavailable_tickers(tmp_path) -> None:
    universe_path = tmp_path / "universe.csv"
    pd.DataFrame({"ticker": ["AAA", "BBB"]}).to_csv(universe_path, index=False)
    _write_price_file(tmp_path, "INDEX_JKSE")
    _write_price_file(tmp_path, "AAA", include_volume=False)

    prices, volumes, loaded, skipped = load_prices(tmp_path, universe_path, "adjclose")

    assert loaded == ["AAA"]
    assert skipped == ["BBB"]
    assert list(prices.columns) == ["^JKSE", "AAA"]
    assert volumes["AAA"].isna().all()


def test_slice_features_preserves_only_requested_dates_after_warmup() -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    features = {
        "frame": pd.DataFrame(np.arange(8).reshape(4, 2), index=dates),
        "series": pd.Series(np.arange(4), index=dates),
    }

    sliced = _slice_features(features, "2020-02-01", "2020-03-31")

    assert sliced["frame"].index.tolist() == list(dates[1:3])
    assert sliced["series"].tolist() == [1, 2]
