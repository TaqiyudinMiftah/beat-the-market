from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.lockbox_research import load_forecast_panel


def test_load_forecast_panel_rejects_duplicate_signal_ticker(tmp_path: Path) -> None:
    path = tmp_path / "forecasts.csv"
    pd.DataFrame(
        [
            {"model": "m", "signal_date": "2026-01-31", "ticker": "AAA", "forecast": 1.0},
            {"model": "m", "signal_date": "2026-01-31", "ticker": "AAA", "forecast": 2.0},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_forecast_panel(path, "m", "2026-08-11")


def test_load_forecast_panel_pivots_and_filters_end(tmp_path: Path) -> None:
    path = tmp_path / "forecasts.csv"
    pd.DataFrame(
        [
            {"model": "m", "signal_date": "2026-01-31", "ticker": "AAA", "forecast": 1.0},
            {"model": "m", "signal_date": "2026-09-30", "ticker": "AAA", "forecast": 2.0},
        ]
    ).to_csv(path, index=False)

    panel = load_forecast_panel(path, "m", "2026-08-11")

    assert panel.index.tolist() == [pd.Timestamp("2026-01-31")]
    assert panel.loc[pd.Timestamp("2026-01-31"), "AAA"] == 1.0
