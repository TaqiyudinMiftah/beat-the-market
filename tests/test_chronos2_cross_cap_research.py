from __future__ import annotations

import numpy as np
import pandas as pd

from src.chronos2_cross_cap_research import _load_forecast_panel, cross_cap_rank_ensemble


def test_cross_cap_rank_ensemble_averages_available_percentile_ranks() -> None:
    index = pd.to_datetime(["2026-01-31"])
    first = pd.DataFrame([[3.0, 2.0, 1.0]], index=index, columns=["AAA", "BBB", "CCC"])
    second = pd.DataFrame([[3.0, 1.0, 2.0]], index=index, columns=["AAA", "BBB", "CCC"])

    result = cross_cap_rank_ensemble({"cap150": first, "cap300": second})

    assert np.isclose(result.loc[index[0], "AAA"], 1.0)
    assert np.isclose(result.loc[index[0], "BBB"], result.loc[index[0], "CCC"])
    assert result.loc[index[0], "AAA"] > result.loc[index[0], "BBB"]


def test_cross_cap_rank_ensemble_does_not_impute_missing_source_forecasts() -> None:
    index = pd.to_datetime(["2026-01-31"])
    first = pd.DataFrame([[3.0, 1.0]], index=index, columns=["AAA", "BBB"])
    second = pd.DataFrame([[np.nan, 2.0]], index=index, columns=["AAA", "BBB"])

    result = cross_cap_rank_ensemble({"cap150": first, "cap300": second})

    assert np.isclose(result.loc[index[0], "AAA"], 1.0)
    assert np.isclose(result.loc[index[0], "BBB"], 0.75)


def test_load_forecast_panel_filters_model_and_pivots_rows(tmp_path) -> None:
    path = tmp_path / "forecasts.csv"
    pd.DataFrame(
        {
            "model": ["other", "chronos2_daily_abs_cross_lower10"],
            "signal_date": ["2026-01-31", "2026-01-31"],
            "ticker": ["AAA", "BBB"],
            "forecast": [99.0, 0.25],
        }
    ).to_csv(path, index=False)

    result = _load_forecast_panel(path)

    assert result.index.tolist() == [pd.Timestamp("2026-01-31")]
    assert result.columns.tolist() == ["BBB"]
    assert np.isclose(result.iloc[0, 0], 0.25)
