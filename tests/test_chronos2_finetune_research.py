from __future__ import annotations

import numpy as np
import pandas as pd

from src.chronos2_finetune_research import (
    build_training_inputs,
    normalized_log_price,
)


def test_normalized_log_price_uses_last_valid_positive_price() -> None:
    values = normalized_log_price(pd.Series([100.0, np.nan, 200.0]))

    assert np.allclose(values, [np.log(100.0 / 200.0), 0.0])


def test_build_training_inputs_excludes_post_cutoff_and_short_series() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    frames = {
        "LONG": pd.DataFrame(
            {"date": dates, "adjclose": [100, 101, 102, 103, 104]},
        ),
        "SHORT": pd.DataFrame(
            {"date": dates[:2], "adjclose": [100, 101]},
        ),
    }

    inputs, training = build_training_inputs(
        frames,
        ["LONG", "SHORT"],
        "adjclose",
        "2020-01-03",
        minimum_length=3,
    )

    assert len(inputs) == 1
    assert training["ticker"].tolist() == ["LONG"]
    assert training["last_observation"].tolist() == ["2020-01-03"]
    assert len(inputs[0]) == 3
