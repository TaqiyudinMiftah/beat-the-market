from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.chronos2_finetune_research import (
    FineTuneConfig,
    build_training_inputs,
    fit_cutoff_for_signal,
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


def test_annual_refit_cutoff_uses_only_prior_calendar_year() -> None:
    base = dict(
        price_field="adjclose",
        train_end="2021-12-31",
        start="2022-01-01",
        end="2026-08-11",
        cap=300,
        context_days=256,
        horizon_days=21,
        top_k=3,
        cost_bps=25.0,
        min_past=256,
        fine_tune_steps=100,
        batch_size=64,
        learning_rate=1e-6,
        seed=1,
        device="cpu",
        model_id="amazon/chronos-2",
        max_signals=0,
        data_dir=Path("data/raw/yahoo_all"),
        universe_path=Path("data/universe_idx_all_2026-08.csv"),
        checkpoint_dir=Path("/tmp/chronos2-test"),
        output_prefix="test",
    )
    config = FineTuneConfig(refit_annual=True, **base)

    assert fit_cutoff_for_signal(pd.Timestamp("2022-01-31"), config) == "2021-12-31"
    assert fit_cutoff_for_signal(pd.Timestamp("2023-01-31"), config) == "2022-12-31"
    assert fit_cutoff_for_signal(pd.Timestamp("2025-07-31"), config) == "2024-12-31"
