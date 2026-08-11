import numpy as np
import pandas as pd

from src.paper_factor_research import paper_score, rolling_beta


def test_rolling_beta_uses_trailing_observations():
    market = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="market")
    stocks = pd.DataFrame({"AAA": market * 2.0})

    beta = rolling_beta(stocks, market, window=3, min_periods=3)

    assert beta.iloc[:2].isna().all().all()
    np.testing.assert_allclose(beta.iloc[2:, 0].to_numpy(), np.full(3, 2.0))


def test_paper_score_ranks_low_beta_and_keeps_screen():
    index = pd.date_range("2024-01-31", periods=1, freq="ME")
    columns = ["AAA", "BBB", "CCC"]
    eligible = pd.DataFrame([[True, True, False]], index=index, columns=columns)
    features = {
        "mom12_1": pd.DataFrame([[0.2, 0.1, 100.0]], index=index, columns=columns),
        "volatility": pd.DataFrame([[0.1, 0.2, 0.0]], index=index, columns=columns),
        "beta_252d": pd.DataFrame([[0.8, 0.4, 0.01]], index=index, columns=columns),
    }

    score = paper_score(features, eligible)

    assert np.isnan(score.loc[index[0], "CCC"])
    assert score.loc[index[0], "AAA"] > score.loc[index[0], "BBB"]
