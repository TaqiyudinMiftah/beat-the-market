# All-listed Indonesian factor-ML research

Run date: 2026-08-12<br>
Universe catalog: `data/universe_idx_all_2026-08.csv`<br>
Downloaded data directory: `data/raw/yahoo_all`<br>
Price field: `close`; top K: `3`; cost: `25.0` bps one-way<br>
Loaded current-catalog tickers: `962`; skipped/unavailable: `0`<br>

## Research basis

[Gu, Kelly, and Xiu](https://www.nber.org/papers/w25398) compare linear, tree, and neural-network methods for cross-sectional return prediction and emphasize nonlinear interactions among momentum, liquidity, and volatility. This run applies their model-comparison idea to a broader Indonesian panel. It is not a literal replication and does not solve the current-universe survivorship problem.

## Design and leakage controls

- Daily OHLCV is loaded for the current IDX catalog and transformed into the repository's monthly momentum, volatility, trend, liquidity, and market-regime features.
- A signal at completed month-end `t` predicts and trades the next completed month's return.
- Ridge and LightGBM use expanding walk-forward training. Training rows are strictly earlier than the signal month; normalization and target winsorization are training-only.
- Ridge alpha is fixed at `10`. LightGBM uses the existing shallow fixed specification and is not tuned on the holdout.
- The fixed candidate set is the existing composite, Ridge, LightGBM, equal rank ensemble, and the equal ensemble with the fixed 10% inverse-volatility exposure rule.
- Validation is 2022–2023. 2024 and 2025–2026 are holdouts. Costs are replayed at 0, 25, 50, and 100 bps.

## Results

| model | period | months | excess_cagr | strategy_sharpe_rf0 | strategy_max_drawdown | average_monthly_turnover |
|---|---|---|---|---|---|---|
| existing_composite | train | 84 | 0.4162 | 1.1584 | -0.3927 | 0.4345 |
| existing_composite | validation | 24 | 0.1281 | 0.5485 | -0.5262 | 0.5417 |
| existing_composite | holdout | 30 | 0.4913 | 0.7799 | -0.6111 | 0.5667 |
| existing_composite | full | 138 | 0.3789 | 0.8580 | -0.6111 | 0.4819 |
| ridge | train | 84 | 0.1587 | 0.5707 | -0.6562 | 0.7103 |
| ridge | validation | 24 | -0.5361 | -1.2995 | -0.7852 | 0.9306 |
| ridge | holdout | 30 | 0.3998 | 0.7083 | -0.5091 | 0.8556 |
| ridge | full | 138 | 0.0395 | 0.3517 | -0.9296 | 0.7802 |
| lightgbm | train | 84 | 0.1639 | 0.5806 | -0.7615 | 0.5437 |
| lightgbm | validation | 24 | 0.1330 | 0.5223 | -0.6009 | 0.6250 |
| lightgbm | holdout | 30 | 0.2345 | 0.5766 | -0.8747 | 0.8778 |
| lightgbm | full | 138 | 0.1745 | 0.5078 | -0.8747 | 0.6304 |
| ensemble | train | 84 | 0.0931 | 0.4655 | -0.6932 | 0.6984 |
| ensemble | validation | 24 | -0.2781 | -0.0502 | -0.6181 | 0.9722 |
| ensemble | holdout | 30 | 0.0781 | 0.3790 | -0.6197 | 0.8667 |
| ensemble | full | 138 | 0.0162 | 0.3359 | -0.7291 | 0.7826 |
| ensemble_volmanaged | train | 84 | 0.0392 | 0.3668 | -0.6744 | 0.6033 |
| ensemble_volmanaged | validation | 24 | -0.2781 | -0.0502 | -0.6181 | 0.9722 |
| ensemble_volmanaged | holdout | 30 | 0.1655 | 0.4090 | -0.5191 | 0.6966 |
| ensemble_volmanaged | full | 138 | 0.0039 | 0.2939 | -0.7876 | 0.6877 |

Validation winner: **lightgbm** under highest validation Sharpe, then excess CAGR. A candidate is considered audit-passing only if it beats the control in validation, remains positive in both holdout blocks at 25 and 100 bps, is positive in at least 60% of rolling 12-month windows, and has a positive 95% block-bootstrap lower CI versus IHSG.

## Fixed audit gate

- **ridge:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; fewer than 60% of trailing 12-month windows have positive excess CAGR; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **lightgbm:** rejects the fixed audit gate (excess CAGR is not positive in holdout_2025_2026 at 25 bps; excess CAGR is not positive in holdout_2025_2026 at 100 bps; fewer than 60% of trailing 12-month windows have positive excess CAGR; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **ensemble:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2025_2026 at 25 bps; excess CAGR is not positive in holdout_2025_2026 at 100 bps; fewer than 60% of trailing 12-month windows have positive excess CAGR; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **ensemble_volmanaged:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2025_2026 at 25 bps; excess CAGR is not positive in holdout_2025_2026 at 100 bps; fewer than 60% of trailing 12-month windows have positive excess CAGR; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).

Preferred research candidate: **none**. No candidate is promoted automatically because the catalog is a current snapshot: it includes survivorship and index-membership look-ahead, and the historical panel does not contain delisted names or historical membership dates.

## Predeclared top-K sensitivity

The default top K is three. The same forecasts are replayed at top K = 1, 3, 5, and 10, with 25 and 100 bps costs, without refitting or using holdout results to change the model. The compact 25 bps view below compares the control, LightGBM, and its rank ensemble with the fixed volatility rule; the complete grid is in `reports/all_stock_ml_sensitivity.csv`.

| model | top K | validation excess CAGR | holdout excess CAGR |
|---|---:|---:|---:|
| ensemble_volmanaged | 1 | -0.7735 | -0.2252 |
| ensemble_volmanaged | 3 | -0.2781 | 0.1655 |
| ensemble_volmanaged | 5 | -0.0541 | 0.2619 |
| ensemble_volmanaged | 10 | -0.0115 | 0.8486 |
| existing_composite | 1 | -0.2962 | -0.1513 |
| existing_composite | 3 | 0.1281 | 0.4913 |
| existing_composite | 5 | 0.1868 | 0.5634 |
| existing_composite | 10 | 0.2341 | 0.9529 |
| lightgbm | 1 | -0.0858 | -0.7227 |
| lightgbm | 3 | 0.1330 | 0.2345 |
| lightgbm | 5 | -0.2647 | 1.2021 |
| lightgbm | 10 | -0.1393 | 1.7857 |

## Paired block bootstrap

The 2024–2026 bootstrap resamples paired three-month circular blocks 5,000 times. The control comparison is against the all-listed existing composite, not the IDX30 composite.

| model | months | market active annualized mean | market 95% CI | control active annualized mean | control 95% CI |
|---|---:|---:|---|---:|---|
| ridge | 30 | 0.5987 | [-0.2600, 1.5649] | -0.1122 | [-1.3719, 1.0558] |
| lightgbm | 30 | 0.9971 | [-0.7410, 3.3714] | 0.2862 | [-1.6456, 2.6450] |
| ensemble | 30 | 0.5273 | [-0.4947, 1.8959] | -0.1835 | [-1.4718, 1.2221] |
| ensemble_volmanaged | 30 | 0.5404 | [-0.3785, 1.8881] | -0.1705 | [-1.3832, 1.2168] |

## Limitations

- A broad current catalog improves cross-sectional sample size but is not a point-in-time universe. A credible deployment study still needs historical IDX membership, delistings, suspensions, corporate actions, and liquidity/execution constraints.
- Yahoo Finance is a convenient research source rather than a licensed exchange feed. Some files have short histories or only one usable row; these symbols naturally drop from feature-ranked signals until enough history exists.
- Small and illiquid stocks can dominate a top-K portfolio. Results must be stress-tested with explicit liquidity, spread, price-limit, and capacity rules before any live use.
- At the default top-three setting, the LightGBM holdout beta is `2.6652` and its maximum drawdown is `-0.8747`. The apparent excess return therefore comes with substantial concentration and market exposure; it is not a low-risk alpha claim.
- No result is investment advice or a guarantee of beating IHSG.

## Reproduction

```bash
python3 src/download_data.py --universe all --output-dir data/raw/yahoo_all   --metadata-path data/raw/yahoo_all_metadata.json --continue-on-error
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.all_stock_ml_research
```

Raw all-stock forecasts are written to ignored `reports/all_stock_ml_close_forecasts.csv`. Committed tables use the same `all_stock_ml_close_` prefix.
