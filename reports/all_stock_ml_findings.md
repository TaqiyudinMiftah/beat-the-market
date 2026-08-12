# All-listed Indonesian factor-ML research

Run date: 2026-08-12<br>
Universe catalog: `data/universe_idx_all_2026-08.csv`<br>
Downloaded data directory: `data/raw/yahoo_all`<br>
Price field: `adjclose`; top K: `3`; cost: `25.0` bps one-way<br>
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
| existing_composite | train | 84 | 0.5186 | 1.3348 | -0.3471 | 0.4464 |
| existing_composite | validation | 24 | -0.0792 | 0.1735 | -0.5575 | 0.6111 |
| existing_composite | holdout | 30 | 0.5243 | 0.8069 | -0.6111 | 0.5778 |
| existing_composite | full | 138 | 0.3966 | 0.8833 | -0.6111 | 0.5036 |
| ridge | train | 84 | 0.2689 | 0.7248 | -0.5876 | 0.7222 |
| ridge | validation | 24 | -0.5093 | -1.2706 | -0.7620 | 0.9444 |
| ridge | holdout | 30 | 0.5802 | 0.8802 | -0.5137 | 0.8444 |
| ridge | full | 138 | 0.1390 | 0.5038 | -0.9034 | 0.7874 |
| lightgbm | train | 84 | 0.0448 | 0.4027 | -0.8068 | 0.5595 |
| lightgbm | validation | 24 | 0.4545 | 0.8126 | -0.6951 | 0.7083 |
| lightgbm | holdout | 30 | 2.0807 | 1.1550 | -0.3302 | 0.7111 |
| lightgbm | full | 138 | 0.4136 | 0.7018 | -0.8068 | 0.6184 |
| ensemble | train | 84 | 0.1988 | 0.6175 | -0.7578 | 0.7063 |
| ensemble | validation | 24 | -0.1604 | 0.0587 | -0.4427 | 0.9444 |
| ensemble | holdout | 30 | 0.4133 | 0.5975 | -0.4911 | 0.8889 |
| ensemble | full | 138 | 0.1725 | 0.5022 | -0.7578 | 0.7874 |
| ensemble_volmanaged | train | 84 | 0.1054 | 0.4919 | -0.7393 | 0.6090 |
| ensemble_volmanaged | validation | 24 | -0.1604 | 0.0587 | -0.4427 | 0.9444 |
| ensemble_volmanaged | holdout | 30 | 0.4356 | 0.5817 | -0.4162 | 0.6989 |
| ensemble_volmanaged | full | 138 | 0.1209 | 0.4276 | -0.7393 | 0.6869 |

Validation winner: **lightgbm** under highest validation Sharpe, then excess CAGR. A candidate is considered audit-passing only if it beats the control in validation, remains positive in both holdout blocks at 25 and 100 bps, is positive in at least 60% of rolling 12-month windows, and has a positive 95% block-bootstrap lower CI versus IHSG.

## Fixed audit gate

- **ridge:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; fewer than 60% of trailing 12-month windows have positive excess CAGR; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **lightgbm:** passes the fixed audit gate (all fixed audit checks passed).
- **ensemble:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2025_2026 at 25 bps; excess CAGR is not positive in holdout_2025_2026 at 100 bps; fewer than 60% of trailing 12-month windows have positive excess CAGR; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **ensemble_volmanaged:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2025_2026 at 25 bps; excess CAGR is not positive in holdout_2025_2026 at 100 bps; fewer than 60% of trailing 12-month windows have positive excess CAGR; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).

Preferred research candidate: **lightgbm**. No candidate is promoted automatically because the catalog is a current snapshot: it includes survivorship and index-membership look-ahead, and the historical panel does not contain delisted names or historical membership dates.

## Predeclared top-K sensitivity

The default top K is three. The same forecasts are replayed at top K = 1, 3, 5, and 10, with 25 and 100 bps costs, without refitting or using holdout results to change the model. The compact 25 bps view below compares the control, LightGBM, and its rank ensemble with the fixed volatility rule; the complete grid is in `reports/all_stock_ml_sensitivity.csv`.

| model | top K | validation excess CAGR | holdout excess CAGR |
|---|---:|---:|---:|
| ensemble_volmanaged | 1 | -0.8164 | 0.9431 |
| ensemble_volmanaged | 3 | -0.1604 | 0.4356 |
| ensemble_volmanaged | 5 | -0.3058 | 0.4880 |
| ensemble_volmanaged | 10 | -0.0609 | 0.5538 |
| existing_composite | 1 | -0.2809 | -0.1462 |
| existing_composite | 3 | -0.0792 | 0.5243 |
| existing_composite | 5 | 0.1244 | 1.1008 |
| existing_composite | 10 | 0.1968 | 0.7926 |
| lightgbm | 1 | -0.5374 | 0.0834 |
| lightgbm | 3 | 0.4545 | 2.0807 |
| lightgbm | 5 | 0.4601 | 1.6442 |
| lightgbm | 10 | -0.1052 | 1.6193 |

## Paired block bootstrap

The 2024–2026 bootstrap resamples paired three-month circular blocks 5,000 times. The control comparison is against the all-listed existing composite, not the IDX30 composite.

| model | months | market active annualized mean | market 95% CI | control active annualized mean | control 95% CI |
|---|---:|---:|---|---:|---|
| ridge | 30 | 0.7025 | [-0.1434, 1.6809] | -0.0278 | [-1.3679, 1.2401] |
| lightgbm | 30 | 1.8523 | [0.5451, 3.6810] | 1.1220 | [-0.7153, 3.0758] |
| ensemble | 30 | 0.8333 | [-0.2779, 2.3147] | 0.1030 | [-1.3684, 1.6877] |
| ensemble_volmanaged | 30 | 0.7741 | [-0.2227, 2.2359] | 0.0438 | [-1.3030, 1.5873] |

## Limitations

- A broad current catalog improves cross-sectional sample size but is not a point-in-time universe. A credible deployment study still needs historical IDX membership, delistings, suspensions, corporate actions, and liquidity/execution constraints.
- Yahoo Finance is a convenient research source rather than a licensed exchange feed. Some files have short histories or only one usable row; these symbols naturally drop from feature-ranked signals until enough history exists.
- Small and illiquid stocks can dominate a top-K portfolio. Results must be stress-tested with explicit liquidity, spread, price-limit, and capacity rules before any live use.
- At the default top-three setting, the LightGBM holdout beta is `2.7195` and its maximum drawdown is `-0.3302`. The apparent excess return therefore comes with substantial concentration and market exposure; it is not a low-risk alpha claim.
- No result is investment advice or a guarantee of beating IHSG.

## Reproduction

```bash
python3 src/download_data.py --universe all --output-dir data/raw/yahoo_all   --metadata-path data/raw/yahoo_all_metadata.json --continue-on-error
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.all_stock_ml_research
```

Raw all-stock forecasts are written to ignored `reports/all_stock_ml_forecasts.csv`. Committed tables are in `reports/all_stock_ml_metrics.csv`, `reports/all_stock_ml_robustness.csv`, `reports/all_stock_ml_rolling_summary.csv`, `reports/all_stock_ml_bootstrap.csv`, `reports/all_stock_ml_training.csv`, and `reports/all_stock_ml_sensitivity.csv`.
