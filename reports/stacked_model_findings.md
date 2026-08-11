# Forecast-stacking research

Run date: 2026-08-12<br>
Research universe: `data/universe_idx30_2026-08.csv`<br>
Signal convention: first-stage forecasts and ranks use information available through completed month-end `t`; the stacker predicts and trades the next completed month's return<br>
Price field: `adjclose`; top K: `3`; cost: `25.0` bps one-way<br>
Common first-stage coverage: `2021-01-31` through `2026-06-30`<br>
Stacker evaluation starts after the `12`-signal-month expanding-window warm-up.

## Research basis

- Gu, Kelly, and Xiu compare regularized linear models, trees, and neural networks for cross-sectional asset-pricing prediction and emphasize nonlinear interactions among momentum, liquidity, and volatility: [NBER Working Paper 25398](https://www.nber.org/papers/w25398).
- Campbell and Thompson stress that return predictors must be judged out of sample and can fail for long periods even when they have in-sample relationships: [NBER Working Paper 11468](https://www.nber.org/papers/w11468).

This is a stacking extension of that cross-sectional setup, not a literal reproduction of either paper. The papers motivate testing nonlinear interactions and strict out-of-sample evaluation; they do not validate these pretrained models or this Indonesian universe.

## First-stage inputs

- `existing_composite`: `66` signal months × `30` tickers; `186` missing symbol-month rows retained
- `chronos`: `66` signal months × `30` tickers; `190` missing symbol-month rows retained
- `timesfm`: `66` signal months × `30` tickers; `190` missing symbol-month rows retained
- `kronos`: `66` signal months × `30` tickers; `190` missing symbol-month rows retained
- `mlp_raw`: `66` signal months × `30` tickers; `186` missing symbol-month rows retained
- `mlp_excess`: `66` signal months × `30` tickers; `186` missing symbol-month rows retained

The stacker uses only primitive rank inputs: existing composite, Chronos, TimesFM, Kronos, raw-return factor MLP, and excess-return factor MLP. Each rank is computed across stocks at the same signal date. Two fixed summary inputs are added: mean rank (agreement) and cross-model rank dispersion (disagreement). Derived foundation/deep ensemble outputs are excluded to avoid duplicate information.

## Leakage controls and fixed models

- Every forecast month `t` is fit using rows with `signal_date < t`; the current month's target is never in its training set. Learned stackers use complete feature rows only; missing model forecasts are not imputed.
- Ridge uses training-only standardization, 1%/99% training-target winsorization, and fixed alpha `10`.
- LightGBM is a shallow fixed-round model: 7 leaves, depth 3, minimum leaf size 30, L1/L2 regularization, fixed seeds, and `100` boosting rounds. Its API follows the official `lgb.Dataset`/`lgb.train` interface.
- `forecast_rank_ensemble` is the equal average of the six base ranks. `stack_ridge_lgbm_50` averages the Ridge and LightGBM rank forecasts; `stack_base_ridge_50` averages the base-rank ensemble and Ridge rank forecast. These are fixed 50/50 combinations; no holdout-selected weight is used.
- Validation is 2022–2023. The 2024 and 2025–2026 blocks are holdouts and are not used to choose a model or parameter.

## Results

| model | period | months | excess_cagr | strategy_sharpe_rf0 | strategy_max_drawdown | average_monthly_turnover |
|---|---|---|---|---|---|---|
| existing_composite | train_2021 | 0 | — | — | — | — |
| existing_composite | validation_2022_2023 | 24 | 0.1887 | 0.8313 | -0.2245 | 0.3194 |
| existing_composite | holdout_2024 | 12 | -0.0868 | -0.4916 | -0.1955 | 0.3611 |
| existing_composite | holdout_2025_2026 | 18 | 0.4403 | 0.8702 | -0.3684 | 0.3333 |
| existing_composite | full | 54 | 0.1996 | 0.6582 | -0.3684 | 0.3333 |
| forecast_rank_ensemble | train_2021 | 0 | — | — | — | — |
| forecast_rank_ensemble | validation_2022_2023 | 24 | -0.0831 | -0.0792 | -0.2509 | 0.7361 |
| forecast_rank_ensemble | holdout_2024 | 12 | -0.1309 | -0.7857 | -0.1759 | 0.7500 |
| forecast_rank_ensemble | holdout_2025_2026 | 18 | 0.3645 | 0.7073 | -0.4215 | 0.8148 |
| forecast_rank_ensemble | full | 54 | 0.0434 | 0.2472 | -0.4215 | 0.7654 |
| stack_ridge_raw | train_2021 | 0 | — | — | — | — |
| stack_ridge_raw | validation_2022_2023 | 24 | -0.0573 | 0.1081 | -0.3576 | 0.6806 |
| stack_ridge_raw | holdout_2024 | 12 | 0.5458 | 1.7451 | -0.0798 | 0.7778 |
| stack_ridge_raw | holdout_2025_2026 | 18 | -0.0162 | -0.0230 | -0.5274 | 0.7407 |
| stack_ridge_raw | full | 54 | 0.0681 | 0.3227 | -0.5274 | 0.7222 |
| stack_ridge_excess | train_2021 | 0 | — | — | — | — |
| stack_ridge_excess | validation_2022_2023 | 24 | -0.0085 | 0.2561 | -0.3129 | 0.6806 |
| stack_ridge_excess | holdout_2024 | 12 | 0.6195 | 1.8901 | -0.0798 | 0.7778 |
| stack_ridge_excess | holdout_2025_2026 | 18 | -0.0217 | -0.0477 | -0.5389 | 0.7222 |
| stack_ridge_excess | full | 54 | 0.1001 | 0.4062 | -0.5389 | 0.7160 |
| stack_lightgbm_raw | train_2021 | 0 | — | — | — | — |
| stack_lightgbm_raw | validation_2022_2023 | 24 | 0.0175 | 0.3546 | -0.2195 | 0.8056 |
| stack_lightgbm_raw | holdout_2024 | 12 | -0.1332 | -0.6888 | -0.2373 | 0.8333 |
| stack_lightgbm_raw | holdout_2025_2026 | 18 | 0.6376 | 1.1338 | -0.3500 | 0.6852 |
| stack_lightgbm_raw | full | 54 | 0.1610 | 0.5632 | -0.3934 | 0.7716 |
| stack_lightgbm_excess | train_2021 | 0 | — | — | — | — |
| stack_lightgbm_excess | validation_2022_2023 | 24 | -0.0000 | 0.2843 | -0.2582 | 0.7778 |
| stack_lightgbm_excess | holdout_2024 | 12 | -0.1567 | -0.7272 | -0.2252 | 0.7778 |
| stack_lightgbm_excess | holdout_2025_2026 | 18 | 0.6940 | 1.1482 | -0.4097 | 0.7037 |
| stack_lightgbm_excess | full | 54 | 0.1590 | 0.5423 | -0.4554 | 0.7531 |
| stack_ridge_lgbm_50 | train_2021 | 0 | — | — | — | — |
| stack_ridge_lgbm_50 | validation_2022_2023 | 24 | -0.0071 | 0.2584 | -0.2507 | 0.7500 |
| stack_ridge_lgbm_50 | holdout_2024 | 12 | 0.1155 | 0.5743 | -0.1342 | 0.6944 |
| stack_ridge_lgbm_50 | holdout_2025_2026 | 18 | 0.5192 | 0.9892 | -0.2853 | 0.7963 |
| stack_ridge_lgbm_50 | full | 54 | 0.1842 | 0.6370 | -0.3110 | 0.7531 |
| stack_base_ridge_50 | train_2021 | 0 | — | — | — | — |
| stack_base_ridge_50 | validation_2022_2023 | 24 | 0.2334 | 0.8902 | -0.2346 | 0.7222 |
| stack_base_ridge_50 | holdout_2024 | 12 | 0.1022 | 0.6099 | -0.1097 | 0.7222 |
| stack_base_ridge_50 | holdout_2025_2026 | 18 | 0.5792 | 1.0054 | -0.2891 | 0.7593 |
| stack_base_ridge_50 | full | 54 | 0.3122 | 0.8673 | -0.2891 | 0.7346 |

The validation winner among stacker candidates was **stack_base_ridge_50** under the pre-declared rule of highest validation Sharpe, then excess CAGR. The existing composite remains the control. A candidate is only called historically audit-passing if it beats the control in validation, has positive excess CAGR in both fixed holdout blocks at both 25 and 100 bps, is positive in at least 60% of rolling 12-month windows, and has a positive 95% circular three-month block-bootstrap lower confidence bound versus IHSG.

The 2021 rows are the stacker's expanding-window warm-up, so learned candidates have no traded evaluation months there. The comparable evaluation begins in January 2022 after 12 prior signal months.

## Fixed audit gate

- **forecast_rank_ensemble:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2024 at 25 bps; excess CAGR is not positive in holdout_2024 at 100 bps; fewer than 60% of trailing 12-month windows have positive excess CAGR; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **stack_ridge_raw:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2025_2026 at 25 bps; excess CAGR is not positive in holdout_2025_2026 at 100 bps; fewer than 60% of trailing 12-month windows have positive excess CAGR; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **stack_ridge_excess:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2025_2026 at 25 bps; excess CAGR is not positive in holdout_2025_2026 at 100 bps; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **stack_lightgbm_raw:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2024 at 25 bps; excess CAGR is not positive in holdout_2024 at 100 bps; fewer than 60% of trailing 12-month windows have positive excess CAGR; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **stack_lightgbm_excess:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2024 at 25 bps; excess CAGR is not positive in holdout_2024 at 100 bps; fewer than 60% of trailing 12-month windows have positive excess CAGR; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **stack_ridge_lgbm_50:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; fewer than 60% of trailing 12-month windows have positive excess CAGR).
- **stack_base_ridge_50:** passes the fixed audit gate (all fixed audit checks passed).

Preferred causal research candidate: **stack_base_ridge_50**. This preference is based on the validation winner surviving the pre-declared audit, not on selecting the best holdout return. Even a passing candidate is not promoted to the web app or live trading because the current IDX30 history has survivorship/index-membership bias and the paired control interval can still include zero.

## Paired block bootstrap

The 2024–2026 comparison resamples paired three-month circular blocks 5,000 times. Market columns compare a candidate with IHSG; control columns compare it with the existing composite. These are uncertainty diagnostics for this sample, not a statistical guarantee under a new universe or regime.

| model | months | market_active_annualized_mean | market_ci | market_positive_probability | control_active_annualized_mean | control_ci |
|---|---|---|---|---|---|---|
| forecast_rank_ensemble | 30 | 0.2081 | [-0.0781, 0.5481] | 0.9100 | -0.0393 | [-0.3753, 0.2756] |
| stack_ridge_raw | 30 | 0.2176 | [-0.1091, 0.5165] | 0.9030 | -0.0299 | [-0.5482, 0.4699] |
| stack_ridge_excess | 30 | 0.2314 | [-0.0928, 0.5330] | 0.9236 | -0.0161 | [-0.5244, 0.4802] |
| stack_lightgbm_raw | 30 | 0.3180 | [-0.0322, 0.6717] | 0.9620 | 0.0706 | [-0.2493, 0.4209] |
| stack_lightgbm_excess | 30 | 0.3448 | [-0.0332, 0.7535] | 0.9596 | 0.0973 | [-0.2299, 0.4700] |
| stack_ridge_lgbm_50 | 30 | 0.3662 | [0.0827, 0.6567] | 0.9944 | 0.1187 | [-0.2649, 0.5129] |
| stack_base_ridge_50 | 30 | 0.3945 | [0.1081, 0.7116] | 0.9982 | 0.1470 | [-0.2417, 0.5313] |

## Stability and costs

The full cost/block table is in `reports/stacked_model_robustness.csv`; rolling windows are in `reports/stacked_model_rolling.csv` and `reports/stacked_model_rolling_summary.csv`.

| model | cost_bps | windows | median_sharpe | positive_excess_fraction | worst_excess_cagr | median_turnover |
|---|---|---|---|---|---|---|
| existing_composite | 25.0 | 43 | 0.5716 | 0.6047 | -0.3344 | 0.3056 |
| forecast_rank_ensemble | 25.0 | 43 | -0.0688 | 0.3488 | -0.2127 | 0.7500 |
| stack_base_ridge_50 | 25.0 | 43 | 0.8986 | 0.9302 | -0.0633 | 0.7500 |
| stack_lightgbm_excess | 25.0 | 43 | 0.1789 | 0.5349 | -0.3966 | 0.7500 |
| stack_lightgbm_raw | 25.0 | 43 | 0.1253 | 0.5581 | -0.3223 | 0.7778 |
| stack_ridge_excess | 25.0 | 43 | 0.9248 | 0.6744 | -0.2372 | 0.7222 |
| stack_ridge_lgbm_50 | 25.0 | 43 | 0.2832 | 0.5581 | -0.2886 | 0.7500 |
| stack_ridge_raw | 25.0 | 43 | 0.6951 | 0.5581 | -0.2865 | 0.7222 |

## Runtime and limitations

LightGBM status: `True`; version: `4.7.0`.<br>
Errors/skips: `{}`.

- The first-stage files were produced by earlier rolling runs; rerun them when changing model IDs, context length, price field, or data dates.
- The 30-name panel is a current-universe snapshot, so historical constituents and delisted stocks are missing. The all-listed catalog in the app is not yet a point-in-time research panel.
- Yahoo Finance data omit or simplify exchange-specific limits, suspensions, taxes, spreads, market impact, and execution timing.
- A model that passes this fixed historical gate may still be overfit to the finite panel. It is not investment advice and does not guarantee beating IHSG.

## Reproduction

```bash
python3 -m src.foundation_research --models all
PYTHONPATH=$PWD /tmp/beat-market-deep/bin/python -m src.deep_ml_research
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.stacked_research
```

The raw stack forecasts are written locally to ignored `reports/stacked_model_forecasts.csv`. Committed audit artifacts are `reports/stacked_model_findings.md`, `reports/stacked_model_metrics.csv`, `reports/stacked_model_robustness.csv`, `reports/stacked_model_rolling_summary.csv`, `reports/stacked_model_bootstrap.csv`, and `reports/stacked_model_training.csv`.
