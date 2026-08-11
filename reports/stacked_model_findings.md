# Forecast-stacking research

Run date: 2026-08-12<br>
Research universe: `data/universe_idx30_2026-08.csv`<br>
Signal convention: first-stage forecasts and ranks use information available through completed month-end `t`; the stacker predicts and trades the next completed month's return<br>
Price field: `adjclose`; top K: `3`; cost: `25.0` bps one-way<br>
Common first-stage coverage: `2018-01-31` through `2026-06-30`<br>
Stacker evaluation starts after the `24`-signal-month expanding-window warm-up.

## Research basis

- Gu, Kelly, and Xiu compare regularized linear models, trees, and neural networks for cross-sectional asset-pricing prediction and emphasize nonlinear interactions among momentum, liquidity, and volatility: [NBER Working Paper 25398](https://www.nber.org/papers/w25398).
- Campbell and Thompson stress that return predictors must be judged out of sample and can fail for long periods even when they have in-sample relationships: [NBER Working Paper 11468](https://www.nber.org/papers/w11468).

This is a stacking extension of that cross-sectional setup, not a literal reproduction of either paper. The papers motivate testing nonlinear interactions and strict out-of-sample evaluation; they do not validate these pretrained models or this Indonesian universe.

## First-stage inputs

- `existing_composite`: `102` signal months × `30` tickers; `366` missing symbol-month rows retained
- `chronos`: `102` signal months × `30` tickers; `370` missing symbol-month rows retained
- `timesfm`: `102` signal months × `30` tickers; `370` missing symbol-month rows retained
- `kronos`: `102` signal months × `30` tickers; `370` missing symbol-month rows retained
- `mlp_raw`: `102` signal months × `30` tickers; `366` missing symbol-month rows retained
- `mlp_excess`: `102` signal months × `30` tickers; `366` missing symbol-month rows retained

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
| existing_composite | train_2021 | 12 | 0.1608 | 0.9632 | -0.1137 | 0.4444 |
| existing_composite | validation_2022_2023 | 24 | 0.1896 | 0.8336 | -0.2245 | 0.2917 |
| existing_composite | holdout_2024 | 12 | -0.0868 | -0.4916 | -0.1955 | 0.3611 |
| existing_composite | holdout_2025_2026 | 18 | 0.4403 | 0.8702 | -0.3684 | 0.3333 |
| existing_composite | full | 78 | 0.2226 | 0.7031 | -0.3684 | 0.3675 |
| forecast_rank_ensemble | train_2021 | 12 | -0.1234 | 0.1650 | -0.2329 | 0.4722 |
| forecast_rank_ensemble | validation_2022_2023 | 24 | -0.0828 | -0.0773 | -0.2509 | 0.7222 |
| forecast_rank_ensemble | holdout_2024 | 12 | -0.1309 | -0.7857 | -0.1759 | 0.7500 |
| forecast_rank_ensemble | holdout_2025_2026 | 18 | 0.3645 | 0.7073 | -0.4215 | 0.8148 |
| forecast_rank_ensemble | full | 78 | 0.1141 | 0.4908 | -0.4215 | 0.7051 |
| stack_ridge_raw | train_2021 | 12 | -0.0144 | 0.5515 | -0.0911 | 0.6111 |
| stack_ridge_raw | validation_2022_2023 | 24 | -0.1020 | -0.0236 | -0.4355 | 0.6528 |
| stack_ridge_raw | holdout_2024 | 12 | 0.1039 | 0.5030 | -0.1036 | 0.5278 |
| stack_ridge_raw | holdout_2025_2026 | 18 | 0.6512 | 1.3280 | -0.3045 | 0.7407 |
| stack_ridge_raw | full | 78 | 0.1252 | 0.5388 | -0.4355 | 0.6496 |
| stack_ridge_excess | train_2021 | 12 | -0.0825 | 0.2956 | -0.1714 | 0.5278 |
| stack_ridge_excess | validation_2022_2023 | 24 | -0.1151 | -0.0508 | -0.4739 | 0.6528 |
| stack_ridge_excess | holdout_2024 | 12 | 0.1039 | 0.5030 | -0.1036 | 0.5278 |
| stack_ridge_excess | holdout_2025_2026 | 18 | 0.2513 | 0.5697 | -0.4220 | 0.7593 |
| stack_ridge_excess | full | 78 | 0.0271 | 0.2652 | -0.4739 | 0.6410 |
| stack_lightgbm_raw | train_2021 | 12 | 0.2219 | 1.0805 | -0.1297 | 0.6111 |
| stack_lightgbm_raw | validation_2022_2023 | 24 | 0.1147 | 0.7556 | -0.1559 | 0.8194 |
| stack_lightgbm_raw | holdout_2024 | 12 | 0.0383 | 0.2190 | -0.1210 | 0.7222 |
| stack_lightgbm_raw | holdout_2025_2026 | 18 | 0.3666 | 0.7214 | -0.4325 | 0.6852 |
| stack_lightgbm_raw | full | 78 | 0.2027 | 0.7197 | -0.4325 | 0.7479 |
| stack_lightgbm_excess | train_2021 | 12 | 0.3643 | 1.4016 | -0.1345 | 0.6944 |
| stack_lightgbm_excess | validation_2022_2023 | 24 | 0.3396 | 1.4896 | -0.1018 | 0.8194 |
| stack_lightgbm_excess | holdout_2024 | 12 | -0.0060 | -0.0206 | -0.1612 | 0.7500 |
| stack_lightgbm_excess | holdout_2025_2026 | 18 | 0.3031 | 0.6247 | -0.3943 | 0.6667 |
| stack_lightgbm_excess | full | 78 | 0.2321 | 0.7833 | -0.3943 | 0.7479 |
| stack_ridge_lgbm_50 | train_2021 | 12 | -0.0924 | 0.2539 | -0.1296 | 0.6111 |
| stack_ridge_lgbm_50 | validation_2022_2023 | 24 | 0.0847 | 0.6313 | -0.1571 | 0.7083 |
| stack_ridge_lgbm_50 | holdout_2024 | 12 | 0.0743 | 0.4081 | -0.1335 | 0.6667 |
| stack_ridge_lgbm_50 | holdout_2025_2026 | 18 | 0.4294 | 0.8770 | -0.3284 | 0.6852 |
| stack_ridge_lgbm_50 | full | 78 | 0.1643 | 0.6530 | -0.3284 | 0.6880 |
| stack_base_ridge_50 | train_2021 | 12 | 0.1486 | 1.0468 | -0.1624 | 0.5833 |
| stack_base_ridge_50 | validation_2022_2023 | 24 | 0.1661 | 0.6953 | -0.3548 | 0.6389 |
| stack_base_ridge_50 | holdout_2024 | 12 | -0.0399 | -0.2659 | -0.1350 | 0.6667 |
| stack_base_ridge_50 | holdout_2025_2026 | 18 | 0.8966 | 1.3034 | -0.2780 | 0.6481 |
| stack_base_ridge_50 | full | 78 | 0.2376 | 0.7425 | -0.3548 | 0.6624 |

The validation winner among stacker candidates was **stack_lightgbm_excess** under the pre-declared rule of highest validation Sharpe, then excess CAGR. The existing composite remains the control. A candidate is only called historically audit-passing if it beats the control in validation, has positive excess CAGR in both fixed holdout blocks at both 25 and 100 bps, is positive in at least 60% of rolling 12-month windows, and has a positive 95% circular three-month block-bootstrap lower confidence bound versus IHSG.

The 2021 rows are the stacker's expanding-window warm-up, so learned candidates have no traded evaluation months there. The comparable evaluation begins in January 2022 after 12 prior signal months.

## Fixed audit gate

- **forecast_rank_ensemble:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2024 at 25 bps; excess CAGR is not positive in holdout_2024 at 100 bps; fewer than 60% of trailing 12-month windows have positive excess CAGR; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **stack_ridge_raw:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite).
- **stack_ridge_excess:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **stack_lightgbm_raw:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2024 at 100 bps).
- **stack_lightgbm_excess:** rejects the fixed audit gate (excess CAGR is not positive in holdout_2024 at 25 bps; excess CAGR is not positive in holdout_2024 at 100 bps; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **stack_ridge_lgbm_50:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite).
- **stack_base_ridge_50:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2024 at 25 bps; excess CAGR is not positive in holdout_2024 at 100 bps).

Preferred causal research candidate: **none**. This preference is based on the validation winner surviving the pre-declared audit, not on selecting the best holdout return. Even a passing candidate is not promoted to the web app or live trading because the current IDX30 history has survivorship/index-membership bias and the paired control interval can still include zero.

## Paired block bootstrap

The 2024–2026 comparison resamples paired three-month circular blocks 5,000 times. Market columns compare a candidate with IHSG; control columns compare it with the existing composite. These are uncertainty diagnostics for this sample, not a statistical guarantee under a new universe or regime.

| model | months | market_active_annualized_mean | market_ci | market_positive_probability | control_active_annualized_mean | control_ci |
|---|---|---|---|---|---|---|
| forecast_rank_ensemble | 30 | 0.2081 | [-0.0781, 0.5481] | 0.9100 | -0.0393 | [-0.3753, 0.2756] |
| stack_ridge_raw | 30 | 0.3993 | [0.1720, 0.6557] | 1.0000 | 0.1518 | [-0.1896, 0.5037] |
| stack_ridge_excess | 30 | 0.2244 | [-0.0236, 0.4890] | 0.9608 | -0.0231 | [-0.3982, 0.3270] |
| stack_lightgbm_raw | 30 | 0.2814 | [0.0252, 0.5648] | 0.9854 | 0.0339 | [-0.3056, 0.3874] |
| stack_lightgbm_excess | 30 | 0.2286 | [-0.0376, 0.5226] | 0.9516 | -0.0189 | [-0.3619, 0.3073] |
| stack_ridge_lgbm_50 | 30 | 0.3024 | [0.0443, 0.5948] | 0.9936 | 0.0550 | [-0.2919, 0.4014] |
| stack_base_ridge_50 | 30 | 0.4715 | [0.0930, 0.8913] | 0.9954 | 0.2240 | [-0.2077, 0.6857] |

## Stability and costs

The full cost/block table is in `reports/stacked_model_robustness.csv`; rolling windows are in `reports/stacked_model_rolling.csv` and `reports/stacked_model_rolling_summary.csv`.

| model | cost_bps | windows | median_sharpe | positive_excess_fraction | worst_excess_cagr | median_turnover |
|---|---|---|---|---|---|---|
| existing_composite | 25.0 | 67 | 0.9804 | 0.7463 | -0.3344 | 0.3611 |
| forecast_rank_ensemble | 25.0 | 67 | 0.2892 | 0.4328 | -0.2127 | 0.7222 |
| stack_base_ridge_50 | 25.0 | 67 | 1.0468 | 0.8209 | -0.2049 | 0.6667 |
| stack_lightgbm_excess | 25.0 | 67 | 1.2686 | 0.8358 | -0.1623 | 0.7500 |
| stack_lightgbm_raw | 25.0 | 67 | 1.0299 | 0.8657 | -0.2182 | 0.7500 |
| stack_ridge_excess | 25.0 | 67 | 0.6627 | 0.6567 | -0.4079 | 0.6111 |
| stack_ridge_lgbm_50 | 25.0 | 67 | 0.6668 | 0.8209 | -0.1078 | 0.6944 |
| stack_ridge_raw | 25.0 | 67 | 0.8022 | 0.7164 | -0.3673 | 0.6111 |

## Runtime and limitations

LightGBM status: `True`; version: `4.7.0`.<br>
Errors/skips: `{}`.

- The first-stage files were produced by earlier rolling runs; rerun them when changing model IDs, context length, price field, or data dates.
- The 30-name panel is a current-universe snapshot, so historical constituents and delisted stocks are missing. The all-listed catalog in the app is not yet a point-in-time research panel.
- Yahoo Finance data omit or simplify exchange-specific limits, suspensions, taxes, spreads, market impact, and execution timing.
- A model that passes this fixed historical gate may still be overfit to the finite panel. It is not investment advice and does not guarantee beating IHSG.

## Reproduction

```bash
python3 -m src.foundation_research --models all --start 2016-01-01 --end 2026-08-11 --kronos-repo /tmp/Kronos
PYTHONPATH=$PWD /tmp/beat-market-deep/bin/python -m src.deep_ml_research
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.stacked_research \
  --start 2016-01-01 --end 2026-08-11 --min-train-months 24
```

The raw stack forecasts are written locally to ignored `reports/stacked_model_forecasts.csv`. Committed audit artifacts are `reports/stacked_model_findings.md`, `reports/stacked_model_metrics.csv`, `reports/stacked_model_robustness.csv`, `reports/stacked_model_rolling_summary.csv`, `reports/stacked_model_bootstrap.csv`, and `reports/stacked_model_training.csv`.
