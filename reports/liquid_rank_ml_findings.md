# Liquidity-filtered rank-target ML research

Run date: 2026-08-12<br>
Universe catalog: `data/universe_idx_all_2026-08.csv`<br>
Data directory: `data/raw/yahoo_all`<br>
Loaded catalog tickers: `962`; skipped/unavailable: `0`<br>
Caps tested: `150, 300, 500, all`; default top K: `3`; cost: `25.0` bps one-way<br>

## Why this experiment exists

The current catalog is not a point-in-time universe. The official [IDX80/LQ45/IDX30 methodology](https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf) describes liquidity, market capitalization, fundamentals, a six-month listing minimum, and consistent trading as part of index selection. Historical free-float and membership data are unavailable here, so this run uses only the observable trailing 60-day median dollar-volume feature to create fixed eligibility caps. It is a robustness proxy, not a reconstruction of IDX membership.

## Paper-backed design

[Gu, Kelly, and Xiu](https://www.nber.org/papers/w25398) motivate comparing regularized linear models and nonlinear trees with strict out-of-sample testing. Each completed month-end `t` uses only data through `t`:

- eligible names are the top fixed liquidity cap at `t`, with no forward volume information;
- raw-return models predict next-month returns; rank-target models predict each eligible stock's percentile rank among the next-month realized returns in historical training months;
- Ridge and shallow LightGBM are refit expanding-window, with training signal dates strictly earlier than the forecast date;
- forecasts are ranked cross-sectionally and the top K names are equally weighted for the following month;
- volatility-managed variants use the existing fixed 10% benchmark-volatility exposure rule;
- caps, top K values, costs, rolling windows, and bootstrap settings are fixed before inspecting holdout results.

Rank targets are a robustness transformation against a few extreme micro-cap returns dominating the raw-return loss. They do not create additional information and are not a guarantee of better prediction.

## Audit summary

The fixed audit gate requires beating the matching liquidity control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% block-bootstrap lower bound versus IHSG, and fixed risk limits of beta ≤ `1.50`, drawdown ≥ `-0.60`, and turnover ≤ `0.90` in the 2025–2026 holdout. Holdout data are not used to choose the variant.

| model | validation excess | 2024 excess | 2025–2026 excess | rolling positive | bootstrap lower | holdout beta | holdout drawdown | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cap150_raw_ridge | -0.3244 | -0.5120 | -0.3346 | 0.3150 | -1.0291 | 1.6566 | -0.7723 | rejects |
| cap150_raw_lightgbm | -0.2018 | 0.7959 | 0.1832 | 0.7087 | -0.2143 | 1.6998 | -0.7361 | rejects |
| cap150_rank_ridge | -0.2566 | 0.1409 | 0.3375 | 0.3858 | -0.0081 | 0.5636 | -0.1247 | rejects |
| cap150_rank_lightgbm | 0.1566 | 0.1308 | -0.0103 | 0.8425 | -0.2174 | 1.2817 | -0.4738 | rejects |
| cap150_raw_ensemble | -0.0344 | 0.4159 | -0.3625 | 0.6378 | -0.7425 | 0.8178 | -0.7807 | rejects |
| cap150_rank_ensemble | 0.0134 | -0.1566 | 0.0639 | 0.4488 | -0.2507 | 1.0547 | -0.3776 | rejects |
| cap150_rank_ensemble_volmanaged | 0.0134 | -0.1455 | 0.0697 | 0.4488 | -0.2509 | 0.6913 | -0.2631 | rejects |
| cap300_raw_ridge | -0.4435 | -0.1201 | 2.1428 | 0.3071 | -0.1147 | 1.3325 | -0.4921 | rejects |
| cap300_raw_lightgbm | -0.5145 | 0.5354 | 0.1868 | 0.5433 | -0.3502 | 1.2960 | -0.7349 | rejects |
| cap300_rank_ridge | 0.1378 | 0.1237 | 0.6135 | 0.6299 | 0.0618 | 0.2949 | -0.0561 | passes |
| cap300_rank_lightgbm | 0.0314 | 0.0684 | 0.4891 | 0.7402 | -0.1805 | 1.0499 | -0.3858 | rejects |
| cap300_raw_ensemble | -0.3352 | 0.2926 | 0.9752 | 0.5039 | -0.0806 | 1.7189 | -0.5023 | rejects |
| cap300_rank_ensemble | -0.0015 | 0.1560 | 0.3028 | 0.8425 | -0.0564 | 0.8123 | -0.2353 | rejects |
| cap300_rank_ensemble_volmanaged | -0.0015 | 0.1477 | 0.2134 | 0.7717 | -0.1119 | 0.4849 | -0.1475 | rejects |
| cap500_raw_ridge | -0.4634 | -0.4569 | -0.3362 | 0.3071 | -0.8129 | 0.4054 | -0.5882 | rejects |
| cap500_raw_lightgbm | 0.3384 | 0.9126 | 0.2247 | 0.6693 | -0.1039 | 1.1277 | -0.4091 | rejects |
| cap500_rank_ridge | -0.0766 | 0.0413 | 1.6308 | 0.5591 | 0.0834 | 0.0717 | -0.0367 | rejects |
| cap500_rank_lightgbm | 0.0611 | 0.0604 | 0.7743 | 0.6772 | 0.0765 | 0.7081 | -0.2031 | rejects |
| cap500_raw_ensemble | 0.2397 | -0.1448 | 0.0043 | 0.5354 | -0.4970 | 1.5847 | -0.5263 | rejects |
| cap500_rank_ensemble | 0.1288 | 0.0659 | 1.2639 | 0.7559 | 0.1426 | 0.1320 | -0.0831 | rejects |
| cap500_rank_ensemble_volmanaged | 0.1288 | 0.0742 | 0.8877 | 0.7402 | 0.0418 | -0.0894 | -0.0522 | rejects |
| capall_raw_ridge | -0.5093 | 0.8872 | 0.4116 | 0.5512 | -0.1236 | 0.4356 | -0.5137 | rejects |
| capall_raw_lightgbm | 0.4545 | 4.9312 | 1.0170 | 0.7087 | 0.5491 | 1.4022 | -0.3269 | passes |
| capall_rank_ridge | -0.1285 | 1.5555 | 0.4583 | 0.6299 | 0.1528 | 0.4672 | -0.1706 | rejects |
| capall_rank_lightgbm | 0.0204 | 0.0675 | 0.7067 | 0.6693 | 0.0881 | 0.4335 | -0.0924 | rejects |
| capall_raw_ensemble | -0.1604 | 1.9545 | -0.1060 | 0.5669 | -0.2745 | 1.1260 | -0.4911 | rejects |
| capall_rank_ensemble | -0.1694 | 0.3207 | 0.3006 | 0.6457 | -0.1994 | -0.0211 | -0.1388 | rejects |
| capall_rank_ensemble_volmanaged | -0.1694 | 0.2928 | 0.3261 | 0.6457 | -0.1822 | -0.1169 | -0.1333 | rejects |

Validation winner under the predeclared rule (highest validation Sharpe, then excess CAGR): **cap500_rank_ensemble**. It is only considered preferred if it passes every audit check: **none**. `28` candidate panels were evaluated; a passing row remains exploratory because the current catalog omits delisted names and historical membership.

## Top-K and cost sensitivity

The complete fixed replay is in `reports/liquid_rank_ml_sensitivity.csv`. It covers top K = 1, 3, 5, and 10, costs = 25 and 100 bps, all tested caps, controls, raw targets, rank targets, ensembles, and volatility-managed variants. No top-K or cap was selected using the holdout.

## Limitations

- This is still a current-universe snapshot. It cannot correct survivorship, delisting, historical suspension, free-float, or corporate-action selection bias.
- Yahoo Finance is a research input, not a licensed exchange execution feed. Spreads, price limits, taxes, impact, capacity, and failed fills are not fully modeled.
- A fixed liquidity rank is not the same as the IDX's transaction-value, free-float, or fundamental screens.
- A backtest result is not investment advice and does not guarantee beating IHSG.

## Reproduction

```bash
python3 src/download_data.py --universe all --start 2015-01-01 --end 2026-08-12 \
  --output-dir data/raw/yahoo_all --metadata-path data/raw/yahoo_all_metadata.json \
  --continue-on-error
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.liquid_rank_ml_research
```

Raw forecasts are written to ignored `reports/liquid_rank_ml_forecasts.csv`. Committed audit tables are `reports/liquid_rank_ml_metrics.csv`, `reports/liquid_rank_ml_robustness.csv`, `reports/liquid_rank_ml_rolling_summary.csv`, `reports/liquid_rank_ml_bootstrap.csv`, `reports/liquid_rank_ml_sensitivity.csv`, and `reports/liquid_rank_ml_training.csv`.
