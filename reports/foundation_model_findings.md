# Foundation-model backtest

Run date: 2026-08-12<br>
Research universe: `data/universe_idx30_2026-08.csv`<br>
Signal: daily history through completed month-end `t`; forecast next 21 trading days; hold next completed month<br>
Price field: `adjclose` for Chronos/TimesFM; raw OHLCV for Kronos<br>
Context: 256 daily observations; top K: 3; cost: 25.0 bps one-way<br>
Device: `cpu`; signals: 2016-01-01 through 2026-08-11

## Model sources

- [Amazon Chronos official repository](https://github.com/amazon-science/chronos-forecasting)
- [Google Research TimesFM official repository](https://github.com/google-research/timesfm)
- [Kronos official repository](https://github.com/shiyu-coder/Kronos) and [Kronos paper](https://arxiv.org/abs/2508.02739)

These are zero-shot forecasts, not fine-tuned models. The models were not trained on this IDX30 panel during this run. A forecast is converted to a cross-sectional score by predicted next-horizon return, then the top K names are equally weighted. The existing composite is retained as the control. The Chronos blend grid uses fixed 25%, 50%, and 75% Chronos-rank weights; no blend weight is tuned on the holdout.

## Runtime status

- **chronos:** available; package `chronos-forecasting` version `2.3.1`; model `amazon/chronos-bolt-tiny`
- **timesfm:** available; package `timesfm` version `2.0.2`; model `google/timesfm-2.5-200m-pytorch`
- **kronos:** available; package `official Kronos repository` version `—`; model `NeoQuasar/Kronos-mini`

## Results

| model | period | months | strategy_cagr | benchmark_cagr | excess_cagr | sharpe | max_drawdown | turnover |
|---|---|---|---|---|---|---|---|---|
| existing_composite | train | 72 | 0.6364 | 0.0623 | 0.5742 | 1.3547 | -0.3810 | 0.3495 |
| existing_composite | validation | 24 | 0.2322 | 0.0426 | 0.1896 | 0.8336 | -0.2245 | 0.2917 |
| existing_composite | holdout | 30 | 0.1510 | -0.0563 | 0.2073 | 0.5463 | -0.3684 | 0.3444 |
| existing_composite | full | 126 | 0.4258 | 0.0291 | 0.3967 | 1.0819 | -0.3810 | 0.3373 |
| chronos | train | 72 | 0.4976 | 0.0623 | 0.4353 | 1.1521 | -0.5461 | 0.5741 |
| chronos | validation | 24 | 0.0484 | 0.0426 | 0.0058 | 0.3061 | -0.2918 | 0.7083 |
| chronos | holdout | 30 | 0.1301 | -0.0563 | 0.1864 | 0.5021 | -0.5323 | 0.6889 |
| chronos | full | 126 | 0.3085 | 0.0291 | 0.2794 | 0.8584 | -0.5461 | 0.6270 |
| timesfm | train | 72 | 0.4069 | 0.0623 | 0.3447 | 1.1114 | -0.3097 | 0.7269 |
| timesfm | validation | 24 | -0.1492 | 0.0426 | -0.1918 | -0.3868 | -0.5229 | 0.6389 |
| timesfm | holdout | 30 | 0.0474 | -0.0563 | 0.1037 | 0.3024 | -0.2758 | 0.6889 |
| timesfm | full | 126 | 0.1917 | 0.0291 | 0.1626 | 0.6836 | -0.5732 | 0.7011 |
| kronos | train | 72 | 0.2250 | 0.0623 | 0.1627 | 0.7205 | -0.5451 | 0.7222 |
| kronos | validation | 24 | -0.0123 | 0.0426 | -0.0549 | 0.0925 | -0.3025 | 0.6667 |
| kronos | holdout | 30 | 0.1060 | -0.0563 | 0.1623 | 0.4832 | -0.3059 | 0.7222 |
| kronos | full | 126 | 0.1475 | 0.0291 | 0.1184 | 0.5683 | -0.5451 | 0.7116 |
| foundation_ensemble | train | 72 | 0.3460 | 0.0623 | 0.2838 | 1.0610 | -0.3957 | 0.7315 |
| foundation_ensemble | validation | 24 | -0.0095 | 0.0426 | -0.0521 | 0.0834 | -0.2458 | 0.7639 |
| foundation_ensemble | holdout | 30 | 0.0430 | -0.0563 | 0.0993 | 0.2855 | -0.2659 | 0.7556 |
| foundation_ensemble | full | 126 | 0.1948 | 0.0291 | 0.1658 | 0.7319 | -0.3957 | 0.7434 |
| foundation_ensemble_volmanaged | train | 72 | 0.3009 | 0.0623 | 0.2386 | 1.0129 | -0.3559 | 0.6467 |
| foundation_ensemble_volmanaged | validation | 24 | -0.0095 | 0.0426 | -0.0521 | 0.0834 | -0.2458 | 0.7639 |
| foundation_ensemble_volmanaged | holdout | 30 | 0.0210 | -0.0563 | 0.0773 | 0.2036 | -0.2172 | 0.5948 |
| foundation_ensemble_volmanaged | full | 126 | 0.1658 | 0.0291 | 0.1367 | 0.6953 | -0.3559 | 0.6567 |
| composite_chronos_blend_25 | train | 72 | 0.5873 | 0.0623 | 0.5250 | 1.2633 | -0.3615 | 0.3611 |
| composite_chronos_blend_25 | validation | 24 | 0.2277 | 0.0426 | 0.1851 | 0.7997 | -0.2051 | 0.3611 |
| composite_chronos_blend_25 | holdout | 30 | 0.1420 | -0.0563 | 0.1983 | 0.5292 | -0.3558 | 0.4222 |
| composite_chronos_blend_25 | full | 126 | 0.3975 | 0.0291 | 0.3684 | 1.0217 | -0.3615 | 0.3757 |
| composite_chronos_blend_50 | train | 72 | 0.6648 | 0.0623 | 0.6025 | 1.4768 | -0.2158 | 0.4120 |
| composite_chronos_blend_50 | validation | 24 | 0.3786 | 0.0426 | 0.3360 | 1.0965 | -0.1839 | 0.4444 |
| composite_chronos_blend_50 | holdout | 30 | 0.0088 | -0.0563 | 0.0651 | 0.2091 | -0.5122 | 0.5000 |
| composite_chronos_blend_50 | full | 126 | 0.4255 | 0.0291 | 0.3964 | 1.0992 | -0.5122 | 0.4392 |
| composite_chronos_blend_75 | train | 72 | 0.6941 | 0.0623 | 0.6318 | 1.4327 | -0.2873 | 0.4630 |
| composite_chronos_blend_75 | validation | 24 | 0.4912 | 0.0426 | 0.4486 | 1.3081 | -0.1700 | 0.5139 |
| composite_chronos_blend_75 | holdout | 30 | 0.0133 | -0.0563 | 0.0695 | 0.2424 | -0.5737 | 0.5889 |
| composite_chronos_blend_75 | full | 126 | 0.4630 | 0.0291 | 0.4339 | 1.1211 | -0.5737 | 0.5026 |

The foundation-model validation winner was **composite_chronos_blend_75** under the pre-declared rule of highest validation Sharpe, then excess CAGR. Its holdout excess CAGR was `0.0695` versus `0.2073` for the existing composite, with holdout maximum drawdown `-0.5737` versus `-0.3684`. Because the validation-selected blend did not generalize, no foundation model is promoted over the existing composite. This does not establish a future edge; validation and holdout results must be stable under point-in-time constituents, costs, and additional unseen data.

## Robustness protocol

The same forecast panels are replayed at 0, 25, 50, and 100 bps one-way costs. Chronological blocks are fixed before looking at the results: 2021 training, 2022–2023 validation, 2024 holdout, and January 2025 through the latest available month holdout. The two holdout blocks are descriptive only and are not used to select a model. The full diagnostics are in `reports/foundation_model_robustness.csv`; raw forecasts are written locally to the ignored `reports/foundation_model_forecasts.csv` artifact.

Cost stress for the validation winner and the existing composite:

| model | cost_bps | period | months | excess_cagr | strategy_sharpe_rf0 | strategy_max_drawdown | average_monthly_turnover |
|---|---|---|---|---|---|---|---|
| composite_chronos_blend_75 | 25.0000 | holdout_2024 | 12 | 0.2956 | 1.3845 | -0.1005 | 0.5556 |
| existing_composite | 25.0000 | holdout_2024 | 12 | -0.0868 | -0.4916 | -0.1955 | 0.3611 |
| composite_chronos_blend_75 | 50.0000 | holdout_2024 | 12 | 0.2750 | 1.3039 | -0.1026 | 0.5556 |
| existing_composite | 50.0000 | holdout_2024 | 12 | -0.0965 | -0.5517 | -0.1962 | 0.3611 |
| composite_chronos_blend_75 | 100.0000 | holdout_2024 | 12 | 0.2346 | 1.1406 | -0.1087 | 0.5556 |
| existing_composite | 100.0000 | holdout_2024 | 12 | -0.1157 | -0.6719 | -0.1977 | 0.3611 |
| composite_chronos_blend_75 | 25.0000 | holdout_2025_2026 | 18 | -0.0501 | -0.0207 | -0.5737 | 0.6111 |
| existing_composite | 25.0000 | holdout_2025_2026 | 18 | 0.4403 | 0.8702 | -0.3684 | 0.3333 |
| composite_chronos_blend_75 | 50.0000 | holdout_2025_2026 | 18 | -0.0662 | -0.0561 | -0.5778 | 0.6111 |
| existing_composite | 50.0000 | holdout_2025_2026 | 18 | 0.4264 | 0.8473 | -0.3737 | 0.3333 |
| composite_chronos_blend_75 | 100.0000 | holdout_2025_2026 | 18 | -0.0975 | -0.1269 | -0.5858 | 0.6111 |
| existing_composite | 100.0000 | holdout_2025_2026 | 18 | 0.3991 | 0.8016 | -0.3840 | 0.3333 |
| composite_chronos_blend_75 | 25.0000 | validation_2022_2023 | 24 | 0.4486 | 1.3081 | -0.1700 | 0.5139 |
| existing_composite | 25.0000 | validation_2022_2023 | 24 | 0.1896 | 0.8336 | -0.2245 | 0.2917 |
| composite_chronos_blend_75 | 50.0000 | validation_2022_2023 | 24 | 0.4262 | 1.2629 | -0.1717 | 0.5139 |
| existing_composite | 50.0000 | validation_2022_2023 | 24 | 0.1792 | 0.8056 | -0.2304 | 0.2917 |
| composite_chronos_blend_75 | 100.0000 | validation_2022_2023 | 24 | 0.3823 | 1.1726 | -0.1750 | 0.5139 |
| existing_composite | 100.0000 | validation_2022_2023 | 24 | 0.1584 | 0.7494 | -0.2422 | 0.2917 |

Chronological block results at the declared 25.0 bps cost:

| model | cost_bps | period | months | excess_cagr | strategy_sharpe_rf0 | strategy_max_drawdown | average_monthly_turnover |
|---|---|---|---|---|---|---|---|
| chronos | 25.0000 | holdout_2024 | 12 | 0.3746 | 1.7848 | -0.0771 | 0.7778 |
| composite_chronos_blend_25 | 25.0000 | holdout_2024 | 12 | 0.0827 | 0.4084 | -0.1798 | 0.3333 |
| composite_chronos_blend_50 | 25.0000 | holdout_2024 | 12 | 0.0493 | 0.2745 | -0.1768 | 0.5000 |
| composite_chronos_blend_75 | 25.0000 | holdout_2024 | 12 | 0.2956 | 1.3845 | -0.1005 | 0.5556 |
| existing_composite | 25.0000 | holdout_2024 | 12 | -0.0868 | -0.4916 | -0.1955 | 0.3611 |
| foundation_ensemble | 25.0000 | holdout_2024 | 12 | 0.0302 | 0.1766 | -0.1653 | 0.7222 |
| foundation_ensemble_volmanaged | 25.0000 | holdout_2024 | 12 | 0.0552 | 0.3186 | -0.1311 | 0.6805 |
| kronos | 25.0000 | holdout_2024 | 12 | -0.2299 | -1.2225 | -0.2523 | 0.6667 |
| timesfm | 25.0000 | holdout_2024 | 12 | 0.0954 | 0.6258 | -0.0820 | 0.7222 |
| chronos | 25.0000 | holdout_2025_2026 | 18 | 0.0821 | 0.2557 | -0.5323 | 0.6296 |
| composite_chronos_blend_25 | 25.0000 | holdout_2025_2026 | 18 | 0.2770 | 0.5957 | -0.3558 | 0.4815 |
| composite_chronos_blend_50 | 25.0000 | holdout_2025_2026 | 18 | 0.0751 | 0.2098 | -0.5122 | 0.5000 |
| composite_chronos_blend_75 | 25.0000 | holdout_2025_2026 | 18 | -0.0501 | -0.0207 | -0.5737 | 0.6111 |
| existing_composite | 25.0000 | holdout_2025_2026 | 18 | 0.4403 | 0.8702 | -0.3684 | 0.3333 |
| foundation_ensemble | 25.0000 | holdout_2025_2026 | 18 | 0.1447 | 0.3354 | -0.1974 | 0.7778 |
| foundation_ensemble_volmanaged | 25.0000 | holdout_2025_2026 | 18 | 0.0912 | 0.1252 | -0.1447 | 0.5376 |
| kronos | 25.0000 | holdout_2025_2026 | 18 | 0.5086 | 1.2378 | -0.1909 | 0.7593 |
| timesfm | 25.0000 | holdout_2025_2026 | 18 | 0.1088 | 0.2206 | -0.2758 | 0.6667 |
| chronos | 25.0000 | train_2021 | 12 | 0.0337 | 0.6064 | -0.1506 | 0.5278 |
| composite_chronos_blend_25 | 25.0000 | train_2021 | 12 | -0.0108 | 0.5010 | -0.2263 | 0.4444 |
| composite_chronos_blend_50 | 25.0000 | train_2021 | 12 | 0.0485 | 0.7092 | -0.1907 | 0.5278 |
| composite_chronos_blend_75 | 25.0000 | train_2021 | 12 | -0.0306 | 0.4399 | -0.1506 | 0.4722 |
| existing_composite | 25.0000 | train_2021 | 12 | 0.1608 | 0.9632 | -0.1137 | 0.4444 |
| foundation_ensemble | 25.0000 | train_2021 | 12 | -0.0832 | 0.2925 | -0.2049 | 0.6944 |
| foundation_ensemble_volmanaged | 25.0000 | train_2021 | 12 | -0.0010 | 0.6454 | -0.1093 | 0.5893 |
| kronos | 25.0000 | train_2021 | 12 | -0.1384 | 0.0663 | -0.2005 | 0.6667 |
| timesfm | 25.0000 | train_2021 | 12 | 0.0476 | 0.5938 | -0.2294 | 0.7222 |
| chronos | 25.0000 | validation_2022_2023 | 24 | 0.0058 | 0.3061 | -0.2918 | 0.7083 |
| composite_chronos_blend_25 | 25.0000 | validation_2022_2023 | 24 | 0.1851 | 0.7997 | -0.2051 | 0.3611 |
| composite_chronos_blend_50 | 25.0000 | validation_2022_2023 | 24 | 0.3360 | 1.0965 | -0.1839 | 0.4444 |
| composite_chronos_blend_75 | 25.0000 | validation_2022_2023 | 24 | 0.4486 | 1.3081 | -0.1700 | 0.5139 |
| existing_composite | 25.0000 | validation_2022_2023 | 24 | 0.1896 | 0.8336 | -0.2245 | 0.2917 |
| foundation_ensemble | 25.0000 | validation_2022_2023 | 24 | -0.0521 | 0.0834 | -0.2458 | 0.7639 |
| foundation_ensemble_volmanaged | 25.0000 | validation_2022_2023 | 24 | -0.0521 | 0.0834 | -0.2458 | 0.7639 |
| kronos | 25.0000 | validation_2022_2023 | 24 | -0.0549 | 0.0925 | -0.3025 | 0.6667 |
| timesfm | 25.0000 | validation_2022_2023 | 24 | -0.1918 | -0.3868 | -0.5229 | 0.6389 |

Validation-block winners (selection audit only):

| period | winner | winner_sharpe | baseline_sharpe |
|---|---|---|---|
| train_2021 | existing_composite | 0.9632 | 0.9632 |
| validation_2022_2023 | composite_chronos_blend_75 | 1.3081 | 0.8336 |

Rolling 12-month stability at the declared cost:

| model | cost_bps | windows | median_sharpe | positive_excess_fraction | worst_excess_cagr | median_turnover |
|---|---|---|---|---|---|---|
| chronos | 25.0000 | 115 | 0.9645 | 0.7565 | -0.5287 | 0.6667 |
| composite_chronos_blend_25 | 25.0000 | 115 | 1.0386 | 0.8435 | -0.1880 | 0.3611 |
| composite_chronos_blend_50 | 25.0000 | 115 | 1.1448 | 0.9043 | -0.1862 | 0.4444 |
| composite_chronos_blend_75 | 25.0000 | 115 | 1.2468 | 0.8783 | -0.2555 | 0.5278 |
| existing_composite | 25.0000 | 115 | 1.1108 | 0.7913 | -0.3344 | 0.3333 |
| foundation_ensemble | 25.0000 | 115 | 0.3076 | 0.5826 | -0.3159 | 0.7500 |
| foundation_ensemble_volmanaged | 25.0000 | 115 | 0.1823 | 0.5478 | -0.2994 | 0.6933 |
| kronos | 25.0000 | 115 | 0.4205 | 0.5217 | -0.3210 | 0.6944 |
| timesfm | 25.0000 | 115 | 0.5938 | 0.6261 | -0.4662 | 0.6944 |

### chronos

Forecasts: 3,284 symbol-months across 126 signal months. Mean rank IC: 0.0416; positive rank-IC fraction: 0.5476.

Model ID: `amazon/chronos-bolt-tiny`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### timesfm

Forecasts: 3,284 symbol-months across 126 signal months. Mean rank IC: 0.0236; positive rank-IC fraction: 0.4921.

Model ID: `google/timesfm-2.5-200m-pytorch`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### kronos

Forecasts: 3,284 symbol-months across 126 signal months. Mean rank IC: 0.0106; positive rank-IC fraction: 0.5317.

Model ID: `NeoQuasar/Kronos-mini`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### foundation_ensemble

Forecasts: 3,284 symbol-months across 126 signal months. Mean rank IC: 0.0387; positive rank-IC fraction: 0.5952.

Model ID: `equal rank ensemble`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### foundation_ensemble_volmanaged

Forecasts: 3,284 symbol-months across 126 signal months. Mean rank IC: 0.0387; positive rank-IC fraction: 0.5952.

Model ID: `equal rank ensemble + fixed 10% volatility target`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### composite_chronos_blend_25

Forecasts: 3,289 symbol-months across 126 signal months. Mean rank IC: 0.0075; positive rank-IC fraction: 0.5714.

Model ID: `75% existing composite + 25% Chronos rank`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### composite_chronos_blend_50

Forecasts: 3,289 symbol-months across 126 signal months. Mean rank IC: 0.0204; positive rank-IC fraction: 0.5635.

Model ID: `50% existing composite + 50% Chronos rank`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### composite_chronos_blend_75

Forecasts: 3,289 symbol-months across 126 signal months. Mean rank IC: 0.0369; positive rank-IC fraction: 0.5635.

Model ID: `25% existing composite + 75% Chronos rank`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.

## Leakage and limitations

- Each context is filtered with `date <= t`; the target is the following completed month's return.
- The forecast calendar is constructed from business days after `t` for model time features; no future prices are supplied.
- The current IDX30 membership is applied historically, creating survivorship and index-membership look-ahead bias.
- Chronos and TimesFM receive adjusted-close log prices, while Kronos receives raw daily OHLCV. This is a useful comparison, not a perfectly identical information set.
- Foundation models are large, stochastic, and pretrained outside Indonesia. Model weights, package versions, CPU/GPU settings, and random seeds must be recorded for any future reproduction.
- No result is investment advice or a guarantee of beating IHSG.

## Reproduction

Install CPU PyTorch first, then the optional packages. For Kronos, clone the official repository and pass its path with `--kronos-repo`.

```bash
python3 -m src.foundation_research --models all --start 2016-01-01 --end 2026-08-11 --kronos-repo /tmp/Kronos
python3 -m src.foundation_research --models chronos --start 2016-01-01 --end 2026-08-11 --max-signals 1
```

Metrics are written to `reports/foundation_model_metrics.csv`, cost/block diagnostics to `reports/foundation_model_robustness.csv`, rolling diagnostics to `reports/foundation_model_rolling.csv`, and this report to `reports/foundation_model_findings.md`.
