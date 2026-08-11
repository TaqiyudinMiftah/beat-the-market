# Foundation-model backtest

Run date: 2026-08-12<br>
Research universe: `data/universe_idx30_2026-08.csv`<br>
Signal: daily history through completed month-end `t`; forecast next 21 trading days; hold next completed month<br>
Price field: `adjclose` for Chronos/TimesFM; raw OHLCV for Kronos<br>
Context: 256 daily observations; top K: 3; cost: 25.0 bps one-way<br>
Device: `cpu`; signals: 2021-01-01 through 2026-08-11

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
| existing_composite | train | 12 | 0.2915 | 0.1311 | 0.1604 | 0.9625 | -0.1137 | 0.4583 |
| existing_composite | validation | 24 | 0.2322 | 0.0426 | 0.1896 | 0.8336 | -0.2245 | 0.2917 |
| existing_composite | holdout | 30 | 0.1510 | -0.0563 | 0.2073 | 0.5463 | -0.3684 | 0.3444 |
| existing_composite | full | 66 | 0.2049 | 0.0113 | 0.1936 | 0.7140 | -0.3684 | 0.3460 |
| chronos | train | 12 | 0.1631 | 0.1311 | 0.0320 | 0.6026 | -0.1506 | 0.5833 |
| chronos | validation | 24 | 0.0484 | 0.0426 | 0.0058 | 0.3061 | -0.2918 | 0.7083 |
| chronos | holdout | 30 | 0.1301 | -0.0563 | 0.1864 | 0.5021 | -0.5323 | 0.6889 |
| chronos | full | 66 | 0.1055 | 0.0113 | 0.0942 | 0.4589 | -0.5323 | 0.6768 |
| timesfm | train | 12 | 0.1778 | 0.1311 | 0.0467 | 0.5921 | -0.2294 | 0.7500 |
| timesfm | validation | 24 | -0.1492 | 0.0426 | -0.1918 | -0.3868 | -0.5229 | 0.6389 |
| timesfm | holdout | 30 | 0.0474 | -0.0563 | 0.1037 | 0.3024 | -0.2758 | 0.6889 |
| timesfm | full | 66 | -0.0079 | 0.0113 | -0.0192 | 0.1181 | -0.5732 | 0.6818 |
| kronos | train | 12 | -0.0088 | 0.1311 | -0.1399 | 0.0588 | -0.2005 | 0.7222 |
| kronos | validation | 24 | -0.0123 | 0.0426 | -0.0549 | 0.0925 | -0.3025 | 0.6667 |
| kronos | holdout | 30 | 0.1060 | -0.0563 | 0.1623 | 0.4832 | -0.3059 | 0.7222 |
| kronos | full | 66 | 0.0405 | 0.0113 | 0.0292 | 0.2805 | -0.4280 | 0.7020 |
| foundation_ensemble | train | 12 | 0.0471 | 0.1311 | -0.0840 | 0.2897 | -0.2049 | 0.7222 |
| foundation_ensemble | validation | 24 | -0.0095 | 0.0426 | -0.0521 | 0.0834 | -0.2458 | 0.7639 |
| foundation_ensemble | holdout | 30 | 0.0430 | -0.0563 | 0.0993 | 0.2855 | -0.2659 | 0.7556 |
| foundation_ensemble | full | 66 | 0.0243 | 0.0113 | 0.0130 | 0.2177 | -0.3048 | 0.7525 |
| foundation_ensemble_volmanaged | train | 12 | 0.0623 | 0.1311 | -0.0689 | 0.3510 | -0.1795 | 0.7139 |
| foundation_ensemble_volmanaged | validation | 24 | -0.0095 | 0.0426 | -0.0521 | 0.0834 | -0.2458 | 0.7639 |
| foundation_ensemble_volmanaged | holdout | 30 | 0.0210 | -0.0563 | 0.0773 | 0.2036 | -0.2172 | 0.5948 |
| foundation_ensemble_volmanaged | full | 66 | 0.0171 | 0.0113 | 0.0058 | 0.1831 | -0.2703 | 0.6779 |
| composite_chronos_blend_25 | train | 12 | 0.1188 | 0.1311 | -0.0124 | 0.4971 | -0.2263 | 0.5000 |
| composite_chronos_blend_25 | validation | 24 | 0.2277 | 0.0426 | 0.1851 | 0.7997 | -0.2051 | 0.3611 |
| composite_chronos_blend_25 | holdout | 30 | 0.1420 | -0.0563 | 0.1983 | 0.5292 | -0.3558 | 0.4222 |
| composite_chronos_blend_25 | full | 66 | 0.1681 | 0.0113 | 0.1568 | 0.6215 | -0.3558 | 0.4141 |
| composite_chronos_blend_50 | train | 12 | 0.1779 | 0.1311 | 0.0468 | 0.7052 | -0.1907 | 0.5833 |
| composite_chronos_blend_50 | validation | 24 | 0.3786 | 0.0426 | 0.3360 | 1.0965 | -0.1839 | 0.4444 |
| composite_chronos_blend_50 | holdout | 30 | 0.0088 | -0.0563 | 0.0651 | 0.2091 | -0.5122 | 0.5000 |
| composite_chronos_blend_50 | full | 66 | 0.1624 | 0.0113 | 0.1511 | 0.5962 | -0.5122 | 0.4949 |
| composite_chronos_blend_75 | train | 12 | 0.0990 | 0.1311 | -0.0322 | 0.4358 | -0.1506 | 0.5278 |
| composite_chronos_blend_75 | validation | 24 | 0.4912 | 0.0426 | 0.4486 | 1.3081 | -0.1700 | 0.5139 |
| composite_chronos_blend_75 | holdout | 30 | 0.0133 | -0.0563 | 0.0695 | 0.2424 | -0.5737 | 0.5889 |
| composite_chronos_blend_75 | full | 66 | 0.1835 | 0.0113 | 0.1722 | 0.6375 | -0.5737 | 0.5505 |

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
| chronos | 25.0000 | train_2021 | 12 | 0.0320 | 0.6026 | -0.1506 | 0.5833 |
| composite_chronos_blend_25 | 25.0000 | train_2021 | 12 | -0.0124 | 0.4971 | -0.2263 | 0.5000 |
| composite_chronos_blend_50 | 25.0000 | train_2021 | 12 | 0.0468 | 0.7052 | -0.1907 | 0.5833 |
| composite_chronos_blend_75 | 25.0000 | train_2021 | 12 | -0.0322 | 0.4358 | -0.1506 | 0.5278 |
| existing_composite | 25.0000 | train_2021 | 12 | 0.1590 | 0.9603 | -0.1137 | 0.5000 |
| foundation_ensemble | 25.0000 | train_2021 | 12 | -0.0840 | 0.2897 | -0.2049 | 0.7222 |
| foundation_ensemble_volmanaged | 25.0000 | train_2021 | 12 | -0.0689 | 0.3510 | -0.1795 | 0.7139 |
| kronos | 25.0000 | train_2021 | 12 | -0.1399 | 0.0588 | -0.2005 | 0.7222 |
| timesfm | 25.0000 | train_2021 | 12 | 0.0467 | 0.5921 | -0.2294 | 0.7500 |
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
| train_2021 | existing_composite | 0.9603 | 0.9603 |
| validation_2022_2023 | composite_chronos_blend_75 | 1.3081 | 0.8336 |

Rolling 12-month stability at the declared cost:

| model | cost_bps | windows | median_sharpe | positive_excess_fraction | worst_excess_cagr | median_turnover |
|---|---|---|---|---|---|---|
| chronos | 25.0000 | 55 | 0.7064 | 0.7636 | -0.3067 | 0.6944 |
| composite_chronos_blend_25 | 25.0000 | 55 | 0.8246 | 0.7636 | -0.1880 | 0.3889 |
| composite_chronos_blend_50 | 25.0000 | 55 | 0.8456 | 0.9091 | -0.1445 | 0.4722 |
| composite_chronos_blend_75 | 25.0000 | 55 | 1.1469 | 0.8727 | -0.1864 | 0.5278 |
| existing_composite | 25.0000 | 55 | 0.8075 | 0.6909 | -0.3344 | 0.3333 |
| foundation_ensemble | 25.0000 | 55 | 0.1534 | 0.4545 | -0.3159 | 0.7500 |
| foundation_ensemble_volmanaged | 25.0000 | 55 | 0.0624 | 0.4545 | -0.2994 | 0.7139 |
| kronos | 25.0000 | 55 | 0.1774 | 0.4000 | -0.3210 | 0.6944 |
| timesfm | 25.0000 | 55 | 0.2708 | 0.4727 | -0.4662 | 0.6944 |

### chronos

Forecasts: 1,790 symbol-months across 66 signal months. Mean rank IC: 0.0256; positive rank-IC fraction: 0.5303.

Model ID: `amazon/chronos-bolt-tiny`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### timesfm

Forecasts: 1,790 symbol-months across 66 signal months. Mean rank IC: -0.0058; positive rank-IC fraction: 0.3939.

Model ID: `google/timesfm-2.5-200m-pytorch`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### kronos

Forecasts: 1,790 symbol-months across 66 signal months. Mean rank IC: 0.0027; positive rank-IC fraction: 0.4545.

Model ID: `NeoQuasar/Kronos-mini`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### foundation_ensemble

Forecasts: 1,790 symbol-months across 66 signal months. Mean rank IC: 0.0122; positive rank-IC fraction: 0.5455.

Model ID: `equal rank ensemble`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### foundation_ensemble_volmanaged

Forecasts: 1,790 symbol-months across 66 signal months. Mean rank IC: 0.0122; positive rank-IC fraction: 0.5455.

Model ID: `equal rank ensemble + fixed 10% volatility target`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### composite_chronos_blend_25

Forecasts: 1,794 symbol-months across 66 signal months. Mean rank IC: 0.0032; positive rank-IC fraction: 0.5606.

Model ID: `75% existing composite + 25% Chronos rank`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### composite_chronos_blend_50

Forecasts: 1,794 symbol-months across 66 signal months. Mean rank IC: 0.0098; positive rank-IC fraction: 0.5758.

Model ID: `50% existing composite + 50% Chronos rank`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### composite_chronos_blend_75

Forecasts: 1,794 symbol-months across 66 signal months. Mean rank IC: 0.0163; positive rank-IC fraction: 0.5303.

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
python3 -m src.foundation_research --models all
python3 -m src.foundation_research --models chronos --max-signals 1
```

Metrics are written to `reports/foundation_model_metrics.csv`, cost/block diagnostics to `reports/foundation_model_robustness.csv`, rolling diagnostics to `reports/foundation_model_rolling.csv`, and this report to `reports/foundation_model_findings.md`.
