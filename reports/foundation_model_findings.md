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
| existing_composite | train | 12 | 0.0000 | 0.1311 | -0.1311 | — | 0.0000 | 0.0000 |
| existing_composite | validation | 24 | 0.2320 | 0.0426 | 0.1894 | 0.8330 | -0.2245 | 0.2986 |
| existing_composite | holdout | 30 | 0.1510 | -0.0563 | 0.2073 | 0.5463 | -0.3684 | 0.3444 |
| existing_composite | full | 66 | 0.1500 | 0.0113 | 0.1387 | 0.5950 | -0.3684 | 0.2652 |
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
| foundation_ensemble_volmanaged | train | 12 | 0.0727 | 0.1311 | -0.0584 | 0.3930 | -0.1654 | 0.7435 |
| foundation_ensemble_volmanaged | validation | 24 | -0.0095 | 0.0426 | -0.0521 | 0.0834 | -0.2458 | 0.7639 |
| foundation_ensemble_volmanaged | holdout | 30 | 0.0210 | -0.0563 | 0.0773 | 0.2036 | -0.2172 | 0.5948 |
| foundation_ensemble_volmanaged | full | 66 | 0.0189 | 0.0113 | 0.0076 | 0.1912 | -0.2703 | 0.6833 |
| composite_chronos_blend_25 | train | 12 | 0.1631 | 0.1311 | 0.0320 | 0.6026 | -0.1506 | 0.5833 |
| composite_chronos_blend_25 | validation | 24 | 0.2262 | 0.0426 | 0.1836 | 0.7962 | -0.2051 | 0.4028 |
| composite_chronos_blend_25 | holdout | 30 | 0.1420 | -0.0563 | 0.1983 | 0.5292 | -0.3558 | 0.4222 |
| composite_chronos_blend_25 | full | 66 | 0.1758 | 0.0113 | 0.1645 | 0.6386 | -0.3558 | 0.4444 |
| composite_chronos_blend_50 | train | 12 | 0.1631 | 0.1311 | 0.0320 | 0.6026 | -0.1506 | 0.5833 |
| composite_chronos_blend_50 | validation | 24 | 0.3769 | 0.0426 | 0.3344 | 1.0933 | -0.1839 | 0.4861 |
| composite_chronos_blend_50 | holdout | 30 | 0.0088 | -0.0563 | 0.0651 | 0.2091 | -0.5122 | 0.5000 |
| composite_chronos_blend_50 | full | 66 | 0.1593 | 0.0113 | 0.1480 | 0.5835 | -0.5122 | 0.5101 |
| composite_chronos_blend_75 | train | 12 | 0.1631 | 0.1311 | 0.0320 | 0.6026 | -0.1506 | 0.5833 |
| composite_chronos_blend_75 | validation | 24 | 0.4906 | 0.0426 | 0.4480 | 1.3070 | -0.1700 | 0.5278 |
| composite_chronos_blend_75 | holdout | 30 | 0.0133 | -0.0563 | 0.0695 | 0.2424 | -0.5737 | 0.5889 |
| composite_chronos_blend_75 | full | 66 | 0.1955 | 0.0113 | 0.1842 | 0.6643 | -0.5737 | 0.5657 |

The foundation-model validation winner was **composite_chronos_blend_75** under the pre-declared rule of highest validation Sharpe, then excess CAGR. Its holdout excess CAGR was `0.0695` versus `0.2073` for the existing composite, with holdout maximum drawdown `-0.5737` versus `-0.3684`. Because the validation-selected blend did not generalize, no foundation model is promoted over the existing composite. This does not establish a future edge; validation and holdout results must be stable under point-in-time constituents, costs, and additional unseen data.

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

Forecasts: 1,794 symbol-months across 66 signal months. Mean rank IC: 0.0083; positive rank-IC fraction: 0.5303.

Model ID: `75% existing composite + 25% Chronos rank`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### composite_chronos_blend_50

Forecasts: 1,794 symbol-months across 66 signal months. Mean rank IC: 0.0144; positive rank-IC fraction: 0.5303.

Model ID: `50% existing composite + 50% Chronos rank`. Package versions: `{"chronos-forecasting": "2.3.1", "timesfm": "2.0.2", "torch": "2.5.1+cpu"}`.
Errors/skips: 0.
### composite_chronos_blend_75

Forecasts: 1,794 symbol-months across 66 signal months. Mean rank IC: 0.0191; positive rank-IC fraction: 0.5152.

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

Metrics are written to `reports/foundation_model_metrics.csv` and this report to `reports/foundation_model_findings.md`.
