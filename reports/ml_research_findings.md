# Paper-backed machine-learning research

Run date: 2026-08-12<br>
Price field: `adjclose`<br>
Research universe: `data/universe_idx30_2026-08.csv`<br>
Signal convention: features are observed at completed month-end `t`; the target is the next month's return and is traded only after `t`.<br>
Latest signal row in the panel: 2026-07-31<br>
Cost: 25.0 bps per unit turnover; top K: 3

## What the papers contributed

- Jegadeesh and Titman document intermediate-horizon momentum and motivate the existing 12–1 feature: [The Journal of Finance (1993)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb04702.x).
- Gu, Kelly, and Xiu compare regularized linear, tree, and neural-network methods for empirical asset pricing and identify momentum, liquidity, and volatility as important predictors: [NBER Working Paper 25398](https://www.nber.org/papers/w25398).
- Moreira and Muir motivate reducing exposure when realized volatility is high: [NBER Working Paper 22208](https://www.nber.org/papers/w22208).

These papers support testable design choices; they do not establish an Indonesian equity edge or guarantee future performance. This run keeps the features, model hyperparameters, top K, cost, and time splits fixed before comparing results.

## Models tested

- **Ridge:** expanding-window pooled cross-sectional regression with training-only standardization, 1%/99% training-target winsorization, and fixed alpha `10`.
- **LightGBM:** shallow boosted trees with fixed depth/leaves, regularization, and 100 estimators. It is included as the nonlinear tree method suggested by the asset-pricing literature.
- **Equal ensemble:** average of the two models' cross-sectional percentile forecasts when both are available.
- **Volatility-managed ensemble:** the equal ensemble with long-only exposure scaled by `min(1, 10% / six-month realized IHSG volatility)`; uninvested exposure is cash with zero return.
- **Existing composite:** the repository's current 12–1/6–1/3-month/low-volatility rank formula, included as a non-ML control.

The models are retrained at every signal month using only earlier panel rows. Validation (2022–2023) is the only selection window; 2024 onward is a holdout and is not used to select weights or hyperparameters.

## Results

| model | period | months | strategy_cagr | benchmark_cagr | excess_cagr | sharpe | max_drawdown | turnover |
|---|---|---|---|---|---|---|---|---|
| existing_composite | train | 84 | 0.5253 | 0.0328 | 0.4924 | 1.2418 | -0.3810 | 0.2996 |
| existing_composite | validation | 24 | 0.2322 | 0.0426 | 0.1896 | 0.8336 | -0.2245 | 0.2917 |
| existing_composite | holdout | 30 | 0.1510 | -0.0563 | 0.2073 | 0.5463 | -0.3684 | 0.3444 |
| existing_composite | full | 138 | 0.3825 | 0.0144 | 0.3680 | 1.0298 | -0.3810 | 0.3080 |
| ridge | train | 84 | 0.1384 | 0.0328 | 0.1055 | 0.5486 | -0.4302 | 0.4325 |
| ridge | validation | 24 | -0.0819 | 0.0426 | -0.1244 | -0.0251 | -0.4709 | 0.3333 |
| ridge | holdout | 30 | 0.2828 | -0.0563 | 0.3391 | 0.8289 | -0.4429 | 0.3778 |
| ridge | full | 138 | 0.1254 | 0.0144 | 0.1110 | 0.5020 | -0.5437 | 0.4034 |
| lightgbm | train | 84 | 0.2803 | 0.0328 | 0.2475 | 0.8763 | -0.3769 | 0.3849 |
| lightgbm | validation | 24 | 0.1950 | 0.0426 | 0.1524 | 0.6591 | -0.2067 | 0.4306 |
| lightgbm | holdout | 30 | 0.3084 | -0.0563 | 0.3647 | 0.8063 | -0.4969 | 0.5778 |
| lightgbm | full | 138 | 0.2711 | 0.0144 | 0.2566 | 0.8215 | -0.4969 | 0.4348 |
| ensemble | train | 84 | 0.1916 | 0.0328 | 0.1588 | 0.7030 | -0.4138 | 0.4246 |
| ensemble | validation | 24 | -0.0187 | 0.0426 | -0.0613 | 0.0919 | -0.3195 | 0.4306 |
| ensemble | holdout | 30 | 0.4350 | -0.0563 | 0.4912 | 1.1828 | -0.3452 | 0.5000 |
| ensemble | full | 138 | 0.1995 | 0.0144 | 0.1851 | 0.7147 | -0.4138 | 0.4420 |
| ensemble_volmanaged | train | 84 | 0.1622 | 0.0328 | 0.1293 | 0.6766 | -0.3352 | 0.3844 |
| ensemble_volmanaged | validation | 24 | -0.0187 | 0.0426 | -0.0613 | 0.0919 | -0.3195 | 0.4306 |
| ensemble_volmanaged | holdout | 30 | 0.2523 | -0.0563 | 0.3086 | 1.0937 | -0.2314 | 0.4432 |
| ensemble_volmanaged | full | 138 | 0.1469 | 0.0144 | 0.1325 | 0.6345 | -0.3539 | 0.4052 |

The all-candidate validation winner under the pre-declared rule (highest Sharpe, then excess CAGR) was **existing_composite**. The ML-only winner was **lightgbm**. That is a selection result, not a claim that either method beats the market live. If the existing composite wins validation, the evidence does not justify replacing it with the ML candidate; compare holdout performance and rerun on a point-in-time universe first.

Panel rows: 4,170. Minimum training rows: 60.<br>
Training row counts are recorded in the run code; no future target is included in a forecast month.

## Foundation-model availability

The requested financial/time-series foundation models are optional because their packages and weights are large and their APIs change independently of this repository:

- **chronos:** not installed (`chronos-forecasting`)
- **timesfm:** not installed (`timesfm`)
- **kronos:** not installed (`Kronos repository model package`)

Chronos, TimesFM, and Kronos should be added only through a separate run that records exact model IDs, package versions, context length, forecast horizon, and download date. A foundation-model forecast must pass the same expanding-window, next-month, transaction-cost, and holdout protocol before it can be compared with these results.

The completed zero-shot rolling comparison is in `reports/foundation_model_findings.md` and is reproduced with `python3 -m src.foundation_research` in the CPU foundation environment.

The separate leakage-safe factor-MLP and causal online-weighting experiment is in `reports/deep_ml_findings.md` and is reproduced with `python3 -m src.deep_ml_research` in the optional deep CPU environment.

## Limitations

- The IDX30 snapshot is a current-universe panel and therefore has survivorship and index-membership look-ahead bias.
- Yahoo Finance is a convenient research source, not a licensed exchange feed; prices, corporate actions, suspensions, limits, taxes, spreads, and market impact are incomplete or unmodeled.
- A 30-stock panel is small for machine learning. The all-listed catalog is now available to the app, but a point-in-time all-stock history is still required for a stronger ML study.
- Out-of-sample backtest performance is not a promise of excess returns or investment advice.

## Reproduction

```bash
python3 -m src.ml_research
python3 -m src.ml_research --price-field close
python3 -m src.ml_research --foundation-status
```

Generated metrics are in `reports/ml_model_metrics.csv` and this report is in `reports/ml_research_findings.md`.
