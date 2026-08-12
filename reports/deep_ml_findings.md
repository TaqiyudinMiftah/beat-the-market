# Deep-learning backtest

Run date: 2026-08-12<br>
Research universe: `data/universe_idx30_2026-08.csv`<br>
Signal: factor rows observed at completed month-end `t`; target is the following month's return<br>
Price field: `adjclose`; top K: 3; cost: 25.0 bps one-way<br>
Evaluation dates: 2015-01-01 through 2026-08-11; device: `cpu`

## Research basis

- [Gu, Kelly, and Xiu, *Empirical Asset Pricing via Machine Learning*](https://www.nber.org/papers/w25398) motivates nonlinear interactions among momentum, liquidity, and volatility and compares trees and neural networks.
- [Abe and Nakayama, *Deep Learning for Forecasting Stock Returns in the Cross-Section*](https://arxiv.org/abs/1801.01777) studies one-month-ahead cross-sectional return forecasts with deep networks.
- The implementation uses the official [PyTorch](https://github.com/pytorch/pytorch) APIs and a small model because this panel is only 30 stocks.

## Leakage controls and fixed design

The MLP is retrained at every signal month using rows with `signal_date < t`. Feature normalization and target winsorization are fit on the training subset only. The most recent 12 historical signal months are used only for early stopping; they are never the current forecast month. Three fixed seeds (`7, 11, 19`) are averaged. Architecture is `(32, 16)` with dropout `0.10`, Adam learning rate `0.01`, weight decay `0.01`, and at most 120 epochs.

The online rules are also causal. They compare the baseline and MLP trailing 12-month Sharpe using only realized returns before `t`; they use either hard selection (`online_best`) or the fixed soft weights `0.25/0.50/0.75` around a fixed Sharpe margin of `0.25` (`online_soft`). Fixed baseline/MLP blend weights are included as overfitting controls.

## Results

| model | period | months | strategy_cagr | benchmark_cagr | excess_cagr | sharpe | max_drawdown | turnover |
|---|---|---|---|---|---|---|---|---|
| existing_composite | train | 84 | 0.5253 | 0.0328 | 0.4924 | 1.2418 | -0.3810 | 0.2996 |
| existing_composite | validation | 24 | 0.2322 | 0.0426 | 0.1896 | 0.8336 | -0.2245 | 0.2917 |
| existing_composite | holdout | 30 | 0.1510 | -0.0563 | 0.2073 | 0.5463 | -0.3684 | 0.3444 |
| existing_composite | full | 138 | 0.3825 | 0.0144 | 0.3680 | 1.0298 | -0.3810 | 0.3080 |
| mlp_raw | train | 84 | 0.0993 | 0.0328 | 0.0664 | 0.4609 | -0.5273 | 0.3571 |
| mlp_raw | validation | 24 | 0.1412 | 0.0426 | 0.0986 | 0.5371 | -0.4247 | 0.5833 |
| mlp_raw | holdout | 30 | 0.4774 | -0.0563 | 0.5337 | 1.1223 | -0.2907 | 0.6222 |
| mlp_raw | full | 138 | 0.1799 | 0.0144 | 0.1655 | 0.6522 | -0.5273 | 0.4541 |
| mlp_excess | train | 84 | 0.1977 | 0.0328 | 0.1649 | 0.7726 | -0.5073 | 0.2778 |
| mlp_excess | validation | 24 | -0.0614 | 0.0426 | -0.1040 | -0.0780 | -0.4130 | 0.6528 |
| mlp_excess | holdout | 30 | 0.2373 | -0.0563 | 0.2936 | 0.8478 | -0.3119 | 0.6222 |
| mlp_excess | full | 138 | 0.1561 | 0.0144 | 0.1417 | 0.6441 | -0.5073 | 0.4179 |
| composite_mlp_blend_25 | train | 84 | 0.4697 | 0.0328 | 0.4368 | 1.1476 | -0.4097 | 0.2976 |
| composite_mlp_blend_25 | validation | 24 | 0.4429 | 0.0426 | 0.4003 | 1.3578 | -0.1442 | 0.4583 |
| composite_mlp_blend_25 | holdout | 30 | 0.0892 | -0.0563 | 0.1455 | 0.4073 | -0.3495 | 0.4889 |
| composite_mlp_blend_25 | full | 138 | 0.3726 | 0.0144 | 0.3582 | 1.0183 | -0.4097 | 0.3671 |
| composite_mlp_blend_50 | train | 84 | 0.5014 | 0.0328 | 0.4686 | 1.2485 | -0.3115 | 0.3333 |
| composite_mlp_blend_50 | validation | 24 | 0.3934 | 0.0426 | 0.3508 | 1.2358 | -0.1324 | 0.6111 |
| composite_mlp_blend_50 | holdout | 30 | 0.0681 | -0.0563 | 0.1244 | 0.3544 | -0.3545 | 0.6222 |
| composite_mlp_blend_50 | full | 138 | 0.3763 | 0.0144 | 0.3619 | 1.0350 | -0.3545 | 0.4444 |
| composite_mlp_blend_75 | train | 84 | 0.4666 | 0.0328 | 0.4338 | 1.2087 | -0.4669 | 0.4087 |
| composite_mlp_blend_75 | validation | 24 | 0.3288 | 0.0426 | 0.2862 | 1.0248 | -0.2004 | 0.7083 |
| composite_mlp_blend_75 | holdout | 30 | 0.2391 | -0.0563 | 0.2954 | 0.6813 | -0.4727 | 0.6444 |
| composite_mlp_blend_75 | full | 138 | 0.3898 | 0.0144 | 0.3753 | 1.0406 | -0.4727 | 0.5121 |
| online_best | train | 84 | 0.4721 | 0.0328 | 0.4393 | 1.2572 | -0.3810 | 0.3333 |
| online_best | validation | 24 | 0.2842 | 0.0426 | 0.2416 | 0.9304 | -0.1442 | 0.4306 |
| online_best | holdout | 30 | 0.1570 | -0.0563 | 0.2133 | 0.5625 | -0.4008 | 0.6667 |
| online_best | full | 138 | 0.3642 | 0.0144 | 0.3498 | 1.0462 | -0.4008 | 0.4227 |
| online_soft | train | 84 | 0.4569 | 0.0328 | 0.4240 | 1.2072 | -0.4263 | 0.3254 |
| online_soft | validation | 24 | 0.3541 | 0.0426 | 0.3115 | 1.1151 | -0.1212 | 0.5278 |
| online_soft | holdout | 30 | 0.1952 | -0.0563 | 0.2515 | 0.6315 | -0.4243 | 0.6333 |
| online_soft | full | 138 | 0.3779 | 0.0144 | 0.3634 | 1.0560 | -0.4263 | 0.4275 |

The validation winner among deep candidates was **composite_mlp_blend_25** under the pre-declared rule of highest validation Sharpe, then excess CAGR. It is not automatically promoted: its later holdout and cost behavior is audited below. A candidate is considered research-promotable only if it beats the existing composite in validation, has positive excess CAGR in both fixed holdout blocks at both 25 and 100 bps, is positive in at least 60% of rolling 12-month windows, and has a positive 95% circular block-bootstrap lower confidence bound for active return versus IHSG over the 2024–2026 holdout. The paired control comparison is reported separately because thirty months is a small sample.

## Fixed audit gate

- **mlp_raw:** rejects the fixed audit gate (validation excess CAGR does not beat the existing composite).
- **mlp_excess:** rejects the fixed audit gate (validation excess CAGR is not positive; validation excess CAGR does not beat the existing composite; excess CAGR is not positive in holdout_2024 at both tested costs).
- **composite_mlp_blend_25:** rejects the fixed audit gate (excess CAGR is not positive in holdout_2024 at both tested costs; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **composite_mlp_blend_50:** rejects the fixed audit gate (excess CAGR is not positive in holdout_2024 at both tested costs; 95% block-bootstrap lower CI for active return versus IHSG is not above zero).
- **composite_mlp_blend_75:** passes the fixed audit gate.
- **online_best:** passes the fixed audit gate.
- **online_soft:** passes the fixed audit gate.

Passed candidates: `composite_mlp_blend_75, online_best, online_soft`. Passing this historical gate is not proof of a future edge and does not remove the current-universe survivorship bias.

Preferred causal research candidate: **online_soft**. This preference is based on the pre-declared online rule and its holdout audit, not on choosing the strongest holdout number. No live/app strategy is changed automatically; the paired control confidence interval still includes zero and a point-in-time all-listed panel is required before deployment consideration.

## Training diagnostics

| model | signals | forecast_rows | median_best_epoch | median_training_rows |
|---|---|---|---|---|
| mlp_excess | 102 | 2694 | 23.5000 | 1857.5000 |
| mlp_raw | 102 | 2694 | 17.0000 | 1857.5000 |

## Paired block-bootstrap check

The holdout bootstrap resamples three-month circular blocks of paired monthly returns 5,000 times. The market columns compare each candidate with IHSG; the control columns compare it with the existing composite. These are uncertainty diagnostics, not a guarantee of statistical significance under a different universe or future regime.

| model | months | market active annualized mean | market 95% CI | market P(active > 0) | control active annualized mean | control 95% CI |
|---|---:|---:|---|---:|---:|---|
| mlp_raw | 30 | 0.5168 | [0.2702, 0.7658] | 1.0000 | 0.2694 | [-0.0383, 0.5954] |
| mlp_excess | 30 | 0.2964 | [0.0794, 0.5243] | 0.9974 | 0.0489 | [-0.3022, 0.3908] |
| composite_mlp_blend_25 | 30 | 0.1867 | [-0.1175, 0.5130] | 0.8790 | -0.0608 | [-0.2461, 0.1242] |
| composite_mlp_blend_50 | 30 | 0.1802 | [-0.1075, 0.4867] | 0.8772 | -0.0672 | [-0.2439, 0.1316] |
| composite_mlp_blend_75 | 30 | 0.3599 | [0.0209, 0.6880] | 0.9816 | 0.1124 | [-0.1835, 0.3892] |
| online_best | 30 | 0.2545 | [0.0124, 0.5015] | 0.9828 | 0.0070 | [-0.3674, 0.4134] |
| online_soft | 30 | 0.2972 | [0.0079, 0.5930] | 0.9788 | 0.0497 | [-0.2008, 0.3121] |

## Cost and chronological diagnostics

The same forecast panels are replayed at 0, 25, 50, and 100 bps and across fixed blocks. Full tables are in `reports/deep_ml_robustness.csv`, `reports/deep_ml_rolling.csv`, `reports/deep_ml_rolling_summary.csv`, and `reports/deep_ml_bootstrap.csv`.

| model | cost_bps | period | months | excess_cagr | strategy_sharpe_rf0 | strategy_max_drawdown | average_monthly_turnover |
|---|---|---|---|---|---|---|---|
| composite_mlp_blend_25 | 25.0000 | holdout_2024 | 12 | -0.0496 | -0.1348 | -0.2094 | 0.6111 |
| composite_mlp_blend_50 | 25.0000 | holdout_2024 | 12 | -0.0575 | -0.1819 | -0.1818 | 0.7222 |
| composite_mlp_blend_75 | 25.0000 | holdout_2024 | 12 | 0.2548 | 1.0107 | -0.1226 | 0.7222 |
| existing_composite | 25.0000 | holdout_2024 | 12 | -0.0868 | -0.4916 | -0.1955 | 0.3611 |
| mlp_excess | 25.0000 | holdout_2024 | 12 | 0.0325 | 0.1858 | -0.1941 | 0.6389 |
| mlp_raw | 25.0000 | holdout_2024 | 12 | 0.1565 | 0.6032 | -0.1387 | 0.6111 |
| online_best | 25.0000 | holdout_2024 | 12 | 0.1565 | 0.6032 | -0.1387 | 0.6111 |
| online_soft | 25.0000 | holdout_2024 | 12 | 0.2548 | 1.0107 | -0.1226 | 0.7222 |
| composite_mlp_blend_25 | 100.0000 | holdout_2024 | 12 | -0.1000 | -0.3486 | -0.2218 | 0.6111 |
| composite_mlp_blend_50 | 100.0000 | holdout_2024 | 12 | -0.1169 | -0.4400 | -0.2031 | 0.7222 |
| composite_mlp_blend_75 | 100.0000 | holdout_2024 | 12 | 0.1784 | 0.7502 | -0.1490 | 0.7222 |
| existing_composite | 100.0000 | holdout_2024 | 12 | -0.1157 | -0.6719 | -0.1977 | 0.3611 |
| mlp_excess | 100.0000 | holdout_2024 | 12 | -0.0246 | -0.0702 | -0.2067 | 0.6389 |
| mlp_raw | 100.0000 | holdout_2024 | 12 | 0.0951 | 0.4038 | -0.1537 | 0.6111 |
| online_best | 100.0000 | holdout_2024 | 12 | 0.0951 | 0.4038 | -0.1537 | 0.6111 |
| online_soft | 100.0000 | holdout_2024 | 12 | 0.1784 | 0.7502 | -0.1490 | 0.7222 |
| composite_mlp_blend_25 | 25.0000 | holdout_2025_2026 | 18 | 0.2881 | 0.6373 | -0.3495 | 0.4074 |
| composite_mlp_blend_50 | 25.0000 | holdout_2025_2026 | 18 | 0.2560 | 0.5557 | -0.3545 | 0.5556 |
| composite_mlp_blend_75 | 25.0000 | holdout_2025_2026 | 18 | 0.3214 | 0.6390 | -0.4727 | 0.5926 |
| existing_composite | 25.0000 | holdout_2025_2026 | 18 | 0.4403 | 0.8702 | -0.3684 | 0.3333 |
| mlp_excess | 25.0000 | holdout_2025_2026 | 18 | 0.4921 | 1.1568 | -0.3119 | 0.6111 |
| mlp_raw | 25.0000 | holdout_2025_2026 | 18 | 0.8369 | 1.3699 | -0.2907 | 0.6296 |
| online_best | 25.0000 | holdout_2025_2026 | 18 | 0.2501 | 0.5525 | -0.4008 | 0.7037 |
| online_soft | 25.0000 | holdout_2025_2026 | 18 | 0.2492 | 0.5394 | -0.4243 | 0.5741 |
| composite_mlp_blend_25 | 100.0000 | holdout_2025_2026 | 18 | 0.2439 | 0.5483 | -0.3671 | 0.4074 |
| composite_mlp_blend_50 | 100.0000 | holdout_2025_2026 | 18 | 0.1995 | 0.4529 | -0.3689 | 0.5556 |
| composite_mlp_blend_75 | 100.0000 | holdout_2025_2026 | 18 | 0.2569 | 0.5479 | -0.4877 | 0.5926 |
| existing_composite | 100.0000 | holdout_2025_2026 | 18 | 0.3991 | 0.8016 | -0.3840 | 0.3333 |
| mlp_excess | 100.0000 | holdout_2025_2026 | 18 | 0.4189 | 1.0025 | -0.3335 | 0.6111 |
| mlp_raw | 100.0000 | holdout_2025_2026 | 18 | 0.7445 | 1.2590 | -0.3021 | 0.6296 |
| online_best | 100.0000 | holdout_2025_2026 | 18 | 0.1796 | 0.4135 | -0.4183 | 0.7037 |
| online_soft | 100.0000 | holdout_2025_2026 | 18 | 0.1906 | 0.4364 | -0.4433 | 0.5741 |
| composite_mlp_blend_25 | 25.0000 | train_2021 | 12 | -0.1388 | 0.1269 | -0.2556 | 0.3333 |
| composite_mlp_blend_50 | 25.0000 | train_2021 | 12 | -0.0373 | 0.4289 | -0.1712 | 0.3889 |
| composite_mlp_blend_75 | 25.0000 | train_2021 | 12 | 0.1911 | 0.9728 | -0.1320 | 0.4722 |
| existing_composite | 25.0000 | train_2021 | 12 | 0.1608 | 0.9632 | -0.1137 | 0.4444 |
| mlp_excess | 25.0000 | train_2021 | 12 | 0.3161 | 1.3841 | -0.1287 | 0.3333 |
| mlp_raw | 25.0000 | train_2021 | 12 | 0.2777 | 1.3645 | -0.1178 | 0.5000 |
| online_best | 25.0000 | train_2021 | 12 | 0.2777 | 1.3645 | -0.1178 | 0.5000 |
| online_soft | 25.0000 | train_2021 | 12 | 0.1634 | 0.9284 | -0.1320 | 0.4444 |
| composite_mlp_blend_25 | 100.0000 | train_2021 | 12 | -0.1687 | 0.0375 | -0.2730 | 0.3333 |
| composite_mlp_blend_50 | 100.0000 | train_2021 | 12 | -0.0751 | 0.3163 | -0.1795 | 0.3889 |
| composite_mlp_blend_75 | 100.0000 | train_2021 | 12 | 0.1363 | 0.8458 | -0.1413 | 0.4722 |
| existing_composite | 100.0000 | train_2021 | 12 | 0.1113 | 0.8385 | -0.1266 | 0.4444 |
| mlp_excess | 100.0000 | train_2021 | 12 | 0.2744 | 1.2836 | -0.1334 | 0.3333 |
| mlp_raw | 100.0000 | train_2021 | 12 | 0.2169 | 1.2003 | -0.1272 | 0.5000 |
| online_best | 100.0000 | train_2021 | 12 | 0.2169 | 1.2003 | -0.1272 | 0.5000 |
| online_soft | 100.0000 | train_2021 | 12 | 0.1128 | 0.8052 | -0.1413 | 0.4444 |
| composite_mlp_blend_25 | 25.0000 | validation_2022_2023 | 24 | 0.4003 | 1.3578 | -0.1442 | 0.4583 |
| composite_mlp_blend_50 | 25.0000 | validation_2022_2023 | 24 | 0.3508 | 1.2358 | -0.1324 | 0.6111 |
| composite_mlp_blend_75 | 25.0000 | validation_2022_2023 | 24 | 0.2862 | 1.0248 | -0.2004 | 0.7083 |
| existing_composite | 25.0000 | validation_2022_2023 | 24 | 0.1896 | 0.8336 | -0.2245 | 0.2917 |
| mlp_excess | 25.0000 | validation_2022_2023 | 24 | -0.1040 | -0.0780 | -0.4130 | 0.6528 |
| mlp_raw | 25.0000 | validation_2022_2023 | 24 | 0.0986 | 0.5371 | -0.4247 | 0.5833 |
| online_best | 25.0000 | validation_2022_2023 | 24 | 0.2416 | 0.9304 | -0.1442 | 0.4306 |
| online_soft | 25.0000 | validation_2022_2023 | 24 | 0.3115 | 1.1151 | -0.1212 | 0.5278 |
| composite_mlp_blend_25 | 100.0000 | validation_2022_2023 | 24 | 0.3445 | 1.2335 | -0.1698 | 0.4583 |
| composite_mlp_blend_50 | 100.0000 | validation_2022_2023 | 24 | 0.2788 | 1.0658 | -0.1412 | 0.6111 |
| composite_mlp_blend_75 | 100.0000 | validation_2022_2023 | 24 | 0.2062 | 0.8344 | -0.2184 | 0.7083 |
| existing_composite | 100.0000 | validation_2022_2023 | 24 | 0.1584 | 0.7494 | -0.2422 | 0.2917 |
| mlp_excess | 100.0000 | validation_2022_2023 | 24 | -0.1576 | -0.2833 | -0.4341 | 0.6528 |
| mlp_raw | 100.0000 | validation_2022_2023 | 24 | 0.0410 | 0.3918 | -0.4463 | 0.5833 |
| online_best | 100.0000 | validation_2022_2023 | 24 | 0.1947 | 0.8184 | -0.1614 | 0.4306 |
| online_soft | 100.0000 | validation_2022_2023 | 24 | 0.2508 | 0.9716 | -0.1495 | 0.5278 |

## Limitations

- The current IDX30 membership is applied historically, so the panel has survivorship and index-membership look-ahead bias.
- Thirty stocks and roughly eleven years of monthly observations are small for a deep network; the model is heavily regularized and should remain research-only until a point-in-time all-listed panel is available.
- Yahoo Finance data do not fully model exchange limits, suspensions, spreads, taxes, market impact, or execution timing.
- No backtest is investment advice or a guarantee of beating IHSG.

## Reproduction

Install the optional CPU environment from `requirements-deep-cpu.txt`, then run:

```bash
python3 -m src.deep_ml_research
```

Metrics are written to `reports/deep_ml_metrics.csv` and this report to `reports/deep_ml_findings.md`.
