# TimesFM all-listed IDX research

Run date: 2026-08-12<br>
Model: google/timesfm-2.5-200m-pytorch; timesfm: 2.0.2<br>
Universe catalog: data/universe_idx_all_2026-08.csv; loaded tickers: 962; skipped: 0<br>
Eligibility: top 300 by trailing 60-day median dollar volume at each completed month-end<br>
Context: 256 daily observations; horizon: 21 business days; batch size: 32<br>
Portfolio: top 3 equal-weight names; monthly signal; 25.0 bps one-way default cost<br>
Signal months attempted: 90; TimesFM forecast values: 97532; recorded errors: 0<br>

## Research basis

The official Google Research TimesFM repository (https://github.com/google-research/timesfm) and the
original TimesFM paper (https://arxiv.org/abs/2310.10688) motivate testing a pretrained
decoder-only time-series forecaster without assuming that zero-shot accuracy
transfers to Indonesian equities. This audit uses the official TimesFM 2.5
PyTorch API and evaluates economic portfolio returns rather than forecast error
alone.

The liquidity proxy follows the official IDX80/LQ45/IDX30 methodology
(https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf). Historical membership and free-float data are not
available here, so the fixed top-300 dollar-volume screen is only a
robustness proxy.

## Fixed variants

Every signal uses only daily observations dated on or before completed
month-end t. TimesFM forecasts the terminal value after 21
business-day steps, which becomes a cross-sectional expected-return score for
the following month. The point and lower-tail variants are declared before
holdout inspection; no holdout-selected quantile or blend weight is used.

| variant | context | selected forecast |
|---|---|---|
| timesfm_daily_abs_point | absolute log price | terminal point forecast |
| timesfm_daily_abs_lower10 | absolute log price | terminal 10th percentile |
| timesfm_daily_norm_point | last-price normalized log price | terminal point forecast |
| timesfm_daily_norm_lower10 | last-price normalized log price | terminal 10th percentile |

## Leakage controls and limitations

- Context rows are filtered with date <= t; the next-month target is never supplied to TimesFM.
- Only stocks eligible at t with a complete trailing daily context are forecast. Missing history is dropped, not forward-filled.
- The rank-Ridge comparator is fit expanding-window with signal dates strictly earlier than each forecast date.
- Costs, top K, rolling windows, bootstrap settings, model variants, and quantile choices are fixed before inspecting holdout results.
- The current catalog omits historical delistings, suspensions, and membership changes. This cannot establish a live edge.
- Yahoo Finance data do not model spreads, taxes, price limits, market impact, failed fills, or capacity.

## Audit summary

The fixed gate requires beating the matching cap300_composite control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025–2026.

| model | validation excess | 2024 excess | 2025–2026 excess | rolling positive | bootstrap lower | holdout beta | holdout drawdown | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cap300_composite | -0.0573 | 0.3178 | -0.0339 | 0.7000 | — | 1.4652 | -0.6248 | rejects |
| cap300_rank_ridge | 0.1702 | 0.2485 | 0.2375 | 0.5875 | -0.0529 | 0.4275 | -0.1063 | rejects |
| timesfm_daily_abs_point | -0.3303 | -0.4655 | 3.4566 | 0.2278 | -0.2870 | 2.5843 | -0.4750 | rejects |
| timesfm_daily_abs_lower10 | -0.2545 | 0.2712 | -0.0556 | 0.3924 | -0.2121 | 0.5348 | -0.3309 | rejects |
| timesfm_daily_norm_point | -0.3303 | -0.4655 | 3.4566 | 0.2278 | -0.2949 | 2.5843 | -0.4750 | rejects |
| timesfm_daily_norm_lower10 | -0.3865 | -0.0664 | 0.5466 | 0.3418 | -0.1134 | 1.6133 | -0.2863 | rejects |

Overall validation winner (highest validation Sharpe, then excess CAGR): cap300_rank_ridge. TimesFM-only validation winner: timesfm_daily_abs_point. Preferred after the fixed gate: none. The validation rule is applied before holdout inspection; a candidate that only looks strong in the holdout is not promoted.

## Reproduction

~~~bash
HF_HOME=/tmp/beat-market-hf PYTHONPATH=$PWD \
  /tmp/beat-market-ml-venv/bin/python -m src.timesfm_all_stock_research
~~~

For a smoke run, add `--max-signals 1`. Raw forecasts remain ignored in
`reports/timesfm_all_forecasts.csv`; committed tables use the same
`timesfm_all_` prefix.

This is research, not investment advice, and no backtest guarantees future performance.
