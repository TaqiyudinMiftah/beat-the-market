# Chronos-2 daily IDX research

Run date: 2026-08-12<br>
Model: amazon/chronos-2; chronos-forecasting: 2.3.1<br>
Universe catalog: data/universe_idx_all_2026-08.csv; loaded tickers: 962; skipped: 0<br>
Eligibility: top 300 by trailing 60-day median dollar volume at each completed month-end<br>
Context: 128 daily observations; horizon: 21 business days; batch size: 100<br>
Portfolio: top 3 equal-weight names; monthly signal; 25.0 bps one-way default cost<br>
Signal months attempted: 90; model forecast values: 128810; recorded errors: 0<br>

## Research basis

The official Chronos repository (https://github.com/amazon-science/chronos-forecasting) documents the long-format predict_df interface, quantile forecasts, cross-learning controls, and batch sizing used here. The Chronos-2 multivariate study (https://arxiv.org/abs/2605.21504) motivates testing cross-series learning while warning that unrelated series can add noise. This audit tests that claim on Indonesian equities.

The liquidity proxy follows the official IDX80/LQ45/IDX30 methodology (https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf). Historical membership and free-float data are not available here, so the fixed top-300 dollar-volume screen is only a robustness proxy.

## Fixed variants

Every signal uses only daily observations dated on or before completed month-end t. Chronos-2 forecasts the terminal value after 21 business-day steps, which becomes a cross-sectional expected-return score for the following month.

| variant | context | inference | selected forecast |
|---|---|---|---|
| chronos2_daily_abs_individual_median | absolute log price | individual | terminal median |
| chronos2_daily_abs_cross_median | absolute log price | cross-learning | terminal median |
| chronos2_daily_abs_cross_lower10 | absolute log price | cross-learning | terminal lower 10% |
| chronos2_daily_norm_cross_median | last-price normalized log price | cross-learning | terminal median |
| chronos2_daily_norm_cross_lower10 | last-price normalized log price | cross-learning | terminal lower 10% |

Normalized variants remove stock price-level scale before inference; their forecast is already a log-return estimate. This transformation is fixed before holdout inspection, and the two representations are not independent evidence because they use the same stocks, dates, and model. No blend weight or holdout-selected threshold is used.

## Leakage controls and limitations

- Context rows are filtered with date <= t; the next-month target is never supplied to Chronos-2.
- Only stocks eligible at t with a complete trailing daily context are forecast. Missing history is dropped; it is not forward-filled.
- The rank-Ridge comparator is fit expanding-window with signal dates strictly earlier than each forecast.
- Top K, costs, rolling windows, bootstrap settings, model variants, and quantile choices are fixed before inspecting holdout results.
- The current catalog omits historical delistings, suspensions, and membership changes. This cannot establish a live edge.
- Yahoo Finance data do not model spreads, taxes, price limits, market impact, failed fills, or capacity.

## Audit summary

The fixed gate requires beating the matching cap300_composite control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025-2026.

| model | validation excess | 2024 excess | 2025–2026 excess | rolling positive | bootstrap lower | holdout beta | holdout drawdown | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cap300_composite | -0.0573 | 0.3178 | -0.0339 | 0.7000 | — | 1.4652 | -0.6248 | rejects |
| cap300_rank_ridge | 0.1702 | 0.2485 | 0.2375 | 0.5875 | -0.0529 | 0.4275 | -0.1063 | rejects |
| chronos2_daily_abs_individual_median | 0.5081 | -0.3967 | 2.0980 | 0.3924 | -0.1541 | 2.1085 | -0.1908 | rejects |
| chronos2_daily_abs_cross_median | 0.0505 | -0.2263 | 0.9922 | 0.3165 | -0.2818 | 1.7109 | -0.3643 | rejects |
| chronos2_daily_abs_cross_lower10 | 0.0241 | 0.9098 | 0.3241 | 0.7468 | 0.0946 | 0.4449 | -0.1210 | passes |
| chronos2_daily_norm_cross_median | 0.0505 | -0.2263 | 0.9922 | 0.3165 | -0.2657 | 1.7109 | -0.3643 | rejects |
| chronos2_daily_norm_cross_lower10 | 0.0241 | 0.9098 | 0.3241 | 0.7468 | 0.0969 | 0.4449 | -0.1210 | passes |

Overall validation winner (highest validation Sharpe, then excess CAGR): cap300_rank_ridge. Chronos-2-only validation winner: chronos2_daily_abs_individual_median. Preferred after the fixed gate: none. 6 candidate panels were evaluated. The lower-tail Chronos-2 panels can pass their individual gate rows while the overall preferred result remains none: the predeclared selection rule chooses the overall validation winner first, and that winner must also pass every gate condition.

## Reproduction

~~~bash
HF_HOME=/tmp/beat-market-hf PYTHONPATH=$PWD \
  /tmp/beat-market-ml-venv/bin/python -m src.chronos2_daily_research
~~~

For a smoke run, add --max-signals 1. Raw forecasts remain ignored in reports/chronos2_daily_ctx128_forecasts.csv. The committed tables use the same chronos2_daily_ctx128_ prefix.

This is research, not investment advice, and no backtest guarantees future performance.
