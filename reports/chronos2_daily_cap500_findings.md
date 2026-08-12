# Chronos-2 daily IDX research

Run date: 2026-08-12<br>
Model: amazon/chronos-2; chronos-forecasting: 2.3.1<br>
Universe catalog: data/universe_idx_all_2026-08.csv; loaded tickers: 962; skipped: 0<br>
Eligibility: top 500 by trailing 60-day median dollar volume at each completed month-end<br>
Context: 256 daily observations; horizon: 21 business days; batch size: 100<br>
Portfolio: top 3 equal-weight names; monthly signal; 25.0 bps one-way default cost<br>
Signal months attempted: 90; model forecast values: 206290; recorded errors: 0<br>

## Research basis

The official Chronos repository (https://github.com/amazon-science/chronos-forecasting) documents the long-format predict_df interface, quantile forecasts, cross-learning controls, and batch sizing used here. The Chronos-2 multivariate study (https://arxiv.org/abs/2605.21504) motivates testing cross-series learning while warning that unrelated series can add noise. This audit tests that claim on Indonesian equities.

The liquidity proxy follows the official IDX80/LQ45/IDX30 methodology (https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf). Historical membership and free-float data are not available here, so the fixed top-500 dollar-volume screen is only a robustness proxy.

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

The fixed gate requires beating the matching cap500_composite control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025-2026.

| model | validation excess | 2024 excess | 2025–2026 excess | rolling positive | bootstrap lower | holdout beta | holdout drawdown | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cap500_composite | 0.1788 | 0.1939 | -0.0409 | 0.6875 | — | 1.2343 | -0.5318 | rejects |
| cap500_rank_ridge | 0.0802 | 0.1001 | 1.4485 | 0.6375 | 0.1061 | 0.1352 | -0.0247 | rejects |
| chronos2_daily_abs_individual_median | -0.5475 | 0.2880 | 2.3341 | 0.2911 | 0.0140 | 2.3572 | -0.4363 | rejects |
| chronos2_daily_abs_cross_median | -0.2151 | 0.1954 | 1.2421 | 0.4051 | -0.1395 | 1.6525 | -0.4209 | rejects |
| chronos2_daily_abs_cross_lower10 | 0.0312 | 0.4312 | 0.7787 | 0.8101 | 0.0593 | 0.4376 | -0.1524 | rejects |
| chronos2_daily_norm_cross_median | -0.2151 | 0.1954 | 1.2421 | 0.4051 | -0.0886 | 1.6525 | -0.4209 | rejects |
| chronos2_daily_norm_cross_lower10 | 0.0312 | 0.4312 | 0.7787 | 0.8608 | 0.0735 | 0.4376 | -0.1524 | rejects |

Overall validation winner (highest validation Sharpe, then excess CAGR): cap500_rank_ridge. Chronos-2-only validation winner: chronos2_daily_abs_cross_lower10. Preferred after the fixed gate: none. 6 candidate panels were evaluated. The lower-tail Chronos-2 panels can pass their individual gate rows while the overall preferred result remains none: the predeclared selection rule chooses the overall validation winner first, and that winner must also pass every gate condition.

## Reproduction

~~~bash
HF_HOME=/tmp/beat-market-hf PYTHONPATH=$PWD \
  /tmp/beat-market-ml-venv/bin/python -m src.chronos2_daily_research
~~~

For a smoke run, add --max-signals 1. Raw forecasts remain ignored in reports/chronos2_daily_cap500_forecasts.csv. The committed tables use the same chronos2_daily_cap500_ prefix.

This is research, not investment advice, and no backtest guarantees future performance.
