# Chronos-2 multivariate IDX research

Run date: 2026-08-12<br>
Model: `amazon/chronos-2`; chronos-forecasting: `2.3.1`<br>
Universe catalog: `data/universe_idx_all_2026-08.csv`; loaded tickers: `962`; skipped: `0`<br>
Eligibility: top `300` by trailing 60-day median dollar volume at each completed month-end<br>
Context: `48` monthly log-price observations; batch size `100`; default top K `3`; cost `25.0` bps one-way<br>
Signal months attempted: `90`; model forecast values: `57504`; recorded errors: `0`<br>

## Research basis

The [Chronos-2 multivariate finance study](https://arxiv.org/abs/2605.21504) reports that cross-series inputs can improve forecasts when the series are related, while warning that noisy cross-series context can hurt. This runner tests that claim on Indonesian equities rather than assuming transfer from another market. It compares Chronos-2's `cross_learning=False` individual forecasts with `cross_learning=True` forecasts in the same monthly, one-step-ahead protocol.

The liquidity proxy follows the [official IDX80/LQ45/IDX30 methodology](https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf), which considers transaction value/frequency, free-float market capitalization, fundamentals, a six-month listing minimum, and consistent trading. Historical membership and free-float data are not available in this repository; therefore the fixed top-`300` dollar-volume cap is only a robustness proxy.

## Leakage controls

- Each signal uses monthly prices through completed month-end `t`; the next-month price is never included in the model context.
- A stock must be eligible at `t` and have a complete shared trailing context; missing history is dropped rather than forward-filled.
- The lower-tail score uses the model's 10th-percentile log-price forecast and is fixed before inspecting holdout results.
- The `cap300_rank_ridge` comparator is refit expanding-window with signal dates strictly earlier than each forecast date.
- Top K, costs, rolling windows, and block bootstrap settings are fixed. Holdout months are descriptive and do not select the preferred model.

## Audit summary

The fixed gate requires beating the `cap300_composite` control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta ≤ `1.50`, drawdown ≥ `-0.60`, and turnover ≤ `0.90` in 2025–2026.

| model | validation excess | 2024 excess | 2025–2026 excess | rolling positive | bootstrap lower | holdout beta | holdout drawdown | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cap300_composite | -0.0573 | 0.3178 | -0.0339 | 0.7969 | — | 1.4652 | -0.6248 | rejects |
| cap300_rank_ridge | 0.1378 | 0.1237 | 0.6135 | 0.6299 | 0.0626 | 0.2949 | -0.0561 | passes |
| chronos2_individual_median | -0.7508 | -0.3553 | 0.1351 | 0.3797 | -0.5852 | 1.1281 | -0.7086 | rejects |
| chronos2_cross_median | -0.6248 | -0.4722 | 0.3910 | 0.2911 | -0.6299 | 1.7099 | -0.7649 | rejects |
| chronos2_cross_lower10 | -0.0109 | -0.0242 | 0.4587 | 0.7595 | -0.0169 | 0.8196 | -0.0524 | rejects |

Overall validation winner (highest validation Sharpe, then excess CAGR): **cap300_rank_ridge**. Chronos-2-only validation winner: **chronos2_cross_lower10**. Preferred research candidate after the fixed gate: **cap300_rank_ridge**. `4` candidate panels were evaluated.

## Interpretation

Chronos-2 is being evaluated as a forecast/ranking input, not as an automatic trading oracle. A passing row would still be exploratory because the current catalog excludes delisted names and historical membership changes. The report must be rerun on a point-in-time universe and future unseen months before any live use.

## Reproduction

```bash
HF_HOME=/tmp/beat-market-hf PYTHONPATH=$PWD \
  /tmp/beat-market-ml-venv/bin/python -m src.chronos2_research
```

For a smoke run without replaying every signal:

```bash
HF_HOME=/tmp/beat-market-hf PYTHONPATH=$PWD \
  /tmp/beat-market-ml-venv/bin/python -m src.chronos2_research --max-signals 2
```

Raw forecasts remain ignored in `reports/chronos2_forecasts.csv`. Committed audit tables are `chronos2_metrics.csv`, `chronos2_robustness.csv`, `chronos2_rolling_summary.csv`, `chronos2_bootstrap.csv`, `chronos2_sensitivity.csv`, and `chronos2_context.csv`.

This is research, not investment advice, and no backtest guarantees future performance.
