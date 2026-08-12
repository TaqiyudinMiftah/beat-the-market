# Chronos-2 cross-cap rank ensemble research

Run date: 2026-08-12<br>
Forecast model: chronos2_daily_abs_cross_lower10; source panels are completed daily Chronos-2 audits<br>
Universe catalog: data/universe_idx_all_2026-08.csv; loaded tickers: 962; skipped: 0<br>
Portfolio: top 3 equal-weight names; monthly signal; 25.0 bps one-way default cost<br>

## Research basis

Chronos-2 cross-learning is tested in the source audits using the official long-format forecasting interface and the multivariate financial-forecasting study. This replay adds a fixed rank-aggregation layer: it does not fit weights, inspect returns, or run new model inference. The liquidity caps follow the official IDX80/LQ45/IDX30 methodology as a proxy because historical membership and free-float data are unavailable.

## Fixed formula

For each signal month t and ticker i, let F be the terminal lower-10% forecast from each completed source panel. The score is:

~~~text
score(t, i) = mean(
    rank_pct(F_cap150(t, i)),
    rank_pct(F_cap300(t, i)),
    rank_pct(F_cap500(t, i)),
)
~~~

The mean uses only available forecasts; no missing value is imputed. Each cap is ranked within its own cross-section before averaging, so a larger universe cannot dominate by forecast scale. The three cap weights are equal and fixed. The fair comparator is the existing composite restricted to the top-500 liquidity proxy, because the ensemble's broadest source is cap 500.

Source coverage:

- cap 150: 90 signals; 12,339 forecast values
- cap 300: 90 signals; 24,383 forecast values
- cap 500: 90 signals; 41,258 forecast values

## Leakage controls and limitations

- Every source forecast was produced using daily observations on or before the completed signal month; this replay only reads those recorded forecasts.
- The ensemble weights and percentile transformation are fixed; no holdout return is used in aggregation.
- The current catalog omits historical delistings, suspensions, and membership changes, so cap robustness does not remove survivorship bias.
- Forecast files are generated from adjusted-price daily Chronos-2 runs. Yahoo data do not model spreads, taxes, price limits, market impact, failed fills, or capacity.

## Audit summary

The fixed gate requires beating the cap-500 composite control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025-2026.

| model | validation excess | 2024 excess | 2025–2026 excess | rolling positive | bootstrap lower | holdout beta | holdout drawdown | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cap500_composite | 0.1788 | 0.1939 | -0.0409 | 0.6875 | — | 1.2343 | -0.5318 | rejects |
| cap150_lower10 | -0.2778 | -0.1193 | 0.2974 | 0.4051 | -0.2020 | 0.3190 | -0.1254 | rejects |
| cap300_lower10 | 0.0715 | 0.3782 | 0.4784 | 0.7848 | 0.0276 | 0.1455 | -0.0216 | rejects |
| cap500_lower10 | 0.0312 | 0.4312 | 0.7787 | 0.8101 | 0.0496 | 0.4376 | -0.1524 | rejects |
| chronos2_cross_cap_rank_ensemble | 0.0472 | 0.5307 | 1.1204 | 0.9241 | 0.1449 | 0.4963 | -0.0811 | rejects |

Overall validation winner: chronos2_cross_cap_rank_ensemble. Preferred after the fixed gate: none. The ensemble is retained as an exploratory rank-aggregation result; it is not promoted unless it survives future point-in-time and unseen-data validation.

## Reproduction

~~~bash
PYTHONPATH=$PWD python3 -m src.chronos2_cross_cap_research
~~~

The runner expects the three ignored source files: reports/chronos2_daily_cap150_forecasts.csv, reports/chronos2_daily_forecasts.csv, and reports/chronos2_daily_cap500_forecasts.csv. Committed outputs use the chronos2_cross_cap_ prefix.

This is research, not investment advice, and no backtest guarantees future performance.
