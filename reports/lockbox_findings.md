# Newest-data model lockbox

Run date: 2026-08-12<br>
Universe: `/home/taqiyudinmiftah/beat-the-market/data/universe_idx_all_2026-08.csv`; loaded tickers: 962; skipped: 0<br>
Liquidity proxy: top 300; portfolio: top 3 equal-weight names; cost: 25.0 bps one-way<br>
Lockbox: 2026-01-01 through 2026-08-11; completed observations: 6<br>

## Purpose

This report is a strict newest-data check, not another model-selection sweep.
It compares the previously generated causal panels from the paper-style MLP,
zero-shot Chronos-2, and annually refit Chronos-2. The 2026 period was not
used to choose their architecture, seeds, forecast tail, or portfolio rule.

## Results

| model | validation excess | 2024 excess | 2025 excess | 2026 lockbox excess | lockbox months | lockbox Sharpe | lockbox drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| cap300_composite | -0.0573 | 0.3178 | 0.4236 | -0.2905 | 6 | -1.9785 | -0.4311 |
| mlp_rank | 0.1980 | 0.5226 | 0.6209 | 0.4886 | 6 | 0.3168 | -0.0712 |
| chronos2_zero_shot_lower10 | 0.0715 | 0.3782 | 0.0807 | 1.1697 | 6 | 6.0738 | 0.0000 |
| chronos2_annual_ft_lower10 | -0.0332 | 0.1035 | 0.1385 | 1.1319 | 6 | 6.7624 | 0.0000 |

Validation winner under the fixed Sharpe-then-excess rule: **mlp_rank**. Descriptive 2026 lockbox winner: **chronos2_annual_ft_lower10**. The lockbox winner is not promoted because it contains only six completed monthly observations.

At the declared cost, the newest lockbox is a useful stress point but not
statistical proof. A short favorable period can reflect market regime, sample
noise, or data-mining even when the forecast was generated causally. Promotion
still requires a longer future sample and a point-in-time universe.

## Reproduction inputs

- `mlp_rank`: `/home/taqiyudinmiftah/beat-the-market/reports/all_stock_deep_forecasts.csv` (`mlp_rank`)
- `chronos2_zero_shot_lower10`: `/home/taqiyudinmiftah/beat-the-market/reports/chronos2_daily_forecasts.csv` (`chronos2_daily_abs_cross_lower10`)
- `chronos2_annual_ft_lower10`: `/home/taqiyudinmiftah/beat-the-market/reports/chronos2_finetune_walkforward_forecasts.csv` (`chronos2_ft_cross_lower10`)

The source runners must be executed before this replay. Committed outputs are
`lockbox_metrics.csv`, `lockbox_costs.csv`,
`lockbox_monthly.csv`, and `lockbox_summary.json`.

This is research, not investment advice, and no backtest guarantees future performance.
