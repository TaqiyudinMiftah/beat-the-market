# Paper-factor audit: momentum, low volatility, and low beta

Run date: 2026-08-11<br>
Price field: `adjclose`; current catalog loaded: `962`; skipped: `0`<br>
Liquidity screen: trailing 60-day median dollar-volume top `300`<br>
Portfolio: top `3` equal-weight names; monthly signal; `25.0` bps one-way default cost<br>

## Research basis

- [Jegadeesh and Titman (1993)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb04702.x) document intermediate-horizon winner-minus-loser momentum. This audit uses their 12-month lookback with the most recent month skipped.
- [Frazzini and Pedersen, Betting Against Beta](https://www.nber.org/papers/w16601) study a long-short, leveraged BAB factor across markets. This repository cannot reproduce that portfolio with its long-only app, so it tests only a low-beta cross-sectional rank combined with momentum and low volatility.
- The liquidity screen is a conservative proxy informed by the [official IDX methodology](https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf); it is not historical IDX30 membership.

## Fixed formula

For each eligible stock at completed month-end t:

~~~text
score = (rank(momentum_12_to_1)
       + rank(-volatility_60d)
       + rank(-beta_252d_vs_IHSG)) / 3
~~~

The ranks are computed within the eligible top-300 screen. The highest 3 scores are held equally during the next month. No weights, top-K, or holdout periods were selected after seeing the holdout.

## Leakage controls and limitations

- Beta and volatility are trailing daily estimates through t; the following month's return is never in the signal.
- The rank-Ridge comparator is fit expanding-window with training signal dates strictly earlier than each forecast.
- The current all-listed catalog still omits historical delistings and membership changes. This is an audit of a proxy universe, not a live signal.
- Yahoo Finance data do not model spreads, taxes, price limits, market impact, failed fills, or capacity.

## Audit summary

The fixed gate requires beating the matching composite control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive trailing 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025-2026.

| model | validation excess | 2024 excess | 2025–2026 excess | rolling positive | bootstrap lower | holdout beta | holdout drawdown | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cap300_composite | -0.0573 | 0.3178 | -0.0339 | 0.7969 | — | 1.4652 | -0.6248 | rejects |
| paper_factor | 0.3195 | -0.2012 | 0.3700 | 0.6094 | -0.2447 | 0.9802 | -0.4476 | rejects |
| cap300_rank_ridge | 0.1378 | 0.1237 | 0.6135 | 0.6299 | 0.0648 | 0.2949 | -0.0561 | passes |

Validation winner under the predeclared rule: **paper_factor**. Preferred after the fixed gate: **none**.

## Reproduction

~~~bash
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.paper_factor_research
~~~

Committed tables are `paper_factor_metrics.csv`, `paper_factor_robustness.csv`, `paper_factor_rolling_summary.csv`, `paper_factor_bootstrap.csv`, `paper_factor_sensitivity.csv`, `paper_factor_context.csv`, and `paper_factor_training.csv`.

This is research, not investment advice, and no backtest guarantees future performance.
