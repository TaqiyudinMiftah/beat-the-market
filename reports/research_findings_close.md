# Indonesian equity research findings

Data field: `close`. One-way cost: `25 bps`. Benchmark: `^JKSE`. The daily panel ends 11 August 2026; incomplete August is excluded from monthly returns, so the holdout ends 31 July 2026.

## Candidate formula

The strongest simple candidate in this first pass is a monthly, equal-weight top-three portfolio ranked by:

```text
score = 0.40 * rank(momentum from t-12 to t-1)
      + 0.30 * rank(momentum from t-6 to t-1)
      + 0.20 * rank(momentum over the latest 3 months)
      + 0.10 * rank(-60-day annualized volatility)

At month-end t, hold the three highest-scoring eligible stocks equally during month t+1.
```

The ranks are cross-sectional within the current IDX30 snapshot. No market-regime filter is used in this candidate.

Because this candidate was chosen during the exploratory robustness sweep, the holdout figures below are descriptive rather than a clean, untouched discovery test.

## Candidate versus controls

| Period | Composite top 3 CAGR | IHSG CAGR | Excess CAGR | Max drawdown | Equal-weight CAGR |
|---|---:|---:|---:|---:|---:|
| 2015-2017 | 96.4% | 7.7% | 88.7% | -6.8% | 26.7% |
| 2018-2020 | 21.5% | -3.9% | 25.4% | -41.6% | 6.2% |
| 2021-2023 | 20.0% | 7.1% | 12.9% | -27.4% | 8.8% |
| 2024-2026_holdout | -2.9% | -5.6% | 2.8% | -53.4% | 7.1% |

Across the four chronological blocks, the candidate had positive excess return in 4/4 blocks; its median excess CAGR was 19.1%, with a worst block of 2.8%.

## Interpretation

- The candidate beats the price-return IHSG benchmark in this sample, including the 2024–2026 holdout.
- It is not a free lunch: concentration and monthly turnover create substantially higher drawdown and implementation risk than the equal-weight control.
- The equal-weight control also beats IHSG, so part of the result is likely exposure to the selected liquid-stock universe rather than the ranking formula itself.
- Adjusted-close results include distributions. Run `python3 src/research.py --price-field close` as a price-only sensitivity; raw close data can be distorted by corporate actions.

## Limitations before live use

This is an exploratory backtest using the current August–October 2026 IDX30 membership applied historically. It therefore has survivorship and index-membership look-ahead bias. It also does not model delistings, suspensions, price limits, bid/ask spreads, taxes, lot sizes, market impact, or exact execution prices. Historical outperformance is not evidence of a guaranteed future edge.

The next research upgrade should use point-in-time IDX30/LQ45 constituents, delisted names, and transaction-level execution assumptions before any capital is placed.
