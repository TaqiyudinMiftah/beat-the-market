# Beat the Indonesian market — reproducible research

This repository tests whether simple, liquid-universe signals have historically
outperformed the IDX Composite (`^JKSE`) after turnover costs. It is research,
not investment advice and not a guarantee of future returns.

## Current first-pass design

- Universe: the 30 stocks in the IDX30 snapshot effective 3 August–30 October
  2026, stored in `data/universe_idx30_2026-08.csv`.
- Data: daily OHLCV from Yahoo Finance's chart API, cached under
  `data/raw/yahoo/`; `adjclose` is the default so splits and distributions are
  reflected in returns.
- Benchmark: IHSG/IDX Composite (`^JKSE`). IDX describes IHSG as measuring the
  price performance of all listed companies on the Main and Development Boards.
- Rebalance: monthly. A signal calculated at month-end `t` is held during the
  following month `t+1`.
- The downloaded daily panel runs through 11 August 2026; incomplete August is
  excluded from monthly return calculations, so the latest completed monthly
  observation is July 2026.
- Cost: 25 bps one-way per unit turnover by default; change with `--cost-bps`.
- Candidate signals: 12–1 month momentum, 6–1 momentum, 3-month momentum,
  low volatility, trend-plus-momentum, a composite rank, and the same composite
  with a market trend regime filter. An equal-weight control is included.

The main composite formula tested is:

```text
score = 0.40 * rank(momentum_12_to_1)
      + 0.30 * rank(momentum_6_to_1)
      + 0.20 * rank(momentum_3)
      + 0.10 * rank(-volatility_60d)

hold the top K stocks equally weighted each month
```

The candidate highlighted by the robustness report uses `K=3`. A separate
`composite_regime` variant applies the additional market trend filter shown in
the source code. The candidate was chosen during this exploratory sweep, so the
reported holdout is informative but not a fully untouched discovery test.

The weights above are a pre-declared candidate, not a claim that these exact
numbers are universal. `src/robustness.py` evaluates the concentrated top-three
composite candidate against the equal-weight control across four chronological
blocks and several cost assumptions. `src/research.py` also screens the small
candidate set using the validation period only, then reports an untouched
2024–2026 test period.

## Run

```bash
python3 src/download_data.py
python3 src/research.py
python3 src/robustness.py
```

Useful sensitivities:

```bash
python3 src/research.py --cost-bps 0
python3 src/research.py --cost-bps 50
python3 src/research.py --price-field close
```

Outputs are written to `reports/`. The current-universe backtest has
survivorship and index-membership look-ahead bias because historical
constituent snapshots and delisted names are not yet included. Treat results as
an exploratory baseline until point-in-time constituents, delistings,
suspensions, bid/ask spreads, taxes, and execution constraints are added.
