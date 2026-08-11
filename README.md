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

## Run the web app locally

The public UI is a Vite/React dashboard backed by a typed FastAPI service. It
uses Lightweight Charts for daily candles, volume, moving averages, signal
inspection, and simple trend/level drawings.

```bash
python3 -m pip install -r requirements.txt
python3 src/download_data.py
uvicorn api.main:app --reload                 # terminal 1, http://localhost:8000
cd web && npm install && npm run dev          # terminal 2, http://localhost:5173
```

The dashboard includes the latest cross-sectional ranking, stock-level factor
explanations, an educational method guide, shareable URLs, and a Strategy Lab.
The lab supports momentum, volatility, trend, and liquidity factor blocks with
validated weights, top-K selection, monthly rebalancing, and one-way costs.

The API exposes `/api/health`, `/api/data-status`, `/api/universe`,
`/api/stocks/{ticker}/ohlcv`, `/api/stocks/{ticker}/features`,
`/api/rankings`, `/api/strategies/default`, and `POST /api/backtests`.
`POST /api/admin/refresh` requires the `X-Refresh-Token` header and is intended
for a scheduler or an operator—not for an unprotected browser action.

## Deployment

For the planned split deployment, create the FastAPI service from
`render.yaml`, set `CORS_ORIGINS` to the Vercel URL, set a strong
`REFRESH_TOKEN`, and attach persistent storage mounted at `data/raw` so cached
CSV files survive deploys. In Vercel, set the project root to `web`, build with
`npm run build`, publish `dist`, and set `VITE_API_URL` to the Render API URL.

The included GitHub Actions workflow runs on weekdays at 18:00 WIB and calls
the protected refresh endpoint. Add repository secrets named `API_URL` and
`REFRESH_TOKEN`; use the workflow dispatch button for a manual refresh.

## Verification

```bash
pytest -q
python3 -m py_compile src/*.py api/*.py
cd web && npm run build && npm test
```
