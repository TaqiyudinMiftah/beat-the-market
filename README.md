# Beat the Indonesian market — reproducible research

This repository tests whether simple, liquid-universe signals have historically
outperformed the IDX Composite (`^JKSE`) after turnover costs. It is research,
not investment advice and not a guarantee of future returns.

## Current first-pass design

- Research universe: the 30 stocks in the IDX30 snapshot effective 3 August–30
  October 2026, stored in `data/universe_idx30_2026-08.csv`.
- App catalog: all 962 stocks returned by the official IDX stock-list endpoint
  on 12 August 2026, stored in `data/universe_idx_all_2026-08.csv`. The app can
  display every catalog ticker, but a chart requires a locally cached price
  file. This separation prevents an incomplete all-stock download from
  silently changing the published IDX30 backtest.
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
python3 -m src.ml_research
```

The paper-backed ML experiment uses the optional environment in
`requirements-ml.txt` and writes `reports/ml_research_findings.md`. It compares
expanding-window Ridge, shallow LightGBM, an equal ensemble, an
inverse-volatility variant, and the existing composite control. The report
links the source papers and records validation/holdout results; it does not
claim that any result is a live edge.

The broader current-catalog experiment applies the same paper-backed
cross-sectional comparison to all 962 tickers returned by the IDX catalog. It
uses the separately cached `data/raw/yahoo_all/` directory so it cannot change
the published IDX30 backtest by accident:

```bash
python3 src/download_data.py --universe all --start 2015-01-01 \
  --end 2026-08-12 --output-dir data/raw/yahoo_all \
  --metadata-path data/raw/yahoo_all_metadata.json --continue-on-error
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.all_stock_ml_research
```

See `reports/all_stock_ml_findings.md` and its committed CSV tables for the
walk-forward LightGBM/Ridge comparison, fixed top-K sensitivity, costs,
rolling windows, and bootstrap diagnostics. The catalog is a current snapshot,
not a point-in-time universe; its results are exploratory and are not promoted
to the app or live trading.

The stricter follow-up applies the official [IDX80/LQ45/IDX30 methodology](https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf)
as a design constraint: it uses trailing 60-day dollar-volume caps of 150, 300,
500, and all eligible names, then compares raw-return and cross-sectional
rank-target Ridge/LightGBM models in expanding walk-forward tests. Its fixed
gate also checks costs, rolling windows, bootstrap uncertainty, beta, drawdown,
and turnover:

```bash
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.liquid_rank_ml_research
```

See `reports/liquid_rank_ml_findings.md` for the full audit. The current
exploratory survivor is `cap300_rank_ridge`; it is not a live recommendation,
and the current catalog still cannot remove survivorship or historical
membership bias.

The foundation-model runner executes real zero-shot rolling forecasts from
Chronos, TimesFM, and Kronos in a separate heavyweight CPU environment:

```bash
python3 -m venv /tmp/beat-market-foundation
/tmp/beat-market-foundation/bin/python -m pip install -r requirements-foundation-cpu.txt
git clone https://github.com/shiyu-coder/Kronos.git /tmp/Kronos
PYTHONPATH=$PWD:/tmp/Kronos /tmp/beat-market-foundation/bin/python -m src.foundation_research \
  --models all --kronos-repo /tmp/Kronos
```

The checked-in foundation report uses fixed model IDs, 256 daily context rows,
a 21-day horizon, top-three selection, 25 bps costs, and 2021–2026 rolling
signals. It keeps holdout data out of model selection. See
`reports/foundation_model_findings.md` for the actual comparison. The runner
also replays the same forecasts at 0/25/50/100 bps and across fixed
chronological blocks; the auditable tables are
`reports/foundation_model_robustness.csv` and
`reports/foundation_model_rolling_summary.csv`. Factor warm-up history is
loaded before the requested evaluation window, while all reported signals are
still restricted to the configured dates.

The deep-learning experiment is a separate optional CPU run. It trains a small
three-seed factor MLP with an internal historical validation slice, then tests
fixed blends and causal online weighting against the existing composite:

```bash
python3 -m venv /tmp/beat-market-deep
/tmp/beat-market-deep/bin/python -m pip install -r requirements-deep-cpu.txt
PYTHONPATH=$PWD /tmp/beat-market-deep/bin/python -m src.deep_ml_research
```

See `reports/deep_ml_findings.md` for the current gate decision. The online
weighting rule may use only realized returns before each signal month; it never
uses the current or later holdout return.

The forecast-stacking experiment combines the completed Chronos, TimesFM,
Kronos, and factor-MLP forecast panels with the existing composite. It tests a
regularized Ridge stacker, shallow LightGBM stacker, and fixed rank blends using
an expanding walk-forward fit. Run it after the two forecast files exist:

```bash
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.stacked_research
```

See `reports/stacked_model_findings.md` for the paper basis, missing-forecast
coverage, cost stress, block bootstrap, and fixed audit gate. A passing
historical candidate is not promoted automatically: the current IDX30 panel
still has survivorship bias and the app's full ticker catalog is not a
point-in-time all-stock research universe.

Refresh the official catalog and, when a full quote snapshot is intended, use:

```bash
python3 src/update_universe.py
python3 src/download_data.py --universe all
```

The all-stock download requests roughly 962 symbols and can take a long time or
encounter Yahoo Finance rate limits. The default downloader remains `--universe
idx30`; use `DATA_UNIVERSE=all` only for an intentional full refresh.

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

`/api/universe` returns the full IDX catalog with `has_data` and
`in_research_universe` flags. Rankings and backtests intentionally remain on
IDX30 until a point-in-time all-stock research panel is built.

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
