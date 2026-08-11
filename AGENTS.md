# Repository Guidelines

## Project Structure

- `src/` contains the Python research workflow:
  `download_data.py` caches Yahoo Finance OHLCV data, `research.py` runs monthly
  backtests, and `robustness.py` produces cost and subperiod comparisons.
- `data/universe_idx30_2026-08.csv` defines the current research universe.
  `data/raw/` stores downloaded source files and provenance metadata.
- `reports/` contains generated metrics, summaries, and findings. Most raw and
  generated outputs are intentionally ignored by Git.
- `README.md` and `DATA_SOURCES.md` document the methodology and provenance.

There is currently no dedicated test directory.

## Development Commands

Use Python 3.10+ from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -m py_compile src/*.py
python3 src/download_data.py
python3 src/research.py
python3 src/robustness.py
python3 src/research.py --price-field close
```

The downloader caches daily data; rerun with `--force` only when a refresh is
intended. Backtests default to adjusted prices, 25 bps one-way costs, monthly
signals, and a three-stock portfolio. Network-dependent commands should record
the retrieval date and source changes in the resulting report.

## Coding Style

Follow standard Python style: four-space indentation, `snake_case` for modules,
functions, and variables, `PascalCase` only for classes, and type hints for
public functions. Keep data transformations explicit and document assumptions
about timing, adjusted prices, costs, and benchmark construction. No formatter
or linter is configured; `py_compile` is the minimum pre-change check.

## Testing Guidelines

There is no automated test suite or coverage target yet. At minimum, compile all
modules and run the research and robustness commands after changing data logic.
Check that reports contain non-empty train, validation, and holdout periods and
that no signal uses the same month’s future return.

## Commits and Pull Requests

No Git history or established commit convention is present. Use concise,
imperative subjects such as `Add point-in-time universe loader`. Pull requests
should explain the research or code change, list commands run, identify data
retrieval dates, and report any changes to costs, benchmark, universe, or
look-ahead protections. Include updated documentation and generated findings
when methodology or results change.

## Data and Security

Do not commit credentials, private market-data keys, or unreviewed external
files. Treat Yahoo Finance and IDX data as research inputs, preserve source
metadata, and do not present backtest results as guaranteed investment advice.
