"""Universe paths and catalog helpers shared by the API and download jobs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IDX30_PATH = ROOT / "data" / "universe_idx30_2026-08.csv"
ALL_STOCKS_PATH = ROOT / "data" / "universe_idx_all_2026-08.csv"


def load_idx30() -> pd.DataFrame:
    return pd.read_csv(IDX30_PATH)


def load_all_stocks() -> pd.DataFrame:
    """Load the official IDX catalog, failing clearly if it has not been refreshed."""
    if not ALL_STOCKS_PATH.exists():
        raise FileNotFoundError(f"Missing {ALL_STOCKS_PATH}; run `python3 src/update_universe.py` first")
    return pd.read_csv(ALL_STOCKS_PATH)
