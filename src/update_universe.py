"""Fetch the current listed-stock catalog from the official IDX market-data endpoint."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "universe_idx_all_2026-08.csv"
METADATA_PATH = ROOT / "data" / "raw" / "idx_universe_metadata.json"
SOURCE_URL = "https://block.idx.id/primary/StockData/GetSecuritiesStock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Destination CSV path")
    parser.add_argument("--language", default="id", choices=("id", "en"))
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def fetch_catalog(language: str = "id", timeout: int = 30) -> pd.DataFrame:
    """Return all securities reported by IDX's public stock-list endpoint."""
    response = requests.get(
        SOURCE_URL,
        params={
            "start": 0,
            "length": 9999,
            "code": "",
            "sector": "",
            "board": "",
            "language": language,
        },
        headers={"User-Agent": "beat-the-market universe research/0.1"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    expected = int(payload.get("recordsFiltered", payload.get("recordsTotal", len(rows))))
    if not rows or len(rows) != expected:
        raise RuntimeError(f"IDX returned {len(rows)} rows but reported {expected}")
    frame = pd.DataFrame(rows).rename(
        columns={
            "Code": "ticker",
            "Name": "company_name",
            "ListingDate": "listing_date",
            "Shares": "shares",
            "ListingBoard": "listing_board",
        }
    )
    required = ["ticker", "company_name", "listing_date", "shares", "listing_board"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(f"IDX response is missing columns: {missing}")
    frame = frame[required].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["company_name"] = frame["company_name"].astype(str).str.strip()
    frame["listing_date"] = pd.to_datetime(frame["listing_date"], errors="coerce").dt.date.astype("string")
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce")
    frame["listing_board"] = frame["listing_board"].astype(str).str.strip()
    frame = frame.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    if frame["ticker"].eq("").any() or frame["ticker"].isna().any():
        raise RuntimeError("IDX returned an empty ticker")
    return frame


def write_catalog(frame: pd.DataFrame, output: Path = DEFAULT_OUTPUT, language: str = "id") -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    metadata = {
        "source": "Indonesia Stock Exchange (IDX) public stock-list endpoint",
        "source_url": SOURCE_URL,
        "language": language,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrieved_on": date.today().isoformat(),
        "rows": int(len(frame)),
        "output": str(output.relative_to(ROOT)),
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    frame = fetch_catalog(args.language, args.timeout)
    output = write_catalog(frame, Path(args.output), args.language)
    print(f"Wrote {len(frame):,} IDX stock tickers to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
