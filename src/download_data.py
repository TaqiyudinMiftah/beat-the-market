"""Download cached daily OHLCV data for an IDX research or catalog universe.

The downloader deliberately uses Yahoo Finance's chart endpoint directly so the
research can be reproduced without a vendor-specific Python package. It is not
intended for real-time trading or guaranteed data completeness.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "data" / "universe_idx30_2026-08.csv"
ALL_UNIVERSE_PATH = ROOT / "data" / "universe_idx_all_2026-08.csv"
OUT_DIR = ROOT / "data" / "raw" / "yahoo"
METADATA_PATH = ROOT / "data" / "raw" / "yahoo_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01", help="Inclusive YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default="2026-08-12",
        help="Exclusive YYYY-MM-DD; defaults to the day after the supplied research date",
    )
    parser.add_argument("--sleep", type=float, default=0.35, help="Seconds between requests")
    parser.add_argument("--force", action="store_true", help="Redownload existing files")
    parser.add_argument(
        "--universe",
        choices=("idx30", "all"),
        default="idx30",
        help="Download the IDX30 research universe or all listed IDX stocks (about 962 symbols)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for downloaded CSV files; use a separate directory for all-stock experiments",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=METADATA_PATH,
        help="Path for the refresh manifest",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record unavailable symbols and continue downloading the remaining universe",
    )
    return parser.parse_args()


def epoch_seconds(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def filename_for(ticker: str) -> str:
    return ticker.replace("^", "INDEX_").replace("/", "_") + ".csv"


def fetch_chart(
    session: requests.Session,
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    yahoo_ticker = ticker if ticker.startswith("^") else f"{ticker}.JK"
    params = {
        "period1": epoch_seconds(start),
        "period2": epoch_seconds(end),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    errors: list[str] = []
    for host in ("query2.finance.yahoo.com", "query1.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{quote(yahoo_ticker, safe='')}"
        for attempt in range(4):
            try:
                response = session.get(url, params=params, timeout=30)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                chart_error = payload.get("chart", {}).get("error")
                if chart_error:
                    raise RuntimeError(str(chart_error))
                result = payload["chart"]["result"][0]
                timestamps = result.get("timestamp", [])
                quote_block = result["indicators"]["quote"][0]
                adjusted_block = result.get("indicators", {}).get("adjclose", [{}])[0]
                frame = pd.DataFrame(
                    {
                        "date": pd.to_datetime(timestamps, unit="s", utc=True)
                        .tz_convert("Asia/Jakarta")
                        .date,
                        "open": quote_block.get("open"),
                        "high": quote_block.get("high"),
                        "low": quote_block.get("low"),
                        "close": quote_block.get("close"),
                        "adjclose": adjusted_block.get("adjclose"),
                        "volume": quote_block.get("volume"),
                    }
                )
                if frame.empty:
                    raise RuntimeError("Yahoo returned no rows")
                frame = frame.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")
                frame["date"] = pd.to_datetime(frame["date"])
                return frame.reset_index(drop=True)
            except (requests.RequestException, KeyError, IndexError, RuntimeError, ValueError) as exc:
                errors.append(f"{host} attempt {attempt + 1}: {exc}")
                time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"Could not download {ticker} ({yahoo_ticker}): {'; '.join(errors)}")


def _universe_path(universe: str) -> Path:
    if universe == "idx30":
        return UNIVERSE_PATH
    if universe == "all":
        if not ALL_UNIVERSE_PATH.exists():
            raise FileNotFoundError(
                f"Missing {ALL_UNIVERSE_PATH}; run `python3 src/update_universe.py` first"
            )
        return ALL_UNIVERSE_PATH
    raise ValueError(f"Unsupported universe: {universe}")


def download_universe(
    start: str = "2015-01-01",
    end: str = "2026-08-12",
    sleep_seconds: float = 0.35,
    universe: str = "idx30",
    output_dir: Path = OUT_DIR,
    metadata_path: Path = METADATA_PATH,
) -> dict[str, object]:
    """Refresh the configured universe and return a compact refresh manifest."""
    universe_path = _universe_path(universe)
    universe_frame = pd.read_csv(universe_path)
    tickers = ["^JKSE", *universe_frame["ticker"].tolist()]
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "beat-the-market research/0.1"})
    downloaded: list[dict[str, object]] = []
    for ticker in tickers:
        output_path = output_dir / filename_for(ticker)
        frame = fetch_chart(session, ticker, start, end)
        frame.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        downloaded.append(
            {
                "ticker": ticker,
                "file": str(output_path.relative_to(ROOT)),
                "rows": int(len(frame)),
                "first_date": str(frame["date"].min().date()),
                "last_date": str(frame["date"].max().date()),
                "status": "downloaded",
            }
        )
        time.sleep(max(sleep_seconds, 0.0))
    metadata = {
        "source": "Yahoo Finance chart API",
        "source_url_template": "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": start,
        "requested_end_exclusive": end,
        "benchmark": "^JKSE",
        "universe": universe,
        "universe_file": str(universe_path.relative_to(ROOT)),
        "files": downloaded,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")

    universe_path = _universe_path(args.universe)
    universe = pd.read_csv(universe_path)
    tickers = ["^JKSE", *universe["ticker"].tolist()]
    output_dir = args.output_dir.resolve()
    metadata_path = args.metadata_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "beat-the-market research/0.1"})
    downloaded: list[dict[str, object]] = []

    for index, ticker in enumerate(tickers, start=1):
        output_path = output_dir / filename_for(ticker)
        if output_path.exists() and not args.force:
            frame = pd.read_csv(output_path, parse_dates=["date"])
            status = "cached"
        else:
            print(f"[{index}/{len(tickers)}] downloading {ticker}", flush=True)
            try:
                frame = fetch_chart(session, ticker, args.start, args.end)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
                downloaded.append(
                    {
                        "ticker": ticker,
                        "file": str(output_path.relative_to(ROOT)),
                        "rows": 0,
                        "first_date": None,
                        "last_date": None,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                time.sleep(max(args.sleep, 0.0))
                continue
            frame.to_csv(output_path, index=False, date_format="%Y-%m-%d")
            status = "downloaded"
            time.sleep(max(args.sleep, 0.0))
        downloaded.append(
            {
                "ticker": ticker,
                "file": str(output_path.relative_to(ROOT)),
                "rows": int(len(frame)),
                "first_date": str(frame["date"].min().date()),
                "last_date": str(frame["date"].max().date()),
                "status": status,
            }
        )
        print(f"  {status}: {len(frame):,} rows through {frame['date'].max().date()}", flush=True)

    metadata = {
        "source": "Yahoo Finance chart API",
        "source_url_template": "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": args.start,
        "requested_end_exclusive": args.end,
        "benchmark": "^JKSE",
        "universe": args.universe,
        "universe_file": str(universe_path.relative_to(ROOT)),
        "files": downloaded,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata to {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
