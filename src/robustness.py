"""Produce subperiod and cost/top-K robustness tables for the first-pass study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research  # noqa: E402


REPORT_DIR = ROOT / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-field", choices=("adjclose", "close"), default="adjclose")
    parser.add_argument("--cost-bps", type=float, default=25.0)
    parser.add_argument("--end", default="2026-08-11")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices, volumes, _ = research.load_prices(args.price_field)
    prices = prices.loc[prices.index <= args.end]
    volumes = volumes.reindex(prices.index)
    features = research.make_features(prices, volumes)

    strategies = [
        "equal_weight",
        "mom12_1",
        "mom6_1",
        "mom3",
        "lowvol",
        "trend_mom",
        "composite",
        "composite_regime",
    ]
    periods = {
        "2015-2017": ("2015-01-01", "2017-12-31"),
        "2018-2020": ("2018-01-01", "2020-12-31"),
        "2021-2023": ("2021-01-01", "2023-12-31"),
        "2024-2026_holdout": ("2024-01-01", args.end),
    }
    top_ks = [3, 5, 8, 10]
    costs = [0.0, 25.0, 50.0]
    grid_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for top_k in top_ks:
        for cost in costs:
            for strategy in strategies:
                simulation = research.simulate(strategy, features, top_k, cost)
                subperiod_excess: list[float] = []
                for period, (start, end) in periods.items():
                    metrics = research.metric_row(simulation, start, end, period)
                    excess = metrics.get("excess_cagr", np.nan)
                    if pd.notna(excess):
                        subperiod_excess.append(float(excess))
                    grid_rows.append(
                        {
                            "strategy": strategy,
                            "top_k": top_k,
                            "cost_bps": cost,
                            "period": period,
                            **metrics,
                            "price_field": args.price_field,
                        }
                    )
                full = research.metric_row(simulation, "2015-01-01", args.end, "full")
                holdout = research.metric_row(simulation, "2024-01-01", args.end, "holdout")
                summary_rows.append(
                    {
                        "strategy": strategy,
                        "top_k": top_k,
                        "cost_bps": cost,
                        "full_cagr": full.get("strategy_cagr", np.nan),
                        "full_excess_cagr": full.get("excess_cagr", np.nan),
                        "full_max_drawdown": full.get("strategy_max_drawdown", np.nan),
                        "full_information_ratio": full.get("information_ratio", np.nan),
                        "holdout_cagr": holdout.get("strategy_cagr", np.nan),
                        "holdout_excess_cagr": holdout.get("excess_cagr", np.nan),
                        "holdout_max_drawdown": holdout.get("strategy_max_drawdown", np.nan),
                        "holdout_information_ratio": holdout.get("information_ratio", np.nan),
                        "positive_excess_periods": int(sum(value > 0 for value in subperiod_excess)),
                        "median_excess_cagr": float(np.median(subperiod_excess)) if subperiod_excess else np.nan,
                        "worst_excess_cagr": float(min(subperiod_excess)) if subperiod_excess else np.nan,
                        "price_field": args.price_field,
                    }
                )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(grid_rows)
    summary = pd.DataFrame(summary_rows)
    grid.to_csv(REPORT_DIR / f"robustness_grid_{args.price_field}.csv", index=False)
    summary.to_csv(REPORT_DIR / f"robustness_summary_{args.price_field}.csv", index=False)

    candidate = summary[(summary["strategy"] == "composite") & (summary["top_k"] == 3) & (summary["cost_bps"] == 25.0)].iloc[0]
    control = summary[(summary["strategy"] == "equal_weight") & (summary["top_k"] == 3) & (summary["cost_bps"] == 25.0)].iloc[0]
    candidate_periods = grid[(grid["strategy"] == "composite") & (grid["top_k"] == 3) & (grid["cost_bps"] == 25.0)]
    control_periods = grid[(grid["strategy"] == "equal_weight") & (grid["top_k"] == 3) & (grid["cost_bps"] == 25.0)]

    def pct(value: object) -> str:
        return "n/a" if pd.isna(value) else f"{float(value):.1%}"

    def num(value: object) -> str:
        return "n/a" if pd.isna(value) else f"{float(value):.2f}"

    lines = [
        "# Indonesian equity research findings",
        "",
        f"Data field: `{args.price_field}`. One-way cost: `{args.cost_bps:.0f} bps`. Benchmark: `^JKSE`. "
        "The daily panel ends 11 August 2026; incomplete August is excluded from monthly returns, so the holdout ends 31 July 2026.",
        "",
        "## Candidate formula",
        "",
        "The strongest simple candidate in this first pass is a monthly, equal-weight top-three portfolio ranked by:",
        "",
        "```text",
        "score = 0.40 * rank(momentum from t-12 to t-1)",
        "      + 0.30 * rank(momentum from t-6 to t-1)",
        "      + 0.20 * rank(momentum over the latest 3 months)",
        "      + 0.10 * rank(-60-day annualized volatility)",
        "",
        "At month-end t, hold the three highest-scoring eligible stocks equally during month t+1.",
        "```",
        "",
        "The ranks are cross-sectional within the current IDX30 snapshot. No market-regime filter is used in this candidate.",
        "",
        "Because this candidate was chosen during the exploratory robustness sweep, the holdout figures below are descriptive rather than a clean, untouched discovery test.",
        "",
        "## Candidate versus controls",
        "",
        "| Period | Composite top 3 CAGR | IHSG CAGR | Excess CAGR | Max drawdown | Equal-weight CAGR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in candidate_periods.iterrows():
        control_row = control_periods[control_periods["period"] == row["period"]].iloc[0]
        lines.append(
            f"| {row['period']} | {pct(row['strategy_cagr'])} | {pct(row['benchmark_cagr'])} | "
            f"{pct(row['excess_cagr'])} | {pct(row['strategy_max_drawdown'])} | {pct(control_row['strategy_cagr'])} |"
        )
    lines += [
        "",
        f"Across the four chronological blocks, the candidate had positive excess return in "
        f"{int(candidate['positive_excess_periods'])}/4 blocks; its median excess CAGR was "
        f"{pct(candidate['median_excess_cagr'])}, with a worst block of {pct(candidate['worst_excess_cagr'])}.",
        "",
        "## Interpretation",
        "",
        "- The candidate beats the price-return IHSG benchmark in this sample, including the 2024–2026 holdout.",
        "- It is not a free lunch: concentration and monthly turnover create substantially higher drawdown and implementation risk than the equal-weight control.",
        "- The equal-weight control also beats IHSG, so part of the result is likely exposure to the selected liquid-stock universe rather than the ranking formula itself.",
        "- Adjusted-close results include distributions. Run `python3 src/research.py --price-field close` as a price-only sensitivity; raw close data can be distorted by corporate actions.",
        "",
        "## Limitations before live use",
        "",
        "This is an exploratory backtest using the current August–October 2026 IDX30 membership applied historically. It therefore has survivorship and index-membership look-ahead bias. It also does not model delistings, suspensions, price limits, bid/ask spreads, taxes, lot sizes, market impact, or exact execution prices. Historical outperformance is not evidence of a guaranteed future edge.",
        "",
        "The next research upgrade should use point-in-time IDX30/LQ45 constituents, delisted names, and transaction-level execution assumptions before any capital is placed.",
        "",
    ]
    findings_path = REPORT_DIR / f"research_findings_{args.price_field}.md"
    findings_path.write_text("\n".join(lines), encoding="utf-8")
    print(summary.sort_values(["positive_excess_periods", "median_excess_cagr"], ascending=False).head(12).to_string(index=False))
    print(f"\nWrote {findings_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
