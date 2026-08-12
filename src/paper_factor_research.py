"""Audit a predeclared momentum, low-volatility, and low-beta factor score.

The experiment adapts two established findings to the repository's long-only
monthly workflow: intermediate-horizon momentum and the betting-against-beta
effect. It is deliberately small and fixed before the holdout is inspected.
The beta paper's original BAB portfolio is long-short and leveraged; this
module tests only a long-only low-beta ranking combined with momentum and a
volatility control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from . import all_stock_ml_research, foundation_robustness, liquid_rank_ml_research
    from . import ml_research, research
except ImportError:  # pragma: no cover
    from src import all_stock_ml_research, foundation_robustness
    from src import liquid_rank_ml_research, ml_research, research


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "yahoo_all"
DEFAULT_UNIVERSE_PATH = ROOT / "data" / "universe_idx_all_2026-08.csv"
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-08-11"
DEFAULT_CAP = 300
DEFAULT_TOP_K = 3
DEFAULT_COST_BPS = 25.0
DEFAULT_MIN_TRAIN_ROWS = 1_000
DEFAULT_ALPHA = 10.0
BETAS_WINDOW_DAYS = 252
BETAS_MIN_PERIODS = 126
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_BLOCK_LENGTH = 3
BOOTSTRAP_SEED = 20260812
TOP_K_SENSITIVITY = (1, 3, 5, 10)
COST_SENSITIVITY_BPS = (25.0, 100.0)
RISK_BETA_LIMIT = 1.50
RISK_DRAWDOWN_LIMIT = -0.60
RISK_TURNOVER_LIMIT = 0.90

MOMENTUM_SOURCE = "https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb04702.x"
BETA_SOURCE = "https://www.nber.org/papers/w16601"
IDX_METHOD_SOURCE = "https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf"


@dataclass(frozen=True)
class PaperFactorConfig:
    price_field: str
    start: str
    end: str
    cap: int
    top_k: int
    cost_bps: float
    min_train_rows: int
    alpha: float
    data_dir: Path
    universe_path: Path


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-field", choices=("adjclose", "close"), default="adjclose")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE_PATH)
    return parser.parse_args()


def rolling_beta(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = BETAS_WINDOW_DAYS,
    min_periods: int = BETAS_MIN_PERIODS,
) -> pd.DataFrame:
    """Estimate trailing stock beta using only returns through each date."""
    if window < 3 or not 2 <= min_periods <= window:
        raise ValueError("window and min_periods are invalid")
    market_mean = market_returns.rolling(window, min_periods=min_periods).mean()
    stock_mean = stock_returns.rolling(window, min_periods=min_periods).mean()
    product_mean = stock_returns.mul(market_returns, axis=0).rolling(
        window,
        min_periods=min_periods,
    ).mean()
    covariance = product_mean.subtract(stock_mean.mul(market_mean, axis=0))
    market_variance = market_returns.pow(2).rolling(
        window,
        min_periods=min_periods,
    ).mean() - market_mean.pow(2)
    return covariance.divide(market_variance.replace(0.0, np.nan), axis=0)


def build_beta_feature(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a month-end trailing beta panel against the IDX Composite."""
    if "^JKSE" not in prices.columns:
        raise KeyError("prices must contain the ^JKSE benchmark")
    stocks = prices.drop(columns="^JKSE")
    returns = prices.pct_change(fill_method=None)
    beta = rolling_beta(returns[stocks.columns], returns["^JKSE"])
    result = research.completed_monthly_last(beta)
    assert isinstance(result, pd.DataFrame)
    return result


def _rank_within(
    values: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    low_is_good: bool = False,
) -> pd.DataFrame:
    prepared = -values if low_is_good else values
    return prepared.where(eligible).rank(axis=1, pct=True, method="average")


def paper_score(
    features: dict[str, pd.DataFrame | pd.Series],
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    """Return the fixed equal-weight paper-factor score within the screen."""
    momentum = features["mom12_1"]
    volatility = features["volatility"]
    beta = features["beta_252d"]
    assert isinstance(momentum, pd.DataFrame)
    assert isinstance(volatility, pd.DataFrame)
    assert isinstance(beta, pd.DataFrame)
    ranks = (
        _rank_within(momentum, eligible)
        + _rank_within(volatility, eligible, low_is_good=True)
        + _rank_within(beta, eligible, low_is_good=True)
    ) / 3.0
    return ranks.where(eligible)


def _metric_periods(end: str, start: str) -> tuple[tuple[str, str, str], ...]:
    return (
        ("train", start, "2021-12-31"),
        ("validation", "2022-01-01", "2023-12-31"),
        ("holdout_2024", "2024-01-01", "2024-12-31"),
        ("holdout_2025_2026", "2025-01-01", end),
        ("full", start, end),
    )


def _format(value: Any) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _blocks(end: str) -> tuple[tuple[str, str, str], ...]:
    return (
        ("train_2021", "2021-01-01", "2021-12-31"),
        ("validation_2022_2023", "2022-01-01", "2023-12-31"),
        ("holdout_2024", "2024-01-01", "2024-12-31"),
        ("holdout_2025_2026", "2025-01-01", end),
    )


def _simulate(
    panels: dict[str, pd.DataFrame],
    features: dict[str, pd.DataFrame | pd.Series],
    top_k: int,
    cost_bps: float,
) -> dict[str, pd.DataFrame]:
    return {
        name: ml_research.simulate_predictions(panel, features, top_k, cost_bps)
        for name, panel in panels.items()
    }


def _metrics(
    simulations: dict[str, pd.DataFrame],
    start: str,
    end: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        name: {
            label: ml_research.period_metrics(simulation, period_start, period_end, label)
            for label, period_start, period_end in _metric_periods(end, start)
        }
        for name, simulation in simulations.items()
    }


def _bootstrap_rows(
    simulations: dict[str, pd.DataFrame],
    end: str,
    control_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position, (name, simulation) in enumerate(simulations.items()):
        if name == control_name:
            continue
        market = simulation[["benchmark_return"]].rename(
            columns={"benchmark_return": "strategy_return"}
        )
        market_result = liquid_rank_ml_research._block_bootstrap_active(
            simulation,
            market,
            "2024-01-01",
            end,
            BOOTSTRAP_SEED + position,
        )
        control_result = liquid_rank_ml_research._block_bootstrap_active(
            simulation,
            simulations[control_name],
            "2024-01-01",
            end,
            BOOTSTRAP_SEED + 100 + position,
        )
        rows.append(
            {
                "model": name,
                "months": market_result["months"],
                "market_active_annualized_mean": market_result["mean"],
                "market_ci_lower": market_result["lower"],
                "market_ci_median": market_result["median"],
                "market_ci_upper": market_result["upper"],
                "market_positive_probability": market_result["positive_probability"],
                "control_active_annualized_mean": control_result["mean"],
                "control_ci_lower": control_result["lower"],
                "control_ci_median": control_result["median"],
                "control_ci_upper": control_result["upper"],
                "control_positive_probability": control_result["positive_probability"],
                "samples": BOOTSTRAP_SAMPLES,
                "block_length": BOOTSTRAP_BLOCK_LENGTH,
            }
        )
    return pd.DataFrame(rows)


def _gate(
    model: str,
    metrics: dict[str, dict[str, Any]],
    diagnostics: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    control_name: str,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    control_validation = diagnostics.loc[
        diagnostics["model"].eq(control_name)
        & diagnostics["period"].eq("validation_2022_2023")
        & np.isclose(diagnostics["cost_bps"], 25.0)
    ]
    candidate_validation = diagnostics.loc[
        diagnostics["model"].eq(model)
        & diagnostics["period"].eq("validation_2022_2023")
        & np.isclose(diagnostics["cost_bps"], 25.0)
    ]
    if (
        control_validation.empty
        or candidate_validation.empty
        or not np.isfinite(candidate_validation.iloc[0]["excess_cagr"])
        or candidate_validation.iloc[0]["excess_cagr"] <= control_validation.iloc[0]["excess_cagr"]
    ):
        reasons.append("validation excess CAGR does not beat the matching composite control")
    for cost in COST_SENSITIVITY_BPS:
        for period in ("holdout_2024", "holdout_2025_2026"):
            rows = diagnostics.loc[
                diagnostics["model"].eq(model)
                & diagnostics["period"].eq(period)
                & np.isclose(diagnostics["cost_bps"], cost)
            ]
            if rows.empty or not np.isfinite(rows.iloc[0]["excess_cagr"]) or rows.iloc[0]["excess_cagr"] <= 0:
                reasons.append(f"excess CAGR is not positive in {period} at {int(cost)} bps")
    rolling = rolling_summary.loc[
        rolling_summary["model"].eq(model) & np.isclose(rolling_summary["cost_bps"], 25.0)
    ]
    if rolling.empty or rolling.iloc[0]["positive_excess_fraction"] < 0.60:
        reasons.append("fewer than 60% of trailing 12-month windows have positive excess CAGR")
    bootstrap_row = bootstrap.loc[bootstrap["model"].eq(model)]
    if bootstrap_row.empty or not np.isfinite(bootstrap_row.iloc[0]["market_ci_lower"]) or bootstrap_row.iloc[0]["market_ci_lower"] <= 0:
        reasons.append("95% block-bootstrap lower CI versus IHSG is not above zero")
    holdout = metrics.get("holdout_2025_2026", {})
    if not np.isfinite(holdout.get("beta_to_benchmark", np.nan)) or holdout["beta_to_benchmark"] > RISK_BETA_LIMIT:
        reasons.append(f"holdout beta exceeds the fixed {RISK_BETA_LIMIT:.2f} risk limit")
    if not np.isfinite(holdout.get("strategy_max_drawdown", np.nan)) or holdout["strategy_max_drawdown"] < RISK_DRAWDOWN_LIMIT:
        reasons.append(f"holdout drawdown is worse than the fixed {RISK_DRAWDOWN_LIMIT:.2f} risk limit")
    if not np.isfinite(holdout.get("average_monthly_turnover", np.nan)) or holdout["average_monthly_turnover"] > RISK_TURNOVER_LIMIT:
        reasons.append(f"holdout turnover exceeds the fixed {RISK_TURNOVER_LIMIT:.2f} risk limit")
    return not reasons, reasons


def _summary_table(
    metrics: dict[str, dict[str, dict[str, Any]]],
    rolling_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    gates: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "| model | validation excess | 2024 excess | 2025–2026 excess | rolling positive | bootstrap lower | holdout beta | holdout drawdown | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, values in metrics.items():
        rolling_row = rolling_summary.loc[
            rolling_summary["model"].eq(model) & np.isclose(rolling_summary["cost_bps"], 25.0)
        ]
        bootstrap_row = bootstrap.loc[bootstrap["model"].eq(model)]
        validation = values["validation"]
        holdout_2024 = values["holdout_2024"]
        holdout = values["holdout_2025_2026"]
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    _format(validation.get("excess_cagr")),
                    _format(holdout_2024.get("excess_cagr")),
                    _format(holdout.get("excess_cagr")),
                    _format(rolling_row.iloc[0]["positive_excess_fraction"] if not rolling_row.empty else np.nan),
                    _format(bootstrap_row.iloc[0]["market_ci_lower"] if not bootstrap_row.empty else np.nan),
                    _format(holdout.get("beta_to_benchmark")),
                    _format(holdout.get("strategy_max_drawdown")),
                    "passes" if gates.get(model, {}).get("passed") else "rejects",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def run(config: PaperFactorConfig) -> dict[str, Any]:
    if config.cap < 30 or config.top_k < 1 or config.top_k > 30:
        raise ValueError("cap must be at least 30 and top_k must be between 1 and 30")
    if config.cost_bps < 0 or config.min_train_rows < 60 or config.alpha <= 0:
        raise ValueError("cost, minimum training rows, and alpha are invalid")
    prices, volumes, loaded_tickers, skipped = all_stock_ml_research.load_prices(
        config.data_dir,
        config.universe_path,
        config.price_field,
    )
    prices = prices.loc[prices.index <= config.end]
    base_features = research.make_features(prices, volumes.reindex(prices.index))
    features = all_stock_ml_research._slice_features(base_features, config.start, config.end)
    features["beta_252d"] = build_beta_feature(prices).loc[config.start:config.end]

    eligible = liquid_rank_ml_research.eligibility_mask(features, config.cap)
    cap_label = str(config.cap)
    control_name = f"cap{cap_label}_composite"
    rank_ridge_name = f"cap{cap_label}_rank_ridge"
    panels: dict[str, pd.DataFrame] = {
        control_name: research.score_frame("composite", features).where(eligible),
        "paper_factor": paper_score(features, eligible),
    }
    base_panel = ml_research.make_panel(features)
    filtered_panel = liquid_rank_ml_research.filter_panel(base_panel, eligible)
    rank_panel = liquid_rank_ml_research.rank_target_panel(filtered_panel)
    predictions, training_counts = ml_research.walk_forward_predictions(
        rank_panel,
        "ridge",
        config.min_train_rows,
        config.alpha,
    )
    panels[rank_ridge_name] = predictions

    simulations = _simulate(panels, features, config.top_k, config.cost_bps)
    metrics = _metrics(simulations, config.start, config.end)
    diagnostics = foundation_robustness.build_cost_and_block_diagnostics(
        panels,
        features,
        config.top_k,
        costs_bps=(0.0, 25.0, 50.0, 100.0),
        blocks=_blocks(config.end),
    )
    rolling = foundation_robustness.build_rolling_diagnostics(
        panels,
        features,
        config.top_k,
        cost_bps=config.cost_bps,
    )
    rolling_summary = foundation_robustness.summarize_rolling_diagnostics(rolling)
    bootstrap = _bootstrap_rows(simulations, config.end, control_name)

    candidate_names = [name for name in panels if name != control_name]
    gates = {
        name: dict(zip(("passed", "reasons"), _gate(
            name,
            metrics[name],
            diagnostics,
            rolling_summary,
            bootstrap,
            control_name,
        )))
        for name in candidate_names
    }
    validation_winner = max(
        candidate_names,
        key=lambda name: (
            metrics[name]["validation"].get("strategy_sharpe_rf0", -np.inf),
            metrics[name]["validation"].get("excess_cagr", -np.inf),
        ),
    )
    preferred = validation_winner if gates[validation_winner]["passed"] else None

    sensitivity_rows: list[dict[str, Any]] = []
    for top_k in TOP_K_SENSITIVITY:
        for cost_bps in COST_SENSITIVITY_BPS:
            replay = _simulate(panels, features, top_k, cost_bps)
            for model, simulation in replay.items():
                for period, period_start, period_end in _metric_periods(config.end, config.start):
                    if period in {"validation", "holdout_2024", "holdout_2025_2026"}:
                        sensitivity_rows.append(
                            {
                                "model": model,
                                "top_k": top_k,
                                "cost_bps": cost_bps,
                                **ml_research.period_metrics(
                                    simulation,
                                    period_start,
                                    period_end,
                                    period,
                                ),
                            }
                        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    context = pd.DataFrame(
        {
            "signal_date": eligible.index,
            "eligible_count": eligible.sum(axis=1),
            "paper_complete_count": panels["paper_factor"].notna().sum(axis=1),
        }
    )
    training = pd.DataFrame(
        [
            {
                "model": rank_ridge_name,
                "signals": int(predictions.notna().any(axis=1).sum()),
                "forecast_rows": int(predictions.notna().sum().sum()),
                "median_training_rows": float(training_counts.dropna().median()),
            }
        ]
    )

    metrics_rows = [
        {"model": model, **values}
        for model, model_values in metrics.items()
        for _, values in model_values.items()
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": REPORT_DIR / "paper_factor_metrics.csv",
        "robustness": REPORT_DIR / "paper_factor_robustness.csv",
        "rolling": REPORT_DIR / "paper_factor_rolling.csv",
        "rolling_summary": REPORT_DIR / "paper_factor_rolling_summary.csv",
        "bootstrap": REPORT_DIR / "paper_factor_bootstrap.csv",
        "sensitivity": REPORT_DIR / "paper_factor_sensitivity.csv",
        "context": REPORT_DIR / "paper_factor_context.csv",
        "training": REPORT_DIR / "paper_factor_training.csv",
        "summary": REPORT_DIR / "paper_factor_summary.json",
        "findings": REPORT_DIR / "paper_factor_findings.md",
    }
    pd.DataFrame(metrics_rows).to_csv(paths["metrics"], index=False)
    diagnostics.to_csv(paths["robustness"], index=False)
    rolling.to_csv(paths["rolling"], index=False)
    rolling_summary.to_csv(paths["rolling_summary"], index=False)
    bootstrap.to_csv(paths["bootstrap"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    context.to_csv(paths["context"], index=False)
    training.to_csv(paths["training"], index=False)

    summary = {
        "run_date_utc": datetime.now(timezone.utc).date().isoformat(),
        "price_field": config.price_field,
        "start": config.start,
        "end": config.end,
        "cap": config.cap,
        "top_k": config.top_k,
        "one_way_cost_bps": config.cost_bps,
        "loaded_tickers": len(loaded_tickers),
        "skipped_tickers": skipped,
        "formula": "equal percentile ranks of 12-1 momentum, negative 60-day volatility, and negative 252-day beta",
        "signal_timing": "signal at completed month-end t; hold during month t+1",
        "validation_winner": validation_winner,
        "preferred_after_gate": preferred,
        "gates": gates,
        "sources": {
            "momentum": MOMENTUM_SOURCE,
            "low_beta": BETA_SOURCE,
            "idx_methodology": IDX_METHOD_SOURCE,
        },
    }
    paths["summary"].write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    report_lines = [
        "# Paper-factor audit: momentum, low volatility, and low beta",
        "",
        f"Run date: {summary['run_date_utc']}<br>",
        f"Price field: `{config.price_field}`; current catalog loaded: `{len(loaded_tickers)}`; skipped: `{len(skipped)}`<br>",
        f"Liquidity screen: trailing 60-day median dollar-volume top `{config.cap}`<br>",
        f"Portfolio: top `{config.top_k}` equal-weight names; monthly signal; `{config.cost_bps}` bps one-way default cost<br>",
        "",
        "## Research basis",
        "",
        f"- [Jegadeesh and Titman (1993)]({MOMENTUM_SOURCE}) document intermediate-horizon winner-minus-loser momentum. This audit uses their 12-month lookback with the most recent month skipped.",
        f"- [Frazzini and Pedersen, Betting Against Beta]({BETA_SOURCE}) study a long-short, leveraged BAB factor across markets. This repository cannot reproduce that portfolio with its long-only app, so it tests only a low-beta cross-sectional rank combined with momentum and low volatility.",
        f"- The liquidity screen is a conservative proxy informed by the [official IDX methodology]({IDX_METHOD_SOURCE}); it is not historical IDX30 membership.",
        "",
        "## Fixed formula",
        "",
        "For each eligible stock at completed month-end t:",
        "",
        "~~~text",
        "score = (rank(momentum_12_to_1)",
        "       + rank(-volatility_60d)",
        "       + rank(-beta_252d_vs_IHSG)) / 3",
        "~~~",
        "",
        f"The ranks are computed within the eligible top-{config.cap} screen. The highest {config.top_k} scores are held equally during the next month. No weights, top-K, or holdout periods were selected after seeing the holdout.",
        "",
        "## Leakage controls and limitations",
        "",
        "- Beta and volatility are trailing daily estimates through t; the following month's return is never in the signal.",
        "- The rank-Ridge comparator is fit expanding-window with training signal dates strictly earlier than each forecast.",
        "- The current all-listed catalog still omits historical delistings and membership changes. This is an audit of a proxy universe, not a live signal.",
        "- Yahoo Finance data do not model spreads, taxes, price limits, market impact, failed fills, or capacity.",
        "",
        "## Audit summary",
        "",
        "The fixed gate requires beating the matching composite control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive trailing 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025-2026.",
        "",
        _summary_table(metrics, rolling_summary, bootstrap, gates),
        "",
        f"Validation winner under the predeclared rule: **{validation_winner}**. Preferred after the fixed gate: **{preferred or 'none'}**.",
        "",
        "## Reproduction",
        "",
        "~~~bash",
        "PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.paper_factor_research",
        "~~~",
        "",
        "Committed tables are `paper_factor_metrics.csv`, `paper_factor_robustness.csv`, `paper_factor_rolling_summary.csv`, `paper_factor_bootstrap.csv`, `paper_factor_sensitivity.csv`, `paper_factor_context.csv`, and `paper_factor_training.csv`.",
        "",
        "This is research, not investment advice, and no backtest guarantees future performance.",
        "",
    ]
    paths["findings"].write_text(
        "\n".join(report_lines).replace("`", chr(96)),
        encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "gates": gates,
        "validation_winner": validation_winner,
        "preferred": preferred,
        "paths": paths,
    }


def main() -> None:
    args = parse_args()
    config = PaperFactorConfig(
        price_field=args.price_field,
        start=args.start,
        end=args.end,
        cap=args.cap,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        min_train_rows=args.min_train_rows,
        alpha=args.alpha,
        data_dir=args.data_dir.resolve(),
        universe_path=args.universe.resolve(),
    )
    result = run(config)
    summary = pd.DataFrame(
        [
            {
                "model": model,
                "validation_excess_cagr": values["validation"].get("excess_cagr"),
                "holdout_2024_excess_cagr": values["holdout_2024"].get("excess_cagr"),
                "holdout_2025_2026_excess_cagr": values["holdout_2025_2026"].get("excess_cagr"),
                "gate": result["gates"].get(model, {}).get("passed"),
            }
            for model, values in result["metrics"].items()
        ]
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    print("\nWrote reports/paper_factor_findings.md and reports/paper_factor_metrics.csv")


if __name__ == "__main__":
    main()
