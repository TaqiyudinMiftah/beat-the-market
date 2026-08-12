# All-listed deep-learning research

Run date: 2026-08-12<br>
Universe catalog: data/universe_idx_all_2026-08.csv; loaded tickers: 962; skipped: 0<br>
Price field: adjclose; liquidity screen: top 500; top K: 3; cost: 25.0 bps one-way<br>
Signal dates: 138; median eligible stocks: 500.0<br>

## Research basis

This audit adapts the cross-sectional deep-learning setup in [Abe and Nakayama's stock-return forecasting study](https://arxiv.org/abs/1801.01777) and the model-comparison discipline of [Gu, Kelly, and Xiu](https://www.nber.org/papers/w25398). PyTorch is used only for a small factor MLP; the model is not pretrained on this IDX panel.

## Fixed design and leakage controls

- At each completed month-end t, eligibility is computed from trailing 60-day median dollar volume and only the top 500 names are retained.
- The target is the next month's cross-sectional return percentile rank. The MLP sees only rows with signal dates strictly earlier than the current forecast date.
- Feature normalization, target winsorization, and early stopping use training history only. The latest 12 historical signal months form an internal validation slice.
- The network is fixed at hidden layers (32, 16), dropout 0.10, Adam learning rate 0.01, weight decay 0.01, seeds (7, 11, 19), and at most 60 epochs.
- `baseline_mlp_blend_50` is a fixed 50/50 percentile-rank blend. `online_best` and `online_soft` use only realized strategy returns before t; `online_soft` uses the fixed 0.25/0.50/0.75 weights around a 0.25 trailing-Sharpe margin.
- Validation is 2022–2023. 2024 and 2025–2026 are holdouts; no holdout result chooses architecture, seed, or blend weight.

## Audit summary

The fixed gate requires beating the matching top-500 composite control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025–2026.

| model | validation excess | 2024 excess | 2025–2026 excess | rolling positive | bootstrap lower | holdout beta | holdout drawdown | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cap500_composite | 0.1788 | 0.1939 | -0.0409 | 0.7874 | — | 1.2343 | -0.5318 | rejects |
| mlp_rank | 0.1831 | 0.6757 | 0.2755 | 0.5354 | 0.0076 | 0.4976 | -0.2639 | rejects |
| baseline_mlp_blend_50 | 0.6035 | 0.2020 | 0.4634 | 0.8425 | -0.2370 | 1.1068 | -0.5484 | rejects |
| online_best | 0.6065 | 0.6757 | -0.1875 | 0.8110 | -0.2387 | 0.7187 | -0.4952 | rejects |
| online_soft | 0.5035 | 0.5237 | 0.4451 | 0.8425 | -0.0136 | 1.1302 | -0.4720 | rejects |

Validation winner: baseline_mlp_blend_50. Preferred after the fixed gate: none. Any passing row remains exploratory because the catalog is a current snapshot with survivorship and historical-membership bias.

## Limitations

- The current all-listed catalog does not reconstruct delisted names, historical suspensions, or point-in-time index membership.
- Yahoo Finance data omit or simplify spreads, taxes, price limits, market impact, failed fills, and capacity.
- Deep-learning results are sensitive to the universe, price field, and training window; this runner must be stress-tested before any deployment consideration.
- This is research, not investment advice, and no backtest guarantees future performance.

## Reproduction

~~~bash
PYTHONPATH=$PWD /tmp/beat-market-ml-venv/bin/python -m src.all_stock_deep_research
~~~

Raw forecasts remain ignored in reports/all_stock_deep_cap500_forecasts.csv. Committed tables use the all_stock_deep_cap500_ prefix.
