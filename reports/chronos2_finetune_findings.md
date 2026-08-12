# Chronos-2 historical-cutoff fine-tuning research

Run date: 2026-08-12<br>
Model: amazon/chronos-2; chronos-forecasting: 2.3.1<br>
Universe catalog: data/universe_idx_all_2026-08.csv; loaded tickers: 962; skipped: 0<br>
Fine-tuning cutoff: 2021-12-31; evaluation: 2022-01-01 through 2026-08-11<br>
Eligibility: top 300 by trailing 60-day median dollar volume; context: 256 days; horizon: 21 business days<br>
Portfolio: top 3 equal-weight names; 25.0 bps one-way default cost<br>

## Research basis

This audit uses the official Chronos-2 `fit` and `predict_df` interfaces documented by [Amazon Science](https://github.com/amazon-science/chronos-forecasting), with the multivariate financial-forecasting motivation from the [Chronos-2 study](https://arxiv.org/abs/2605.21504). It adapts the pretrained model to Indonesian daily price histories before the evaluation period, rather than selecting a holdout-tuned model.

## Fixed design and leakage controls

- The model is fine-tuned exactly once on normalized daily log-price series with observations no later than 2021-12-31. The frozen checkpoint is then used for every evaluation month.
- Training uses 650 current-catalog series with at least 277 pre-cutoff observations, 100 full-model steps, learning rate 1e-06, batch size 64, and seed 20260812.
- At each completed month-end t, only the top 300 names by trailing 60-day median dollar volume are forecast. Context rows are filtered to dates <= t.
- The two recorded variants are terminal cross-learning median and terminal cross-learning 10th-percentile forecasts. Their ranks and top-3 portfolio rule are fixed before inspecting validation or holdout results.
- Validation is 2022–2023. 2024 and 2025–2026 are holdouts. No holdout return is used for fine-tuning, model choice, or tail selection.

## Audit summary

The fixed gate requires beating the matching cap300_composite control in validation, positive excess CAGR in both holdout blocks at 25 and 100 bps, at least 60% positive rolling 12-month windows, a positive 95% three-month block-bootstrap lower bound versus IHSG, beta <= 1.50, drawdown >= -0.60, and turnover <= 0.90 in 2025–2026.

| model | validation excess | 2024 excess | 2025–2026 excess | rolling positive | bootstrap lower | holdout beta | holdout drawdown | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cap300_composite | -0.0580 | 0.3178 | -0.0339 | 0.6136 | — | 1.4652 | -0.6248 | rejects |
| chronos2_ft_cross_median | 0.0496 | -0.5791 | 1.6675 | 0.6364 | -0.4168 | 1.3135 | -0.5664 | rejects |
| chronos2_ft_cross_lower10 | -0.0328 | 0.0977 | 0.4581 | 0.6818 | -0.0240 | 0.1342 | -0.0475 | rejects |

Validation winner: chronos2_ft_cross_median. Preferred after the fixed gate: none. Any passing row remains exploratory because the catalog is a current snapshot with survivorship and historical-membership bias.

## Limitations

- The current catalog omits historical delistings, suspensions, and membership changes.
- Full-model CPU fine-tuning is reproducible but expensive; it is not a substitute for a point-in-time, multi-market training panel.
- Yahoo Finance data do not model spreads, taxes, price limits, market impact, failed fills, or capacity.
- This is research, not investment advice, and no backtest guarantees future performance.

## Reproduction

~~~bash
HF_HOME=/tmp/beat-market-hf PYTHONPATH=$PWD \
  /tmp/beat-market-ml-venv/bin/python -m src.chronos2_finetune_research
~~~

The generated checkpoint remains outside Git at `/tmp/beat-market-chronos2-finetuned`. Raw forecasts remain ignored in reports/chronos2_finetune_forecasts.csv; committed tables use the `chronos2_finetune_` prefix.
