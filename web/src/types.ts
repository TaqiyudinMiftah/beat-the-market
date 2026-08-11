export type FactorType = "momentum" | "volatility" | "trend" | "liquidity";
export type Direction = "high" | "low";

export interface FactorBlock {
  id?: string;
  type: FactorType;
  lookback_months?: number;
  skip_months?: number;
  window_days?: number;
  weight: number;
  direction: Direction;
  label?: string;
}
export interface StrategyConfig {
  version: number;
  name: string;
  factors: FactorBlock[];
  top_k: number;
  rebalance: "monthly";
  cost_bps: number;
  price_field: "adjclose" | "close";
  benchmark: "^JKSE";
}

export interface DataStatus {
  available: boolean;
  source: string;
  benchmark: string;
  universe_size: number;
  fetched_at_utc?: string;
  requested_start?: string;
  requested_end_exclusive?: string;
  last_trading_date?: string;
  missing_tickers: string[];
  stale: boolean;
}

export interface UniverseStock {
  ticker: string;
  company_name: string;
  index_effective_from: string;
  index_effective_to: string;
  source: string;
}

export interface Candle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  adjclose: number;
  volume: number;
}

export interface FactorContribution {
  label: string;
  raw: number | null;
  rank: number | null;
  contribution: number | null;
}

export interface RankingRow {
  ticker: string;
  rank: number;
  score: number;
  selected: boolean;
  factors: Record<string, FactorContribution>;
}

export interface StockFeatures {
  ticker: string;
  signal_date: string;
  latest: { date: string; price: number };
  rank: number;
  score: number;
  selected: boolean;
  weight: number;
  factors: Record<string, FactorContribution>;
  monthly_return: number | null;
}

export interface EquityPoint {
  date: string;
  strategy: number;
  benchmark: number;
  drawdown: number;
  turnover: number;
}

export interface MetricRow {
  period: string;
  months: number;
  strategy_cagr?: number;
  benchmark_cagr?: number;
  excess_cagr?: number;
  strategy_sharpe_rf0?: number;
  strategy_max_drawdown?: number;
  average_monthly_turnover?: number;
  information_ratio?: number;
}

export interface HoldingSnapshot {
  signal_date: string;
  positions: { ticker: string; weight: number }[];
}

export interface BacktestResponse {
  strategy: StrategyConfig;
  signal_timing: string;
  data_status: DataStatus;
  metrics: Record<string, MetricRow>;
  equity_curve: EquityPoint[];
  holdings: HoldingSnapshot[];
  latest_factors: RankingRow[];
  warnings: string[];
}
