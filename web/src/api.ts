import type {
  BacktestResponse,
  Candle,
  DataStatus,
  RankingRow,
  StockFeatures,
  StrategyConfig,
  UniverseStock,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function getDataStatus() {
  return request<DataStatus>("/api/data-status");
}

export function getDefaultStrategy() {
  return request<StrategyConfig>("/api/strategies/default");
}

export function getUniverse() {
  return request<{ benchmark: string; stocks: UniverseStock[] }>("/api/universe");
}

export function getRankings(strategy: StrategyConfig) {
  const params = new URLSearchParams({
    top_k: String(strategy.top_k),
    cost_bps: String(strategy.cost_bps),
    price_field: strategy.price_field,
  });
  return request<{ as_of: string; rankings: RankingRow[] }>(`/api/rankings?${params}`);
}

export function getOhlcv(ticker: string) {
  return request<{ ticker: string; interval: string; data: Candle[] }>(
    `/api/stocks/${encodeURIComponent(ticker)}/ohlcv`,
  );
}

export function getFeatures(ticker: string) {
  return request<StockFeatures>(`/api/stocks/${encodeURIComponent(ticker)}/features`);
}

export function runBacktest(strategy: StrategyConfig, start = "2015-01-01") {
  return request<BacktestResponse>("/api/backtests", {
    method: "POST",
    body: JSON.stringify({ strategy, start }),
  });
}
