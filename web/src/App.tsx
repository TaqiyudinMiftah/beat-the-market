import { useEffect, useMemo, useState } from "react";
import {
  getDataStatus,
  getDefaultStrategy,
  getFeatures,
  getOhlcv,
  getRankings,
  getUniverse,
  runBacktest,
} from "./api";
import MarketChart from "./components/MarketChart";
import StrategyLab from "./components/StrategyLab";
import type {
  BacktestResponse,
  Candle,
  DataStatus,
  FactorContribution,
  RankingRow,
  StockFeatures,
  StrategyConfig,
  UniverseStock,
} from "./types";

type View = "overview" | "workspace" | "lab" | "method";
const VALID_VIEWS: View[] = ["overview", "workspace", "lab", "method"];

const FALLBACK_STRATEGY: StrategyConfig = {
  version: 1,
  name: "Composite momentum",
  factors: [
    { id: "momentum_12_1", type: "momentum", lookback_months: 12, skip_months: 1, weight: 0.4, direction: "high", label: "12–1 month momentum" },
    { id: "momentum_6_1", type: "momentum", lookback_months: 6, skip_months: 1, weight: 0.3, direction: "high", label: "6–1 month momentum" },
    { id: "momentum_3", type: "momentum", lookback_months: 3, skip_months: 0, weight: 0.2, direction: "high", label: "3-month momentum" },
    { id: "volatility_60", type: "volatility", window_days: 60, weight: 0.1, direction: "low", label: "Low 60-day volatility" },
  ],
  top_k: 3,
  rebalance: "monthly",
  cost_bps: 25,
  price_field: "adjclose",
  benchmark: "^JKSE",
};

function urlParam(name: string): string | null {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get(name);
}

function decodeStrategy(value: string | null): StrategyConfig | null {
  if (!value) return null;
  try {
    return JSON.parse(decodeURIComponent(window.atob(value))) as StrategyConfig;
  } catch {
    return null;
  }
}

function encodeStrategy(strategy: StrategyConfig): string {
  return window.btoa(encodeURIComponent(JSON.stringify(strategy)));
}

function formatDate(value?: string | null): string {
  if (!value) return "—";
  const parsed = new Date(`${value.slice(0, 10)}T00:00:00`);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function formatPrice(value?: number | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
}

function formatNumber(value?: number | null, digits = 2): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(value);
}

function formatPercent(value?: number | null, digits = 1): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function App() {
  const sharedStrategy = decodeStrategy(urlParam("strategy"));
  const requestedView = urlParam("view");
  const [view, setView] = useState<View>(requestedView && VALID_VIEWS.includes(requestedView as View) ? requestedView as View : "overview");
  const [ticker, setTicker] = useState((urlParam("ticker") || "BBCA").toUpperCase());
  const [strategy, setStrategy] = useState<StrategyConfig>(sharedStrategy ?? FALLBACK_STRATEGY);
  const [status, setStatus] = useState<DataStatus | null>(null);
  const [universe, setUniverse] = useState<UniverseStock[]>([]);
  const [rankings, setRankings] = useState<RankingRow[]>([]);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [features, setFeatures] = useState<StockFeatures | null>(null);
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [loadingMarket, setLoadingMarket] = useState(false);
  const [loadingShell, setLoadingShell] = useState(true);
  const [backtestBusy, setBacktestBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shareMessage, setShareMessage] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all([getDataStatus(), getUniverse(), getDefaultStrategy()])
      .then(([nextStatus, nextUniverse, nextDefault]) => {
        if (!active) return;
        setStatus(nextStatus);
        setUniverse(nextUniverse.stocks);
        if (!sharedStrategy) setStrategy(nextDefault);
        const availableTickers = nextUniverse.stocks.map((stock) => stock.ticker);
        if (!availableTickers.includes(ticker)) setTicker(availableTickers.includes("BBCA") ? "BBCA" : availableTickers[0] ?? "BBCA");
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "The API could not be reached.");
      })
      .finally(() => {
        if (active) setLoadingShell(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!status?.available) return undefined;
    let active = true;
    setLoadingMarket(true);
    Promise.all([getOhlcv(ticker), getFeatures(ticker)])
      .then(([ohlcv, nextFeatures]) => {
        if (!active) return;
        setCandles(ohlcv.data);
        setFeatures(nextFeatures);
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "The stock data could not be loaded.");
      })
      .finally(() => {
        if (active) setLoadingMarket(false);
      });
    return () => {
      active = false;
    };
  }, [status?.available, ticker]);

  useEffect(() => {
    if (!status?.available) return undefined;
    let active = true;
    getRankings(strategy)
      .then((response) => {
        if (active) setRankings(response.rankings);
      })
      .catch(() => {
        // The stock workspace remains useful if the ranking snapshot is unavailable.
      });
    return () => {
      active = false;
    };
  }, [status?.available, strategy.cost_bps, strategy.price_field, strategy.top_k]);

  useEffect(() => {
    if (!status?.available || backtest) return undefined;
    let active = true;
    setBacktestBusy(true);
    runBacktest(strategy)
      .then((result) => {
        if (active) setBacktest(result);
      })
      .catch(() => {
        // The lab shows its own error when the user runs a custom experiment.
      })
      .finally(() => {
        if (active) setBacktestBusy(false);
      });
    return () => {
      active = false;
    };
  }, [status?.available]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    params.set("ticker", ticker);
    params.set("view", view);
    params.set("strategy", encodeStrategy(strategy));
    window.history.replaceState({}, "", `${window.location.pathname}?${params.toString()}`);
  }, [ticker, view, strategy]);

  const selectedRanking = useMemo(() => rankings.find((row) => row.ticker === ticker), [rankings, ticker]);
  const title = view === "overview" ? "A calmer way to study Indonesian equities" : view === "workspace" ? "Market workspace" : view === "lab" ? "Strategy lab" : "Learn the method";

  function selectTicker(nextTicker: string) {
    setTicker(nextTicker.toUpperCase());
    setView("overview");
    setError(null);
  }

  async function shareCurrentView() {
    if (typeof window === "undefined") return;
    const link = window.location.href;
    try {
      await navigator.clipboard?.writeText(link);
      setShareMessage("Share link copied");
    } catch {
      setShareMessage("The share link is ready in your address bar");
    }
    window.setTimeout(() => setShareMessage(null), 2400);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => setView("overview")} aria-label="Go to overview">
          <span className="brand-mark">↗</span>
          <span><strong>Beat the Market</strong><small>Indonesian equity research</small></span>
        </button>
        <div className="topbar-actions">
          <label className="ticker-picker"><span>⌕</span><select value={ticker} onChange={(event) => selectTicker(event.target.value)} aria-label="Select stock">
            {universe.length === 0 ? <option value={ticker}>{ticker}</option> : universe.map((stock) => <option value={stock.ticker} key={stock.ticker}>{stock.ticker} · {stock.company_name}</option>)}
          </select></label>
          <div className={`status-pill ${status?.available ? "fresh" : "offline"}`}><i />{status?.available ? `EOD ${formatDate(status.last_trading_date)}` : loadingShell ? "Connecting…" : "Data needed"}</div>
          <button className="share-button" onClick={shareCurrentView}>Share view</button>
        </div>
      </header>

      <div className="app-body">
        <aside className="sidebar">
          <div className="sidebar-section-label">Workspace</div>
          <nav className="side-nav" aria-label="Primary navigation">
            <NavButton active={view === "overview"} icon="◈" label="Overview" onClick={() => setView("overview")} />
            <NavButton active={view === "workspace"} icon="⌁" label="Chart workspace" onClick={() => setView("workspace")} />
            <NavButton active={view === "lab"} icon="✦" label="Strategy lab" onClick={() => setView("lab")} />
            <NavButton active={view === "method"} icon="◎" label="How it works" onClick={() => setView("method")} />
          </nav>
          <div className="sidebar-spacer" />
          <div className="sidebar-card">
            <span className="eyebrow">Research mode</span>
            <strong>Explain every score</strong>
            <p>Every signal is visible, testable, and separated from the return it predicts.</p>
            <span className="sidebar-footnote">Daily EOD · monthly rebalance</span>
          </div>
        </aside>

        <main className="main-content">
          <div className="page-heading">
            <div><span className="eyebrow">Public research workspace</span><h1>{title}</h1></div>
            <div className="page-heading-meta"><span className="live-dot" />{status?.universe_size ?? 30} stocks · IHSG benchmark</div>
          </div>
          {error && <div className="error-banner">{error}<button onClick={() => setError(null)} aria-label="Dismiss error">×</button></div>}
          {loadingShell ? <LoadingState /> : !status?.available ? <DataSetupState status={status} /> : view === "overview" ? <OverviewView ticker={ticker} features={features} candles={candles} rankings={rankings} selectedRanking={selectedRanking} loadingMarket={loadingMarket} onSelectTicker={selectTicker} /> : view === "workspace" ? <WorkspaceView ticker={ticker} features={features} candles={candles} loadingMarket={loadingMarket} /> : view === "lab" ? <StrategyLab strategy={strategy} onStrategyChange={setStrategy} onShare={shareCurrentView} result={backtest} onResult={setBacktest} busy={backtestBusy} setBusy={setBacktestBusy} /> : <MethodView onOpenLab={() => setView("lab")} />}
          {shareMessage && <div className="toast">✓ {shareMessage}</div>}
        </main>
      </div>
    </div>
  );
}

function NavButton({ active, icon, label, onClick }: { active: boolean; icon: string; label: string; onClick: () => void }) {
  return <button className={`nav-button ${active ? "active" : ""}`} onClick={onClick}><span>{icon}</span>{label}{active && <b>›</b>}</button>;
}

function LoadingState() {
  return <div className="panel loading-state"><div className="loading-orb" /><h2>Connecting to the market snapshot</h2><p>Loading the latest research metadata and configured universe.</p></div>;
}

function DataSetupState({ status }: { status: DataStatus | null }) {
  return <div className="panel setup-state"><div className="setup-icon">◌</div><span className="eyebrow">Data snapshot required</span><h2>Download the market data to open the workspace</h2><p>The app uses daily end-of-day Yahoo Finance data cached locally. This keeps the research reproducible and makes the signal timing visible.</p><pre>python3 src/download_data.py{`\n`}uvicorn api.main:app --reload</pre>{status?.missing_tickers.length ? <p className="muted">Missing {status.missing_tickers.length} files: {status.missing_tickers.slice(0, 5).join(", ")}{status.missing_tickers.length > 5 ? "…" : ""}</p> : null}</div>;
}

interface MarketViewProps {
  ticker: string;
  features: StockFeatures | null;
  candles: Candle[];
  loadingMarket: boolean;
}

function OverviewView({ ticker, features, candles, rankings, selectedRanking, loadingMarket, onSelectTicker }: MarketViewProps & { rankings: RankingRow[]; selectedRanking?: RankingRow; onSelectTicker: (ticker: string) => void }) {
  return <>
    <section className="intro-grid">
      <div className="panel intro-card"><span className="eyebrow">The idea</span><h2>Find strength that is broad enough to trust.</h2><p>Rank the configured universe by a blend of medium-term momentum and risk. Inspect the inputs, then test the exact rule on history.</p><div className="intro-links"><span>Signal at month-end</span><span>Held next month</span><span>25 bps costs</span></div></div>
      <div className="panel intro-stat"><span className="eyebrow">Default candidate</span><strong>0.40</strong><span>weight on 12–1 momentum</span><div className="mini-spark"><i /><i /><i /><i /><i /><i /><i /></div><small>Four ranked factors · top 3 holdings</small></div>
      <div className="panel intro-stat accent"><span className="eyebrow">Latest signal</span><strong>{features?.signal_date ? formatDate(features.signal_date) : "—"}</strong><span>{features?.selected ? `${ticker} is selected` : `${ticker} is outside the top 3`}</span><div className="signal-status"><i className={features?.selected ? "positive-dot" : "neutral-dot"} />{features?.selected ? `${formatPercent(features.weight)} portfolio weight` : "Watchlist"}</div></div>
    </section>
    <div className="dashboard-grid">
      <RankingPanel rankings={rankings} selectedTicker={ticker} onSelectTicker={onSelectTicker} />
      <div className="market-column">
        <StockSummary ticker={ticker} features={features} ranking={selectedRanking} />
        {loadingMarket ? <ChartLoading /> : candles.length ? <MarketChart ticker={ticker} candles={candles} /> : <ChartEmpty />}
        <FactorPanel features={features} />
      </div>
    </div>
  </>;
}

function WorkspaceView({ ticker, features, candles, loadingMarket }: MarketViewProps) {
  return <div className="workspace-view">
    <section className="workspace-banner panel"><div><span className="eyebrow">Chart workspace</span><h2>Read price, volume, and the method in one place.</h2><p>Use Cursor to inspect candles, Trend to draw a two-point line, or Level to mark a price. The moving averages are context—not trading signals.</p></div><div className="workspace-badges"><span>SMA20</span><span>SMA50</span><span>SMA200</span><span>Volume</span></div></section>
    <StockSummary ticker={ticker} features={features} />
    {loadingMarket ? <ChartLoading expanded /> : candles.length ? <MarketChart ticker={ticker} candles={candles} expanded /> : <ChartEmpty />}
    <FactorPanel features={features} />
  </div>;
}

function RankingPanel({ rankings, selectedTicker, onSelectTicker }: { rankings: RankingRow[]; selectedTicker: string; onSelectTicker: (ticker: string) => void }) {
  return <section className="panel ranking-panel"><div className="section-heading"><div><span className="eyebrow">Signal board</span><h2>Latest ranking</h2></div><span className="tiny-label">Top {rankings.filter((row) => row.selected).length || 3}</span></div><p className="panel-description">Cross-sectional score across the configured IDX30 snapshot.</p><div className="ranking-table"><div className="table-header ranking-header"><span>#</span><span>Stock</span><span>Score</span><span>State</span></div>{rankings.length === 0 ? <div className="table-empty">No ranking snapshot yet.</div> : rankings.map((row) => <button className={`table-row ranking-row ${row.ticker === selectedTicker ? "current" : ""}`} key={row.ticker} onClick={() => onSelectTicker(row.ticker)}><span className="rank-number">{String(row.rank).padStart(2, "0")}</span><span><strong>{row.ticker}</strong><small>{row.selected ? "Selected" : "Watchlist"}</small></span><span className="score-value">{formatNumber(row.score, 3)}</span><span className={row.selected ? "selected-chip" : "watch-chip"}>{row.selected ? "IN" : "—"}</span></button>)}</div><div className="panel-footer"><span>Score is a weighted percentile</span><span>Monthly</span></div></section>;
}

function StockSummary({ ticker, features, ranking }: { ticker: string; features: StockFeatures | null; ranking?: RankingRow }) {
  return <section className="panel stock-summary"><div><span className="eyebrow">Selected instrument</span><div className="stock-title"><h2>{ticker}</h2><span className={features?.selected ? "selected-chip" : "watch-chip"}>{features?.selected ? "Portfolio" : "Watchlist"}</span></div><p>{features?.latest.date ? `Latest close · ${formatDate(features.latest.date)}` : "Loading latest quote…"}</p></div><div className="stock-quote"><strong>{formatPrice(features?.latest.price)}</strong><span>{formatPercent(features?.monthly_return)} latest completed month</span></div><div className="stock-score"><span>Composite score</span><strong>{formatNumber(features?.score, 3)}</strong><small>Rank #{features?.rank ?? ranking?.rank ?? "—"}</small></div></section>;
}

function FactorPanel({ features }: { features: StockFeatures | null }) {
  const factors = features ? Object.entries(features.factors) : [];
  return <section className="panel factor-panel"><div className="section-heading"><div><span className="eyebrow">Explainability</span><h2>What drives this score?</h2></div><span className="tiny-label">Percentile ranks</span></div>{factors.length === 0 ? <div className="table-empty">Select a stock to inspect its factor contribution.</div> : <div className="factor-cards">{factors.map(([factorId, factor]) => <FactorCard key={factorId} factor={factor} />)}</div>}</section>;
}

function FactorCard({ factor }: { factor: FactorContribution }) {
  const percentile = factor.rank ?? 0;
  return <div className="factor-card"><div className="factor-card-top"><span>{factor.label}</span><strong>{formatPercent(factor.contribution)}</strong></div><div className="factor-bar"><i style={{ width: `${Math.max(3, percentile * 100)}%` }} /></div><div className="factor-card-meta"><span>Raw {formatFactorRaw(factor.raw, factor.label)}</span><span>{formatPercent(factor.rank)} percentile</span></div></div>;
}

function formatFactorRaw(value: number | null, label: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (label.toLowerCase().includes("liquidity")) return formatNumber(value, 0);
  return formatPercent(value);
}

function ChartLoading({ expanded = false }: { expanded?: boolean }) {
  return <section className={`chart-card chart-placeholder ${expanded ? "expanded" : ""}`}><div className="loading-line" /><div className="loading-line short" /><div className="chart-placeholder-grid" /></section>;
}

function ChartEmpty() {
  return <section className="chart-card chart-placeholder"><div className="setup-icon">⌁</div><h3>No chart data for this ticker</h3><p>Refresh the local Yahoo Finance snapshot and try again.</p></section>;
}

function MethodView({ onOpenLab }: { onOpenLab: () => void }) {
  return <div className="method-view">
    <section className="method-hero panel"><span className="eyebrow">The method, without the mystery</span><h2>Score strength. Control risk. Wait one month.</h2><p>The app does not forecast a price or promise an edge. It creates a transparent ranking experiment that can be audited from raw daily data to portfolio returns.</p><div className="formula"><span>Score</span><b>0.40 × rank(12–1 momentum)</b><b>+ 0.30 × rank(6–1 momentum)</b><b>+ 0.20 × rank(3-month momentum)</b><b>+ 0.10 × rank(low 60-day volatility)</b></div></section>
    <section className="method-steps"><MethodStep number="01" title="Measure" text="At each completed month-end, calculate the factors from data that was available then. The 12–1 signal skips the most recent month to reduce short-term reversal noise." /><MethodStep number="02" title="Rank" text="Rank every stock against the other stocks in the configured universe. Percentile ranks put momentum, volatility, trend, and liquidity on the same scale." /><MethodStep number="03" title="Select" text="Add the weighted ranks, choose the top K stocks, and give each selected stock an equal portfolio weight." /><MethodStep number="04" title="Lag" text="Apply the signal to the following month, subtracting the configured turnover cost. This lag is the key protection against look-ahead bias." /></section>
    <section className="method-grid"><div className="panel"><span className="eyebrow">How to use it</span><h3>Change one assumption at a time</h3><p>Open Strategy Lab, adjust a factor weight or top K, then compare requested-period, validation, and holdout metrics. A strategy that only works in one window is a clue—not a conclusion.</p><button className="secondary-button" onClick={onOpenLab}>Explore the lab →</button></div><div className="panel warning-panel"><span className="eyebrow">Read the fine print</span><h3>Backtests are evidence, not certainty</h3><p>Current-universe data introduces survivorship and index-membership look-ahead bias. Taxes, suspensions, price limits, spreads, impact, and live execution are not modeled.</p><span className="warning-label">⚠ Research only · not investment advice</span></div></section>
  </div>;
}

function MethodStep({ number, title, text }: { number: string; title: string; text: string }) {
  return <article className="method-step"><span className="step-number">{number}</span><h3>{title}</h3><p>{text}</p></article>;
}

export default App;
