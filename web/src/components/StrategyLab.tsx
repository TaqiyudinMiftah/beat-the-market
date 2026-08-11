import { useMemo, useState } from "react";
import { runBacktest } from "../api";
import type { BacktestResponse, FactorBlock, FactorType, StrategyConfig } from "../types";

interface StrategyLabProps {
  strategy: StrategyConfig;
  onStrategyChange: (strategy: StrategyConfig) => void;
  onShare: () => void;
  result: BacktestResponse | null;
  onResult: (result: BacktestResponse) => void;
  busy: boolean;
  setBusy: (busy: boolean) => void;
}
const labels: Record<FactorType, string> = {
  momentum: "Momentum",
  volatility: "Volatility",
  trend: "Trend",
  liquidity: "Liquidity",
};

function makeFactor(type: FactorType): FactorBlock {
  if (type === "momentum") return { id: `momentum_${Date.now()}`, type, lookback_months: 12, skip_months: 1, weight: 0.2, direction: "high" };
  return { id: `${type}_${Date.now()}`, type, window_days: type === "trend" ? 200 : 60, weight: 0.2, direction: type === "volatility" ? "low" : "high" };
}

function percent(value: number | undefined) {
  return value === undefined || Number.isNaN(value) ? "—" : `${(value * 100).toFixed(1)}%`;
}

export default function StrategyLab({ strategy, onStrategyChange, onShare, result, onResult, busy, setBusy }: StrategyLabProps) {
  const [error, setError] = useState<string | null>(null);
  const weightTotal = useMemo(() => strategy.factors.reduce((sum, factor) => sum + factor.weight, 0), [strategy.factors]);

  function updateFactor(index: number, patch: Partial<FactorBlock>) {
    const factors = strategy.factors.map((factor, factorIndex) => factorIndex === index ? { ...factor, ...patch } : factor);
    onStrategyChange({ ...strategy, factors });
  }

  async function submit() {
    setError(null);
    if (Math.abs(weightTotal - 1) > 0.001) {
      setError("Factor weights must add up to 100% before running the lab.");
      return;
    }
    setBusy(true);
    try {
      onResult(await runBacktest(strategy));
    } catch (err) {
      setError(err instanceof Error ? err.message : "The backtest could not be completed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="lab-layout">
      <section className="panel lab-controls">
        <div className="section-heading"><div><span className="eyebrow">Strategy lab</span><h2>Build a ranking you can explain</h2></div><button className="ghost-button" onClick={onShare}>Share URL</button></div>
        <p className="body-copy">Use supported factor blocks. The engine ranks every factor cross-sectionally, combines the weighted ranks, selects the top K, and holds the portfolio for the next month.</p>
        <div className="control-grid two">
          <label>Portfolio name<input value={strategy.name} onChange={(event) => onStrategyChange({ ...strategy, name: event.target.value })} /></label>
          <label>Top K<input type="number" min={1} max={30} value={strategy.top_k} onChange={(event) => onStrategyChange({ ...strategy, top_k: Number(event.target.value) })} /></label>
          <label>Cost (bps, one-way)<input type="number" min={0} max={500} value={strategy.cost_bps} onChange={(event) => onStrategyChange({ ...strategy, cost_bps: Number(event.target.value) })} /></label>
          <label>Price field<select value={strategy.price_field} onChange={(event) => onStrategyChange({ ...strategy, price_field: event.target.value as StrategyConfig["price_field"] })}><option value="adjclose">Adjusted close</option><option value="close">Close</option></select></label>
        </div>
        <div className="factor-list">
          <div className="factor-list-header"><span>Factor blocks</span><span className={Math.abs(weightTotal - 1) < 0.001 ? "valid" : "warning"}>{(weightTotal * 100).toFixed(0)}% total</span></div>
          {strategy.factors.map((factor, index) => (
            <div className="factor-row" key={factor.id ?? index}>
              <div className="factor-row-main">
                <select value={factor.type} onChange={(event) => updateFactor(index, makeFactor(event.target.value as FactorType))}>{Object.entries(labels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select>
                <input className="weight-input" type="number" min={0.01} max={1} step={0.05} value={factor.weight} onChange={(event) => updateFactor(index, { weight: Number(event.target.value) })} aria-label={`${factor.type} weight`} />
                <select value={factor.direction} onChange={(event) => updateFactor(index, { direction: event.target.value as FactorBlock["direction"] })}><option value="high">Higher is better</option><option value="low">Lower is better</option></select>
                <button className="icon-button" onClick={() => onStrategyChange({ ...strategy, factors: strategy.factors.filter((_, factorIndex) => factorIndex !== index) })} aria-label={`Remove ${factor.type}`}>×</button>
              </div>
              <div className="factor-row-detail">
                {factor.type === "momentum" ? <><label>Lookback (months)<input type="number" min={1} max={36} value={factor.lookback_months ?? 12} onChange={(event) => updateFactor(index, { lookback_months: Number(event.target.value) })} /></label><label>Skip latest (months)<input type="number" min={0} max={12} value={factor.skip_months ?? 0} onChange={(event) => updateFactor(index, { skip_months: Number(event.target.value) })} /></label></> : <label>Window (days)<input type="number" min={20} max={400} value={factor.window_days ?? 60} onChange={(event) => updateFactor(index, { window_days: Number(event.target.value) })} /></label>}
              </div>
            </div>
          ))}
        </div>
        <div className="button-row"><button className="secondary-button" onClick={() => onStrategyChange({ ...strategy, factors: [...strategy.factors, makeFactor("momentum")] })}>+ Add factor</button><button className="primary-button" onClick={submit} disabled={busy}>{busy ? "Running…" : "Run backtest"}</button></div>
        {error && <div className="error-banner">{error}</div>}
      </section>
      <section className="panel lab-results">
        <div className="section-heading"><div><span className="eyebrow">Results</span><h2>Does the method survive?</h2></div></div>
        {!result ? <div className="empty-state"><div className="empty-icon">↗</div><h3>Run your first experiment</h3><p>Start with the default formula, then change one variable at a time. The warning banner is part of the result, not an afterthought.</p></div> : <>
          <div className="metric-grid"><Metric label="Requested CAGR" value={percent(result.metrics.requested?.strategy_cagr)} /><Metric label="IHSG CAGR" value={percent(result.metrics.requested?.benchmark_cagr)} tone="muted" /><Metric label="Excess CAGR" value={percent(result.metrics.requested?.excess_cagr)} tone="positive" /><Metric label="Max drawdown" value={percent(result.metrics.requested?.strategy_max_drawdown)} tone="negative" /></div>
          <EquityStrip points={result.equity_curve} />
          <div className="result-table"><div className="table-header"><span>Period</span><span>Strategy</span><span>IHSG</span><span>Sharpe</span></div>{Object.entries(result.metrics).map(([period, metric]) => <div className="table-row" key={period}><span>{period}</span><span>{percent(metric.strategy_cagr)}</span><span>{percent(metric.benchmark_cagr)}</span><span>{metric.strategy_sharpe_rf0?.toFixed(2) ?? "—"}</span></div>)}</div>
          <div className="warning-stack">{result.warnings.map((warning) => <div className="warning-banner" key={warning}>⚠ {warning}</div>)}</div>
        </>}
      </section>
    </div>
  );
}

function Metric({ label, value, tone = "default" }: { label: string; value: string; tone?: string }) {
  return <div className={`metric-card ${tone}`}><span>{label}</span><strong>{value}</strong></div>;
}

function EquityStrip({ points }: { points: { date: string; strategy: number; benchmark: number }[] }) {
  if (points.length < 2) return null;
  const values = points.flatMap((point) => [point.strategy, point.benchmark]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = 720;
  const height = 150;
  const line = (key: "strategy" | "benchmark") => points.map((point, index) => `${(index / (points.length - 1)) * width},${height - ((point[key] - min) / Math.max(max - min, 0.001)) * height}`).join(" ");
  return <div className="equity-strip"><div className="equity-legend"><span><i className="legend-dot strategy-line" />Strategy</span><span><i className="legend-dot benchmark-line" />IHSG</span></div><svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none"><polyline points={line("benchmark")} fill="none" stroke="#607894" strokeWidth="2" /><polyline points={line("strategy")} fill="none" stroke="#38d39f" strokeWidth="3" /></svg></div>;
}
