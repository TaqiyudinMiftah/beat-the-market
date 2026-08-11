import { useEffect, useRef, useState } from "react";
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type MouseEventParams,
  type Time,
} from "lightweight-charts";
import type { Candle } from "../types";

interface MarketChartProps {
  candles: Candle[];
  ticker: string;
  expanded?: boolean;
}

type Drawing = { tool: "trend"; points: { time: string; value: number }[] } | { tool: "level"; value: number };

function sma(candles: Candle[], window: number) {
  const output: { time: string; value: number }[] = [];
  for (let index = window - 1; index < candles.length; index += 1) {
    const values = candles.slice(index - window + 1, index + 1).map((candle) => candle.close);
    output.push({ time: candles[index].date, value: values.reduce((a, b) => a + b, 0) / window });
  }
  return output;
}

export default function MarketChart({ candles, ticker, expanded = false }: MarketChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const [tool, setTool] = useState<"cursor" | "trend" | "level">("cursor");
  const [drawings, setDrawings] = useState<Drawing[]>([]);
  const [pendingTrend, setPendingTrend] = useState<{ time: string; value: number }[]>([]);

  useEffect(() => {
    if (!containerRef.current || candles.length === 0) return undefined;
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: expanded ? 520 : 380,
      layout: { background: { type: ColorType.Solid, color: "#0b1628" }, textColor: "#9bb0ca" },
      grid: { vertLines: { color: "#14253b" }, horzLines: { color: "#14253b" } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#213651" },
      timeScale: { borderColor: "#213651", timeVisible: false },
    });
    chartRef.current = chart;
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#38d39f",
      downColor: "#ff6b7a",
      borderVisible: false,
      wickUpColor: "#38d39f",
      wickDownColor: "#ff6b7a",
    });
    candleRef.current = candleSeries;
    candleSeries.setData(candles.map((candle) => ({
      time: candle.date,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    })) as never);
    const volume = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "",
      color: "#31577c",
      priceLineVisible: false,
      lastValueVisible: false,
    });
    volume.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    volume.setData(candles.map((candle) => ({
      time: candle.date,
      value: candle.volume,
      color: candle.close >= candle.open ? "#255c58" : "#653844",
    })) as never);
    const ma20 = chart.addSeries(LineSeries, { color: "#f6c85f", lineWidth: 1, title: "SMA20" });
    ma20.setData(sma(candles, 20) as never);
    const ma50 = chart.addSeries(LineSeries, { color: "#7da9ff", lineWidth: 1, title: "SMA50" });
    ma50.setData(sma(candles, 50) as never);
    const ma200 = chart.addSeries(LineSeries, { color: "#c58cff", lineWidth: 2, title: "SMA200" });
    ma200.setData(sma(candles, 200) as never);
    for (const drawing of drawings) {
      if (drawing.tool === "level") {
        candleSeries.createPriceLine({ price: drawing.value, color: "#ffcf70", lineWidth: 1, lineStyle: 2, title: "Level" });
      } else if (drawing.points.length === 2) {
        const line = chart.addSeries(LineSeries, { color: "#ffcf70", lineWidth: 2, lineStyle: 2, title: "Trend" });
        line.setData(drawing.points as never);
      }
    }
    chart.timeScale().fitContent();
    const resize = () => chart.applyOptions({ width: containerRef.current?.clientWidth ?? 800 });
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
    };
  }, [candles, drawings, expanded]);

  function handleClick(param: MouseEventParams<Time>) {
    if (tool === "cursor" || !param.point || param.time === undefined || !candleRef.current) return;
    const value = candleRef.current.coordinateToPrice(param.point.y);
    if (value === null) return;
    const point = { time: String(param.time), value };
    if (tool === "level") {
      setDrawings((current) => [...current, { tool: "level", value }]);
      setTool("cursor");
    } else if (pendingTrend.length === 0) {
      setPendingTrend([point]);
    } else {
      setDrawings((current) => [...current, { tool: "trend", points: [pendingTrend[0], point] }]);
      setPendingTrend([]);
      setTool("cursor");
    }
  }

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return undefined;
    const handler = (param: MouseEventParams<Time>) => handleClick(param);
    chart.subscribeClick(handler);
    return () => chart.unsubscribeClick(handler);
  }, [tool, pendingTrend]);

  return (
    <section className="chart-card">
      <div className="chart-toolbar">
        <div>
          <span className="eyebrow">Price & volume</span>
          <h2>{ticker} market structure</h2>
        </div>
        <div className="chart-actions">
          <button className={tool === "cursor" ? "tool active" : "tool"} onClick={() => setTool("cursor")}>Cursor</button>
          <button className={tool === "trend" ? "tool active" : "tool"} onClick={() => setTool("trend")}>Trend</button>
          <button className={tool === "level" ? "tool active" : "tool"} onClick={() => setTool("level")}>Level</button>
          <button className="tool" onClick={() => { setDrawings([]); setPendingTrend([]); }}>Clear</button>
        </div>
      </div>
      {tool !== "cursor" && <div className="chart-hint">{tool === "trend" ? "Click two points to draw a trend line." : "Click the chart to mark a price level."}</div>}
      <div ref={containerRef} className="chart-canvas" aria-label={`${ticker} candlestick chart`} />
      <div className="legend-row">
        <span><i className="legend-dot sma20" />SMA20</span>
        <span><i className="legend-dot sma50" />SMA50</span>
        <span><i className="legend-dot sma200" />SMA200</span>
        <span className="muted">Daily EOD · adjusted close signals</span>
      </div>
    </section>
  );
}
