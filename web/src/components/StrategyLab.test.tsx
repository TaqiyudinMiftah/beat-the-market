import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import StrategyLab from "./StrategyLab";
import type { StrategyConfig } from "../types";

const strategy: StrategyConfig = {
  version: 1,
  name: "Test strategy",
  factors: [{ type: "momentum", lookback_months: 12, skip_months: 1, weight: 0.6, direction: "high" }],
  top_k: 3,
  rebalance: "monthly",
  cost_bps: 25,
  price_field: "adjclose",
  benchmark: "^JKSE",
};

describe("StrategyLab", () => {
  it("explains that factor weights must balance before running", () => {
    render(<StrategyLab strategy={strategy} onStrategyChange={vi.fn()} onShare={vi.fn()} result={null} onResult={vi.fn()} busy={false} setBusy={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Run backtest" }));

    expect(screen.getByText("Factor weights must add up to 100% before running the lab.")).toBeTruthy();
  });
});
