"use client";

import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { CausalityResult } from "@/lib/api";
import { formatNumber } from "@/lib/format";

interface CausalityPanelProps {
  result: CausalityResult | null;
}

/** Faiz orani -> konut kredisi Granger nedensellik vitrini. Lag korelasyonu
 * bar olarak, en iyi lag kirmizi vurgulanmis - dataviz skill'inin "renk
 * rutbeyi degil kimligi takip eder" kuralina uygun: sadece en iyi lag'in
 * kendisi (kategori kimligi) renklenir, siralama degil. */
export default function CausalityPanel({ result }: CausalityPanelProps) {
  if (!result) {
    return (
      <div
        className="flex h-64 items-center justify-center rounded-lg border text-sm"
        style={{ borderColor: "var(--border)", color: "var(--text-dim)", background: "var(--surface)" }}
      >
        Nedensellik testi icin yeterli veri yok
      </div>
    );
  }

  const data = result.lag_correlations.map((c) => ({
    lag: c.lag,
    correlation: c.correlation,
    isBest: result.best_lag?.lag === c.lag,
  }));

  const hasSignificant = result.granger_significant_lags.length > 0;

  return (
    <div className="card">
      <h3 className="mb-1 text-sm font-semibold" style={{ color: "var(--text)" }}>
        Nedensellik: Faiz Orani &rarr; Konut Kredisi
      </h3>
      <p className="mb-3 text-xs" style={{ color: "var(--text-dim)" }}>
        {result.shared_periods} ortak donem &middot; kirmizi = en iyi lag
      </p>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{ top: 20, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
          <XAxis dataKey="lag" tick={{ fontSize: 11, fill: "var(--text-dim)" }} stroke="var(--border)" />
          <YAxis tick={{ fontSize: 11, fill: "var(--text-dim)" }} stroke="var(--border)" width={48} domain={[0, 1]} />
          <Tooltip
            formatter={(value) => [formatNumber(Number(value), 4), "Korelasyon"]}
            labelFormatter={(lag) => `Lag ${lag}`}
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              fontSize: 12,
              color: "var(--text)",
            }}
          />
          <Bar dataKey="correlation" radius={[4, 4, 0, 0]} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.lag} fill={d.isBest ? "var(--critical)" : "var(--s1)"} />
            ))}
            {/* All correlations here cluster near 0.95-0.98 - a 0-1 axis
                alone makes the bars look nearly identical, so the exact
                value is also written directly above each bar. */}
            <LabelList
              dataKey="correlation"
              position="top"
              formatter={(v) => formatNumber(Number(v), 2)}
              style={{ fontSize: 10, fill: "var(--text-dim)" }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div
        className="mt-3 rounded-md p-2 text-xs"
        style={{
          background: hasSignificant ? "var(--gold-soft)" : "var(--surface-2)",
          color: "var(--text)",
        }}
      >
        {hasSignificant
          ? `Granger testi: lag ${result.granger_significant_lags.join(", ")} icin anlamli (p<0.05).`
          : "Granger testinde anlamli bir lag bulunamadi."}{" "}
        <span style={{ color: "var(--text-dim)" }}>{result.warning}</span>
      </div>
    </div>
  );
}
