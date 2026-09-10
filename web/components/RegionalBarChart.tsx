"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { RegionalTopRow } from "@/lib/api";
import { formatNumber } from "@/lib/format";

interface RegionalBarChartProps {
  rows: RegionalTopRow[];
}

export default function RegionalBarChart({ rows }: RegionalBarChartProps) {
  if (rows.length === 0) {
    return (
      <div
        className="flex h-64 items-center justify-center rounded-lg border text-sm"
        style={{ borderColor: "var(--border)", color: "var(--text-dim)", background: "var(--surface)" }}
      >
        Bolgesel veri yok
      </div>
    );
  }

  const data = rows.map((r) => ({
    bolge: r.Bolge,
    krediBinTl: r.Toplam_Nakdi_Kredi_Bin_TL ?? 0,
  }));

  return (
    <div className="card">
      <h3 className="mb-3 text-sm font-semibold" style={{ color: "var(--text)" }}>
        Bolgesel Nakdi Kredi (Top {rows.length})
      </h3>
      <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 32)}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 4 }}>
          <CartesianGrid stroke="var(--chart-grid)" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fontSize: 11, fill: "var(--text-dim)" }}
            stroke="var(--border)"
            // Raw values are in BIN TL (thousand TL), so /1e6 lands in
            // billions of TL, not millions - dividing by 1e6 and
            // labeling "M" (million) was silently off by 1000x.
            tickFormatter={(v) => `${formatNumber(v / 1_000_000, 1)} Mlyr`}
          />
          <YAxis
            type="category"
            dataKey="bolge"
            tick={{ fontSize: 11, fill: "var(--text-dim)" }}
            stroke="var(--border)"
            width={90}
          />
          <Tooltip
            formatter={(value) => [`${formatNumber(Number(value), 0)} bin TL`, "Nakdi Kredi"]}
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              fontSize: 12,
              color: "var(--text)",
            }}
          />
          <Bar dataKey="krediBinTl" fill="var(--bronze)" radius={[0, 4, 4, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
