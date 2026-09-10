"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";

export interface LineSeries {
  key: string;
  label: string;
  color: string;
  data: { period: string; value: number | null }[];
}

interface SeriesLineChartProps {
  title: string;
  series: LineSeries[];
  height?: number;
  yLabel?: string;
}

/** One-axis, at-most-two-series time chart (recharts). Merges every series
 * onto a shared Period axis so a single missing point on one line doesn't
 * misalign the others; nulls render as gaps, not zeros. */
export default function SeriesLineChart({ title, series, height = 280, yLabel }: SeriesLineChartProps) {
  const nonEmpty = series.filter((s) => s.data.length > 0);
  if (nonEmpty.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border text-sm"
        style={{ height, borderColor: "var(--border)", color: "var(--text-dim)", background: "var(--surface)" }}
      >
        {title}: veri yok
      </div>
    );
  }

  const periodSet = new Set<string>();
  nonEmpty.forEach((s) => s.data.forEach((d) => periodSet.add(d.period)));
  const periods = Array.from(periodSet).sort();

  const merged = periods.map((period) => {
    const row: Record<string, string | number | null> = { period };
    nonEmpty.forEach((s) => {
      const point = s.data.find((d) => d.period === period);
      row[s.key] = point ? point.value : null;
    });
    return row;
  });

  return (
    <div className="rounded-lg border p-4" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
      <h3 className="mb-3 text-sm font-semibold" style={{ color: "var(--text)" }}>
        {title}
      </h3>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={merged} margin={{ top: 4, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
          <XAxis
            dataKey="period"
            tick={{ fontSize: 11, fill: "var(--text-dim)" }}
            stroke="var(--border)"
            minTickGap={24}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "var(--text-dim)" }}
            stroke="var(--border)"
            width={56}
            label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--text-dim)" } : undefined}
          />
          <Tooltip
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              fontSize: 12,
              color: "var(--text)",
            }}
          />
          {nonEmpty.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
          {nonEmpty.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={s.color}
              strokeWidth={2}
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
