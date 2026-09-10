import type { ChangeEntry } from "@/lib/api";

interface ChangeTableProps {
  changes: ChangeEntry[];
}

export default function ChangeTable({ changes }: ChangeTableProps) {
  if (changes.length === 0) {
    return (
      <p className="text-xs" style={{ color: "var(--text-dim)" }}>
        Degisim verisi yok
      </p>
    );
  }

  return (
    <table className="w-full text-left text-xs">
      <thead>
        <tr style={{ color: "var(--text-dim)" }}>
          <th className="border-b pb-2 pr-3 font-medium" style={{ borderColor: "var(--border)" }}>
            Donem
          </th>
          <th className="border-b pb-2 pr-3 font-medium" style={{ borderColor: "var(--border)" }}>
            Mutlak Degisim
          </th>
          <th className="border-b pb-2 font-medium" style={{ borderColor: "var(--border)" }}>
            Yuzde Degisim
          </th>
        </tr>
      </thead>
      <tbody>
        {changes.map((c) => (
          <tr key={c.period} className="border-b" style={{ borderColor: "var(--border)" }}>
            <td className="py-1.5 pr-3 mono" style={{ color: "var(--text)" }}>
              {c.period}
            </td>
            <td className="py-1.5 pr-3 mono" style={{ color: c.absolute_change >= 0 ? "var(--good)" : "var(--critical)" }}>
              {c.absolute_change >= 0 ? "+" : ""}
              {c.absolute_change.toFixed(2)}
            </td>
            <td className="py-1.5 mono" style={{ color: c.percent_change >= 0 ? "var(--good)" : "var(--critical)" }}>
              {c.percent_change >= 0 ? "+" : ""}
              {c.percent_change.toFixed(1)}%
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
