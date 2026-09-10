import type { ChangeEntry } from "@/lib/api";
import { formatPercent, formatSigned } from "@/lib/format";

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
          <tr key={c.period} className="border-b transition-colors hover:bg-[var(--surface-2)]" style={{ borderColor: "var(--border)" }}>
            <td className="py-1.5 pr-3 mono" style={{ color: "var(--text)" }}>
              {c.period}
            </td>
            <td className="py-1.5 pr-3 mono" style={{ color: c.absolute_change >= 0 ? "var(--good)" : "var(--critical)" }}>
              {formatSigned(c.absolute_change, 2)}
            </td>
            <td className="py-1.5 mono" style={{ color: c.percent_change >= 0 ? "var(--good)" : "var(--critical)" }}>
              {formatPercent(c.percent_change, 1)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
