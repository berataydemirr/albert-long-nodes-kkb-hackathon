"use client";

import { useMemo, useState } from "react";

import type { CatalogRow } from "@/lib/api";
import { downloadCsv } from "@/lib/csv";

interface SeriesCatalogTableProps {
  rows: CatalogRow[];
}

type SortKey = "Kaynak" | "Seri_Kod" | "Seri_Adi" | "Birim" | "Tip";

const COLUMNS: { key: SortKey; label: string }[] = [
  { key: "Kaynak", label: "Kaynak" },
  { key: "Seri_Kod", label: "Seri Kodu" },
  { key: "Seri_Adi", label: "Seri Adi" },
  { key: "Birim", label: "Birim" },
  { key: "Tip", label: "Tip" },
];

export default function SeriesCatalogTable({ rows }: SeriesCatalogTableProps) {
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("Kaynak");
  const [sortAsc, setSortAsc] = useState(true);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const base = q
      ? rows.filter(
          (r) =>
            r.Seri_Kod.toLowerCase().includes(q) ||
            r.Seri_Adi.toLowerCase().includes(q) ||
            r.Kaynak.toLowerCase().includes(q)
        )
      : rows;
    const sorted = [...base].sort((a, b) => {
      const av = (a[sortKey] ?? "").toString();
      const bv = (b[sortKey] ?? "").toString();
      return av.localeCompare(bv, "tr");
    });
    if (!sortAsc) sorted.reverse();
    return sorted;
  }, [rows, query, sortKey, sortAsc]);

  function handleSort(key: SortKey) {
    console.log("[SeriesCatalogTable] sort", { key, wasKey: sortKey, wasAsc: sortAsc });
    if (key === sortKey) {
      setSortAsc((a) => !a);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  }

  function handleExport() {
    downloadCsv(
      `seri_katalogu_${new Date().toISOString().slice(0, 10)}.csv`,
      filtered as unknown as Record<string, unknown>[],
      ["Kaynak", "Seri_Kod", "Seri_Adi", "Birim", "Tip"]
    );
  }

  return (
    <div className="card">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>
          Seri Katalogu ({filtered.length}/{rows.length})
        </h3>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="ara: seri kodu, adi, kaynak..."
            className="rounded-md border px-2 py-1 text-xs outline-none"
            style={{ borderColor: "var(--border)", background: "var(--bg)", color: "var(--text)" }}
          />
          <button
            type="button"
            onClick={handleExport}
            disabled={filtered.length === 0}
            className="rounded-md border px-2 py-1 text-xs transition-colors hover:border-[var(--bronze)] hover:text-[var(--bronze)] disabled:opacity-40"
            style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}
          >
            CSV indir
          </button>
        </div>
      </div>
      <div className="max-h-80 overflow-y-auto overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr style={{ color: "var(--text-dim)" }} className="sticky top-0">
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className="cursor-pointer select-none border-b pb-2 pr-3 font-medium last:pr-0"
                  style={{ borderColor: "var(--border)", background: "var(--surface)" }}
                >
                  {col.label}
                  {sortKey === col.key && <span className="ml-1">{sortAsc ? "▲" : "▼"}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              // Kaynak+Seri_Kod alone isn't unique: bddk_finturk's
              // tablo_1.toplam_nakdi_kredi_ulusal appears twice with
              // different Tip values ("flow" vs "flow_quarter_ffill" -
              // two legitimately different derived variants of the same
              // underlying series), which collided as a React key here.
              <tr key={`${r.Kaynak}-${r.Seri_Kod}-${r.Tip}`} className="border-b transition-colors hover:bg-[var(--surface-2)]" style={{ borderColor: "var(--border)" }}>
                <td className="py-1.5 pr-3" style={{ color: "var(--text-dim)" }}>
                  {r.Kaynak}
                </td>
                <td className="mono py-1.5 pr-3" style={{ color: "var(--text)" }}>
                  {r.Seri_Kod}
                </td>
                <td className="py-1.5 pr-3" style={{ color: "var(--text)" }}>
                  {r.Seri_Adi}
                </td>
                <td className="py-1.5 pr-3" style={{ color: r.Birim ? "var(--text-dim)" : "var(--critical)" }}>
                  {r.Birim ?? "dogrulanmamis"}
                </td>
                <td className="py-1.5" style={{ color: "var(--text-dim)" }}>
                  {r.Tip}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={5} className="py-4 text-center" style={{ color: "var(--text-dim)" }}>
                  Sonuc yok
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
