"use client";

import { useMemo, useState } from "react";

import type { CatalogRow } from "@/lib/api";

interface SeriesCatalogTableProps {
  rows: CatalogRow[];
}

export default function SeriesCatalogTable({ rows }: SeriesCatalogTableProps) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter(
      (r) =>
        r.Seri_Kod.toLowerCase().includes(q) ||
        r.Seri_Adi.toLowerCase().includes(q) ||
        r.Kaynak.toLowerCase().includes(q)
    );
  }, [rows, query]);

  return (
    <div className="rounded-lg border p-4" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>
          Seri Katalogu ({filtered.length}/{rows.length})
        </h3>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="ara: seri kodu, adi, kaynak..."
          className="rounded-md border px-2 py-1 text-xs outline-none"
          style={{ borderColor: "var(--border)", background: "var(--bg)", color: "var(--text)" }}
        />
      </div>
      <div className="max-h-80 overflow-y-auto overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr style={{ color: "var(--text-dim)" }} className="sticky top-0" >
              <th className="border-b pb-2 pr-3 font-medium" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
                Kaynak
              </th>
              <th className="border-b pb-2 pr-3 font-medium" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
                Seri Kodu
              </th>
              <th className="border-b pb-2 pr-3 font-medium" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
                Seri Adi
              </th>
              <th className="border-b pb-2 pr-3 font-medium" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
                Birim
              </th>
              <th className="border-b pb-2 font-medium" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
                Tip
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr key={`${r.Kaynak}-${r.Seri_Kod}`} className="border-b" style={{ borderColor: "var(--border)" }}>
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
