/** Minimal CSV export - no library needed for this data shape. Escapes
 * only what CSV actually requires (quote fields containing a comma,
 * quote, or newline; double up embedded quotes) rather than quoting
 * everything, so the exported file stays readable. */

function escapeCsvField(value: unknown): string {
  if (value === null || value === undefined) return "";
  const str = String(value);
  if (/[",\n]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export function toCsv(rows: Record<string, unknown>[], columns?: string[]): string {
  if (rows.length === 0) return "";
  const cols = columns ?? Object.keys(rows[0]);
  const header = cols.map(escapeCsvField).join(",");
  const lines = rows.map((row) => cols.map((c) => escapeCsvField(row[c])).join(","));
  return [header, ...lines].join("\r\n");
}

export function downloadCsv(filename: string, rows: Record<string, unknown>[], columns?: string[]) {
  console.log("[csv] downloadCsv", { filename, rowCount: rows.length, columns });
  const csv = toCsv(rows, columns);
  if (!csv) {
    console.warn("[csv] downloadCsv: no rows, skipping download", { filename });
    return;
  }
  // Prefix with a UTF-8 BOM so Excel (still the most common consumer of
  // these exports) doesn't mangle Turkish characters (Birim/Seri_Adi
  // columns) when it guesses the file's encoding.
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  try {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    console.log("[csv] downloadCsv: triggered", { filename });
  } finally {
    URL.revokeObjectURL(url);
  }
}
