"use client";

import { useState } from "react";

import { ApiError, askQuestion, type AskResponse } from "@/lib/api";
import SeriesLineChart from "@/components/SeriesLineChart";

interface AskPanelProps {
  miaConfigured: boolean;
}

// Deliberately adversarial examples - each one targets a specific guardrail
// (out-of-range forecast, a series too sparse to answer, comparative
// multi-hop reasoning, an unconfirmed-unit table) rather than a question
// the agent trivially answers with one lookup.
const EXAMPLE_QUESTIONS = [
  "2027 yilinda politika faizi ne olur?",
  "Altin fiyati ile TL kredi faizi arasindaki korelasyon nedir?",
  "Istanbul ve Ankara'nin 2021'den 2026'ya nakdi kredi buyume oranini karsilastir, hangisi daha hizli buyudu?",
  "M2 para arzi ile TCMB politika faizi arasinda 12 lag ile Granger nedensellik testi yap.",
  "Net kar verisindeki en buyuk anomaliyi bul ve sebebini acikla.",
  "BDDK haftalik konut kredisi ile aylik tuketici kredisini karsilastir, birimleri tutarli mi?",
];

export default function AskPanel({ miaConfigured }: AskPanelProps) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);

  async function runQuestion(q: string) {
    if (!q.trim() || loading) return;
    console.log("[AskPanel] submitting question", { question: q });
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await askQuestion(q.trim());
      console.log("[AskPanel] got response", response);
      setResult(response);
    } catch (err) {
      const message = err instanceof ApiError ? err.detail : "Beklenmeyen bir hata olustu.";
      console.error("[AskPanel] request failed", err);
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    runQuestion(question);
  }

  function handleExampleClick(example: string) {
    setQuestion(example);
    runQuestion(example);
  }

  if (!miaConfigured) {
    return (
      <div className="card text-sm" style={{ color: "var(--critical)" }}>
        MIA_API_KEY .env dosyasinda tanimli degil - bu sekme calismaz. &quot;Genel Bakis&quot; sekmesi API anahtari
        gerektirmez.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Konut kredisi faiz orani 2021-2026 arasinda nasil degisti?"
          maxLength={2000}
          className="flex-1 rounded-md border px-3 py-2 text-sm outline-none"
          style={{ borderColor: "var(--border)", background: "var(--surface)", color: "var(--text)" }}
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="rounded-md px-4 py-2 text-sm font-medium text-white transition-opacity disabled:opacity-50"
          style={{ background: "var(--bronze)" }}
        >
          {loading ? "Calisiyor..." : "Sor"}
        </button>
      </form>

      {!result && !loading && !error && (
        <div className="flex flex-col gap-2">
          <span className="text-xs uppercase tracking-wide" style={{ color: "var(--text-dim)" }}>
            Zorlu ornek sorular
          </span>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_QUESTIONS.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => handleExampleClick(example)}
                className="rounded-full border px-3 py-1.5 text-left text-xs transition-colors hover:border-[var(--bronze)] hover:text-[var(--bronze)]"
                style={{ borderColor: "var(--border)", color: "var(--text-dim)", background: "var(--surface)" }}
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      )}

      {loading && (
        <p className="text-sm" style={{ color: "var(--text-dim)" }}>
          MIA agent calisiyor (birkac tur surebilir)...
        </p>
      )}

      {error && (
        <div
          className="rounded-md border p-3 text-sm"
          style={{ borderColor: "var(--critical)", color: "var(--critical)", background: "var(--surface)" }}
        >
          Hata: {error}
        </div>
      )}

      {result && (
        <div className="flex flex-col gap-4">
          <div className="card">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>
                Cevap
              </h3>
              <span className="mono text-xs" style={{ color: "var(--text-dim)" }}>
                {result.elapsed_sec}s
              </span>
            </div>
            <p className="whitespace-pre-wrap text-sm" style={{ color: "var(--text)" }}>
              {result.answer}
            </p>
          </div>

          <div className="card">
            <h3 className="mb-2 text-sm font-semibold" style={{ color: "var(--text)" }}>
              Kaynaklar
            </h3>
            {!result.grounded ? (
              <p className="text-sm" style={{ color: "var(--critical)" }}>
                Kaynak yok: {result.citation_warning}
              </p>
            ) : (
              <ul className="flex flex-col gap-1 text-sm">
                {result.citations.map((c, i) => (
                  <li key={i} style={{ color: "var(--text)" }}>
                    {c.tool === "lakehouse_query" ? (
                      <>
                        Lakehouse sorgusu
                        {c.tables && c.tables.length > 0 && ` - tablo: ${c.tables.join(", ")}`}
                        {c.seri_kodlar && c.seri_kodlar.length > 0 && ` - seri: ${c.seri_kodlar.join(", ")}`}
                      </>
                    ) : (
                      <>
                        {c.tool}: {c.detail}
                      </>
                    )}
                  </li>
                ))}
              </ul>
            )}
            {result.tool_failures.length > 0 && (
              <div className="mt-3">
                <p className="text-xs font-medium" style={{ color: "var(--critical)" }}>
                  Basarisiz tool cagrilari (goz ardi edilmedi):
                </p>
                <ul className="text-xs" style={{ color: "var(--critical)" }}>
                  {result.tool_failures.map((f, i) => (
                    <li key={i}>
                      {f.tool}: {f.error}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {result.charts.length > 0 && (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {result.charts.map((chart) => (
                <SeriesLineChart
                  key={chart.seri_kod}
                  title={chart.seri_kod}
                  series={[
                    {
                      key: "value",
                      label: chart.seri_kod,
                      color: "var(--s1)",
                      data: chart.periods.map((p, i) => ({ period: p, value: chart.values[i] })),
                    },
                  ]}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
