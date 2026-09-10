"use client";

import { useEffect, useState } from "react";

import { ApiError, getDashboard, getHealth, type DashboardResponse, type HealthResponse } from "@/lib/api";
import OverviewPanel from "@/components/OverviewPanel";
import AskPanel from "@/components/AskPanel";
import ThemeToggle from "@/components/ThemeToggle";

type Tab = "ask" | "overview";

export default function Home() {
  const [tab, setTab] = useState<Tab>("overview");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      console.log("[page] loading health + dashboard");
      setLoading(true);
      setError(null);
      try {
        const [healthResult, dashboardResult] = await Promise.all([getHealth(), getDashboard()]);
        if (cancelled) return;
        console.log("[page] loaded", { healthResult, panelCounts: {
          konut_demo: dashboardResult.konut_demo.length,
          catalog: dashboardResult.catalog.length,
          health: dashboardResult.health.length,
          regional_top: dashboardResult.regional_top.length,
          net_profit: dashboardResult.net_profit.length,
        } });
        setHealth(healthResult);
        setDashboard(dashboardResult);
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof ApiError ? err.detail : "Beklenmeyen bir hata olustu.";
        console.error("[page] load failed", err);
        setError(message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="flex flex-1 flex-col">
      <div className="h-[3px] w-full" style={{ background: "linear-gradient(90deg, var(--bronze), var(--silver), var(--gold))" }} />
      <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-4 py-8 sm:px-8">
        <header className="flex flex-wrap items-start justify-between gap-2">
          <div className="flex flex-col gap-1">
            <h1 className="text-xl font-semibold" style={{ color: "var(--text)" }}>
              KKB Agentic Data Analytics
            </h1>
            <p className="text-sm" style={{ color: "var(--text-dim)" }}>
              Lakehouse gold katmani uzerinde dogal dil soru-cevap, kaynak gosterimi ve KPI paneli.
            </p>
          </div>
          <div className="flex items-center gap-3">
            {health && (
              <div className="flex items-center gap-1.5 text-xs" style={{ color: "var(--text-dim)" }}>
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ background: health.gold_db_exists ? "var(--good)" : "var(--critical)" }}
                />
                Backend bagli{health.mia_configured ? " · MIA aktif" : " · MIA anahtari yok"}
              </div>
            )}
            <ThemeToggle />
          </div>
        </header>

        <nav className="flex gap-1 border-b" style={{ borderColor: "var(--border)" }}>
          <TabButton active={tab === "overview"} onClick={() => setTab("overview")}>
            Genel Bakis
          </TabButton>
          <TabButton active={tab === "ask"} onClick={() => setTab("ask")}>
            Soru Sor
          </TabButton>
        </nav>

        {loading && (
          <div className="flex items-center gap-2 text-sm" style={{ color: "var(--text-dim)" }}>
            <span
              className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2"
              style={{ borderColor: "var(--border)", borderTopColor: "var(--bronze)" }}
            />
            Yukleniyor...
          </div>
        )}

        {error && (
          <div className="card text-sm" style={{ borderColor: "var(--critical)", color: "var(--critical)" }}>
            Backend&apos;e ulasilamadi: {error}
            <div className="mt-1 text-xs" style={{ color: "var(--text-dim)" }}>
              `uvicorn api.main:app --reload --port 8000` calisiyor mu kontrol edin.
            </div>
          </div>
        )}

        {!loading && !error && dashboard && (
          <>
            {tab === "overview" && (
              dashboard.gold_db_exists ? (
                <OverviewPanel data={dashboard} />
              ) : (
                <p className="text-sm" style={{ color: "var(--critical)" }}>
                  Gold veritabani bulunamadi. `python -m src.gold.build_gold` calistirin.
                </p>
              )
            )}
            {tab === "ask" && <AskPanel miaConfigured={health?.mia_configured ?? false} />}
          </>
        )}
      </div>
    </div>
  );
}

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className="border-b-2 px-4 py-2 text-sm font-medium transition-colors hover:text-[var(--text)]"
      style={{
        borderColor: active ? "var(--bronze)" : "transparent",
        color: active ? "var(--text)" : "var(--text-dim)",
      }}
    >
      {children}
    </button>
  );
}
