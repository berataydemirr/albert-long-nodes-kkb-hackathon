/**
 * Typed fetch client for the FastAPI backend (api/main.py). Shapes here
 * are hand-mirrored from the live Pydantic/dict responses verified during
 * backend testing - see api/gold_queries.py and api/main.py for the source
 * of truth on the Python side. Every function logs before/after so a
 * network or parsing failure shows up in the browser console with enough
 * context to trace back to the exact call that failed.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

function log(...args: unknown[]) {
  console.log("[api]", ...args);
}

function logError(...args: unknown[]) {
  console.error("[api]", ...args);
}

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API error ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  log("request start", { url, method: init?.method ?? "GET" });
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (err) {
    logError("network failure", { url, err });
    throw new ApiError(0, `Backend'e ulasilamadi (${API_BASE}). Sunucu calisiyor mu?`);
  }

  let body: unknown = null;
  const text = await response.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch (err) {
      logError("response was not valid JSON", { url, text: text.slice(0, 500), err });
      throw new ApiError(response.status, "Sunucu gecersiz JSON dondurdu.");
    }
  }

  if (!response.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : response.statusText;
    logError("request failed", { url, status: response.status, detail });
    throw new ApiError(response.status, detail);
  }

  log("request ok", { url, status: response.status });
  return body as T;
}

// ---------------------------------------------------------------------
// /api/health
// ---------------------------------------------------------------------

export interface HealthResponse {
  status: string;
  gold_db_exists: boolean;
  mia_configured: boolean;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

// ---------------------------------------------------------------------
// /api/dashboard
// ---------------------------------------------------------------------

export interface KonutDemoRow {
  Period: string;
  Konut_Kredisi_Milyon_TL: number | null;
  Konut_Kredisi_Reel_Milyon_TL: number | null;
  [key: string]: unknown;
}

export interface CatalogRow {
  Kaynak: string;
  Seri_Kod: string;
  Seri_Adi: string;
  Birim: string | null;
  Tip: string;
}

export interface HealthRow {
  Period: string;
  Bolge: string;
  Kaynak: string;
  Seri_Kod: string;
  Seri_Adi: string;
  Deger: number | null;
  Birim: string;
  Tip: string;
}

export interface RegionalTopRow {
  Bolge: string;
  Toplam_Nakdi_Kredi_Bin_TL: number | null;
}

export interface NetProfitRow {
  Period: string;
  Value_Monthly: number | null;
}

export interface LagCorrelation {
  lag: number;
  correlation: number;
}

export interface GrangerPValue {
  lag: number;
  p_value: number;
}

export interface CausalityResult {
  shared_periods: number;
  lag_correlations: LagCorrelation[];
  best_lag: LagCorrelation | null;
  granger_pvalues: GrangerPValue[];
  granger_significant_lags: number[];
  warning: string;
}

export interface ChangeEntry {
  period: string;
  absolute_change: number;
  percent_change: number;
}

export interface PolicyRateSeriesRow {
  Period: string;
  Deger: number | null;
}

export interface PolicyRatePanel {
  series: PolicyRateSeriesRow[];
  change_detection: { largest_changes: ChangeEntry[] };
}

export interface DashboardResponse {
  konut_demo: KonutDemoRow[];
  catalog: CatalogRow[];
  health: HealthRow[];
  regional_top: RegionalTopRow[];
  net_profit: NetProfitRow[];
  causality: CausalityResult | null;
  policy_rate: PolicyRatePanel;
  gold_db_exists: boolean;
}

export function getDashboard(): Promise<DashboardResponse> {
  return request<DashboardResponse>("/api/dashboard");
}

// ---------------------------------------------------------------------
// /api/ask
// ---------------------------------------------------------------------

export interface Citation {
  tool: string;
  tables?: string[];
  seri_kodlar?: string[];
  detail?: string;
}

export interface ToolFailure {
  tool: string;
  error: string;
}

export interface ChartData {
  seri_kod: string;
  periods: string[];
  values: (number | null)[];
}

export interface AskResponse {
  answer: string;
  grounded: boolean;
  citations: Citation[];
  tool_failures: ToolFailure[];
  citation_warning: string | null;
  charts: ChartData[];
  elapsed_sec: number;
}

export function askQuestion(question: string): Promise<AskResponse> {
  return request<AskResponse>("/api/ask", {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}
