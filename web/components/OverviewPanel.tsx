import type { DashboardResponse } from "@/lib/api";
import StatTile from "@/components/StatTile";
import SeriesLineChart from "@/components/SeriesLineChart";
import CausalityPanel from "@/components/CausalityPanel";
import RegionalBarChart from "@/components/RegionalBarChart";
import SeriesCatalogTable from "@/components/SeriesCatalogTable";
import ChangeTable from "@/components/ChangeTable";

interface OverviewPanelProps {
  data: DashboardResponse;
}

export default function OverviewPanel({ data }: OverviewPanelProps) {
  const latestKonut = data.konut_demo.at(-1);
  const latestHealth = data.health.at(-1);
  const latestPolicyRate = data.policy_rate.series.at(-1);

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile
          label="Konut Kredisi (Nominal)"
          value={latestKonut?.Konut_Kredisi_Milyon_TL != null ? `${latestKonut.Konut_Kredisi_Milyon_TL.toLocaleString("tr-TR")} M TL` : "-"}
          hint={latestKonut?.Period}
          accent="bronze"
        />
        <StatTile
          label="Politika Faizi"
          value={latestPolicyRate?.Deger != null ? `%${latestPolicyRate.Deger}` : "-"}
          hint={latestPolicyRate?.Period}
          accent="gold"
        />
        <StatTile
          label="Beklenen Yasam Suresi"
          value={latestHealth?.Deger != null ? `${latestHealth.Deger} yil` : "-"}
          hint={latestHealth?.Period}
          accent="silver"
        />
        <StatTile label="Seri Sayisi" value={String(data.catalog.length)} hint="gold_fact_long" accent="bronze" />
      </div>

      <SeriesLineChart
        title="Konut Kredisi: Nominal vs Reel"
        yLabel="Milyon TL"
        series={[
          {
            key: "nominal",
            label: "Nominal",
            color: "var(--s1)",
            data: data.konut_demo.map((r) => ({ period: r.Period, value: r.Konut_Kredisi_Milyon_TL })),
          },
          {
            key: "reel",
            label: "Reel (2021-01 fiyatlariyla)",
            color: "var(--s2)",
            data: data.konut_demo.map((r) => ({ period: r.Period, value: r.Konut_Kredisi_Reel_Milyon_TL })),
          },
        ]}
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <CausalityPanel result={data.causality} />
        <RegionalBarChart rows={data.regional_top} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <SeriesLineChart
          title="Politika Faizi"
          yLabel="%"
          series={[
            {
              key: "faiz",
              label: "Politika Faizi",
              color: "var(--bronze)",
              data: data.policy_rate.series.map((r) => ({ period: r.Period, value: r.Deger })),
            },
          ]}
        />
        <div className="rounded-lg border p-4" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
          <h3 className="mb-3 text-sm font-semibold" style={{ color: "var(--text)" }}>
            En Buyuk Degisimler (Politika Faizi)
          </h3>
          <ChangeTable changes={data.policy_rate.change_detection.largest_changes} />
        </div>
      </div>

      <SeriesLineChart
        title="Net Kar (BDDK, Aylik)"
        yLabel="TL"
        series={[
          {
            key: "net_profit",
            label: "Net Kar",
            color: "var(--silver)",
            data: data.net_profit.map((r) => ({ period: r.Period, value: r.Value_Monthly })),
          },
        ]}
      />

      <SeriesCatalogTable rows={data.catalog} />
    </div>
  );
}
