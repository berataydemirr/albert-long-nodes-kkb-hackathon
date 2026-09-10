"""
TUIK SDMX series registry.

TUIK's SDMX API is considerably more complex than EVDS: each series is
addressed by a "series key" built from multiple dimensions (REF_AREA,
FREQ, INDICATOR, ...). Keys were not guessed - each one was resolved
and confirmed via the dataflow -> DSD -> codelist chain (see the
comment in src/collectors/tuik.py).

NOTE: this SDMX catalog (427 dataflows, checked 2026-09-09) contains
NO dataflow related to "konut" (housing) or "emlak" (real estate) -
this was confirmed, not a search miss (all 427 titles were scanned).
Rely on BDDK/EVDS for housing data instead.

Health data (2026-09-10 pass, satisfies the hackathon's "not just
finance" requirement): the catalog also has NO dataflow matching
"saglik" (health/hospital) directly. Found instead: DF_DNG_YS (life
expectancy at birth). A second candidate, DF_DOGUM_KDH_C (crude birth
rate by province), was investigated but NOT added - it shares a
19-dimension birth-statistics DSD with several other DF_DOGUM_* /
DF_BD_* dataflows, and even after resolving every dimension's "_T"
(Toplam) code via the DSD's own codelists, the query still returns
"NoRecordsFound" - the real data appears to require a specific,
non-total value on at least one dimension (likely YAYIM_DONEMI, a
528-value release-period dimension with no total code) that wasn't
identified. Left as a follow-up, not guessed at.
"""

SERIES: list[dict] = [
    {
        "dataflow_id": "DF_ISGUCU_AYLIK_TEMEL_ISGUCU_C",
        "version": "1.0",
        "key": "TR.M.ISGI_TIG.8.Y_GE15.N._T._Z.",
        "label": "Unemployment Rate (%) - Turkey, Monthly, Age 15+, Not Seasonally Adjusted",
        "code": "TUIK_UNEMPLOYMENT_RATE",
    },
    {
        "dataflow_id": "DF_DNG_YS",
        "version": "1.0",
        "key": "NG_YS.TR._T.A.TR",
        "label": "Life Expectancy at Birth (Years) - Turkey, Both Sexes, Annual",
        "code": "TUIK_LIFE_EXPECTANCY",
    },
]
