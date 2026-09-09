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
"""

SERIES: list[dict] = [
    {
        "dataflow_id": "DF_ISGUCU_AYLIK_TEMEL_ISGUCU_C",
        "version": "1.0",
        "key": "TR.M.ISGI_TIG.8.Y_GE15.N._T._Z.",
        "label": "Unemployment Rate (%) - Turkey, Monthly, Age 15+, Not Seasonally Adjusted",
        "code": "TUIK_UNEMPLOYMENT_RATE",
    },
]
