"""
EVDS series registry.

Selection rationale: series were chosen to match the hackathon brief's
own example scenario ("how did house prices change as interest rates
moved from X to Y?"). Each one was found via `--discover --search
<keyword>` against real EVDS data groups (bie_kfe, bie_kt210a,
bie_bispolfaiz, bie_tukfiy2003) and confirmed individually with
`evds.evdsAPI.get_series(datagroup_code)` - no series code was guessed.

    python -m src.collectors.evds --discover --search konut
    python -m src.collectors.evds --discover --search faiz
    python -m src.collectors.evds --discover --search "tuketici fiyat"
"""

SERIES: list[dict] = [
    {
        "code": "TP.KFE.TR",
        "label": "House Price Index (Turkey, nationwide)",
        "category": "housing",
    },
    {
        "code": "TP.BKR.TRY.18",
        "label": "Housing Loan Interest Rate (TRY, Stock, %)",
        "category": "interest_rate",
    },
    {
        "code": "TP.BISPOLFAIZ.TUR",
        "label": "CBRT Policy Interest Rate",
        "category": "interest_rate",
    },
    {
        "code": "TP.GENENDEKS.T1",
        "label": "CPI General Index (2003=100)",
        "category": "inflation",
    },
    {
        "code": "TP.DK.USD.A.YTL",
        "label": "USD Buying Rate",
        "category": "fx",
    },
    {
        "code": "TP.TSANAYMT2021.BCD",
        "label": "Industrial Production Index - Total Industry (Seasonally/Calendar Adjusted)",
        "category": "economic_activity",
    },
]
