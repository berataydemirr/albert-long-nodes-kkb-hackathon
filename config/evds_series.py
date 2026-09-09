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
    # Added in a later pass, same rule as above (no guessed codes) - found via:
    #   python -m src.collectors.evds --discover --search "mevduat faiz"
    #   python -m src.collectors.evds --discover --search "kredi kartı"
    #   python -m src.collectors.evds --discover --search "reel kesim"
    # then confirmed with client.get_series(datagroup_code). Chosen to widen
    # the gold layer beyond the housing-loan demo scenario: a deposit rate
    # (loan rate's natural counterpart), a real-economy spending proxy, and
    # a leading-indicator confidence index - all plausible Causality/Anomaly
    # Tool inputs beyond the single worked example in the hackathon brief.
    {
        "code": "TP.MT210AGS.TRY.MT06",
        "label": "TRY Deposit Interest Rate (Stock, Total excl. Demand Deposits, %)",
        "category": "interest_rate",
    },
    {
        "code": "TP.KKHARTUT.KT1",
        "label": "Credit/Debit Card Spending, Total (Flow)",
        "category": "consumption",
    },
    {
        "code": "TP.GY1.N2",
        "label": "Real Sector Confidence Index",
        "category": "sentiment",
    },
    # Third pass, same rule (no guessed codes), found via:
    #   python -m src.collectors.evds --discover --search "altin"
    #   python -m src.collectors.evds --discover --search "para arzi"
    #   python -m src.collectors.evds --discover --search "faiz oran"
    # then confirmed with client.get_series(datagroup_code). Gold price
    # directly matches the hackathon brief's own second worked example
    # (Borsa Istanbul precious metals, see slide 13); M2 and the broad
    # consumer loan rate widen the macro picture beyond the single
    # housing-only interest rate already in the registry.
    # DATA QUALITY WARNING - collected but NOT wired into the gold layer:
    # only 207 irregular observations over 2021-2026 (gaps up to 28 days,
    # zero rows in 2026 at all - last observation 2025-11-24). This looks
    # like ad-hoc/spot reporting, not a routine daily series, so treating
    # it as monthly (forward-filling the gaps) would fabricate data. Kept
    # here for reference; the brief's own demo (slide 13) points at
    # sourcing Borsa Istanbul gold prices live via the Web URL Agent Tool
    # instead, which is the more honest fit for this specific need.
    {
        "code": "TP.ALTINPIYASA.KAP05",
        "label": "Gold Closing Price (Borsa Istanbul, TRY/gram) - SPARSE, see comment above",
        "category": "commodity_price",
    },
    {
        "code": "TP.HPBITABLO1.11",
        "label": "M2 Money Supply",
        "category": "money_supply",
    },
    {
        "code": "TP.BKR.TRY.KTFTUK",
        "label": "Consumer Loan Interest Rate, Combined Personal+Auto+Housing (TL, Stock, %)",
        "category": "interest_rate",
    },
]
