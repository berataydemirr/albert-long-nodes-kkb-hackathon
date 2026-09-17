"""Fetch source data into versioned DuckDB tables; never import a preview."""
import hashlib
import json
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.agent.data_import import import_dataset, _write_dataset
from src.agent.web_url import _download
from src.collectors.evds import _get_client, _year_chunks


def import_url_dataset(url: str, sheet: str | None = None) -> dict:
    final_url, content_type, content = _download(url)
    suffix = Path(urlparse(final_url).path).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        if "csv" in content_type:
            suffix = ".csv"
        elif "spreadsheetml" in content_type:
            suffix = ".xlsx"
        elif "excel" in content_type:
            suffix = ".xls"
        else:
            raise ValueError("Doğrudan CSV/Excel indirme URL'si gerekli; HTML için web_url_extract kullanın.")
    digest = hashlib.sha256(content + (sheet or "").encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="kkb-import-") as directory:
        path = Path(directory) / ("source" + suffix)
        path.write_bytes(content)
        result = import_dataset(path, "url_" + digest[:24], sheet=sheet, source_label=final_url)
    return {**result, "source_url": final_url, "sha256": digest,
            "content_trust": "untrusted_remote_content"}


def search_evds_series(query: str = "", datagroup: str | None = None) -> dict:
    client = _get_client()
    frame = client.get_series(datagroup) if datagroup else client.get_sub_categories()
    if query:
        def fold(value):
            return str(value).casefold().replace("i̇", "i").translate(str.maketrans("çğıöşü", "cgiosu"))
        needle = fold(query)
        frame = frame[frame.apply(lambda row: needle in fold(" ".join(map(str, row))), axis=1)]
    rows = json.loads(frame.head(60).to_json(orient="records", force_ascii=False))
    return {"source_url": "https://evds2.tcmb.gov.tr/", "rows": rows,
            "row_count": len(rows), "truncated": len(frame) > 60,
            "next_step": "DATAGROUP_CODE ile tekrar ara; yalnızca doğrulanmış SERIE_CODE ile fetch_evds_series çağır."}


def fetch_evds_series(datagroup: str, seri_kod: str, start_period: str,
                      end_period: str, aggregation: str = "mean") -> dict:
    if aggregation not in {"mean", "last", "sum"}:
        raise ValueError("aggregation: mean, last veya sum olmalı.")
    for period in (start_period, end_period):
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", period):
            raise ValueError("Dönem YYYY-MM biçiminde olmalı.")
    start, end = pd.Period(start_period, freq="M"), pd.Period(end_period, freq="M")
    if end < start or end.ordinal - start.ordinal > 120:
        raise ValueError("Dönem aralığı sıralı ve en fazla 121 ay olmalı.")
    client = _get_client()
    catalog = client.get_series(datagroup)
    matches = catalog[catalog["SERIE_CODE"] == seri_kod]
    if matches.empty:
        raise ValueError("Seri kodu EVDS grubunda doğrulanamadı; search_evds_series kullanın.")
    parts = []
    for first, last in _year_chunks(start.start_time.strftime("%d-%m-%Y"), end.end_time.strftime("%d-%m-%Y")):
        frame = client.get_data([seri_kod], startdate=first, enddate=last)
        if frame is not None and not frame.empty:
            parts.append(frame)
    if not parts:
        raise ValueError("EVDS istenen dönemde veri döndürmedi.")
    raw = pd.concat(parts, ignore_index=True).drop_duplicates()
    value_column = seri_kod.replace(".", "_")
    if value_column not in raw or "Tarih" not in raw:
        raise ValueError("EVDS tarih/değer şeması beklenenden farklı; veri dönüştürülmedi.")
    dates = raw["Tarih"].astype(str)
    monthly = dates.str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])")
    parsed = pd.to_datetime(dates.where(~monthly), dayfirst=True, errors="coerce", format="mixed")
    periods = parsed.dt.strftime("%Y-%m").where(~monthly, dates)
    values = pd.to_numeric(raw[value_column], errors="coerce")
    if periods.isna().any() or (raw[value_column].notna() & values.isna()).any():
        raise ValueError("EVDS tarih veya sayısal değerleri kayıpsız ayrıştırılamadı.")
    work = pd.DataFrame({"Period": periods, "Value": values, "Date": dates})
    if work.duplicated("Date").any():
        raise ValueError("Aynı tarih için çelişkili gözlemler var.")
    work = work[(work.Period >= start_period) & (work.Period <= end_period)].sort_values("Date", key=lambda s: pd.to_datetime(s, format="mixed", dayfirst=True))
    grouped = work.groupby("Period").Value
    series = grouped.sum(min_count=1) if aggregation == "sum" else getattr(grouped, aggregation)()
    result_frame = series.reset_index()
    digest = hashlib.sha256((seri_kod + aggregation + raw.to_json(orient="split")).encode()).hexdigest()
    raw_name, name = "evds_raw_" + digest[:24], "evds_" + digest[:24]
    _write_dataset(raw_name, "TCMB EVDS: " + seri_kod, raw)
    imported_at = _write_dataset(name, "TCMB EVDS: " + seri_kod, result_frame)
    missing = sorted(set(pd.period_range(start, end, freq="M").astype(str)) - set(result_frame.dropna().Period))
    return {"dataset_name": name, "raw_dataset_name": raw_name, "source_url": "https://evds2.tcmb.gov.tr/",
            "seri_kod": seri_kod, "metadata": json.loads(matches.to_json(orient="records", force_ascii=False)),
            "aggregation": aggregation, "imported_at": imported_at, "sha256": digest,
            "start_period": start_period, "end_period": end_period, "missing_periods": missing,
            "columns": ["Period", "Value"], "rows": json.loads(result_frame.to_json(orient="records")),
            "row_count": len(result_frame), "warning": "Faiz yıllıklandırılmış olabilir; TÜFE endeksi enflasyon oranı değildir. Birimi metadata ile doğrulayın."}
