"""Fetch source data into versioned DuckDB tables; never import a preview."""
import difflib
import hashlib
import json
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from config.settings import LAKEHOUSE_DB_PATH
from src.agent.data_import import import_dataset, _write_dataset
from src.agent.db_utils import connect_with_retry
from src.agent.web_url import _download
from src.collectors.evds import _get_client, _year_chunks
from src.common.logging_setup import get_logger

logger = get_logger("finance_fetch")


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


def _fold(value) -> str:
    return str(value).casefold().replace("i̇", "i").translate(str.maketrans("çğıöşü", "cgiosu"))


# Word-level similarity threshold for the fuzzy fallback below - below this,
# a row is unrelated noise rather than a plausible near-miss (calibrated by
# hand against real EVDS catalog rows: "ihtiyaç kredisi" vs "İhtiyaç
# Kredisi Faiz Oranı" scores ~0.9 per word; unrelated rows score <0.3).
_FUZZY_FLOOR = 0.6


def _closest_rows(frame, needle: str, limit: int = 20):
    """Falls back to per-word letter-similarity when the exact substring
    search finds nothing - confirmed live: "ihtiyaç kredisi faiz oranı"
    returned 0 rows even though EVDS has a close series, purely because
    its own naming doesn't contain that exact phrase as a substring.
    Scored per query-word against the best-matching word anywhere in the
    row (not the whole joined row against the whole query) so word order
    and extra words in either side don't tank the score the way a
    whole-string ratio would."""
    needle_words = needle.split()
    if not needle_words:
        return frame.iloc[0:0]

    def row_score(row) -> float:
        text_words = _fold(" ".join(map(str, row))).split()
        if not text_words:
            return 0.0
        return sum(
            max((difflib.SequenceMatcher(None, w, tw).ratio() for tw in text_words), default=0.0)
            for w in needle_words
        ) / len(needle_words)

    scores = frame.apply(row_score, axis=1)
    ranked = scores[scores >= _FUZZY_FLOOR].sort_values(ascending=False)
    return frame.loc[ranked.index[:limit]]


def _imported_evds_series(con, needle: str) -> list[dict]:
    """EVDS columns (TP_*) inside imported datasets, described by the
    file's own "Seri Açıklamaları" rows. Live: the model only found
    TP_KTF17 (TL commercial loan rate) in an uploaded EVDS export after the
    user said to look there."""
    found = []
    for dataset, source in con.execute(
            "SELECT dataset_name, source_file FROM _import_registry ORDER BY dataset_name").fetchall():
        try:
            columns = [row[0] for row in con.execute(f'DESCRIBE "{dataset}"').fetchall()]
        except Exception:
            continue
        codes = [c for c in columns if c.upper().startswith("TP_")]
        if not codes or len(columns) < 2:
            continue
        # EVDS exports list descriptions below the data: code in the date
        # column, description in the next one.
        first, second = columns[0], columns[1]
        # The first mention wins: a later "Notlar" block repeats each code
        # with "Veri Kaynağı" etc. in the same column.
        described = {}
        for code, text in con.execute(
                f'SELECT CAST("{first}" AS VARCHAR), CAST("{second}" AS VARCHAR) FROM "{dataset}" '
                f'WHERE CAST("{first}" AS VARCHAR) LIKE \'TP.%\' AND "{second}" IS NOT NULL ORDER BY rowid').fetchall():
            described.setdefault(code, text)
        for code in codes:
            description = described.get(code.replace("_", ".")) or described.get(code)
            found.append({"dataset_name": dataset, "source_file": source,
                          "kolon": code, "aciklama": description})
    words = [w for w in needle.replace("_", ".").split() if len(w) >= 3]
    if words:
        def hits(row):
            text = _fold(f"{row['kolon']} {row['aciklama'] or ''}".replace("_", "."))
            return sum(w in text for w in words)
        best = max((hits(row) for row in found), default=0)
        return [row for row in found if hits(row) == best][:60] if best else []
    return found[:60]


def _evds_unavailable(needle: str = "") -> dict:
    """Result for a missing EVDS_API_KEY. A raw ValueError sent the model
    off searching the web for series the lakehouse already holds (live:
    a CPI question needing only TP.GENENDEKS.T1); list those, and any EVDS
    columns inside imported files, instead."""
    rows, imported = [], []
    try:
        con = connect_with_retry(LAKEHOUSE_DB_PATH, read_only=True)
        try:
            rows = con.execute(
                "SELECT Seri_Kod, any_value(Seri_Adi), min(Period), max(Period) FROM gold_fact_long "
                "WHERE Kaynak = 'evds' GROUP BY Seri_Kod ORDER BY Seri_Kod").fetchall()
            imported = _imported_evds_series(con, _fold(needle))
        finally:
            con.close()
    except Exception:
        logger.exception("_evds_unavailable: offline EVDS catalog lookup failed")
    series = [{"seri_kod": code, "seri_adi": name, "start_period": start, "end_period": end}
              for code, name, start, end in rows]
    return {
        "error": "Canlı EVDS erişimi kapalı: EVDS_API_KEY tanımlı değil.",
        "evds_available": False,
        "lakehouse_evds_series": series,
        "imported_evds_series": imported,
        "next_step": (
            "İstenen seri lakehouse_evds_series listesindeyse gold_fact_long üzerinden lakehouse_query ile "
            "oku (kaynak: 'TCMB EVDS, lakehouse'a önceden yüklenmiş'). Değilse imported_evds_series'e bak: "
            "açıklaması istenen kavramla eşleşen kolonu query_imported_dataset ile oku; tarih kolonu metin "
            "olabilir ve dosyanın altındaki açıklama satırlarını (TP.xxx) veri sayma. Bunun kullanıcı "
            "yüklemesi olduğunu ve doğrulanmadığını belirt. İkisinde de yoksa canlı EVDS'ye erişilemediğini "
            "söyle; seri kodu veya değer tahmin etme, EVDS'yi web_search/web_url_extract ile arama "
            "(sayfaları JavaScript ile yüklenir)."
        ),
    }


def _evds_client():
    try:
        return _get_client()
    except ValueError as exc:
        if "EVDS_API_KEY" in str(exc):
            return None
        raise


def search_evds_series(query: str = "", datagroup: str | None = None) -> dict:
    client = _evds_client()
    if client is None:
        return _evds_unavailable(query)
    try:
        frame = client.get_series(datagroup) if datagroup else client.get_sub_categories()
    except KeyError:
        # The evds package indexes the response with a fixed column list
        # (SERIE_CODE/SERIE_NAME/START_DATE) and raises a raw KeyError
        # instead of a clear error when `datagroup` is not a real
        # DATAGROUP_CODE - confirmed live: the model guessed "BKR" (a
        # SERIE_CODE prefix, not a datagroup code) and this propagated as
        # an opaque tool_error with a pandas traceback. Turn it into a
        # helpful result that points back at the real discovery flow
        # instead of a crash the model has no way to act on.
        raise ValueError(
            f"datagroup={datagroup!r} geçerli bir EVDS DATAGROUP_CODE değil (SERIE_CODE öneki "
            f"'BKR' gibi bir şeyle karıştırılmış olabilir). Önce search_evds_series'i datagroup "
            f"OLMADAN çağır (get_sub_categories) - gerçek DATAGROUP_CODE değerleri (ör. "
            f"'bie_kt210a') sonuçta görünür; sonra o kodla tekrar dene."
        ) from None
    fuzzy = False
    if query:
        needle = _fold(query)
        exact = frame[frame.apply(lambda row: needle in _fold(" ".join(map(str, row))), axis=1)]
        if not exact.empty:
            frame = exact
        else:
            frame = _closest_rows(frame, needle)
            fuzzy = True
    rows = json.loads(frame.head(60).to_json(orient="records", force_ascii=False))
    next_step = (
        "Tam eşleşme bulunamadı; bunlar harf benzerliğine göre en yakın adaylardır (YAKLAŞIK "
        "eşleşme) - kullanmadan önce SERIE_NAME'i oku ve gerçekten istenen kavramı karşılıyor mu "
        "değerlendir, sonra fetch_evds_series ile doğrula."
        if fuzzy else
        "DATAGROUP_CODE ile tekrar ara; yalnızca doğrulanmış SERIE_CODE ile fetch_evds_series çağır."
    )
    return {"source_url": "https://evds2.tcmb.gov.tr/", "rows": rows,
            "row_count": len(rows), "truncated": len(frame) > 60,
            "fuzzy_match": fuzzy, "next_step": next_step}


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
    client = _evds_client()
    if client is None:
        return _evds_unavailable(seri_kod)
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
