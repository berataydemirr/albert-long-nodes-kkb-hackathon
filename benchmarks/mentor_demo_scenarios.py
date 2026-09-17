"""Mentor demo dry-run: three 4-question chained scenarios (team's own
"Sistem Sorunlari ve Cozum Planimiz" doc), each testing whether the agent
can use ONE question's real output as grounding for the NEXT question -
the exact shape mentor questions will take on demo day ("üstüne koya koya
gidecek", each step building on the previous answer).

Unlike benchmarks/agent_benchmark.py (single-shot questions), this drives
ask() with history=[...] carried forward between calls, exactly like the
browser tab does via historyForRequest() in static/design/index.html - so
a scenario genuinely exercises cross-turn memory, not just one isolated
call.

CI'da/pytest'te calismaz (network + token maliyeti + non-deterministik,
same reasoning as agent_benchmark.py).

Calistir (repo root'tan, -m ile - plain `python benchmarks/....py` "src"
paketini bulamaz, bu proje editable install degil):
  python -m benchmarks.mentor_demo_scenarios
  python -m benchmarks.mentor_demo_scenarios --only senaryo_1_enflasyon_kredi_nedensellik
"""
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agent.orchestrator import ask
from src.common.logging_setup import get_logger

logger = get_logger("mentor_demo_scenarios")
REPORT_PATH = Path(__file__).parent / "mentor_demo_scenarios_report.json"

SCENARIOS = [
    {
        "id": "senaryo_1_enflasyon_kredi_nedensellik",
        "title": "Enflasyon ve Kredi Hacmi Nedenselliği",
        "questions": [
            "TCMB web bağlantısı üzerinden son iki yılın aylık ihtiyaç kredisi faiz oranlarını çek ve listele. "
            "Yazdığın koda her adımı terminale yazdıracak bol log ve hata ayıklama satırı ekle.",
            "Aynı zaman dilimi için aylık enflasyon verilerini yerel veri tabanından getir.",
            "İlk soruda bulduğun faiz oranları ile az önce çektiğin enflasyon verisini birleştirip reel faiz "
            "oranını hesapla. Hata ile karşılaşırsan log kayıtlarından algoritmanın nerede koptuğunu detaylıca bulalım.",
            "Reel faizin negatif olduğu dönemlerde hanehalkı kredi kullanım hacmi nasıl değişmiş incele. "
            "Çıktı olarak bu ilişkinin nedenselliğini gösteren bir grafik hazırla.",
        ],
        "expected_output": "Veri kaynaklarının doğrulandığı log kayıtları ile birlikte faiz ve kredi hacmi zıtlığını gösteren net bir grafikli rapor.",
    },
    {
        "id": "senaryo_2_dis_veri_kirilganlik",
        "title": "Dış Veri Entegrasyonu ve Kırılganlık Analizi",
        "questions": [
            "Sisteme yüklenen şirket bilançoları Excel dosyasını oku ve teknoloji şirketlerinin kar marjlarını "
            "hesapla. İşlem adımlarını takip edebilmemiz için kodun içine detaylı hata ayıklama satırları yerleştir.",
            "Lakehouse içindeki ulusal serilerden güncel döviz kuru oynaklığını bul.",
            "Döviz kurundaki en sert sıçramaların yaşandığı aykırı dönemleri anomali testi ile tespit et.",
            "Bu aykırı dönemlerde teknoloji şirketlerinin karlılık oranlarındaki değişimi ilk adımdaki verileri "
            "kullanarak kıyasla. Kur şoklarına karşı en kırılgan üç şirketi belirle.",
        ],
        "expected_output": "Kur şokları ve şirket karlılığı arasındaki anomaliyi eşleştiren detaylı bir kırılganlık analizi tablosu.",
        # Real prerequisite this scenario assumes but this repo does not
        # ship: an uploaded "şirket bilançoları" Excel with tech-company
        # margin data. Without it, question 1 is expected to fail honestly
        # ("no such dataset") rather than fabricate figures - that failure
        # is itself part of what this dry run is meant to catch.
        "requires_upload": "şirket bilançoları (teknoloji şirketleri) Excel dosyası - bu repoda yok, demo öncesi yüklenmeli",
    },
    {
        "id": "senaryo_3_bolgesel_isi_haritasi",
        "title": "Bölgesel Isı Haritası ve Kredi Batık Oranları",
        "questions": [
            "BDDK verilerinden Türkiye geneli son çeyrek takipteki konut kredisi oranlarını çek. Arama işlemi "
            "sırasında takılırsan arka planda ne olduğunu anlamamız için bol debug kodu kullan.",
            "Bu veriler ışığında batık kredi oranı en yüksek on ili sırala.",
            "İkinci soruda bulduğun on ilin TÜİK verilerinden güncel işsizlik oranlarını getir.",
            "Batık kredi oranları ile işsizlik verilerini birleştirerek bir eşik değer filtresi oluştur. İki "
            "verinin de ülke ortalamasının üzerinde olduğu illeri vurgulayan analitik bir metin yaz.",
        ],
        "expected_output": "Bölgesel riskleri şeffaf log kayıtlarıyla kanıtlayan kusursuz bir veri birleştirme ve çapraz analiz metni.",
    },
]


def run_scenario(scenario: dict) -> dict:
    logger.info(f"run_scenario: START {scenario['id']}")
    history: list[dict] = []
    turns = []
    for i, question in enumerate(scenario["questions"], start=1):
        t0 = time.monotonic()
        try:
            answer, trace, timing = ask(question, history=history, return_trace=True, return_timing=True)
            error = None
        except Exception as exc:
            logger.exception(f"run_scenario: {scenario['id']} step {i} raised")
            answer, trace, timing, error = "", [], None, f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - t0
        turn = {
            "step": i,
            "question": question,
            "answer": answer,
            "tools_called": [t["name"] for t in trace],
            "tool_errors": [{"name": t["name"], "error": t["error"]} for t in trace if t.get("error")],
            "elapsed_sec": round(elapsed, 1),
            "timing_summary": timing["summary"] if timing else None,
            "error": error,
        }
        turns.append(turn)
        # Only (question, answer) carries forward - exactly what a real
        # browser tab sends via historyForRequest() in
        # static/design/index.html. A scenario step that raised is NOT
        # added to history: an empty/error turn would just poison every
        # later step with nothing to build on.
        if not error:
            history.append({"question": question, "answer": answer})
        logger.info(
            f"run_scenario: {scenario['id']} step {i}/{len(scenario['questions'])} done in {elapsed:.1f}s, "
            f"tools={turn['tools_called']}, error={error}"
        )
    logger.info(f"run_scenario: DONE {scenario['id']}")
    return {
        "id": scenario["id"], "title": scenario["title"], "turns": turns,
        "expected_output": scenario["expected_output"],
        "requires_upload": scenario.get("requires_upload"),
    }


def main() -> None:
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    scenarios = [s for s in SCENARIOS if only is None or s["id"] == only]
    if not scenarios:
        print(f"No scenario matches --only {only!r}. Known ids: {[s['id'] for s in SCENARIOS]}")
        return

    results = [run_scenario(s) for s in scenarios]
    REPORT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("MENTOR DEMO SCENARIO DRY RUN")
    print("=" * 60)
    for r in results:
        print(f"\n[{r['id']}] {r['title']}")
        for t in r["turns"]:
            flag = "OK" if not t["error"] and not t["tool_errors"] else "DİKKAT"
            print(f"  [{flag}] step {t['step']}: {t['elapsed_sec']}s, tools={t['tools_called']}")
            if t["error"]:
                print(f"          error: {t['error']}")
            if t["tool_errors"]:
                print(f"          tool_errors: {t['tool_errors']}")
    print(f"\nDetaylı rapor: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
