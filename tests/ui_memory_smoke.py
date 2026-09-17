"""Browser smoke test with labelled fixture responses; no paid model calls.

Run the app on 127.0.0.1:8013, then python tests/ui_memory_smoke.py.
"""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    requests = []
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1360, "height": 950})
        page.on("pageerror", lambda error: errors.append(str(error)))

        def respond(route):
            requests.append(route.request.post_data_json)
            number = len(requests)
            route.fulfill(json={"answer": f"Test verisi {number}\n\n| Dönem | Değer |\n|---|---|\n|2025-01|10|",
                "route": {"name": "analysis", "reason": "browser fixture"}, "trace": [], "charts": [],
                "turn_id": f"{number:032x}", "turn_note": "test kaydı",
                "artifacts": [{"evidence_id": f"{number + 10:032x}", "row_count": 1}],
                "timing": {"summary": {"total_ms": 10, "tool_ms": 5}, "context": {"compaction_runs": 1}}})

        page.route("**/api/ask", respond)
        page.goto("http://127.0.0.1:8013/lab/", wait_until="networkidle")
        field = page.locator("textarea")
        field.fill("Bellek testi: faiz verilerini listele")
        field.press("Enter")
        page.get_by_text("Test verisi 1", exact=True).wait_for()
        page.get_by_text("1 veri kaydı korundu · Bağlam sıkıştırıldı (%80)", exact=True).wait_for()
        field.fill("Aynı dönemin enflasyonunu getir")
        field.press("Enter")
        page.get_by_text("Test verisi 2", exact=True).wait_for()
        assert requests[1]["turn_ids"] == [f"{1:032x}"]
        page.reload(wait_until="networkidle")
        page.get_by_text("Bellek testi: faiz verilerini listele", exact=True).first.click()
        page.get_by_text("Test verisi 2", exact=True).wait_for()
        page.locator("textarea").fill("İlk iki verinin korelasyonunu hesapla")
        page.locator("textarea").press("Enter")
        page.get_by_text("Test verisi 3", exact=True).wait_for()
        assert requests[2]["turn_ids"] == [f"{1:032x}", f"{2:032x}"]
        assert not errors, errors
        Path(".playtest").mkdir(exist_ok=True)
        page.screenshot(path=".playtest/progressive-finance.png", full_page=True, animations="disabled")
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        browser.close()
    print(json.dumps({"passed": True, "turns": len(requests), "reload_preserved_ids": True,
                      "page_errors": errors, "mobile_overflow": False}))


if __name__ == "__main__":
    main()
