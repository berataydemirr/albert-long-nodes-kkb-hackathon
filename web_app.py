"""Local browser UI for exercising routing and function calling."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agent.orchestrator import ask
from src.agent.router import route_question
from src.agent.tools import TOOL_SCHEMAS

app = FastAPI(title="KKB Agentic Analytics", docs_url=None, redoc_url=None)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KKB Agent Lab</title><style>
:root{color-scheme:light;--bg:#f4f7fb;--panel:#fff;--line:#e3e9f2;--text:#172235;--muted:#66758c;--blue:#2563eb;--green:#087f5b;--red:#c92a2a}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#eef4ff,#f8fafc 55%,#eefbf6);color:var(--text);font:15px ui-sans-serif,system-ui,-apple-system,sans-serif}.shell{max-width:1120px;margin:auto;padding:42px 20px 64px}h1{font:750 30px system-ui;margin:0 0 7px;letter-spacing:-.03em}.lead{color:var(--muted);margin:0 0 26px}.grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(320px,.85fr);gap:18px}.panel{background:rgba(255,255,255,.9);box-shadow:0 12px 34px rgba(38,65,105,.08);border:1px solid var(--line);border-radius:16px;padding:20px}.examples{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 18px}.example{background:#f7f9fd;border:1px solid #d7e1ef;color:#29446b;border-radius:9px;padding:8px 11px;cursor:pointer;font:500 13px system-ui}.example:hover{border-color:var(--blue);color:var(--blue);background:#f0f5ff}textarea{width:100%;min-height:135px;resize:vertical;background:#fbfcfe;color:var(--text);border:1px solid #d7e0ec;border-radius:10px;padding:13px;font:inherit;line-height:1.5}#submit{margin-top:10px;background:var(--blue);color:#fff;border:0;border-radius:9px;padding:10px 18px;font:650 14px system-ui;cursor:pointer;box-shadow:0 5px 12px rgba(37,99,235,.2)}button:disabled{opacity:.55}.label{color:var(--muted);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:9px}.section-title{font:700 16px system-ui;margin:0 0 10px}.answer{white-space:pre-wrap;line-height:1.6;min-height:110px}.route{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:12px}.badge{padding:5px 9px;border-radius:999px;background:#e6f7ef;color:var(--green);font-size:12px;font-weight:700}.muted{color:var(--muted)}#trace{display:grid;gap:10px}.call{border-left:3px solid var(--blue);background:#f8faff;border-radius:7px;padding:11px}.call.error{border-color:var(--red)}.callhead{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px}.tool{color:var(--blue);font-weight:700}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#34445d;background:#f3f6fa;border-radius:6px;padding:9px;margin:7px 0 0;font-size:12px}.empty{color:var(--muted);padding:12px 0}.tool-list{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 18px}.tool-chip{border:1px solid #d7e1ef;background:#f7f9fd;border-radius:999px;padding:5px 8px;color:#48617f;font:12px ui-monospace,monospace}@media(max-width:820px){.grid{grid-template-columns:1fr}}
</style></head><body><main class="shell"><h1>KKB Agentic Analytics</h1><p class="lead">Agent Lab — router kararını ve modelin gerçek function-call zincirini canlı izle.</p>
<section class="grid"><div class="panel"><div class="section-title">Örnek Promptlar</div><div class="examples">
<button class="example" data-prompt="Lakehouse'daki tabloları ve mevcut serileri listele.">Lakehouse</button>
<button class="example" data-prompt="Konut kredisi faiz oranı serisindeki aykırı dönemleri anomaly_analysis ile bul ve açıkla.">C4 Anomaly</button>
<button class="example" data-prompt="Konut kredisi faiz oranındaki en büyük değişimleri bul ve yorumla.">Change detection</button>
<button class="example" data-prompt="Konut kredisi faiz oranı ile konut kredisi hacmi arasında nedensellik var mı? Kaynak dönemleriyle analiz et.">Causality</button>
<button class="example" data-prompt="TCMB'nin güncel politika faizini web'de ara; güvenilir kaynak URL'lerini ver.">Web search</button>
<button class="example" data-prompt="https://www.tcmb.gov.tr adresindeki içeriği çıkar ve ana başlıkları özetle.">URL extract</button>
<button class="example" data-prompt="Güncel politika faizini web'de bul, en güvenilir sonuç URL'sini aç ve yerel konut kredisi serisiyle kıyasla.">Tool chain</button>
</div><form id="ask-form"><textarea id="question" maxlength="2000" required placeholder="Sorunu yaz veya yukarıdan örnek seç…"></textarea><button id="submit" type="submit">Çalıştır</button></form></div>
<div class="panel"><div class="section-title">Kullanılabilir Tool'lar</div><div id="tool-list" class="tool-list"><span class="empty">Tool registry yükleniyor…</span></div><div class="section-title" style="margin-top:18px">Router</div><div id="route" class="empty">Henüz bir istek yok.</div><div class="section-title" style="margin-top:18px">Function calls</div><div id="trace"><div class="empty">Tool çağrıları burada görünecek.</div></div></div></section>
<section class="panel" style="margin-top:16px"><div class="label">Final answer</div><div id="answer" class="answer" aria-live="polite">Yanıt burada görünecek.</div></section></main>
<script>
const form=document.querySelector('#ask-form'),q=document.querySelector('#question'),submit=document.querySelector('#submit'),answer=document.querySelector('#answer'),trace=document.querySelector('#trace'),route=document.querySelector('#route');
document.querySelectorAll('.example').forEach(x=>x.addEventListener('click',()=>{q.value=x.dataset.prompt;q.focus()}));
fetch('/api/tools').then(r=>r.json()).then(d=>{document.querySelector('#tool-list').innerHTML=d.tools.map(t=>`<span class="tool-chip" title="${esc(t.description)}">${esc(t.name)}</span>`).join('')}).catch(()=>{document.querySelector('#tool-list').innerHTML='<span class="muted">Tool registry yüklenemedi.</span>'});
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pretty=s=>{try{return JSON.stringify(JSON.parse(s),null,2)}catch{return s||'{}'}};
form.addEventListener('submit',async e=>{e.preventDefault();submit.disabled=true;answer.textContent='Yanıt hazırlanıyor…';trace.innerHTML='<div class="empty">Model çalışıyor…</div>';route.innerHTML='<span class="muted">Router bekleniyor…</span>';try{const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q.value})}),d=await r.json();if(!r.ok)throw new Error(d.detail||'İstek başarısız oldu.');answer.textContent=d.answer;route.innerHTML=`<div class="route"><span class="badge">${esc(d.route.name)}</span><span class="muted">${esc(d.route.reason)}</span></div><div class="muted">Erişilebilir tool (${d.route.allowed_tools.length}): ${d.route.allowed_tools.map(esc).join(', ')}</div>`;trace.innerHTML=d.trace.length?d.trace.map((x,i)=>`<article class="call ${x.error?'error':''}"><div class="callhead"><span class="tool">${i+1}. ${esc(x.name)}</span><span class="muted">round ${esc(x.round)}</span></div><div class="muted">arguments</div><pre>${esc(pretty(x.arguments))}</pre><div class="muted" style="margin-top:8px">result preview</div><pre>${esc(pretty(x.output_preview))}</pre>${x.error?`<pre>${esc(x.error)}</pre>`:''}</article>`).join(''):'<div class="empty">Model hiçbir tool çağırmadı.</div>'}catch(err){answer.textContent='Hata: '+err.message;trace.innerHTML='<div class="empty">Trace üretilemedi.</div>'}finally{submit.disabled=false}});
</script></body></html>"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/tools")
def list_tools() -> dict:
    """Expose the model-visible registry for local MVP inspection."""
    return {
        "count": len(TOOL_SCHEMAS),
        "tools": [
            {
                "name": schema["function"]["name"],
                "description": schema["function"].get("description", ""),
                "parameters": schema["function"].get("parameters", {}),
            }
            for schema in TOOL_SCHEMAS
        ],
    }


@app.post("/api/ask")
def answer_question(request: AskRequest) -> dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Soru boş olamaz.")
    try:
        decision = route_question(question, TOOL_SCHEMAS)
        answer, trace = ask(question, return_trace=True)
        return {
            "answer": answer,
            "route": {
                "name": decision.route.value,
                "reason": decision.reason,
                "available": decision.available,
                "allowed_tools": list(decision.allowed_tool_names),
            },
            "trace": trace,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent yanıt üretemedi. Sunucu günlüklerini kontrol edin.") from exc
