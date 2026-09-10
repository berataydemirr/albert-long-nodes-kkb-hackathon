"""Minimal browser UI for the KKB MIA agent.

Run locally with ``uvicorn web_app:app --reload`` and open the printed URL.
The API exposes only a question-to-answer operation; tools and credentials
remain server-side.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agent.orchestrator import ask

app = FastAPI(title="KKB Agentic Analytics", docs_url=None, redoc_url=None)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>KKB Agentic Analytics</title><style>
body{font:16px system-ui,sans-serif;background:#f6f8fb;color:#162235;max-width:800px;margin:0 auto;padding:3rem 1.25rem}h1{margin:0 0 .5rem}p{color:#536273}form{display:grid;gap:.75rem;margin-top:2rem}textarea{min-height:110px;padding:.8rem;font:inherit;border:1px solid #b7c1ce;border-radius:8px}button{justify-self:start;background:#1259b0;color:white;border:0;border-radius:7px;padding:.7rem 1rem;font:inherit;cursor:pointer}button:disabled{opacity:.6}#answer{white-space:pre-wrap;background:white;border:1px solid #dce2ea;border-radius:8px;padding:1rem;margin-top:1.5rem;min-height:2rem}</style>
</head><body><h1>KKB Agentic Analytics</h1><p>Güncel veya harici bir soru sorarak web aramasını test edin. Cevaptaki kaynak URL'lerini kontrol edin.</p>
<form id="ask-form"><textarea id="question" maxlength="2000" required placeholder="Örn. TCMB'nin güncel politika faizi nedir? Kaynaklarıyla ver."></textarea><button id="submit" type="submit">Sor</button></form><section id="answer" aria-live="polite">Yanıt burada görünecek.</section>
<script>const form=document.querySelector('#ask-form'),q=document.querySelector('#question'),b=document.querySelector('#submit'),a=document.querySelector('#answer');form.addEventListener('submit',async e=>{e.preventDefault();b.disabled=true;a.textContent='Yanıt hazırlanıyor…';try{const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q.value})}),d=await r.json();if(!r.ok)throw new Error(d.detail||'İstek başarısız oldu.');a.textContent=d.answer}catch(err){a.textContent='Hata: '+err.message}finally{b.disabled=false}});</script></body></html>"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask")
def answer_question(request: AskRequest) -> dict[str, str]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Soru boş olamaz.")
    try:
        return {"answer": ask(question)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent yanıt üretemedi. Sunucu günlüklerini kontrol edin.") from exc
