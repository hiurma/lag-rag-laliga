from __future__ import annotations

import os, sys, pathlib, sqlite3, time, re
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# 0) SETUP BÁSICO
# =========================
load_dotenv()

# Asegura imports desde /src
ROOT = os.path.dirname(__file__)  # .../src
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Importa tus módulos
from chat_agent import ChatAgent
from rag_sql import ask_rag
from report_builder import build_report

app = FastAPI(title="LaLiga RAG SQL API", version="1.0")

# =========================
# 1) CORS (CORRECTO)
# =========================
# IMPORTANTE:
# - NO uses "*" con allow_credentials=True
# - Para GitHub Pages, basta con permitir tu dominio exacto
origins = [
    "https://hiurma.github.io",
    "https://hiurma.github.io/laliga-chat-web",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,   # CLAVE para que no bloquee preflight
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 2) MODELOS Pydantic
# =========================
class ChatBody(BaseModel):
    pregunta: str

class ReportBody(BaseModel):
    temporada: Optional[str] = None
    preguntas: Optional[List[str]] = None
    formato: Optional[str] = "md"  # "md" | "html" | "both"

# =========================
# 3) AGENTE
# =========================
agent = ChatAgent()

# =========================
# 4) ENDPOINTS
# =========================
@app.get("/")
def root():
    return {"message": "LaLiga RAG SQL API is running."}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat_endpoint(body: ChatBody):
    q = (body.pregunta or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La 'pregunta' no puede estar vacía")
    return agent.chat_query(q)

# ---- Helpers de report ----
def _leer_bateria(temporada: str | None):
    p = pathlib.Path("data/prompts/bateria.md")
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        q = line.lstrip("-").strip().strip('"').strip("'")
        if temporada and "{temporada}" in q:
            q = q.replace("{temporada}", temporada)
        out.append(q)
    return out

def _normalize_temporada(t_raw: str | None) -> str | None:
    if not t_raw:
        return None
    t = t_raw.strip()
    # "2025/2026" o "2025-2026" -> "2025/2026"
    m = re.match(r"^(\d{4})\s*[-/]\s*(\d{4})$", t)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # "2025" -> "2025/2026"
    m2 = re.match(r"^(\d{4})$", t)
    if m2:
        y1 = int(m2.group(1))
        return f"{y1}/{y1+1}"
    return t

@app.post("/report")
def generar_report(body: ReportBody):
    temporada_norm = _normalize_temporada(body.temporada)

    preguntas = body.preguntas or _leer_bateria(temporada_norm)
    if not preguntas:
        raise HTTPException(
            status_code=400,
            detail="No hay preguntas. Pasa 'preguntas' o crea data/prompts/bateria.md"
        )

    md_path, html_path, md_text, html_text = build_report(temporada_norm, preguntas)

    formato = (body.formato or "md").lower()
    if formato == "html":
        return HTMLResponse(content=html_text, status_code=200)
    if formato == "md":
        return PlainTextResponse(content=md_text, status_code=200)

    return {
        "ok": True,
        "temporada_usada": temporada_norm,
        "report_md": md_path,
        "report_html": html_path,
        "url_html": f"/reports/{os.path.basename(html_path)}",
        "total_preguntas": len(preguntas),
    }

@app.get("/diag/sql")
def diag_sql(sql: str = Query(..., description="Ejecuta una consulta SQL manual (solo SELECT)")):
    try:
        con = sqlite3.connect(os.getenv("DB_PATH", "data/laliga.sqlite"))
        cur = con.cursor()
        cur.execute(sql)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
        con.close()
        return {"ok": True, "columnas": cols, "resultados": rows[:10]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# =========================
# 5) REPORTS ESTÁTICOS
# =========================
pathlib.Path("reports").mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

