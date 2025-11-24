from __future__ import annotations

import os
import sys
import time
import pathlib
import sqlite3
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# --------------------
# PATHS / ENV
# --------------------
load_dotenv()

BASE_DIR = pathlib.Path(__file__).resolve().parent          # .../src
ROOT_DIR = BASE_DIR.parent                                 # .../ (raíz repo)
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
STATIC_DIR = BASE_DIR / "static"

REPORTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# asegura que src esté en el path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# --------------------
# IMPORTS internos
# --------------------
from chat_agent import ChatAgent
from rag_sql import ask_rag
from report_builder import build_report

agent = ChatAgent()

app = FastAPI(title="LaLiga RAG SQL API", version="1.0")

# --------------------
# CORS
# --------------------
# En producción puedes quitar "*" si quieres seguridad.
origins = [
    "https://hiurma.github.io",
    "https://hiurma.github.io/",
    "https://hiurma.github.io/laliga-chat-web",
    "https://hiurma.github.io/laliga-chat-web/",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# MODELOS
# --------------------
class ChatBody(BaseModel):
    pregunta: str

class AskBody(BaseModel):
    pregunta: Optional[str] = None
    question: Optional[str] = None

class ReportBody(BaseModel):
    temporada: Optional[str] = None
    preguntas: Optional[List[str]] = None
    formato: Optional[str] = "md"   # "md" | "html" | "both"


# --------------------
# ROOT / HEALTH / FRONT
# --------------------
@app.get("/")
def root():
    """
    Si existe src/static/index.html, lo sirve.
    Si no, devuelve JSON simple.
    """
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "LaLiga RAG SQL API is running."}

@app.get("/health")
def health():
    return {"ok": True}


# --------------------
# CHAT
# --------------------
@app.post("/chat")
def chat_endpoint(body: ChatBody):
    q = (body.pregunta or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La 'pregunta' no puede estar vacía")

    try:
        return agent.chat_query(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------
# RAG SQL (opcional)
# --------------------
@app.post("/ask")
def ask_endpoint(body: AskBody):
    q = (body.pregunta or body.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Falta 'pregunta'")

    try:
        return ask_rag(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------
# REPORT
# --------------------
def _leer_bateria(temporada: str | None):
    p = DATA_DIR / "prompts" / "bateria.md"
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
    import re
    m = re.match(r"^(\d{4})\s*[-/]\s*(\d{4})$", t)
    if m:
        y1, y2 = m.group(1), m.group(2)
        return f"{y1}/{y2}"
    m2 = re.match(r"^(\d{4})$", t)
    if m2:
        y1 = int(m2.group(1))
        return f"{y1}/{y1+1}"
    return t

@app.post("/report")
def generar_report(body: ReportBody):
    temporada_norm = _normalize_temporada(body.temporada)
    if not temporada_norm:
        raise HTTPException(status_code=400, detail="Falta 'temporada'")

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
        "preview_markdown": md_text[:500],
        "total": len(preguntas),
    }


@app.get("/reports/list")
def listar_reports():
    files = []
    for f in sorted(REPORTS_DIR.glob("report_*.*")):
        st = f.stat()
        files.append({
            "name": f.name,
            "path": str(f),
            "size_bytes": st.st_size,
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
        })
    return {"count": len(files), "items": files}


# --------------------
# STATIC MOUNTS
# --------------------
# Sirve /reports/*
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")
# Si quieres servir más assets estáticos:
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
