from __future__ import annotations

import logging
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd

import sys

# Aseguramos que el directorio src esté en el PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from chat_agent import ChatAgent

# -----------------------------------------------------------------------------
# FastAPI APP
# -----------------------------------------------------------------------------

app = FastAPI(title="LaLiga RAG API", version="1.0")

# CORS – GitHub Pages + cualquier origen (para ir tranquilos)
origins = [
    "https://hiurma.github.io",
    "https://hiurma.github.io/",
    "https://hiurma.github.io/laliga-chat-web",
    "https://hiurma.github.io/laliga-chat-web/",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = ChatAgent()

STATIC_DIR = os.path.join(CURRENT_DIR, "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")
logger = logging.getLogger(__name__)

if not os.path.isdir(STATIC_DIR):
    logger.warning("Static directory not found at %s; landing page won't load.", STATIC_DIR)
if not os.path.isfile(INDEX_FILE):
    logger.warning("Landing page missing at %s; root will return an error.", INDEX_FILE)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatBody(BaseModel):
    pregunta: str


@app.get("/")
def root():
    if os.path.isfile(INDEX_FILE):
        return FileResponse(INDEX_FILE)

    raise HTTPException(status_code=500, detail="Landing page no encontrada en el servidor")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat_endpoint(body: ChatBody):
    q = (body.pregunta or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La 'pregunta' no puede estar vacía")

    try:
        return agent.chat_query(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
