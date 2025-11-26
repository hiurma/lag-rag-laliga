from __future__ import annotations

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


class ChatBody(BaseModel):
    pregunta: str


@app.get("/")
def root():
    return {
        "message": "LaLiga RAG API está viva.",
        "endpoints": ["/health", "/chat"],
    }


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
