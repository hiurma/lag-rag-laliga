from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from chat_agent import ChatAgent

# -------------------------------------------------------------------
#   APP PRINCIPAL
# -------------------------------------------------------------------
app = FastAPI(title="LaLiga RAG API", version="1.0")

# -------------------------------------------------------------------
#   DIRECTORIOS
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# -------------------------------------------------------------------
#   SERVIR ARCHIVOS ESTÁTICOS (HTML / JS / CSS)
# -------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def serve_home():
    """Sirve la página web del chat."""
    html_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(html_path)

# -------------------------------------------------------------------
#   CORS (REALMENTE YA NO ES NECESARIO SI WEB Y API ESTÁN EN RENDER)
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Puedes restringirlo después si quieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
#   MODELOS
# -------------------------------------------------------------------
class ChatRequest(BaseModel):
    pregunta: str

# -------------------------------------------------------------------
#   AGENTE RAG
# -------------------------------------------------------------------
agent = ChatAgent()

# -------------------------------------------------------------------
#   ENDPOINTS
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat_endpoint(body: ChatRequest):
    try:
        q = body.pregunta.strip()
        if not q:
            raise HTTPException(status_code=400, detail="La pregunta está vacía")

        result = agent.chat_query(q)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   
   
    # --- app_fastapi.py ------------------------------------------------


# Dominios permitidos (tu GitHub Pages + comodín por si acaso)
origins: List[str] = [
    "https://hiurma.github.io",
    "https://hiurma.github.io/",
    "https://hiurma.github.io/laliga-chat-web",
    "https://hiurma.github.io/laliga-chat-web/",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # si lo quieres súper estricto, quita "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatBody(BaseModel):
    pregunta: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(body: ChatBody):
    q = (body.pregunta or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La 'pregunta' no puede estar vacía")
    return agent.chat_query(q)
