from fastapi import FastAPI
from pydantic import BaseModel
import os
from .config import DB_URL

app = FastAPI(title="LaLiga RAG SQL API")

class AskBody(BaseModel):
    question: str

@app.get("/")
def root():
    return {"mensaje": "API activa"}

@app.post("/ask")
def ask(body: AskBody):
    return {"pregunta_recibida": body.question}
