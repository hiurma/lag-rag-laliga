from __future__ import annotations
import os
import re
import pandas as pd

from web_rag import fetch_standings_espn, fetch_laliga_titles_wikipedia
from rag_sql import ask_rag

try:
    from openai import OpenAI
except:
    OpenAI = None

def _md(df):
    try:
        return df.to_markdown(index=False)
    except:
        return df.to_string(index=False)

class ChatAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = "gpt-4o-mini"

    def chat_query(self, question: str):
        q = question.strip()
        q_low = q.lower()

        # 1️⃣ SQL RAG + PRONÓSTICOS
        sql_res = ask_rag(q)
        if sql_res:
            if sql_res["tipo"] == "pronostico":
                L = sql_res["local"]
                V = sql_res["visitante"]
                P = sql_res["pronostico"]
                probs = sql_res["probabilidades"]

                txt = (f"**Pronóstico {L} vs {V}**\n\n"
                       f"Marcador probable: **{P}**\n\n"
                       f"Probabilidades:\n"
                       f"- {L}: {probs['local']}%\n"
                       f"- Empate: {probs['empate']}%\n"
                       f"- {V}: {probs['visitante']}%\n")

                return {"mode": "sql-pronostico", "respuesta": txt}

        # 2️⃣ Clasificación
        if "clasific" in q_low or "tabla" in q_low:
            df, meta = fetch_standings_espn()
            return {
                "mode": "web",
                "respuesta": "Clasificación actual de LaLiga:",
                "tabla": _md(df)
            }

        # 3️⃣ Palmarés
        if "titulo" in q_low or "palmar" in q_low:
            df, url = fetch_laliga_titles_wikipedia()
            return {
                "mode": "web",
                "respuesta": "Títulos históricos por club:",
                "tabla": _md(df)
            }

        # 4️⃣ LLM fallback
        if not self.client:
            return {"mode": "llm", "respuesta": "(Sin LLM configurado)"}

        comp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Eres analista de fútbol español."},
                {"role": "user", "content": q}
            ]
        )

        return {"mode": "llm", "respuesta": comp.choices[0].message.content}