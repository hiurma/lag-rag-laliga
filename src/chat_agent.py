# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
from typing import Any, Dict, Optional
import pandas as pd

from rag_sql import ask_rag
from poisson_model import predict_match_poisson

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


SEASON_RE = re.compile(r"(20\d{2})\s*[-/]\s*(20\d{2})", re.I)
PRONO_HINT = re.compile(r"\b(prono|pronó|pronós|pronost|predic|predi[cs])", re.I)
SPLIT_RE = re.compile(r"\s+(?:vs\.?|v|contra|frente\s+a)\s+", re.I)


def _extract_season(text: str) -> Optional[str]:
    m = SEASON_RE.search(text)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "(sin filas)"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return "\n" + df.head(max_rows).to_string(index=False) + "\n"


def _parse_match(q: str) -> Optional[tuple[str, str, Optional[str]]]:
    text = q.strip()
    if not PRONO_HINT.search(text):
        return None

    season = _extract_season(text)
    if season:
        text = SEASON_RE.sub("", text)

    text = re.sub(r"^\s*(pron[oó]stico|prono|predice|haz un pron[oó]stico)\s*[:\-]?\s*", "", text, flags=re.I)

    parts = SPLIT_RE.split(text)
    if len(parts) >= 2:
        h = parts[0].strip(" .,:;-").lower()
        a = parts[1].strip(" .,:;-").lower()
        if h and a:
            return h, a, season
    return None


class ChatAgent:

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # -------------------------------------------------------------------------
    # 🧠 LLM: interpreta resultados SQL
    # -------------------------------------------------------------------------
    def _sql_to_llm(self, question: str, resumen: str, tabla: str) -> str:
        if not self.client:
            return resumen  # fallback

        prompt = f"""
Eres un analista de datos de LaLiga. Responde en 3 frases, claras y útiles.

Pregunta del usuario:
{question}

Resumen numérico:
{resumen}

Tabla de datos:
{tabla}

Escribe la interpretación humana, sin inventarte datos que no estén ahí.
"""

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Analista de datos experto en LaLiga, conciso y fiel a los datos."},
                    {"role": "user", "content": prompt},
                ]
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return resumen

    # -------------------------------------------------------------------------
    # CHAT PRINCIPAL
    # -------------------------------------------------------------------------
    def chat_query(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()

        # 1) PRONÓSTICO (Poisson)
        parsed = _parse_match(q)
        if parsed:
            home_raw, away_raw, season = parsed
            try:
                pred = predict_match_poisson(home_raw, away_raw, season)

                msg = (
                    f"Pronóstico (Poisson) {pred['temporada']}: "
                    f"{pred['home']} vs {pred['away']} → **{pred['pred_score']}**\n"
                    f"Prob. local: {pred['p_home']:.0%} | "
                    f"Empate: {pred['p_draw']:.0%} | "
                    f"Visitante: {pred['p_away']:.0%}"
                )

                # ✨ Contexto táctico opcional
                if self.client:
                    ctx = self._ctx_llm_pred(pred)
                    if ctx:
                        msg += "\n\n" + ctx

                return {"mode": "poisson", "respuesta": msg}

            except Exception as e:
                return {"mode": "poisson", "respuesta": f"No pude calcular el pronóstico: {e}"}

        # 2) CONSULTAS SQL
        try:
            res = ask_rag(q)
            if not res.get("ok"):
                raise RuntimeError(res.get("error"))

            rows = res.get("resultados", [])
            cols = res.get("columnas", [])
            tabla = _md_table(pd.DataFrame(rows, columns=cols))
            resumen = res.get("resumen", "")

            # 🧠 Generar interpretación con LLM
            interpretacion = self._sql_to_llm(q, resumen, tabla)

            return {
                "mode": "sql",
                "respuesta": interpretacion,
                "tabla": tabla,
            }

        except Exception as e:
            # 3) TODO LO DEMÁS → LLM normal
            return self._fallback_llm(q, e)

    # -------------------------------------------------------------------------
    # FALLBACK LLM
    # -------------------------------------------------------------------------
    def _fallback_llm(self, pregunta: str, error: Exception) -> Dict[str, Any]:
        if not self.client:
            return {"mode": "llm", "respuesta": f"(LLM no configurado) No pude procesar la consulta. {error}"}

        prompt = f"""
El usuario hizo esta pregunta: {pregunta}
Hubo un error del motor SQL ({error}), así que respóndele tú como modelo LLM.
Da una respuesta clara y útil.
"""

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Asistente experto en LaLiga."},
                    {"role": "user", "content": prompt},
                ]
            )
            return {"mode": "llm", "respuesta": r.choices[0].message.content.strip()}
        except Exception:
            return {"mode": "llm", "respuesta": "No pude generar respuesta con LLM."}

    # -------------------------------------------------------------------------
    # CONTEXTO EXTRA PARA PRONÓSTICO
    # -------------------------------------------------------------------------
    def _ctx_llm_pred(self, pred: Dict[str, Any]) -> str:
        if not self.client:
            return ""

        prompt = f"""
Escribe 2 frases analíticas sobre el partido:
{pred['home']} vs {pred['away']} ({pred['temporada']}).
Di algo sobre la dinámica reciente y estilo de juego SIN inventar lesiones ni fichajes futuros.
"""

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Analista de fútbol profesional."},
                    {"role": "user", "content": prompt},
                ]
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return ""
