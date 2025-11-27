# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import pandas as pd

from rag_sql import ask_rag
from poisson_model import predict_match_poisson

# LLM opcional (OpenAI)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

SEASON_RE = re.compile(r"(20\d{2})\s*[-/]\s*(20\d{2})", re.I)


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


SPLIT_RE = re.compile(r"\s+(?:vs\.?|v|contra|frente\s+a)\s+", re.I)
PRONO_HINT = re.compile(r"\b(prono|pronó|pronós|pronost|predic|predi[cs])", re.I)


def _parse_match(q: str) -> Optional[tuple[str, str, Optional[str]]]:
    """
    Devuelve (home, away, season) o None si no detecta partido.
    Acepta:
      - "pronostico real madrid vs fc barcelona 2025/2026"
      - "predice madrid v barça"
      - "pronóstico Girona contra Sevilla 2025-2026"
    """
    text = q.strip()
    if not PRONO_HINT.search(text):
        return None

    season = _extract_season(text)
    if season:
        text = SEASON_RE.sub("", text)

    text = re.sub(
        r"^\s*(pron[oó]stico|prono|predice|predic[eir]?|haz un pron[oó]stico)\s*[:\-]?\s*",
        "",
        text,
        flags=re.I,
    )

    parts = SPLIT_RE.split(text)
    if len(parts) >= 2:
        home = parts[0].strip(" .,:;–-").lower()
        away = parts[1].strip(" .,:;–-").lower()
        if home and away:
            return home, away, season
    return None


class ChatAgent:
    """
    Router de alto nivel:
      - Pronósticos → Poisson (+ contexto LLM si hay API key)
      - Consultas de datos (clasificación, pichichi, fichajes, resultados) → SQL + LLM que interpreta la tabla
      - Small talk / “hola” → solo LLM
    """

    def __init__(self) -> None:
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key)
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def chat_query(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()
        q_low = q.lower()

        # 1) PRONÓSTICO (Poisson)
        parsed = _parse_match(q)
        if parsed:
            home_raw, away_raw, season = parsed
            try:
                pred = predict_match_poisson(home_raw, away_raw, season)
                msg = (
                    f"Pronóstico {pred['temporada']}: "
                    f"{pred['home']} vs {pred['away']} → "
                    f"**{pred['pred_score']}**\n"
                    f"Prob. local: {pred['p_home']:.0%}  |  "
                    f"Empate: {pred['p_draw']:.0%}  |  "
                    f"Visitante: {pred['p_away']:.0%}"
                )

                ctx = self._ctx_llm(pred) if self.client else ""
                if ctx:
                    msg += "\n\n" + ctx

                return {"mode": "poisson", "respuesta": msg}
            except Exception as e:
                return {
                    "mode": "poisson",
                    "respuesta": f"No pude calcular el pronóstico: {e}",
                }

        # 2) Palabras que indican que hay que tirar de BD (SQL)
        SQL_HINT_WORDS = [
            "clasific",
            "tabla",
            "puntos",
            "jornada",
            "resultado",
            "marcador",
            "pichichi",
            "goleador",
            "goles",
            "fichaje",
            "traspaso",
            "transfer",
            "valor",
            "clubes más caros",
            "clubes mas caros",
        ]
        if any(w in q_low for w in SQL_HINT_WORDS):
            try:
                return self._sql_with_llm(q)
            except Exception as e:
                return {
                    "mode": "llm",
                    "respuesta": f"No pude consultar la BD local: {e}",
                }

        # 3) Resto → LLM (small talk)
        return self._fallback_llm(
            system=(
                "Eres un asistente experto en LaLiga. "
                "Si el usuario saluda o pregunta algo general, "
                "respóndele de forma amable y breve."
            ),
            user=q or "hola",
        )

    # ------------------- SQL + LLM ------------------------------------
    def _sql_with_llm(self, question: str) -> Dict[str, Any]:
        """
        Ejecuta RAG SQL y deja que el LLM sea quien redacta la explicación
        PRINCIPAL usando la tabla como contexto.
        """
        res = ask_rag(question)
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "error en BD"))

        rows = res.get("resultados", [])
        cols = res.get("columnas", [])
        df = pd.DataFrame(rows, columns=cols)
        tabla = _md_table(df)
        base_resumen = (
            res.get("resumen")
            or res.get("descripcion")
            or "Consulta a BD local."
        )

        # Si no hay LLM o la tabla está vacía → devolvemos solo SQL
        if not self.client or df.empty:
            return {
                "mode": "sql",
                "respuesta": base_resumen,
                "tabla": tabla,
            }

        # Pedimos al LLM que explique la tabla en 3–4 frases
        explicacion = self._summarize_sql_llm(question, tabla)

        if not explicacion:
            # fallback: solo resumen SQL
            return {
                "mode": "sql",
                "respuesta": base_resumen,
                "tabla": tabla,
            }

        # Aquí la respuesta principal YA es del LLM
        respuesta_final = explicacion

        return {
            "mode": "sql",
            "respuesta": respuesta_final,
            "tabla": tabla,
        }

    def _summarize_sql_llm(self, question: str, tabla_md: str) -> str:
        """
        El LLM recibe la tabla en markdown y la pregunta original,
        y devuelve una explicación natural de 3–4 frases.
        """
        if not self.client:
            return ""
        try:
            prompt = (
                "El usuario ha hecho una pregunta sobre LaLiga.\n"
                f"Pregunta: {question}\n\n"
                "Esta es la tabla de resultados (markdown):\n"
                f"{tabla_md}\n\n"
                "Interpreta los datos en 3-4 frases EN ESPAÑOL, claras y concisas. "
                "Explica quién va por delante, qué jugador o club destaca, "
                "o qué patrón importante se ve. No repitas toda la tabla, "
                "solo los insights clave."
            )
            comp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un analista de datos de fútbol que resume tablas de forma clara y breve.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return (comp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    # ------------------- Fallback LLM general -------------------------
    def _fallback_llm(self, system: str, user: str) -> Dict[str, Any]:
        if not self.client:
            return {
                "mode": "llm",
                "respuesta": "(LLM no configurado) Dime qué necesitas de LaLiga.",
            }
        try:
            comp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = (comp.choices[0].message.content or "").strip()
            return {"mode": "llm", "respuesta": text}
        except Exception as e:
            return {"mode": "llm", "respuesta": f"(Error LLM: {e})"}

    # ------------------- Contexto táctico Poisson ---------------------
    def _ctx_llm(self, pred: Dict[str, Any]) -> str:
        if not self.client:
            return ""
        try:
            prompt = (
                "Escribe 2 frases de contexto táctico sobre este partido de LaLiga:\n"
                f"{pred['home']} vs {pred['away']} en la temporada {pred['temporada']}.\n"
                "Habla del momento de forma reciente y de la importancia del partido, "
                "sin inventar lesiones concretas."
            )
            comp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un analista táctico de fútbol, breve y objetivo.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return (comp.choices[0].message.content or "").strip()
        except Exception:
            return ""
