# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import pandas as pd

# RAG SQL (BD local)
from rag_sql import ask_rag
# Modelo Poisson para pronósticos
from poisson_model import predict_match_poisson

# LLM opcional (OpenAI)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ----------------- Utilidades comunes ---------------------------------

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


# Split tolerante para detectar "EquipoA vs EquipoB"
SPLIT_RE = re.compile(r"\s+(?:vs\.?|v|contra|frente\s+a)\s+", re.I)

# Palabras que indican intención de pronóstico
PRONO_HINT = re.compile(r"\b(prono|pronó|pronós|pronost|predic|predi[cs])", re.I)


def _parse_match(q: str) -> Optional[tuple[str, str, Optional[str]]]:
    """
    Devuelve (home, away, season) o None si no detecta partido.

    Ejemplos aceptados:
      - "pronostico real madrid vs fc barcelona 2025/2026"
      - "predice madrid v barça"
      - "pronóstico Girona contra Sevilla 2025-2026"
    """
    text = q.strip()
    if not PRONO_HINT.search(text):
        return None

    season = _extract_season(text)
    if season:
        text = SEASON_RE.sub("", text)  # quitamos la temporada del texto

    # quitamos prefijos típicos (“pronóstico…”, “prono…”, etc.)
    text = re.sub(
        r"^\s*(pron[oó]stico|prono|predice|predic[eir]?|haz un pron[oó]stico)\s*[:\-]?\s*",
        "",
        text,
        flags=re.I,
    )

    # buscamos el “vs / contra / v”
    parts = SPLIT_RE.split(text)
    if len(parts) >= 2:
        home = parts[0].strip(" .,:;–-").lower()
        away = parts[1].strip(" .,:;–-").lower()
        if home and away:
            return home, away, season
    return None


# ----------------- Agente principal ------------------------------------


class ChatAgent:
    """
    Router de alto nivel:
      - Si detecta PRONÓSTICO → Poisson (+ LLM contexto táctico)
      - Si detecta palabras de BD → SQL (ask_rag) + LLM que interpreta la tabla
      - Si no, → LLM (small talk / preguntas generales tipo “hola”)
    """

    def __init__(self) -> None:
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key)
        # Modelo por defecto
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ------------------------------------------------------------------
    def chat_query(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()
        q_low = q.lower()

        # 1) Intent de PRONÓSTICO (Poisson)
        match = _parse_match(q)
        if match:
            home_raw, away_raw, season = match
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

                # Contexto táctico con LLM (si hay clave)
                ctx = self._ctx_llm(pred) if self.client else ""
                if ctx:
                    msg += "\n\n" + ctx

                # Por compatibilidad con tu front, podrías añadir aquí un campo "pronostico" si quieres:
                # pron_dict = {
                #     "ganador_probable": ...,
                #     "marcador_estimado": pred["pred_score"],
                #     "comentario": ctx,
                # }
                # return {"mode": "poisson", "respuesta": msg, "pronostico": pron_dict}

                return {"mode": "poisson", "respuesta": msg}

            except Exception as e:
                return {
                    "mode": "poisson",
                    "respuesta": f"No pude calcular el pronóstico: {e}",
                }

        # 2) Intent de BD (RAG SQL): clasificación, pichichi, fichajes, etc.
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
                # si falla la BD, pasamos a LLM explicando el problema
                return {
                    "mode": "llm",
                    "respuesta": f"No pude consultar la BD local: {e}",
                }

        # 3) Resto → LLM (small talk, preguntas generales tipo “hola”)
        return self._fallback_llm(
            system=(
                "Eres un asistente experto en LaLiga. "
                "Si el usuario hace una pregunta muy general como 'hola', "
                "respóndele de forma amable y breve."
            ),
            user=q or "hola",
        )

    # ------------------------------------------------------------------
    #  SQL + LLM: la BD responde, el LLM interpreta
    # ------------------------------------------------------------------
    def _sql_with_llm(self, question: str) -> Dict[str, Any]:
        """
        Ejecuta la consulta SQL y, si hay modelo LLM disponible,
        genera una explicación en lenguaje natural de la tabla.
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

        # Sin LLM → nos quedamos con el resumen base
        if not self.client or df.empty:
            return {
                "mode": "sql",
                "respuesta": base_resumen,
                "tabla": tabla,
            }

        # Con LLM → pedir una explicación breve usando la tabla
        extra = self._summarize_sql_llm(question, base_resumen, tabla)
        if extra:
            respuesta = base_resumen + "\n\n" + extra
        else:
            respuesta = base_resumen

        return {
            "mode": "sql",
            "respuesta": respuesta,
            "tabla": tabla,
        }

    def _summarize_sql_llm(
        self, question: str, resumen: str, tabla_md: str
    ) -> str:
        """
        Pide al LLM que explique de forma breve el resultado SQL.
        """
        if not self.client:
            return ""
        try:
            prompt = (
                "El usuario ha hecho una pregunta sobre LaLiga.\n"
                f"Pregunta: {question}\n\n"
                "Este es el resumen automático de la consulta SQL:\n"
                f"{resumen}\n\n"
                "Y esta es la tabla de resultados en formato markdown:\n"
                f"{tabla_md}\n\n"
                "Escribe en 3-4 frases, en español, una explicación clara y sintética "
                "de lo que muestran los datos (quién va primero, diferencias de puntos, "
                "si hay un pichichi destacado, si un fichaje es muy caro, etc.). "
                "No repitas literalmente toda la tabla, solo interpreta los datos más importantes."
            )
            comp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un analista de datos de fútbol. "
                                   "Explicas tablas de forma clara y breve.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return (comp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    #  Fallback LLM general
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    #  Contexto extra para el pronóstico Poisson
    # ------------------------------------------------------------------
    def _ctx_llm(self, pred: Dict[str, Any]) -> str:
        """Dos frases de contexto táctico para acompañar el Poisson."""
        if not self.client:
            return ""
        try:
            prompt = (
                "Escribe 2 frases de contexto táctico (sin repetir números) "
                "sobre este partido de LaLiga:\n"
                f"{pred['home']} vs {pred['away']} en la temporada {pred['temporada']}. "
                "Menciona estado reciente e importancia del partido, "
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
