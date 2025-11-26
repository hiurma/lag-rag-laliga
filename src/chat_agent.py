# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import pandas as pd

from rag_sql import ask_rag
from poisson_model import predict_match_poisson

# LLM opcional
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# --------------------------- Helpers básicos ---------------------------------

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
    Acepta: "pronostico real madrid vs FC Barcelona 2025/2026"
            "predice madrid v barça"
            "pronóstico Girona contra Sevilla 2025-2026"
    """
    text = q.strip()
    if not PRONO_HINT.search(text):
        return None

    season = _extract_season(text)
    if season:
        # Quitamos la temporada del texto para que no moleste al parseo de equipos
        text = SEASON_RE.sub("", text)

    # Quitamos prefijos típicos
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


# ------------------------------- Agente --------------------------------------


class ChatAgent:
    def __init__(self) -> None:
        # Cliente OpenAI opcional
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key)

        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Entrada principal
    def chat_query(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()

        # 1) Intent de PRONÓSTICO (Poisson)
        parsed = _parse_match(q)
        if parsed:
            home_raw, away_raw, season = parsed
            try:
                pred = predict_match_poisson(home_raw, away_raw, season)

                # Ganador probable por probabilidad
                winner = "Empate"
                if pred["p_home"] >= pred["p_draw"] and pred["p_home"] >= pred["p_away"]:
                    winner = pred["home"]
                elif pred["p_away"] >= pred["p_draw"]:
                    winner = pred["away"]

                comentario = (
                    f"Probabilidades aproximadas — "
                    f"Victoria {pred['home']}: {pred['p_home']:.0%}, "
                    f"Empate: {pred['p_draw']:.0%}, "
                    f"Victoria {pred['away']}: {pred['p_away']:.0%}."
                )

                base_msg = (
                    f"Pronóstico (Poisson) {pred['temporada']}: "
                    f"{pred['home']} vs {pred['away']} → {pred['pred_score']}"
                )

                # Contexto táctico corto con LLM si está configurado
                extra = ""
                if self.client:
                    extra = self._ctx_llm(pred)
                full_text = base_msg + (("\n\n" + extra) if extra else "")

                return {
                    "mode": "poisson",
                    "respuesta": full_text,
                    "pronostico": {
                        "ganador_probable": winner,
                        "marcador_estimado": pred["pred_score"],
                        "comentario": comentario,
                    },
                }
            except Exception as e:
                return {
                    "mode": "poisson",
                    "respuesta": f"No pude calcular el pronóstico: {e}",
                }

        # 2) Resto de preguntas → BD local (RAG SQL)
        try:
            res = ask_rag(q)
            if not res.get("ok"):
                raise RuntimeError(res.get("error", "error"))

            rows = res.get("resultados", [])
            cols = res.get("columnas", [])
            tabla = _md_table(pd.DataFrame(rows, columns=cols))

            return {
                "mode": "sql",
                "respuesta": res.get("resumen")
                or res.get("descripcion")
                or "Consulta a BD local.",
                "tabla": tabla,
            }
        except Exception as e:
            # Si falla SQL y NO hay LLM configurado
            return {
                "mode": "llm",
                "respuesta": f"(LLM no configurado) Dime qué necesitas de LaLiga. [{e}]",
            }

    # ---------------------- Contexto táctico con LLM -------------------------
    def _ctx_llm(self, pred: Dict[str, Any]) -> str:
        try:
            prompt = (
                "Escribe 2 frases de contexto táctico (sin repetir los números de probabilidad) "
                "sobre este partido:\n"
                f"{pred['home']} vs {pred['away']} en {pred['temporada']}. "
                "Menciona estado reciente e importancia del rival, sin inventar lesiones."
            )
            comp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "Analista de fútbol, breve y objetivo."},
                    {"role": "user", "content": prompt},
                ],
            )
            return (comp.choices[0].message.content or "").strip()
        except Exception:
            return ""