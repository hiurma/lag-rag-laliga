from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import pandas as pd

from rag_sql import ask_rag
from poisson_model import predict_match_poisson

# Web RAG (ESPN / Wikipedia) – si fallan, hacemos fallback a LLM o SQL
try:
    from web_rag import fetch_standings_espn, fetch_laliga_titles_wikipedia
except Exception:  # si no existe el módulo en Render no pasa nada
    def fetch_standings_espn():
        raise RuntimeError("web_rag.fetch_standings_espn no disponible")

    def fetch_laliga_titles_wikipedia():
        raise RuntimeError("web_rag.fetch_laliga_titles_wikipedia no disponible")


# LLM opcional
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "(sin filas)"
    try:
        # to_markdown necesita tabulate, por si acaso hacemos fallback
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return "\n" + df.head(max_rows).to_string(index=False) + "\n"


def _extract_season(text: str) -> Optional[str]:
    m = re.search(r"(20\d{2})\s*[-/]\s*(20\d{2})", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


# ---------------------------------------------------------------------
# ChatAgent
# ---------------------------------------------------------------------
class ChatAgent:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        self.client = OpenAI(api_key=api_key) if (OpenAI and api_key) else None
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # --------------------------------------------------------------
    def chat_query(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()
        q_low = q.lower()
        season = _extract_season(q_low)

        # 1) Pronósticos de partidos (Poisson)
        if any(k in q_low for k in ["pronostic", "pronóstico", "predic", "resultado del partido"]):
            return self._handle_prediction(q)

        # 2) Resumen / “cómo va la liga”
        if any(k in q_low for k in ["resumen", "analiza", "cómo va la liga", "como va la liga"]):
            return self._handle_clasificacion_resumen(q, season)

        # 3) Clasificación directa
        if any(k in q_low for k in ["clasific", "tabla", "posiciones", "standings"]):
            return self._handle_clasificacion_tabla(q, season)

        # 4) Palmarés / títulos históricos
        if any(
            k in q_low
            for k in ["títulos", "titulos", "palmarés", "palmares", "ligas ganadas", "mas ligas", "más ligas"]
        ):
            return self._handle_titulos(q)

        # 5) Preguntas sobre datos históricos (pichichi, fichajes, valor, etc.) → RAG SQL
        if any(k in q_low for k in ["pichichi", "goleador", "fichaje", "traspaso", "transfer", "valor"]):
            return self._handle_sql_rag(q)

        # 6) Resto → LLM directo
        return self._fallback_llm(
            system="Eres un asistente experto en fútbol español (LaLiga). Responde breve, claro y útil.",
            user=q or "Hola",
        )

    # --------------------------------------------------------------
    # Handlers
    # --------------------------------------------------------------
    def _handle_clasificacion_resumen(self, q: str, season: Optional[str]) -> Dict[str, Any]:
        try:
            df, meta = fetch_standings_espn()
            txt = self._summarize_standings_with_llm(
                df,
                season,
                meta.get("source", "espn"),
            )
            return {
                "mode": "web",
                "respuesta": txt,
                "tabla": _md_table(df),
                "meta": meta,
            }
        except Exception as e:
            # Fallback: explicación genérica
            return self._fallback_llm(
                system="Eres un periodista deportivo experto en LaLiga.",
                user=(
                    f"No pude obtener la clasificación en vivo para hacer el resumen ({e}). "
                    "Explica al usuario cómo consultar la tabla actual y cómo se interpreta "
                    "la clasificación (puestos europeos, descenso, etc.)."
                ),
            )

    def _handle_clasificacion_tabla(self, q: str, season: Optional[str]) -> Dict[str, Any]:
        # Primero intentamos ESPN (web)
        try:
            df, meta = fetch_standings_espn()
            return {
                "mode": "web",
                "respuesta": f"Clasificación LaLiga {season or '(actual)'} (fuente: {meta.get('source', 'espn')}).",
                "tabla": _md_table(df),
                "meta": meta,
            }
        except Exception:
            # Si falla, tiramos de base local (rag_sql)
            rag = ask_rag(q)
            if not rag.get("ok"):
                return self._fallback_llm(
                    system="Eres un asistente de fútbol.",
                    user=(
                        "No he podido obtener la clasificación ni en web ni en la base local. "
                        "Explica al usuario cómo puede consultarla en internet de forma fiable."
                    ),
                )

            df = pd.DataFrame(rag["resultados"], columns=rag["columnas"])
            return {
                "mode": "sql",
                "respuesta": rag["resumen"],
                "tabla": _md_table(df),
                "meta": {"fuente": "sqlite"},
            }

    def _handle_titulos(self, q: str) -> Dict[str, Any]:
        try:
            titles_df, cite_url = fetch_laliga_titles_wikipedia()

            title_col = None
            if "Títulos" in titles_df.columns:
                title_col = "Títulos"
            elif "Titulos" in titles_df.columns:
                title_col = "Titulos"
            else:
                title_col = next(
                    (c for c in titles_df.columns if "titul" in c.lower()),
                    None,
                )

            resumen_extra = ""
            q_low = q.lower()
            ask_rm = ("real madrid" in q_low) or (" madrid" in q_low)
            ask_fcb = ("barcelona" in q_low) or ("fc barcelona" in q_low)

            if title_col:
                club_col = next(
                    (c for c in titles_df.columns if c.lower() in ("club", "equipo", "team")),
                    None,
                )
                if club_col:
                    t = titles_df.copy()
                    t["club_lc"] = t[club_col].astype(str).str.lower()

                    def _get_titles(mask):
                        vals = t.loc[mask, title_col]
                        if vals.empty:
                            return 0
                        try:
                            return int(vals.max())
                        except Exception:
                            return int(pd.to_numeric(vals, errors="coerce").fillna(0).max())

                    if ask_rm or ask_fcb:
                        rm = _get_titles(t["club_lc"].str.contains("madrid", na=False))
                        fcb = _get_titles(t["club_lc"].str.contains("barcelona", na=False))
                        ganador = "Real Madrid" if rm >= fcb else "FC Barcelona"
                        resumen_extra = f"\n\nComparativa → Real Madrid: {rm} ligas | FC Barcelona: {fcb} ligas. Más títulos: {ganador}."

            return {
                "mode": "web",
                "respuesta": (
                    "Títulos históricos de LaLiga por club (fuente: Wikipedia).\n"
                    f"Fuente original: {cite_url}{resumen_extra}"
                ),
                "tabla": _md_table(titles_df, 30),
                "meta": {"source": "wikipedia", "url": cite_url},
            }
        except Exception:
            return self._fallback_llm(
                system="Eres un asistente de fútbol.",
                user=(
                    "No pude extraer la tabla de títulos por club en Wikipedia. "
                    "Resume qué clubes tienen más ligas y cómo comprobarlo con enlaces fiables."
                ),
            )

    def _handle_sql_rag(self, q: str) -> Dict[str, Any]:
        rag = ask_rag(q)
        if not rag.get("ok"):
            return {
                "mode": "sql",
                "respuesta": f"No pude responder con la base local: {rag.get('error')}",
            }
        df = pd.DataFrame(rag["resultados"], columns=rag["columnas"])
        return {
            "mode": "sql",
            "respuesta": rag["resumen"],
            "tabla": _md_table(df),
            "meta": {"fuente": "sqlite"},
        }

    def _handle_prediction(self, q: str) -> Dict[str, Any]:
        q_low = q.lower()

        # Muy simple: buscamos "equipo1 vs equipo2"
        m = re.search(r"(.+?)\s+vs\s+(.+)", q_low)
        if not m:
            return self._fallback_llm(
                system="Eres un analista de fútbol.",
                user=(
                    "El usuario quiere un pronóstico de partido pero no detecto bien los equipos. "
                    "Explícale que use formato 'Pronóstico Real Madrid vs FC Barcelona 2025/2026'. "
                    f"Pregunta original: {q}"
                ),
            )

        home_raw = m.group(1).strip()
        away_raw = m.group(2).strip()

        try:
            pred = predict_match_poisson(home_raw, away_raw)
        except Exception as e:
            # Si no se encuentran equipos en la BD o cualquier otra cosa
            return self._fallback_llm(
                system="Eres un analista de fútbol. No inventes datos históricos concretos.",
                user=(
                    f"No he podido calcular un pronóstico Poisson para '{q}'. "
                    f"El error técnico fue: {e}. Da un análisis cualitativo del partido, "
                    "hablando de estilos de juego, factores clave, etc., pero sin fingir que "
                    "has usado estadísticas exactas de goles."
                ),
            )

        score = pred["marcador_mas_probable"]
        probs = pred["probabilidades_1X2"]
        equipos = pred["equipos_resueltos"]

        texto = (
            f"Pronóstico Poisson (usando tu base de datos histórica):\n\n"
            f"- Partido: {equipos['local']} vs {equipos['visitante']}\n"
            f"- Marcador más probable: {score['local']}–{score['visitante']}\n"
            f"- Prob. victoria local:  {probs['local']*100:0.1f}%\n"
            f"- Prob. empate:          {probs['empate']*100:0.1f}%\n"
            f"- Prob. victoria visitante: {probs['visitante']*100:0.1f}%\n\n"
            f"Goles esperados (xG aproximados): "
            f"{pred['xg_local']:0.2f} para el local, {pred['xg_visitante']:0.2f} para el visitante."
        )

        return {
            "mode": "poisson",
            "respuesta": texto,
            "meta": pred,
        }

    # --------------------------------------------------------------
    # LLM helpers
    # --------------------------------------------------------------
    def _summarize_standings_with_llm(
        self, df: pd.DataFrame, season: Optional[str], source: str
    ) -> str:
        if df is None or df.empty:
            return "(No hay datos disponibles para resumir.)"

        if not self.client:
            return "(LLM no configurado para generar el resumen, pero la tabla se ha obtenido correctamente.)"

        try:
            tabla_txt = df.head(10).to_string(index=False)
            prompt = (
                f"Resume brevemente la clasificación de LaLiga {season or 'actual'}, "
                f"destacando líder, puestos europeos y descenso. Fuente: {source}.\n\n"
                f"Tabla (primeros 10):\n{tabla_txt}"
            )
            comp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "Eres un periodista deportivo experto en LaLiga."},
                    {"role": "user", "content": prompt},
                ],
            )
            return (comp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"(Error al generar resumen: {e})"

    def _fallback_llm(self, system: str, user: str) -> Dict[str, Any]:
        if not self.client:
            return {"mode": "llm", "respuesta": user}

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