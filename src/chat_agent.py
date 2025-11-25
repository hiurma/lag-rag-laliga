# --- chat_agent.py --------------------------------
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import pandas as pd

# Web RAG: ESPN (clasificación) + Wikipedia (títulos)
from web_rag import fetch_standings_espn, fetch_laliga_titles_wikipedia

# RAG SQL: tu base de datos local (clasificaciones, goleadores, resultados, fichajes…)
from rag_sql import ask_rag as ask_rag_sql

# (Opcional) LLM para redacción natural / fallback
try:
    from openai import OpenAI
except Exception:  # si no está instalado en Render
    OpenAI = None


# -------------------- helpers -------------------------

def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Devuelve un preview en markdown."""
    if df is None or df.empty:
        return "(sin filas)"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return "\n" + df.head(max_rows).to_string(index=False) + "\n"


def _extract_season(text: str) -> Optional[str]:
    # 2024/2025 ó 2024-2025
    m = re.search(r"(20\d{2})\s*[-/]\s*(20\d{2})", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


def _is_pichichi_question(q_low: str) -> bool:
    return any(
        k in q_low
        for k in [
            "pichichi",
            "máximo goleador",
            "maximo goleador",
            "goleador actual",
        ]
    )


def _is_prediction_question(q_low: str) -> bool:
    return any(
        k in q_low
        for k in [
            "pronóstico",
            "pronostico",
            "quien gana",
            "quién gana",
            "quien ganará",
            "quién ganará",
            "quien ganara",
            "resultado del partido",
            "resultado del clasico",
            "marcador del partido",
        ]
    ) and " vs " in q_low  # para asegurarnos de que es un partido concreto


# -------------------- agente principal -------------------------

class ChatAgent:
    def __init__(self) -> None:
        # Cliente OpenAI opcional
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key)

        # Modelo por defecto
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ------------------------ Entrada principal -----------------------------
    def chat_query(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()
        q_low = q.lower()
        season = _extract_season(q_low)

        # 0️⃣ Intento PICHICHI / GOLEADORES (SQL)
        if _is_pichichi_question(q_low):
            sql_data = ask_rag_sql(q)
            if not sql_data.get("ok"):
                return {
                    "mode": "sql",
                    "respuesta": f"No pude recuperar los goleadores: {sql_data.get('error')}",
                }

            cols = sql_data.get("columnas", [])
            rows = sql_data.get("resultados", [])
            df = pd.DataFrame(rows, columns=cols)

            tabla_md = _md_table(df)
            desc = sql_data.get(
                "descripcion", "Pichichi / top goleadores según tu base local."
            )
            return {
                "mode": "sql",
                "respuesta": f"🏆 {desc}",
                "tabla": tabla_md,
                "meta": {
                    "tipo": "goleadores",
                    "sql": sql_data.get("consulta"),
                    "parametros": sql_data.get("parametros"),
                },
            }

        # 1️⃣ Intento PRONÓSTICO (usa SQL + LLM)
        if _is_prediction_question(q_low):
            # 1) saco contexto estadístico de tu SQL (resultados, clasificación, etc.)
            sql_data = ask_rag_sql(q)
            if not sql_data.get("ok"):
                # si la SQL falla, caigo a LLM “a pelo”
                return self._fallback_llm(
                    system=(
                        "Eres un analista de fútbol. Da un pronóstico prudente "
                        "del partido preguntado, sin usar cuotas ni probabilidades numéricas."
                    ),
                    user=q,
                )

            cols = sql_data.get("columnas", [])
            rows = sql_data.get("resultados", [])
            df = pd.DataFrame(rows, columns=cols)
            tabla_md = _md_table(df) if not df.empty else None
            resumen_stats = sql_data.get("resumen", "")

            # 2) construyo el pronóstico con LLM utilizando ese contexto
            if not self.client:
                # Sin LLM configurado: devuelvo el contexto y aviso
                texto = (
                    "No tengo LLM configurado en el servidor, pero estos son algunos datos "
                    "históricos relevantes de tu base local:\n"
                    f"{resumen_stats or '(sin datos)'}\n\n"
                    "Con ellos podrías estimar el favorito comparando goles, victorias y estado de ambos equipos."
                )
                return {
                    "mode": "sql_pred",
                    "respuesta": texto,
                    "tabla": tabla_md,
                    "meta": {
                        "tipo": "pronostico_sql_sin_llm",
                        "sql": sql_data.get("consulta"),
                        "parametros": sql_data.get("parametros"),
                    },
                }

            # Con LLM → pronóstico estilo casas de apuestas (pero sin cuotas numéricas)
            prompt = f"""
Eres un analista de fútbol y tipster. Debes dar UN pronóstico para el partido de LaLiga
que se describe en la pregunta del usuario.

Pregunta del usuario:
{q}

Te doy un pequeño resumen estadístico sacado de una base de datos histórica local:
{resumen_stats or '(no hay mucho dato, pronostica con prudencia)'}

Con eso, responde en 3-4 frases en español:
- Explica qué equipo ves ligeramente favorito y por qué (diferencia de goles, resultados previos, factor campo, etc.).
- Da un marcador probable (por ejemplo: "Mi pronóstico es victoria del Real Madrid 2-1").
- NO menciones cuotas, porcentajes ni probabilidades numéricas.
- Mantén un tono prudente, dejando claro que es solo una estimación y que el fútbol es impredecible.
"""

            try:
                comp = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Eres un analista de fútbol español que da pronósticos prudentes.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                texto = (comp.choices[0].message.content or "").strip()
            except Exception as e:
                texto = f"(Error al generar el pronóstico con el LLM: {e})"

            return {
                "mode": "sql_pred",
                "respuesta": texto,
                "tabla": tabla_md,
                "meta": {
                    "tipo": "pronostico_sql_llm",
                    "sql": sql_data.get("consulta"),
                    "parametros": sql_data.get("parametros"),
                },
            }

        # 2️⃣ RESUMEN / ANÁLISIS DE LA CLASIFICACIÓN (web ESPN + LLM)
        if any(k in q_low for k in ["resumen", "analiza", "cómo va la liga", "como va la liga"]):
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
                return self._fallback_llm(
                    system="Eres un periodista deportivo experto en LaLiga.",
                    user=f"No pude obtener la clasificación para generar el resumen: {e}",
                )

        # 3️⃣ CLASIFICACIÓN DIRECTA (web ESPN)
        if any(k in q_low for k in ["clasificación", "clasificacion", "tabla", "posiciones", "standings"]):
            try:
                df, meta = fetch_standings_espn()
                return {
                    "mode": "web",
                    "respuesta": f"Clasificación LaLiga {season or '(actual)'} (fuente: {meta.get('source', 'espn')}).",
                    "tabla": _md_table(df),
                    "meta": meta,
                }
            except Exception as e:
                return self._fallback_llm(
                    system=(
                        "Eres un asistente de fútbol. Si no tienes datos en vivo, "
                        "explica cómo consultarlos (web oficial, ESPN, etc.)."
                    ),
                    user=(
                        f"No pude obtener la clasificación {season or 'actual'} de LaLiga. "
                        f"Explica de forma útil cómo conseguirla y qué significan las columnas. ({e})"
                    ),
                )

        # 4️⃣ TÍTULOS / PALMARÉS (web Wikipedia)
        if any(
            k in q_low
            for k in [
                "títulos",
                "titulos",
                "palmarés",
                "palmares",
                "más ligas",
                "mas ligas",
                "ligas ganadas",
                "campeonatos",
            ]
        ):
            try:
                titles_df, cite_url = fetch_laliga_titles_wikipedia()

                # Determina la columna de títulos
                if "Títulos" in titles_df.columns:
                    title_col = "Títulos"
                elif "Titulos" in titles_df.columns:
                    title_col = "Titulos"
                else:
                    title_col = next(
                        (c for c in titles_df.columns if "titul" in c.lower()),
                        None,
                    )

                resumen = ""
                ask_rm = ("real madrid" in q_low) or (" madrid" in q_low)
                ask_fcb = ("barcelona" in q_low) or ("fc barcelona" in q_low)

                if title_col:
                    t = titles_df.copy()
                    club_col = next(
                        (c for c in t.columns if c.lower() in ("club", "equipo", "team")),
                        None,
                    )
                    if club_col:
                        t["club_lc"] = t[club_col].astype(str).str.lower()
                        if ask_rm or ask_fcb:
                            rm_val = t.loc[
                                t["club_lc"].str.contains("madrid", na=False), title_col
                            ].max()
                            fcb_val = t.loc[
                                t["club_lc"].str.contains("barcelona", na=False),
                                title_col,
                            ].max()

                            def _to_int(v):
                                try:
                                    return int(v)
                                except Exception:
                                    return int(
                                        pd.to_numeric(
                                            pd.Series([v]), errors="coerce"
                                        ).fillna(0).iloc[0]
                                    )

                            rm = _to_int(rm_val)
                            fcb = _to_int(fcb_val)
                            ganador = "Real Madrid" if rm >= fcb else "FC Barcelona"
                            resumen = (
                                f"\n\nComparativa → Real Madrid: {rm} | "
                                f"FC Barcelona: {fcb}. Más títulos: {ganador}."
                            )

                return {
                    "mode": "web",
                    "respuesta": (
                        "Títulos históricos de LaLiga por club (fuente: Wikipedia)\n"
                        f"Fuente: {cite_url}{resumen}"
                    ),
                    "tabla": _md_table(titles_df, 30),
                    "meta": {"source": "wikipedia", "url": cite_url},
                }
            except Exception:
                return self._fallback_llm(
                    system="Eres un asistente de fútbol que explica palmarés con contexto.",
                    user=(
                        "No pude extraer la tabla de títulos por club en Wikipedia. "
                        "Resume qué clubes tienen más ligas y cómo comprobarlo con enlaces fiables."
                    ),
                )

        # 5️⃣ Small talk / resto → LLM (si disponible)
        return self._fallback_llm(
            system="Eres un asistente de fútbol: breve, claro y útil.",
            user=q or "Hola",
        )

    # ------------------- Resumen clasificación con LLM -----------------------
    def _summarize_standings_with_llm(
        self, df: pd.DataFrame, season: Optional[str], source: str
    ) -> str:
        if df is None or df.empty:
            return "(No hay datos disponibles para resumir.)"

        if not self.client:
            return (
                "(LLM no configurado para generar el resumen, pero la tabla se ha "
                "obtenido correctamente.)"
            )

        try:
            tabla_md = df.head(10).to_markdown(index=False)
            prompt = f"""
Eres un periodista deportivo analítico. Resume brevemente la clasificación de LaLiga
{season or 'actual'}, destacando quién lidera la tabla, qué equipos están en posiciones
europeas y quiénes en descenso.

Clasificación (fuente: {source}):

{tabla_md}

Da una respuesta de 3-4 frases, clara y concisa.
"""
            completion = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un periodista experto en fútbol español.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as e:
            return f"(Error al generar resumen: {e})"

    # ------------------------ Helper LLM genérico ---------------------------
    def _fallback_llm(self, system: str, user: str) -> Dict[str, Any]:
        if not self.client:
            return {
                "mode": "llm",
                "respuesta": "(LLM no configurado) " + user,
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