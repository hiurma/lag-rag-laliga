# --- chat_agent.py --------------------------------
from __future__ import annotations

import os
import re
import sqlite3
import pathlib
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List

# Web RAG: ESPN (clasificación) + Wikipedia (títulos)
try:
    from .web_rag import fetch_standings_espn, fetch_laliga_titles_wikipedia
except Exception:
    from web_rag import fetch_standings_espn, fetch_laliga_titles_wikipedia

# Motor de RAG SQL / pronósticos
try:
    from .rag_sql import ask_rag
except Exception:
    try:
        from rag_sql import ask_rag
    except Exception:
        ask_rag = None  # por si no está disponible

# (Opcional) LLM para redacción natural / fallback
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
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
    # caso "2025" -> "2025/2026"
    m2 = re.search(r"\b(20\d{2})\b", text)
    if m2:
        y1 = int(m2.group(1))
        return f"{y1}/{y1+1}"
    return None


def _db_path() -> str:
    return os.getenv("DB_PATH", "data/laliga.sqlite")


class ChatAgent:
    def __init__(self) -> None:
        # Cliente OpenAI opcional
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key)

        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ------------------------ Entrada principal -----------------------------
    def chat_query(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()
        q_low = q.lower()
        season = _extract_season(q_low)

        # 1) RESUMEN CLASIFICACIÓN
        if any(k in q_low for k in ["resumen", "analiza", "cómo va la liga", "como va la liga"]):
            try:
                df, meta = fetch_standings_espn()
                txt = self._summarize_standings_with_llm(df, season, meta.get("source", "espn"))
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

        # 2) CLASIFICACIÓN DIRECTA
        if any(k in q_low for k in ["clasificación", "clasificacion", "tabla", "posiciones", "standings"]):
            try:
                df, meta = fetch_standings_espn()
                return {
                    "mode": "web",
                    "respuesta": f"Clasificación LaLiga {season or '(actual)'} (fuente: {meta.get('source', 'espn')})",
                    "tabla": _md_table(df),
                    "meta": meta,
                }
            except Exception as e:
                return self._fallback_llm(
                    system="Eres un asistente de fútbol.",
                    user=f"No pude obtener la clasificación {season or 'actual'} de LaLiga. ({e})",
                )

        # 3) PICHICHI / GOLEADORES (SQLite local)
        if any(k in q_low for k in ["pichichi", "goleador", "goleadores", "máximo goleador", "maximo goleador", "top goleadores"]):
            try:
                df, meta_txt = self._get_top_scorers_from_db(season)
                if df is not None and not df.empty:
                    return {
                        "mode": "local_db",
                        "respuesta": meta_txt,
                        "tabla": _md_table(df, 30),
                        "meta": {"source": "sqlite", "db": _db_path()},
                    }
                return {
                    "mode": "local_db",
                    "respuesta": (
                        "No tengo datos de goleadores actualizados en tu base local. "
                        "Asegúrate de que la tabla 'goleadores' de data/laliga.sqlite "
                        "incluya la temporada que preguntas."
                    ),
                }
            except Exception as e:
                return {
                    "mode": "local_db",
                    "respuesta": f"No pude leer goleadores desde SQLite: {e}",
                }

        # 4) PRONÓSTICOS / PREDICCIONES (RAG SQL)
        if any(k in q_low for k in ["pronóstico", "pronostico", "predicción", "prediccion", "quién ganará", "quien ganara", "quien va a ganar", "quién va a ganar"]):
            if not ask_rag:
                return {
                    "mode": "rag_sql",
                    "respuesta": (
                        "El motor de pronósticos RAG SQL (ask_rag) no está disponible. "
                        "Comprueba que rag_sql.py existe y se importa correctamente."
                    ),
                }
            try:
                out = ask_rag(q)

                # si ask_rag ya devuelve un dict listo, lo respetamos
                if isinstance(out, dict):
                    # nos aseguramos de que haya al menos 'mode' y 'respuesta'
                    if "mode" not in out:
                        out["mode"] = out.get("mode", "rag_sql")
                    if "respuesta" not in out and "answer" in out:
                        out["respuesta"] = out["answer"]
                    return out

                # si es solo texto
                return {
                    "mode": "rag_sql",
                    "respuesta": str(out),
                }
            except Exception as e:
                return self._fallback_llm(
                    system="Eres un analista que habla de pronósticos de LaLiga con mucha prudencia.",
                    user=f"No he podido ejecutar el motor de pronósticos sobre la base de datos. Explica qué factores se usarían para predecir y por qué no puedes dar un valor exacto ahora. (Error: {e})",
                )

        # 5) TÍTULOS / PALMARÉS (Wikipedia)
        if any(
            k in q_low
            for k in ["títulos", "titulos", "palmarés", "palmares",
                      "más ligas", "mas ligas", "ligas ganadas", "campeonatos"]
        ):
            try:
                titles_df, cite_url = fetch_laliga_titles_wikipedia()

                title_col = None
                if "Títulos" in titles_df.columns:
                    title_col = "Títulos"
                elif "Titulos" in titles_df.columns:
                    title_col = "Titulos"
                else:
                    title_col = next((c for c in titles_df.columns if "titul" in c.lower()), None)

                resumen = ""
                ask_rm = ("real madrid" in q_low) or (" madrid" in q_low)
                ask_fcb = ("barcelona" in q_low) or ("fc barcelona" in q_low)

                if title_col and (ask_rm or ask_fcb):
                    t = titles_df.copy()
                    club_col = next((c for c in t.columns if c.lower() in ("club", "equipo", "team")), None)
                    if club_col:
                        t["club_lc"] = t[club_col].astype(str).str.lower()
                        rm_val = t.loc[t["club_lc"].str.contains("madrid", na=False), title_col].max()
                        fcb_val = t.loc[t["club_lc"].str.contains("barcelona", na=False), title_col].max()

                        rm = int(pd.to_numeric(pd.Series([rm_val]), errors="coerce").fillna(0).iloc[0])
                        fcb = int(pd.to_numeric(pd.Series([fcb_val]), errors="coerce").fillna(0).iloc[0])

                        ganador = "Real Madrid" if rm >= fcb else "FC Barcelona"
                        resumen = f"\n\nComparativa → Real Madrid: {rm} | FC Barcelona: {fcb}. Más títulos: {ganador}."

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

        # 6) Small talk / resto
        return self._fallback_llm(
            system="Eres un asistente de fútbol: breve, claro y útil.",
            user=q or "Hola",
        )

    # ------------------- GOLEADORES desde SQLite ----------------------------
    def _get_top_scorers_from_db(self, season: Optional[str]) -> Tuple[pd.DataFrame, str]:
        db = _db_path()
        if not pathlib.Path(db).exists():
            return pd.DataFrame(), f"No existe la base de datos en {db}"

        con = sqlite3.connect(db)

        if season:
            temporada = season
        else:
            temporada_df = pd.read_sql_query(
                "SELECT Temporada, MAX(Jornada) as Jmax "
                "FROM goleadores GROUP BY Temporada "
                "ORDER BY Temporada DESC LIMIT 1;",
                con,
            )
            temporada = temporada_df["Temporada"].iloc[0] if not temporada_df.empty else None

        if not temporada:
            con.close()
            return pd.DataFrame(), "No hay temporadas en la tabla goleadores."

        jmax_df = pd.read_sql_query(
            "SELECT MAX(Jornada) as Jmax FROM goleadores WHERE Temporada = ?;",
            con,
            params=[temporada],
        )
        jmax = int(jmax_df["Jmax"].iloc[0]) if not jmax_df.empty and jmax_df["Jmax"].iloc[0] is not None else None

        if jmax is not None:
            df = pd.read_sql_query(
                """
                SELECT Jugador, Club, Goles, Partidos, Minutos
                FROM goleadores
                WHERE Temporada = ? AND Jornada = ?
                ORDER BY Goles DESC, Partidos ASC, Minutos ASC
                LIMIT 30;
                """,
                con,
                params=[temporada, jmax],
            )
            meta_txt = f"Pichichi / Top goleadores de LaLiga {temporada} (jornada {jmax}) según tu base local."
        else:
            df = pd.read_sql_query(
                """
                SELECT Jugador, Club, Goles, Partidos, Minutos
                FROM goleadores
                WHERE Temporada = ?
                ORDER BY Goles DESC, Partidos ASC, Minutos ASC
                LIMIT 30;
                """,
                con,
                params=[temporada],
            )
            meta_txt = f"Pichichi / Top goleadores de LaLiga {temporada} según tu base local."

        con.close()
        return df, meta_txt

    # ------------------- Resumen con LLM ------------------------------------
    def _summarize_standings_with_llm(self, df: pd.DataFrame, season: Optional[str], source: str) -> str:
        if df is None or df.empty:
            return "(No hay datos disponibles para resumir.)"

        if not self.client:
            return "(LLM no configurado para generar resumen, pero la tabla se obtuvo correctamente.)"

        try:
            tabla_md = df.head(10).to_markdown(index=False)
            prompt = f"""
Eres un periodista deportivo analítico. Resume brevemente la clasificación de LaLiga {season or 'actual'},
destacando quién lidera la tabla, qué equipos están en posiciones europeas y quiénes en descenso.
Clasificación (fuente: {source}):

{tabla_md}

Da una respuesta de 3-4 frases, clara y concisa.
"""
            completion = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "Eres un periodista experto en fútbol español."},
                    {"role": "user", "content": prompt},
                ],
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as e:
            return f"(Error al generar resumen: {e})"

    # ------------------------ Helper LLM ------------------------------------
    def _fallback_llm(self, system: str, user: str) -> Dict[str, Any]:
        if not self.client:
            return {
                "mode": "llm",
                "respuesta": (
                    "No tengo datos suficientes para responder con precisión. "
                    "Prueba a preguntar por clasificación, títulos, pichichi o pronósticos "
                    "con datos que tenga en la base."
                ),
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