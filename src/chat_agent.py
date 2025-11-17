# --- chat_agent.py --------------------------------
from __future__ import annotations

import os
import re
import pandas as pd
from typing import Any, Dict, Optional

# Web RAG: ESPN (clasificación) + Wikipedia (títulos)
from web_rag import fetch_standings_espn, fetch_laliga_titles_wikipedia

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
    return None


class ChatAgent:
    def __init__(self) -> None:
        # Cliente OpenAI opcional
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key)

        # Modelo por defecto (puedes cambiarlo en .env con OPENAI_MODEL)
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ------------------------ Entrada principal -----------------------------
    def chat_query(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()
        q_low = q.lower()
        season = _extract_season(q_low)

        # 1️⃣ RESUMEN / ANÁLISIS DE LA CLASIFICACIÓN
        if any(k in q_low for k in ["resumen", "analiza", "cómo va la liga", "como va la liga"]):
            try:
                df, meta = fetch_standings_espn()
                txt = self._summarize_standings_with_llm(
                    df,
                    season,
                    meta.get("source", "espn")
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

        # 2️⃣ CLASIFICACIÓN DIRECTA (RAG web → ESPN)
        if any(k in q_low for k in ["clasificación", "clasificacion", "tabla", "posiciones", "standings"]):
            try:
                # El fetcher trae SIEMPRE la clasificación actual (o por temporada si lo soporta)
                df, meta = fetch_standings_espn()
                return {
                    "mode": "web",
                    "respuesta": f"Clasificación LaLiga {season or '(actual)'} (fuente: {meta.get('source', 'espn')})",
                    "tabla": _md_table(df),
                    "meta": meta,
                }
            except Exception as e:
                return self._fallback_llm(
                    system="Eres un asistente de fútbol. Si no tienes datos en vivo, explica cómo consultarlos (web oficial, ESPN, etc.).",
                    user=f"No pude obtener la clasificación {season or 'actual'} de LaLiga. Explica de forma útil cómo conseguirla y qué significan las columnas. ({e})",
                )

        # 3️⃣ TÍTULOS / PALMARÉS (RAG web → Wikipedia)
        if any(
            k in q_low
            for k in ["títulos", "titulos", "palmarés", "palmares",
                      "más ligas", "mas ligas", "ligas ganadas", "campeonatos"]
        ):
            try:
                titles_df, cite_url = fetch_laliga_titles_wikipedia()

                # Determina la columna de títulos con o sin acento
                title_col = None
                if "Títulos" in titles_df.columns:
                    title_col = "Títulos"
                elif "Titulos" in titles_df.columns:
                    title_col = "Titulos"
                else:
                    title_col = next(
                        (c for c in titles_df.columns if "titul" in c.lower()),
                        None
                    )

                resumen = ""
                ask_rm = ("real madrid" in q_low) or (" madrid" in q_low)
                ask_fcb = ("barcelona" in q_low) or ("fc barcelona" in q_low)

                if title_col and (ask_rm or ask_fcb):
                    t = titles_df.copy()
                    # Normaliza nombre del club
                    club_col = next(
                        (c for c in t.columns if c.lower() in ("club", "equipo", "team")),
                        None
                    )
                    if club_col:
                        t["club_lc"] = t[club_col].astype(str).str.lower()
                        rm_val = t.loc[t["club_lc"].str.contains("madrid", na=False), title_col].max()
                        fcb_val = t.loc[t["club_lc"].str.contains("barcelona", na=False), title_col].max()

                        try:
                            rm = int(rm_val)
                        except Exception:
                            rm = int(pd.to_numeric(pd.Series([rm_val]), errors="coerce").fillna(0).iloc[0])

                        try:
                            fcb = int(fcb_val)
                        except Exception:
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

        # 4️⃣ Small talk / resto → LLM (si disponible)
        return self._fallback_llm(
            system="Eres un asistente de fútbol: breve, claro y útil.",
            user=q or "Hola",
        )

    # ------------------- Resumen con LLM ------------------------------------
    def _summarize_standings_with_llm(self, df: pd.DataFrame, season: Optional[str], source: str) -> str:
        if df is None or df.empty:
            return "(No hay datos disponibles para resumir.)"

        if not self.client:
            # Por si no hay clave de API configurada
            return "(LLM no configurado para generar el resumen, pero la tabla se ha obtenido correctamente.)"

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


# Nota: Nada de prints ni llamadas a fetch_* aquí arriba. Si quieres probar manualmente,
# hazlo en un bloque CLI aparte:
# if __name__ == "__main__":
#     ag = ChatAgent()
#     print(ag.chat_query("Dame la clasificación actual de LaLiga 2025/2026"))
#     print(ag.chat_query("¿Quién tiene más ligas, el Real Madrid o el Barcelona?"))