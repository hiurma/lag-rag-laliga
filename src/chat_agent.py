# --- chat_agent.py --------------------------------
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import pandas as pd

# Web RAG: ESPN (clasificación actual) + Wikipedia (títulos)
from web_rag import fetch_standings_espn, fetch_laliga_titles_wikipedia

# RAG SQL sobre tu base local (incluye modo "pronostico")
from rag_sql import ask_rag

# (Opcional) LLM para redacción natural / fallback
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ------------------------ Utils -----------------------------

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


# -----------------------------------------------------------
# ChatAgent
# -----------------------------------------------------------

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

        # ---------------- 0) Small talk / nada de fútbol --------------------
        futbol_keywords = [
            "liga", "laliga", "clasific", "goleador", "goles", "pichichi",
            "fichaje", "traspaso", "valor", "club", "partido", "resultado",
            "pronostic", "probabilidad", "apuesta", "marcador", "jornada",
            "título", "titulos", "títulos", "palmarés", "palmares"
        ]
        if not any(k in q_low for k in futbol_keywords):
            # No parece una pregunta de fútbol → LLM directo
            return self._fallback_llm(
                system="Eres un asistente general pero con especialización en fútbol.",
                user=q or "Hola",
            )

        # ---------------- 1) CLASIFICACIÓN ACTUAL (ESPN) -------------------
        # Si preguntan por "clasificación" sin temporada concreta o usando
        # palabras tipo "actual", "ahora", "esta temporada" → web (ESPN)
        if "clasific" in q_low:
            pide_actual = (
                ("actual" in q_low)
                or ("ahora" in q_low)
                or ("esta temporada" in q_low)
                or (season is None)  # sin año específico → asumimos actual
            )

            if pide_actual:
                try:
                    df, meta = fetch_standings_espn()
                    return {
                        "mode": "web",
                        "respuesta": f"Clasificación LaLiga (temporada actual, fuente: {meta.get('source', 'espn')})",
                        "tabla": _md_table(df),
                        "meta": meta,
                    }
                except Exception as e:
                    # Si falla el scraping usamos LLM para explicar
                    return self._fallback_llm(
                        system="Eres un asistente de fútbol. Si no tienes datos en vivo, explica cómo consultarlos (web oficial, ESPN, etc.).",
                        user=f"No pude obtener la clasificación actual de LaLiga. Explica de forma útil cómo conseguirla y qué significan las columnas. ({e})",
                    )
            else:
                # Pregunta de clasificación con temporada específica → usar RAG SQL
                rag = ask_rag(q)
                return self._format_rag_response(rag)

        # ---------------- 2) TÍTULOS / PALMARÉS (Wikipedia) -----------------
        if any(
            k in q_low
            for k in [
                "títulos", "titulos", "palmarés", "palmares",
                "más ligas", "mas ligas", "ligas ganadas", "campeonatos"
            ]
        ):
            try:
                titles_df, cite_url = fetch_laliga_titles_wikipedia()

                # Determina la columna de títulos con o sin acento
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

                        def _to_int(v):
                            try:
                                return int(v)
                            except Exception:
                                return int(pd.to_numeric(pd.Series([v]), errors="coerce").fillna(0).iloc[0])

                        rm = _to_int(rm_val)
                        fcb = _to_int(fcb_val)

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

        # ---------------- 3) TODO LO DEMÁS → RAG SQL -----------------------
        # Aquí entran:
        #   - goleadores
        #   - valor clubes
        #   - fichajes
        #   - resultados
        #   - y PRONÓSTICOS históricos (modo 'pronostico')
        rag = ask_rag(q)
        return self._format_rag_response(rag)

    # ------------------- Formatear respuesta RAG ---------------------------
    def _format_rag_response(self, rag: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convierte la salida de ask_rag en algo homogéneo para el frontend:
          - modo 'pronostico' → texto con probabilidades
          - modo 'sql'        → resumen + tabla markdown
          - error             → mensaje sencillo
        """
        if not isinstance(rag, dict):
            return {
                "mode": "rag_error",
                "respuesta": "Error inesperado en el módulo RAG.",
            }

        if not rag.get("ok", False):
            return {
                "mode": "rag_error",
                "respuesta": rag.get("error", "No se pudieron obtener datos."),
                "meta": rag,
            }

        modo = rag.get("modo")

        # ---- PRONÓSTICO histórico 1-X-2 (sin cuotas) ----
        if modo == "pronostico":
            partido = rag.get("partido", {}) or {}
            probs = rag.get("probabilidades", {}) or {}
            resumen = rag.get("resumen", "")

            local = partido.get("local", "Local")
            visit = partido.get("visitante", "Visitante")
            temporada = partido.get("temporada", "histórico")

            p_local = probs.get("p_local", 0.0)
            p_empate = probs.get("p_empate", 0.0)
            p_visit = probs.get("p_visitante", 0.0)

            texto = (
                f"**Pronóstico histórico para {local} vs {visit} ({temporada})**\n"
                f"{resumen}\n\n"
                f"Probabilidades aproximadas (basadas solo en tu histórico local):\n"
                f"- 1 (gana {local}): {p_local:.1%}\n"
                f"- X (empate): {p_empate:.1%}\n"
                f"- 2 (gana {visit}): {p_visit:.1%}"
            )

            return {
                "mode": "pronostico",
                "respuesta": texto,
                "meta": rag,
            }

        # ---- Consultas SQL (top goleadores, valor, fichajes, tabla, etc.) ----
        if modo == "sql" or modo is None:
            tabla_md = rag.get("tabla_md", "")
            resumen = rag.get("resumen", "Consulta ejecutada sobre tu base de datos local.")
            return {
                "mode": "sql",
                "respuesta": resumen,
                "tabla": tabla_md,
                "meta": {
                    "consulta": rag.get("consulta"),
                    "parametros": rag.get("parametros"),
                    "descripcion": rag.get("descripcion"),
                },
            }

        # ---- fallback raro ----
        return {
            "mode": "rag",
            "respuesta": rag.get("resumen", "Consulta realizada."),
            "meta": rag,
        }

    # ------------------- Resumen standings con LLM -------------------------
    def _summarize_standings_with_llm(
        self,
        df: pd.DataFrame,
        season: Optional[str],
        source: str,
    ) -> str:
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