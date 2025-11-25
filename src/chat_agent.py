# --- chat_agent.py --------------------------------
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import pandas as pd

# Web RAG: ESPN (clasificación) + Wikipedia (títulos)
try:
    from web_rag import fetch_standings_espn, fetch_laliga_titles_wikipedia
except Exception:
    fetch_standings_espn = None
    fetch_laliga_titles_wikipedia = None

# RAG SQL: tu base local (goleadores, fichajes, etc. + contexto para pronósticos)
try:
    from rag_sql import ask_rag
except Exception:
    ask_rag = None

# (Opcional) LLM para redacción natural / fallback
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ------------------- Helpers generales -----------------------------

def _table_text(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Devuelve la tabla en texto plano (sin depender de 'tabulate'),
    perfecta para meter dentro de <pre> en el frontend.
    """
    if df is None or df.empty:
        return "(sin filas)"
    try:
        return "\n" + df.head(max_rows).to_string(index=False) + "\n"
    except Exception:
        # fallback ultra-seguro
        return "\n" + "\n".join(str(r) for _, r in df.head(max_rows).iterrows()) + "\n"


def _extract_season(text: str) -> Optional[str]:
    # 2024/2025 ó 2024-2025
    m = re.search(r"(20\d{2})\s*[-/]\s*(20\d{2})", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


# ------------------- Clase principal -------------------------------

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

        # 0️⃣ INTENCIÓN: PRONÓSTICO DE PARTIDO
        if ("pronostic" in q_low or "resultado probable" in q_low) and (
            " vs " in q_low or " contra " in q_low
        ):
            return self._handle_prediction(q)

        # 1️⃣ RESUMEN / ANÁLISIS DE LA CLASIFICACIÓN (usando web RAG + LLM)
        if fetch_standings_espn and any(
            k in q_low for k in ["resumen", "analiza", "cómo va la liga", "como va la liga"]
        ):
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
                    "tabla": _table_text(df),
                    "meta": meta,
                }
            except Exception as e:
                return self._fallback_llm(
                    system="Eres un periodista deportivo experto en LaLiga.",
                    user=f"No pude obtener la clasificación para generar el resumen: {e}",
                )

        # 2️⃣ CLASIFICACIÓN DIRECTA (tabla actual desde ESPN)
        if fetch_standings_espn and any(
            k in q_low for k in ["clasificación", "clasificacion", "tabla", "posiciones", "standings"]
        ):
            try:
                df, meta = fetch_standings_espn()
                return {
                    "mode": "web",
                    "respuesta": f"Clasificación LaLiga {season or '(actual)'} (fuente: {meta.get('source', 'espn')})",
                    "tabla": _table_text(df),
                    "meta": meta,
                }
            except Exception as e:
                return self._fallback_llm(
                    system="Eres un asistente de fútbol. Si no tienes datos en vivo, explica cómo consultarlos (web oficial, ESPN, etc.).",
                    user=f"No pude obtener la clasificación {season or 'actual'} de LaLiga. Explica de forma útil cómo conseguirla y qué significan las columnas. ({e})",
                )

        # 3️⃣ TÍTULOS / PALMARÉS (Wikipedia)
        if fetch_laliga_titles_wikipedia and any(
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
                    "tabla": _table_text(titles_df),
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

        # 4️⃣ PREGUNTAS SOBRE GOLEADORES / FICHAJES / VALOR (RAG SQL)
        if ask_rag and any(
            k in q_low
            for k in [
                "goleador", "pichichi", "maximo goleador", "máximo goleador",
                "fichaje", "traspaso", "transfer",
                "valor de", "club más caro", "club mas caro",
                "resultados", "marcador de la jornada"
            ]
        ):
            try:
                info = ask_rag(q)
                if not info.get("ok", False) or not info.get("resultados"):
                    return self._fallback_llm(
                        system="Eres un asistente de fútbol ligado a una base de datos histórica de LaLiga.",
                        user=f"No encontré datos claros para la consulta: {q}. Explica al usuario cómo podría obtenerlos.",
                    )

                cols = info.get("columnas", [])
                rows = info.get("resultados", [])
                df = pd.DataFrame(rows, columns=cols)
                tabla = _table_text(df)
                resumen = info.get("resumen", "(sin resumen generado)")

                return {
                    "mode": "sql",
                    "respuesta": resumen,
                    "tabla": tabla,
                    "meta": {
                        "descripcion": info.get("descripcion"),
                        "consulta": info.get("consulta"),
                        "parametros": info.get("parametros"),
                    },
                }
            except Exception as e:
                return self._fallback_llm(
                    system="Eres un asistente de LaLiga con conocimientos estadísticos.",
                    user=f"Ha fallado el módulo SQL RAG al responder a: {q}. Error técnico: {e}. Explica algo útil al usuario igualmente.",
                )

        # 5️⃣ Small talk / resto → LLM (si disponible)
        return self._fallback_llm(
            system="Eres un asistente de fútbol: breve, claro y útil.",
            user=q or "Hola",
        )

    # ------------------- Pronósticos ----------------------------------------

    def _handle_prediction(self, question: str) -> Dict[str, Any]:
        """
        Usa la base de datos (vía rag_sql) como contexto y el LLM para
        generar un pronóstico estilo casa de apuestas:
        - Marcador probable
        - Probabilidades (local / empate / visitante)
        - Explicación breve
        """
        ctx_md = ""
        ctx_txt = ""
        home = ""
        away = ""
        temporada = _extract_season(question.lower())

        if ask_rag:
            try:
                info = ask_rag(question)
                # En modo pronóstico, rag_sql devuelve intent == "prediction"
                if info.get("intent") == "prediction" and info.get("ok", False):
                    ctx_md = info.get("context_markdown", "") or ""
                    ctx_txt = info.get("context_text", "") or ""
                    home = info.get("home_team", "") or ""
                    away = info.get("away_team", "") or ""
                    if not temporada:
                        temporada = info.get("temporada")
            except Exception:
                # Si falla el RAG SQL, seguimos solo con el LLM
                pass

        # Si no tenemos LLM configurado devolvemos algo sencillo
        if not self.client:
            texto_basico = (
                "Pronóstico orientativo (sin modelo de IA activo): "
                "veo un partido igualado, con ligera ventaja para el equipo local."
            )
            return {
                "mode": "prediccion",
                "respuesta": texto_basico,
                "tabla": ctx_md or None,
            }

        # Prompt para el LLM
        system_msg = (
            "Eres un analista de apuestas deportivas especializado en LaLiga. "
            "Debes dar un pronóstico en formato muy estructurado, sin andarte por las ramas."
        )

        context_block = ""
        if ctx_txt or ctx_md:
            context_block = f"""
Contexto estadístico aproximado extraído de la base de datos histórica de LaLiga:

{ctx_txt}

Tabla resumen:
{ctx_md}
"""

        user_msg = f"""
Quiero un pronóstico del partido con este texto de usuario:

\"\"\"{question}\"\"\".

{context_block}

Instrucciones de formato (respóndelas SIEMPRE):
1. Empieza con una línea: **RESULTADO PROBABLE:** Real_Madrid 2 - 1 FC_Barcelona (por ejemplo).
2. Segunda línea: **PROBABILIDADES:** Local X% | Empate Y% | Visitante Z%.
   - Asegúrate de que X + Y + Z ≈ 100 (no hace falta que sea exacto al 1%).
3. Después, 3-5 frases breves explicando el porqué del pronóstico
   (momento de forma, nivel ofensivo/defensivo, historial, etc.).
4. NO menciones casas de apuestas reales ni des consejos de apostar dinero.
"""

        completion = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()

        return {
            "mode": "prediccion",
            "respuesta": text,
            "tabla": ctx_md or None,
            "meta": {
                "home": home,
                "away": away,
                "temporada": temporada,
            },
        }

    # ------------------- Resumen clasificación con LLM ----------------------

    def _summarize_standings_with_llm(
        self,
        df: pd.DataFrame,
        season: Optional[str],
        source: str
    ) -> str:
        if df is None or df.empty:
            return "(No hay datos disponibles para resumir.)"

        if not self.client:
            # Por si no hay clave de API configurada
            return "(LLM no configurado para generar el resumen, pero la tabla se ha obtenido correctamente.)"

        try:
            try:
                tabla_txt = df.head(10).to_string(index=False)
            except Exception:
                tabla_txt = str(df.head(10))

            prompt = f"""
Eres un periodista deportivo analítico. Resume brevemente la clasificación de LaLiga {season or 'actual'},
destacando quién lidera la tabla, qué equipos están en posiciones europeas y quiénes en descenso.
Clasificación (fuente: {source}):

{tabla_txt}

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
                "respuesta": "(LLM no configurado) " + (user or ""),
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