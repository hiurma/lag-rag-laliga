# --- chat_agent.py -------------------------------------------------
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# --------- Web RAG: ESPN (clasificación) + Wikipedia (títulos) ----------
try:
    from web_rag import fetch_standings_espn, fetch_laliga_titles_wikipedia
except Exception:
    fetch_standings_espn = None
    fetch_laliga_titles_wikipedia = None

# --------- Motor SQL RAG (goleadores, valor, fichajes, etc.) -----------
try:
    from rag_sql import ask_rag
except Exception:
    ask_rag = None

# --------- Cliente OpenAI (opcional) -----------------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


DB_PATH = Path("data/laliga.sqlite")


# ====================== Helpers generales ===============================

def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "(sin filas)"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return "\n" + df.head(max_rows).to_string(index=False) + "\n"


def _extract_season(text: str) -> Optional[str]:
    """
    Detecta temporadas tipo '2024/2025' o '2024-2025' y normaliza a '2024/2025'.
    """
    m = re.search(r"(20\d{2})\s*[-/]\s*(20\d{2})", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


def _detect_match_teams(question: str) -> Optional[tuple[str, str]]:
    """
    Intenta extraer equipos de patrones tipo:
      - 'Real Madrid vs FC Barcelona'
      - 'Real Madrid contra FC Barcelona'
    Devuelve (local, visitante) o None.
    """
    q = question.strip()

    m = re.search(r"(.+?)\s+vs\.?\s+(.+)", q, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"(.+?)\s+contra\s+(.+)", q, flags=re.IGNORECASE)

    if not m:
        return None

    local = m.group(1).strip()
    visitante = m.group(2).strip()

    # Quitamos coletillas tipo "en el Bernabéu" del segundo equipo
    visitante = re.sub(r"\s+en\s+el\s+.+$", "", visitante, flags=re.IGNORECASE).strip()
    visitante = re.sub(r"\s+en\s+la\s+.+$", "", visitante, flags=re.IGNORECASE).strip()

    if not local or not visitante:
        return None
    return local, visitante


def _normalize_season_sql(colname: str) -> str:
    """
    Expresión SQL para igualar temporada ignorando '-' vs '/' y espacios.
    """
    return f"REPLACE(TRIM({colname}), '-', '/') = REPLACE(TRIM(?), '-', '/')"


# =================== Clase principal: ChatAgent =========================

class ChatAgent:
    def __init__(self) -> None:
        # Cliente OpenAI opcional
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key)

        # Modelo por defecto (puedes cambiarlo en .env con OPENAI_MODEL)
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ------------------------------------------------------------------
    # Entrada principal
    # ------------------------------------------------------------------
    def chat_query(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()
        q_low = q.lower()
        season = _extract_season(q_low)

        # 0️⃣ Pronóstico de partido con marcador + probabilidades
        if (
            ("pronostic" in q_low or "quién gana" in q_low or "quien gana" in q_low)
            and (" vs " in q_low or " contra " in q_low)
        ):
            pred = self._predict_match_with_db_and_llm(question=q, season=season)
            if pred is not None:
                return pred
            # Si no se pudo usar la BD, al menos que el LLM diga algo
            return self._fallback_llm(
                system=(
                    "Eres un analista de apuestas deportivas. "
                    "Da un pronóstico razonado para el partido mencionado."
                ),
                user=q,
            )

        # 1️⃣ RESUMEN / ANÁLISIS DE LA CLASIFICACIÓN (ESPN + LLM)
        if any(k in q_low for k in ["resumen", "analiza", "cómo va la liga", "como va la liga"]):
            try:
                if not fetch_standings_espn:
                    raise RuntimeError("fetch_standings_espn no disponible")
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

        # 2️⃣ CLASIFICACIÓN DIRECTA (tabla ESPN)
        if any(k in q_low for k in ["clasificación", "clasificacion", "tabla", "posiciones", "standings"]):
            try:
                if not fetch_standings_espn:
                    raise RuntimeError("fetch_standings_espn no disponible")
                df, meta = fetch_standings_espn()
                return {
                    "mode": "web",
                    "respuesta": f"Clasificación LaLiga {season or '(actual)'} (fuente: {meta.get('source', 'espn')})",
                    "tabla": _md_table(df),
                    "meta": meta,
                }
            except Exception as e:
                return self._fallback_llm(
                    system=(
                        "Eres un asistente de fútbol. "
                        "Si no tienes datos en vivo, explica cómo consultarlos (web oficial, ESPN, etc.)."
                    ),
                    user=f"No pude obtener la clasificación {season or 'actual'} de LaLiga. Explica de forma útil cómo conseguirla. ({e})",
                )

        # 3️⃣ GOLEADORES / PICHICHI / CONSULTAS SQL CON RAG
        if any(k in q_low for k in ["pichichi", "goleador", "goleadores", "máximo goleador", "maximo goleador"]):
            if ask_rag is None or not DB_PATH.exists():
                return self._fallback_llm(
                    system="Eres un asistente de fútbol.",
                    user=(
                        "No tengo acceso a la base de datos local, pero explica quiénes suelen ser "
                        "los máximos goleadores de LaLiga en los últimos años."
                    ),
                )
            rag_res = ask_rag(q)
            if not rag_res.get("ok"):
                return {"mode": "sql", "respuesta": rag_res.get("error", "Error en consulta SQL RAG")}

            cols = rag_res.get("columnas", [])
            rows = rag_res.get("resultados", [])
            df = pd.DataFrame(rows, columns=cols)
            texto = rag_res.get("resumen") or "Pichichi / top goleadores según tu base local."

            return {
                "mode": "sql",
                "respuesta": texto,
                "tabla": _md_table(df),
                "meta": {"sql": rag_res.get("consulta", "")},
            }

        # 4️⃣ TÍTULOS / PALMARÉS (Wikipedia)
        if any(
            k in q_low
            for k in [
                "títulos", "titulos", "palmarés", "palmares",
                "más ligas", "mas ligas", "ligas ganadas", "campeonatos",
            ]
        ):
            try:
                if not fetch_laliga_titles_wikipedia:
                    raise RuntimeError("fetch_laliga_titles_wikipedia no disponible")
                titles_df, cite_url = fetch_laliga_titles_wikipedia()

                # Determina la columna de títulos con o sin acento
                if "Títulos" in titles_df.columns:
                    title_col = "Títulos"
                elif "Titulos" in titles_df.columns:
                    title_col = "Titulos"
                else:
                    title_col = next(
                        (c for c in titles_df.columns if "titul" in c.lower()),
                        None,
                    )

                resumen_comp = ""
                ask_rm = ("real madrid" in q_low) or (" madrid" in q_low)
                ask_fcb = ("barcelona" in q_low) or ("fc barcelona" in q_low)

                if title_col and (ask_rm or ask_fcb):
                    t = titles_df.copy()
                    club_col = next(
                        (c for c in t.columns if c.lower() in ("club", "equipo", "team")),
                        None,
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
                        resumen_comp = f"\n\nComparativa → Real Madrid: {rm} | FC Barcelona: {fcb}. Más títulos: {ganador}."

                return {
                    "mode": "web",
                    "respuesta": (
                        "Títulos históricos de LaLiga por club (fuente: Wikipedia)\n"
                        f"Fuente: {cite_url}{resumen_comp}"
                    ),
                    "tabla": _md_table(titles_df, 30),
                    "meta": {"source": "wikipedia", "url": cite_url},
                }
            except Exception as e:
                return self._fallback_llm(
                    system="Eres un asistente de fútbol que explica palmarés con contexto.",
                    user=(
                        "No pude extraer la tabla de títulos por club en Wikipedia. "
                        "Resume qué clubes tienen más ligas y cómo comprobarlo con enlaces fiables. "
                        f"(Error: {e})"
                    ),
                )

        # 5️⃣ Cualquier otra cosa → LLM genérico de fútbol
        return self._fallback_llm(
            system="Eres un asistente de fútbol: breve, claro y útil.",
            user=q or "Hola",
        )

    # ==================================================================
    # Pronóstico con marcador + probabilidades usando la BD local
    # ==================================================================
    def _predict_match_with_db_and_llm(
        self,
        question: str,
        season: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        equipos = _detect_match_teams(question)
        if not equipos:
            return None

        local, visitante = equipos

        if not DB_PATH.exists():
            return None

        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()

        # Histórico de la temporada concreta (si viene)
        params = []
        where = "1=1"
        if season:
            where = _normalize_season_sql("Temporada")
            params.append(season)

        sql_hist = f"""
            SELECT GolesLocal, GolesVisitante
            FROM resultados
            WHERE {where}
              AND Local = ?
              AND Visitante = ?;
        """
        params_hist = params + [local, visitante]
        cur.execute(sql_hist, params_hist)
        rows = cur.fetchall()

        # Si no hay datos esa temporada, usamos todo el histórico
        if not rows:
            sql_hist_all = """
                SELECT GolesLocal, GolesVisitante
                FROM resultados
                WHERE Local = ?
                  AND Visitante = ?;
            """
            cur.execute(sql_hist_all, (local, visitante))
            rows = cur.fetchall()

        if not rows:
            con.close()
            # Heurística neutra
            p_local = 0.45
            p_emp = 0.25
            p_visit = 0.30
            g_local = 2
            g_visit = 1
        else:
            import numpy as np

            goles_local = [r[0] for r in rows]
            goles_visit = [r[1] for r in rows]

            total = len(rows)
            vict_local = sum(1 for gl, gv in rows if gl > gv)
            vict_visit = sum(1 for gl, gv in rows if gl < gv)
            empates = sum(1 for gl, gv in rows if gl == gv)

            total_adj = total + 3  # suavizado
            p_local = (vict_local + 1) / total_adj
            p_visit = (vict_visit + 1) / total_adj
            p_emp = (empates + 1) / total_adj

            s = p_local + p_visit + p_emp
            p_local, p_emp, p_visit = p_local / s, p_emp / s, p_visit / s

            g_local = int(round(float(np.mean(goles_local))))
            g_visit = int(round(float(np.mean(goles_visit))))
            g_local = max(0, min(g_local, 4))
            g_visit = max(0, min(g_visit, 4))

        con.close()

        # Probabilidades en %
        pL = int(round(p_local * 100))
        pE = int(round(p_emp * 100))
        pV = int(round(p_visit * 100))
        diff = 100 - (pL + pE + pV)
        if diff != 0:
            pL += diff  # pequeño ajuste

        base_text = (
            f"Pronóstico basado en tu base de datos de LaLiga.\n\n"
            f"Resultado probable: **{local} {g_local} – {g_visit} {visitante}**.\n"
            f"Probabilidades aproximadas:\n"
            f"- Victoria {local}: {pL}%\n"
            f"- Empate: {pE}%\n"
            f"- Victoria {visitante}: {pV}%"
        )

        if self.client:
            try:
                extra = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un analista de apuestas deportivas. "
                                "Recibes un pronóstico ya calculado (marcador y probabilidades) "
                                "y SOLO debes añadir 2-3 frases de explicación en español, "
                                "sin cambiar cifras ni resultado."
                            ),
                        },
                        {"role": "user", "content": base_text},
                    ],
                )
                extra_text = (extra.choices[0].message.content or "").strip()
                full_text = base_text + "\n\n" + extra_text
            except Exception:
                full_text = base_text
        else:
            full_text = base_text

        return {
            "mode": "sql_prediccion",
            "respuesta": full_text,
            "tabla": None,
            "meta": {
                "local": local,
                "visitante": visitante,
                "temporada": season,
                "db": str(DB_PATH),
            },
        }

    # ==================================================================
    # Resumen de clasificación con LLM
    # ==================================================================
    def _summarize_standings_with_llm(
        self,
        df: pd.DataFrame,
        season: Optional[str],
        source: str,
    ) -> str:
        if df is None or df.empty:
            return "(No hay datos disponibles para resumir.)"

        if not self.client:
            return (
                "(LLM no configurado para generar el resumen, "
                "pero la tabla de clasificación se ha obtenido correctamente.)"
            )

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

    # ==================================================================
    # Fallback LLM genérico
    # ==================================================================
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