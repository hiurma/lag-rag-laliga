# --- chat_agent.py -------------------------------------------------
from __future__ import annotations

import os
import re
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Web RAG: clasificación actual y títulos históricos
from web_rag import fetch_standings_espn, fetch_laliga_titles_wikipedia

# LLM opcional (para resúmenes y small-talk)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# Ruta a tu BD local
DB_PATH = Path(os.getenv("DB_PATH", "data/laliga.sqlite"))


# -------------------------------------------------------------------
# Helpers genéricos
# -------------------------------------------------------------------
def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "(sin filas)"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return "\n" + df.head(max_rows).to_string(index=False) + "\n"


def _extract_season(text: str) -> Optional[str]:
    """
    Busca cosas tipo '2024/2025' o '2024-2025' y devuelve '2024/2025'.
    """
    m = re.search(r"(20\d{2})\s*[-/]\s*(20\d{2})", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


def _season_filter(colname: str) -> str:
    """
    Expresión SQL para comparar temporadas ignorando '-' vs '/' y espacios.
    """
    return f"REPLACE(TRIM({colname}),'-','/') = REPLACE(TRIM(?),'-','/')"


def _canon_team(name: str) -> Optional[str]:
    """
    Intenta encontrar el nombre “canónico” del equipo según la BD local.
    Busca en resultados + clasificaciones.
    """
    if not name:
        return None
    s = name.strip()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    try:
        row = cur.execute(
            """
            SELECT t FROM (
              SELECT DISTINCT Local AS t FROM resultados
              UNION
              SELECT DISTINCT Visitante AS t FROM resultados
              UNION
              SELECT DISTINCT Club AS t FROM clasificaciones
            )
            WHERE LOWER(t) LIKE LOWER(?)
            ORDER BY LENGTH(t) ASC
            LIMIT 1;
            """,
            (f"%{s}%",),
        ).fetchone()
    finally:
        con.close()

    return row[0] if row else None


def _extract_match(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extrae (local, visitante, temporada) de frases tipo:
    - "Real Madrid vs Girona 2024/2025"
    - "Pronostica Barcelona - Real Madrid"
    - "Atlético contra Sevilla 2025-2026"
    """
    # Primero pillamos posible temporada
    season = _extract_season(text)

    # Quitamos la temporada del texto para no molestar al regex
    t_clean = text
    if season:
        t_clean = t_clean.replace(season, "")

    m = re.search(
        r"(.+?)\s+(?:vs\.?|contra|frente a|-)\s+(.+)$",
        t_clean,
        re.IGNORECASE,
    )
    if not m:
        return None, None, season

    home_raw = m.group(1).strip()
    away_raw = m.group(2).strip()
    return home_raw, away_raw, season


# -------------------------------------------------------------------
# Clase principal
# -------------------------------------------------------------------
class ChatAgent:
    def __init__(self) -> None:
        # Cliente OpenAI opcional
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key)

        # Modelo por defecto (puedes cambiarlo en .env con OPENAI_MODEL)
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ----------------------------------------------------------------
    # Entrada principal
    # ----------------------------------------------------------------
    def chat_query(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()
        q_low = q.lower()

        # 0) Detectamos temporada si aparece en el texto (para mensajes)
        season = _extract_season(q_low)

        # 1️⃣ RESUMEN / ANÁLISIS DE LA CLASIFICACIÓN (RAG web + LLM)
        if any(k in q_low for k in ["resumen", "analiza", "cómo va la liga", "como va la liga"]):
            try:
                df, meta = fetch_standings_espn()  # clasificación actual
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
                df, meta = fetch_standings_espn()  # siempre liga actual
                return {
                    "mode": "web",
                    "respuesta": f"Clasificación LaLiga {season or '(actual)'} (fuente: {meta.get('source', 'espn')})",
                    "tabla": _md_table(df),
                    "meta": meta,
                }
            except Exception as e:
                return self._fallback_llm(
                    system="Eres un asistente de fútbol. Si no tienes datos en vivo, explica cómo consultarlos (web oficial, ESPN, etc.).",
                    user=(
                        f"No pude obtener la clasificación {season or 'actual'} de LaLiga. "
                        f"Explica de forma útil cómo conseguirla y qué significan las columnas. ({e})"
                    ),
                )

        # 3️⃣ TÍTULOS / PALMARÉS (RAG web → Wikipedia)
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

                resumen = ""
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

        # 4️⃣ PRONÓSTICO DE PARTIDOS (Poisson sobre tu BD)
        if any(k in q_low for k in [" vs ", "vs ", "contra", "frente a", "partido", "pronóstic", "pronostic", "marcador", "resultado"]):
            home_raw, away_raw, season_match = _extract_match(q)
            temporada = season_match or season  # si lo detectamos antes

            if home_raw and away_raw:
                home_canon = _canon_team(home_raw)
                away_canon = _canon_team(away_raw)

                if not home_canon or not away_canon:
                    return {
                        "mode": "prediccion",
                        "respuesta": (
                            f"No puedo predecir bien ese partido porque no encuentro a los equipos "
                            f"en la base de datos local: '{home_raw}' vs '{away_raw}'. "
                            f"Prueba con el nombre oficial (por ejemplo 'Real Madrid', 'FC Barcelona', 'Girona FC')."
                        ),
                    }

                pred = self._predict_poisson(home_canon, away_canon, temporada)
                return {
                    "mode": "prediccion",
                    "respuesta": self._format_prediction(
                        home_canon,
                        away_canon,
                        temporada,
                        pred,
                    ),
                    "meta": pred,
                }

        # 5️⃣ Small talk / resto → LLM (si disponible)
        return self._fallback_llm(
            system="Eres un asistente de fútbol: breve, claro y útil.",
            user=q or "Hola",
        )
    def _extract_match(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:

        original_text = text.lower().strip()

    # 1) Detectar temporada
        season = _extract_season(original_text)

    # 2) Lista de palabras basura
        stopwords = [
            "pronostico", "pronóstico", "prediccion", "predicción",
            "resultado", "partido", "jornada", "del", "vs", "versus",
            "de la", "de", "temporada", "para", "el", "la", "en",
            "dame", "hazme", "quiero", "pronostica"
        ]

        clean = original_text

        # 3) Quitar temporada del texto
        if season:
            clean = clean.replace(season.lower(), "")

        # 4) Eliminar palabras basura
        for w in stopwords:
            clean = clean.replace(w, " ")

            clean = re.sub(r"\s+", " ", clean).strip()
        # 5) Regex para capturar el partido
        pattern = r"(.+?)\s+(?:vs\.?|contra|frente a|-|vs)\s+(.+)$"
        m = re.search(pattern, clean, re.IGNORECASE)

        if not m:
            return None, None, season

        home = m.group(1).strip()
        away = m.group(2).strip()

        # 6) ✨ SUPER-FILTRO DE SEGURIDAD ✨
        # Nunca devolver cadenas que empiecen por "pronostico", "prediccion", etc.
        bad_prefixes = ["pronostico", "pronóstico", "prediccion", "predicción"]

        def clean_name(t):
            t = t.strip()
            for b in bad_prefixes:
                if t.startswith(b):
                    t = t.replace(b, "").strip()
            return t

        home = clean_name(home)
        away = clean_name(away)

        return home, away, season
    # ----------------------------------------------------------------
    # PRONÓSTICO: modelo Poisson real con datos de tu BD
    # ----------------------------------------------------------------
    def _league_means(self, temporada: Optional[str]) -> Tuple[float, float]:
        """
        Medias de goles de la liga (local y visitante) para la temporada dada
        o para todo histórico si temporada es None.
        """
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        try:
            params = ()
            where = ""
            if temporada:
                where = "WHERE " + _season_filter("Temporada")
                params = (temporada,)

            row = cur.execute(
                f"SELECT AVG(GolesLocal), AVG(GolesVisitante) FROM resultados {where};",
                params,
            ).fetchone()
        finally:
            con.close()

        if not row:
            return 1.4, 1.1
        m_home = row[0] if row[0] is not None else 1.4
        m_away = row[1] if row[1] is not None else 1.1
        return float(m_home), float(m_away)

    def _team_profile(self, team: str, temporada: Optional[str]) -> Dict[str, Optional[float]]:
        """
        Devuelve ataque/defensa en casa y fuera para un equipo.
        """
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        try:
            params = (team,)
            where = "WHERE Local=?"
            if temporada:
                where += " AND " + _season_filter("Temporada")
                params = (team, temporada)

            # Ataque y defensa como local
            row_home = cur.execute(
                f"""
                SELECT AVG(GolesLocal), AVG(GolesVisitante)
                FROM resultados
                {where};
                """,
                params,
            ).fetchone()

            # Ataque y defensa como visitante
            params = (team,)
            where = "WHERE Visitante=?"
            if temporada:
                where += " AND " + _season_filter("Temporada")
                params = (team, temporada)

            row_away = cur.execute(
                f"""
                SELECT AVG(GolesVisitante), AVG(GolesLocal)
                FROM resultados
                {where};
                """,
                params,
            ).fetchone()
        finally:
            con.close()

        atk_home = row_home[0] if row_home and row_home[0] is not None else None
        def_home = row_home[1] if row_home and row_home[1] is not None else None
        atk_away = row_away[0] if row_away and row_away[0] is not None else None
        def_away = row_away[1] if row_away and row_away[1] is not None else None

        return {
            "atk_home": float(atk_home) if atk_home is not None else None,
            "def_home": float(def_home) if def_home is not None else None,
            "atk_away": float(atk_away) if atk_away is not None else None,
            "def_away": float(def_away) if def_away is not None else None,
        }

    def _predict_poisson(self, home: str, away: str, temporada: Optional[str]) -> Dict[str, Any]:
        """
        Calcula lambdas y probabilidades de marcador usando un modelo Poisson
        calibrado con:
          - medias de liga
          - ataque/defensa reciente de cada equipo
          - ligera ventaja local
        """
        Lh, La = self._league_means(temporada)
        home_prof = self._team_profile(home, temporada)
        away_prof = self._team_profile(away, temporada)

        def fallback(v, default):
            return default if v is None or math.isnan(v) else v

        atk_home = fallback(home_prof["atk_home"], Lh)
        def_home = fallback(home_prof["def_home"], La)
        atk_away = fallback(away_prof["atk_away"], La)
        def_away = fallback(away_prof["def_away"], Lh)

        # Ventaja local suave
        HOME_ADV = 1.10

        # Estilo Dixon-Coles simplificado
        lam_home = HOME_ADV * Lh * (atk_home / Lh) * (def_away / Lh)
        lam_away = La * (atk_away / La) * (def_home / La)

        # límites razonables
        lam_home = float(max(0.2, min(4.0, lam_home)))
        lam_away = float(max(0.2, min(4.0, lam_away)))

        # Matriz de probabilidad de marcadores 0–6
        max_goals = 6

        def pois(k: int, lam: float) -> float:
            return math.exp(-lam) * (lam ** k) / math.factorial(k)

        grid = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                grid[i, j] = pois(i, lam_home) * pois(j, lam_away)

        prob_draw = float(np.sum(np.diag(grid)))
        prob_home = float(np.sum(np.tril(grid, -1)))
        prob_away = float(np.sum(np.triu(grid, 1)))
        i_max, j_max = np.unravel_index(np.argmax(grid), grid.shape)

        return {
            "lambda_home": round(lam_home, 3),
            "lambda_away": round(lam_away, 3),
            "prob_home": round(prob_home, 3),
            "prob_draw": round(prob_draw, 3),
            "prob_away": round(prob_away, 3),
            "score_most_likely": f"{i_max}-{j_max}",
            "matrix_max_goals": max_goals,
        }

    def _format_prediction(
        self,
        home: str,
        away: str,
        temporada: Optional[str],
        pred: Dict[str, Any],
    ) -> str:
        return (
            f"**Pronóstico {home} vs {away} ({temporada or 'histórico en tu BD'})**\n"
            f"- λ local: {pred['lambda_home']:.2f}\n"
            f"- λ visitante: {pred['lambda_away']:.2f}\n\n"
            f"**Probabilidades (modelo Poisson sobre tu base de datos):**\n"
            f"- Victoria {home}: {pred['prob_home']*100:.1f}%\n"
            f"- Empate: {pred['prob_draw']*100:.1f}%\n"
            f"- Victoria {away}: {pred['prob_away']*100:.1f}%\n\n"
            f"**Marcador más probable:** {pred['score_most_likely']}"
        )

    # ----------------------------------------------------------------
    # Resumen clasificación con LLM
    # ----------------------------------------------------------------
    def _summarize_standings_with_llm(self, df: pd.DataFrame, season: Optional[str], source: str) -> str:
        if df is None or df.empty:
            return "(No hay datos disponibles para resumir.)"

        if not self.client:
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

    # ----------------------------------------------------------------
    # Fallback LLM genérico
    # ----------------------------------------------------------------
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