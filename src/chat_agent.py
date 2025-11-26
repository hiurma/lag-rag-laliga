# -*- coding: utf-8 -*-
from __future__ import annotations

import os, re, sqlite3
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List

# --- RAG web ya existente (clasificación/títulos) ---
from web_rag import fetch_standings_espn, fetch_laliga_titles_wikipedia

# --- Poisson y normalizador de nombres ---
from poisson_model import predict_match_poisson, bulk_predict_rm
from names import canon_team

# --- LLM opcional ---
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
    m = re.search(r"(20\d{2})\s*[-/]\s*(20\d{2})", text)
    return f"{m.group(1)}/{m.group(2)}" if m else None

# ----------------- parse "Pronóstico A vs B ..." -----------------

MATCH_RE = re.compile(
    r"(?:pronost[aí]co|predicci[oó]n)\s+([a-z0-9 .\-_/áéíóúñ]+?)\s+vs\.?\s+([a-z0-9 .\-_/áéíóúñ]+?)(?:\s+(20\d{2}[/\-]20\d{2}))?$",
    re.I
)

def _parse_match_query(q: str) -> Optional[Tuple[str,str,Optional[str]]]:
    m = MATCH_RE.search(q)
    if not m:
        return None
    a = m.group(1).strip()
    b = m.group(2).strip()
    s = _extract_season(q) or (m.group(3).strip() if m.group(3) else None)
    return a,b,s

# ----------------- ChatAgent -----------------

class ChatAgent:
    def __init__(self) -> None:
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key)
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ---------- helpers contexto LLM ----------
    def _contexto_llm(self, season: str, home: str, away: str, pred: Dict) -> str:
        """
        Genera 3-4 frases de contexto: forma reciente (últimos 5), media de goles, punto táctico.
        Si no hay LLM, devuelve un texto heurístico.
        """
        # Stats rápidas desde resultados (últimos 5 de cada uno)
        con = sqlite3.connect("data/laliga.sqlite")
        sql = """
        SELECT Jornada, Local, Visitante, GolesLocal, GolesVisitante
        FROM resultados
        WHERE REPLACE(TRIM(Temporada),'-','/') = REPLACE(TRIM(?),'-','/')
          AND (Local=? OR Visitante=?)
          AND GolesLocal IS NOT NULL AND GolesVisitante IS NOT NULL
        ORDER BY Jornada DESC
        LIMIT 10
        """
        df = pd.read_sql_query(sql, con, params=[season, home, home])
        df2 = pd.read_sql_query(sql, con, params=[season, away, away])
        con.close()

        def _mini(df: pd.DataFrame) -> Dict:
            if df.empty:
                return {"pj":0,"gf":0,"ga":0,"prom":0.0}
            pj = len(df)
            gf = int((df["GolesLocal"]*(df["Local"]==home)).sum() + (df["GolesVisitante"]*(df["Visitante"]==home)).sum())
            ga = int((df["GolesVisitante"]*(df["Local"]==home)).sum() + (df["GolesLocal"]*(df["Visitante"]==home)).sum())
            prom = (gf+ga)/max(1,pj)
            return {"pj":pj,"gf":gf,"ga":ga,"prom":round(prom,2)}

        h = _mini(df)
        a = _mini(df2)

        base_bullets = (
            f"- λ esperadas: {home} {pred['lambda_home']} vs {away} {pred['lambda_away']}. "
            f"1X2: {pred['p_home']}-{pred['p_draw']}-{pred['p_away']}.\n"
            f"- Últimos {h['pj']} del {home}: {h['gf']}-{h['ga']} (media {h['prom']} g/p). "
            f"Últimos {a['pj']} del {away}: {a['gf']}-{a['ga']} (media {a['prom']} g/p)."
        )

        if not self.client:
            return ("Contexto (heurístico):\n" + base_bullets)

        prompt = f"""
        Eres un analista de fútbol. Con los datos siguientes, escribe 3-5 frases concisas,
        neutrales y útiles sobre el partido {home} vs {away} de la temporada {season}.
        Incorpora el pronóstico Poisson, la tendencia de goles y un punto táctico general.

        {base_bullets}
        """

        try:
            comp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role":"system","content":"Eres un analista deportivo objetivo y claro."},
                    {"role":"user","content":prompt},
                ],
                temperature=0.3,
            )
            return comp.choices[0].message.content.strip()
        except Exception as e:
            return "Contexto (LLM no disponible): " + base_bullets + f" [{e}]"

    # ------------------------- entrada principal -------------------------

    def chat_query(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()
        q_low = q.lower()
        season = _extract_season(q_low) or "2025/2026"

        # 1) Pronóstico concreto (Poisson) — permitimos CUALQUIER rival,
        #    pero para tu TFG lo usaremos sobre todo con el Madrid.
        m = _parse_match_query(q_low)
        if m:
            home_raw, away_raw, s2 = m
            season_use = s2 or season
            home = canon_team(home_raw)
            away = canon_team(away_raw)

            # estimamos fuerzas SOLO con lo jugado hasta jornada 8 (tu DB)
            try:
                pred = predict_match_poisson(season_use, home, away, upto_matchday=8)
                ctx = self._contexto_llm(season_use, home, away, pred)
                texto = (
                    f"**Pronóstico Poisson** {season_use}: **{pred['score_mas_probable']}**\n"
                    f"Prob. 1X2 → 1={pred['p_home']:.2f}  X={pred['p_draw']:.2f}  2={pred['p_away']:.2f}\n"
                    f"{ctx}"
                )
                return {"mode":"pred", "respuesta": texto, "meta": pred}
            except Exception as e:
                return {"mode":"pred", "respuesta": f"No pude calcular el Poisson: {e}"}

        # 2) Pronósticos RM “hasta hoy” (bloque)  → detecta “hasta hoy”, “hasta la jornada X”
        if "hasta hoy" in q_low or ("hasta la jornada" in q_low and "real madrid" in q_low):
            # intenta detectar jornada actual por web; si falla, usa 38
            try:
                _, meta = fetch_standings_espn()
                j_act = int(meta.get("matchday") or meta.get("jornada") or 38)
            except Exception:
                j_act = 38
            try:
                dfp = bulk_predict_rm(season, 9, j_act, upto_matchday_for_fit=8)
                if dfp.empty:
                    return {"mode":"pred","respuesta":"No encuentro calendario del Real Madrid en tu DB para esas jornadas."}
                return {
                    "mode":"pred",
                    "respuesta": f"Pronósticos Real Madrid (Poisson con datos hasta J8) {season}:\n\n" + _md_table(dfp, 50),
                }
            except Exception as e:
                return {"mode":"pred","respuesta": f"No pude generar la tanda de pronósticos: {e}"}

        # ----------------- rutas generales que ya tenías -----------------

        if any(k in q_low for k in ["resumen", "analiza", "cómo va la liga", "como va la liga"]):
            try:
                df, meta = fetch_standings_espn()
                tabla_md = _md_table(df)
                # resumen con LLM si está disponible (opcional)
                if self.client and not df.empty:
                    try:
                        prompt = f"Resume en 3-4 frases la clasificación: {tabla_md}"
                        comp = self.client.chat.completions.create(
                            model=self.openai_model,
                            messages=[
                                {"role":"system","content":"Eres un periodista deportivo conciso."},
                                {"role":"user","content":prompt},
                            ],
                            temperature=0.2,
                        )
                        txt = comp.choices[0].message.content.strip()
                    except Exception:
                        txt = "(No pude generar resumen LLM)."
                else:
                    txt = "(Resumen LLM no configurado)."
                return {"mode":"web","respuesta":txt,"tabla":tabla_md,"meta":meta}
            except Exception as e:
                return {"mode":"llm","respuesta":f"(Error al obtener clasificación: {e})"}

        if any(k in q_low for k in ["clasificación","clasificacion","tabla","posiciones","standings"]):
            try:
                df, meta = fetch_standings_espn()
                return {"mode":"web","respuesta":f"Clasificación LaLiga (fuente: {meta.get('source','espn')})","tabla":_md_table(df),"meta":meta}
            except Exception as e:
                return {"mode":"llm","respuesta":f"(No pude obtener la clasificación: {e})"}

        if any(k in q_low for k in ["títulos","titulos","palmarés","palmares","más ligas","mas ligas","ligas ganadas","campeonatos"]):
            try:
                titles_df, cite_url = fetch_laliga_titles_wikipedia()
                return {
                    "mode":"web",
                    "respuesta": f"Títulos históricos por club (Wikipedia). Fuente: {cite_url}",
                    "tabla": _md_table(titles_df, 30),
                    "meta": {"source":"wikipedia","url":cite_url}
                }
            except Exception as e:
                return {"mode":"llm","respuesta":f"(No pude extraer palmarés: {e})"}

        # fallback genérico
        if self.client:
            try:
                comp = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role":"system","content":"Eres un asistente de fútbol: breve, claro y útil."},
                        {"role":"user","content": q or "Hola"},
                    ],
                )
                return {"mode":"llm","respuesta":(comp.choices[0].message.content or "").strip()}
            except Exception as e:
                return {"mode":"llm","respuesta":f"(Error LLM: {e})"}

        return {"mode":"llm","respuesta":"(LLM no configurado) Dime qué necesitas de LaLiga."}