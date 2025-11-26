# -*- coding: utf-8 -*-
from __future__ import annotations
import sqlite3, math
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd

DB_PATH = Path("data/laliga.sqlite")

# ------------------------- util Poisson -------------------------

def _pois_pmf(lmbda: float, k: int) -> float:
    # pmf = e^-λ λ^k / k!
    try:
        return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)
    except OverflowError:
        return 0.0

def _score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 6):
    """Matriz de probabilidades P(Home=i, Away=j)."""
    ph = [ _pois_pmf(lambda_home, i) for i in range(max_goals + 1) ]
    pa = [ _pois_pmf(lambda_away, j) for j in range(max_goals + 1) ]
    # producto externo
    mat = [[ph[i]*pa[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]
    return mat

def _most_probable_score(mat) -> Tuple[int,int,float]:
    best = (0,0,0.0)
    for i,row in enumerate(mat):
        for j,p in enumerate(row):
            if p > best[2]:
                best = (i,j,p)
    return best

# -------------------- carga & fuerzas de equipos --------------------

def _read_results_until(season: str, upto_matchday: Optional[int]) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    q = """
        SELECT Temporada, Jornada, Local, Visitante, GolesLocal, GolesVisitante
        FROM resultados
        WHERE REPLACE(TRIM(Temporada),'-','/') = REPLACE(TRIM(?),'-','/')
          AND GolesLocal IS NOT NULL AND GolesVisitante IS NOT NULL
    """
    params: List = [season]
    if upto_matchday is not None:
        q += " AND Jornada <= ?"
        params.append(upto_matchday)
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    # normaliza textos
    for c in ["Local","Visitante"]:
        df[c] = df[c].astype(str).str.strip()
    return df

def _league_averages(df: pd.DataFrame) -> Tuple[float,float]:
    # medias por partido
    if df.empty:
        return 1.4, 1.1  # valores razonables por defecto
    home_avg = df["GolesLocal"].mean()
    away_avg = df["GolesVisitante"].mean()
    return float(home_avg), float(away_avg)

def _smooth_rate(goals: float, matches: float, prior_rate: float, alpha: float = 3.0) -> float:
    """
    Shrink: (g + α*μ) / (m + α), robusto con pocas jornadas (8).
    """
    return (goals + alpha*prior_rate) / max(1.0, (matches + alpha))

def _team_strengths(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calcula rates por equipo separados casa/fuera:
      att_home, def_home, att_away, def_away (relativos a medias liga).
    """
    if df.empty:
        return {}

    mu_h, mu_a = _league_averages(df)

    # agregados por equipo
    home = df.groupby("Local").agg(
        gf=("GolesLocal","sum"),
        ga=("GolesVisitante","sum"),
        pj=("Local","count")
    ).rename_axis("Team").reset_index()

    away = df.groupby("Visitante").agg(
        gf=("GolesVisitante","sum"),
        ga=("GolesLocal","sum"),
        pj=("Visitante","count")
    ).rename_axis("Team").reset_index()

    teams = sorted(set(home["Team"]) | set(away["Team"]))
    out: Dict[str, Dict[str, float]] = {}
    for t in teams:
        h = home[home["Team"]==t]
        a = away[away["Team"]==t]

        gf_h = float(h["gf"].iloc[0]) if not h.empty else 0.0
        ga_h = float(h["ga"].iloc[0]) if not h.empty else 0.0
        pj_h = float(h["pj"].iloc[0]) if not h.empty else 0.0

        gf_a = float(a["gf"].iloc[0]) if not a.empty else 0.0
        ga_a = float(a["ga"].iloc[0]) if not a.empty else 0.0
        pj_a = float(a["pj"].iloc[0]) if not a.empty else 0.0

        # tasas suavizadas
        rate_gf_h = _smooth_rate(gf_h, pj_h, mu_h)
        rate_ga_h = _smooth_rate(ga_h, pj_h, mu_a)
        rate_gf_a = _smooth_rate(gf_a, pj_a, mu_a)
        rate_ga_a = _smooth_rate(ga_a, pj_a, mu_h)

        # fortalezas relativas
        att_home = rate_gf_h / max(mu_h, 1e-6)
        def_home = rate_ga_a / max(mu_a, 1e-6)  # ojo: concedidos a domicilio referenciados a media away
        att_away = rate_gf_a / max(mu_a, 1e-6)
        def_away = rate_ga_h / max(mu_h, 1e-6)

        out[t] = {
            "att_home": float(att_home),
            "def_home": float(def_home),
            "att_away": float(att_away),
            "def_away": float(def_away),
        }
    # medias liga
    out["_mu_home"] = mu_h
    out["_mu_away"] = mu_a
    return out

def build_strengths(season: str, upto_matchday: Optional[int]) -> Dict[str, Dict[str,float]]:
    df = _read_results_until(season, upto_matchday)
    return _team_strengths(df)

# --------------------------- predicción ---------------------------

def predict_match_poisson(
    season: str,
    home_team: str,
    away_team: str,
    upto_matchday: Optional[int] = None,
    max_goals: int = 6
) -> Dict:
    """
    Predice un partido usando fortalezas calculadas SOLO con los datos hasta `upto_matchday`.
    """
    strengths = build_strengths(season, upto_matchday)
    mu_h = strengths.get("_mu_home", 1.4)
    mu_a = strengths.get("_mu_away", 1.1)

    th = strengths.get(home_team)
    ta = strengths.get(away_team)
    if not th or not ta:
        raise ValueError(f"No tengo fortalezas para {home_team} o {away_team} (temporada {season}).")

    # λ = μ * ataque_home * defensa_away  (y análogo para visitantes)
    lam_home = mu_h * th["att_home"] * ta["def_away"]
    lam_away = mu_a * ta["att_away"] * th["def_home"]

    mat = _score_matrix(lam_home, lam_away, max_goals=max_goals)
    i,j,pij = _most_probable_score(mat)

    # prob 1X2
    p_home = sum(mat[i2][j2] for i2 in range(max_goals+1) for j2 in range(max_goals+1) if i2>j2)
    p_draw = sum(mat[i2][j2] for i2 in range(max_goals+1) for j2 in range(max_goals+1) if i2==j2)
    p_away = 1.0 - p_home - p_draw

    return {
        "temporada": season,
        "lambda_home": round(lam_home, 3),
        "lambda_away": round(lam_away, 3),
        "score_mas_probable": f"{home_team} {i}-{j} {away_team}",
        "p_home": round(p_home, 3),
        "p_draw": round(p_draw, 3),
        "p_away": round(p_away, 3),
        "matrix_max_goals": max_goals,
    }

# -------------------- pronósticos en bloque (Real Madrid) --------------------

def fixtures_rm_from_db(season: str, desde_j: int, hasta_j: int) -> pd.DataFrame:
    """
    Lee de la tabla resultados los partidos del Real Madrid entre jornadas dadas,
    aunque no tengan marcador (sirven de calendario).
    """
    con = sqlite3.connect(DB_PATH)
    q = """
    SELECT Jornada, Local, Visitante, Marcador
    FROM resultados
    WHERE REPLACE(TRIM(Temporada),'-','/') = REPLACE(TRIM(?),'-','/')
      AND Jornada BETWEEN ? AND ?
      AND (Local='Real Madrid' OR Visitante='Real Madrid')
    ORDER BY Jornada
    """
    df = pd.read_sql_query(q, con, params=[season, desde_j, hasta_j])
    con.close()
    for c in ["Local","Visitante"]:
        df[c] = df[c].astype(str).str.strip()
    return df

def bulk_predict_rm(season: str, desde_j: int, hasta_j: int, upto_matchday_for_fit: int) -> pd.DataFrame:
    """
    Predice todas las jornadas de RM en [desde_j, hasta_j] usando
    fortalezas estimadas hasta `upto_matchday_for_fit` (por ejemplo 8).
    """
    cal = fixtures_rm_from_db(season, desde_j, hasta_j)
    if cal.empty:
        # no hay calendario en DB → devolvemos DF vacío con columnas estándar
        return pd.DataFrame(columns=["Jornada","Local","Visitante","Pronóstico","P(1)","P(X)","P(2)"])

    rows = []
    for _,r in cal.iterrows():
        try:
            pred = predict_match_poisson(season, r["Local"], r["Visitante"], upto_matchday=upto_matchday_for_fit)
            rows.append({
                "Jornada": int(r["Jornada"]),
                "Local": r["Local"],
                "Visitante": r["Visitante"],
                "Pronóstico": pred["score_mas_probable"],
                "P(1)": pred["p_home"],
                "P(X)": pred["p_draw"],
                "P(2)": pred["p_away"],
            })
        except Exception as e:
            rows.append({
                "Jornada": int(r["Jornada"]),
                "Local": r["Local"],
                "Visitante": r["Visitante"],
                "Pronóstico": f"(error: {e})",
                "P(1)": None, "P(X)": None, "P(2)": None
            })

    out = pd.DataFrame(rows).sort_values("Jornada")
    return out