# -*- coding: utf-8 -*-
from __future__ import annotations
import sqlite3
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import math

DB_PATH = Path("data/laliga.sqlite")

# --------------------------------------------------
# Normalización de equipos
# --------------------------------------------------
TEAM_ALIASES = {
    "real madrid": "Real Madrid",
    "r. madrid": "Real Madrid",
    "realmadrid": "Real Madrid",
    "rm": "Real Madrid",

    "fc barcelona": "FC Barcelona",
    "barcelona": "FC Barcelona",
    "barça": "FC Barcelona",
    "fcb": "FC Barcelona",

    "atletico de madrid": "Atlético Madrid",
    "atlético de madrid": "Atlético Madrid",
    "atletico madrid": "Atlético Madrid",
    "atm": "Atlético Madrid",

    "sevilla": "Sevilla FC",
    "sevilla fc": "Sevilla FC",

    "girona": "Girona FC",
    "girona fc": "Girona FC",

    "osasuna": "Osasuna",
    "real sociedad": "Real Sociedad",

    "betis": "Real Betis",
    "real betis": "Real Betis",

    "rayo": "Rayo Vallecano",
    "rayo vallecano": "Rayo Vallecano",

    "villarreal": "Villarreal",
    "valencia": "Valencia",
    "getafe": "Getafe",

    "celta de vigo": "Celta de Vigo",
    "celta": "Celta de Vigo",

    "espanyol": "Espanyol",
    "real oviedo": "Real Oviedo",
    "alaves": "Alavés",
    "alavés": "Alavés",
    "mallorca": "Mallorca",
    "elche": "Elche",
    "granada": "Granada",
    "leganes": "Leganés",
    "leganés": "Leganés",
    "levante": "Levante",
}

def norm_team(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return TEAM_ALIASES.get(s, name.strip())


# --------------------------------------------------
# Utilidades Poisson
# --------------------------------------------------
def _poisson_prob(lmbd: float, k: int) -> float:
    if lmbd <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lmbd) * (lmbd ** k) / math.factorial(k)


def _grid_probs(mu_h: float, mu_a: float, max_goals: int = 6):
    best_score = (0, 0)
    best_p = 0
    p_home = p_draw = p_away = 0.0

    for gh in range(max_goals + 1):
        for ga in range(max_goals + 1):
            p = _poisson_prob(mu_h, gh) * _poisson_prob(mu_a, ga)
            if p > best_p:
                best_p = p
                best_score = (gh, ga)
            if gh > ga:
                p_home += p
            elif gh == ga:
                p_draw += p
            else:
                p_away += p

    return {
        "best_score": best_score,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
    }


# --------------------------------------------------
# Temporadas recientes
# --------------------------------------------------
def _get_recent_seasons(con: sqlite3.Connection, max_seasons: int = 3) -> List[str]:
    cur = con.cursor()
    rows = cur.execute(
        "SELECT DISTINCT Temporada FROM resultados ORDER BY Temporada DESC"
    ).fetchall()
    return [r[0] for r in rows][:max_seasons]


def _get_current_season(con: sqlite3.Connection) -> Optional[str]:
    cur = con.cursor()
    row = cur.execute("SELECT MAX(Temporada) FROM resultados").fetchone()
    return row[0] if row and row[0] else None


# --------------------------------------------------
# Fuerza de equipos con historial + suavizado bayesiano + decay
# --------------------------------------------------
def _team_strengths_multiseason(
    con: sqlite3.Connection,
    temporada_objetivo: str,
    max_seasons=3,
    decay=0.7,
    prior_k=5.0
):
    seasons = _get_recent_seasons(con, max_seasons=max_seasons)
    if not seasons:
        raise RuntimeError("No hay temporadas en BD.")

    # Pesos: temporada más actual → peso 1
    season_w = {t: decay ** i for i, t in enumerate(seasons)}

    # Cargar todos los partidos relevantes
    qmarks = ",".join("?" for _ in seasons)
    rows = con.execute(
        f"SELECT Temporada, Local, Visitante, GolesLocal, GolesVisitante "
        f"FROM resultados WHERE Temporada IN ({qmarks})",
        seasons
    ).fetchall()

    teams = {}
    total_w_matches = 0
    sum_gl = 0
    sum_gv = 0

    for t, loc, vis, gl, gv in rows:
        w = season_w[t]
        gl = gl or 0
        gv = gv or 0

        sum_gl += w * gl
        sum_gv += w * gv
        total_w_matches += w

        def init_team(team):
            return {
                "home_scored": 0, "home_conc": 0, "home_played": 0,
                "away_scored": 0, "away_conc": 0, "away_played": 0
            }

        if loc not in teams:
            teams[loc] = init_team(loc)
        if vis not in teams:
            teams[vis] = init_team(vis)

        teams[loc]["home_scored"] += w * gl
        teams[loc]["home_conc"] += w * gv
        teams[loc]["home_played"] += w

        teams[vis]["away_scored"] += w * gv
        teams[vis]["away_conc"] += w * gl
        teams[vis]["away_played"] += w

    avg_home = sum_gl / max(total_w_matches, 1)
    avg_away = sum_gv / max(total_w_matches, 1)

    def shrink(value, n):
        return (value * n + 1.0 * prior_k) / (n + prior_k)

    strengths = {}
    for team, st in teams.items():
        n_h = st["home_played"]
        n_a = st["away_played"]

        atk_h_raw = (st["home_scored"] / n_h) / avg_home if n_h > 0 else 1
        def_h_raw = (st["home_conc"] / n_h) / avg_away if n_h > 0 else 1
        atk_a_raw = (st["away_scored"] / n_a) / avg_away if n_a > 0 else 1
        def_a_raw = (st["away_conc"] / n_a) / avg_home if n_a > 0 else 1

        atk_h = shrink(atk_h_raw, n_h)
        def_h = shrink(def_h_raw, n_h)
        atk_a = shrink(atk_a_raw, n_a)
        def_a = shrink(def_a_raw, n_a)

        strengths[team] = (atk_h, def_h, atk_a, def_a)

    return avg_home, avg_away, strengths


# --------------------------------------------------
# Predicción principal
# --------------------------------------------------
def predict_match_poisson(home_raw: str, away_raw: str, temporada: Optional[str] = None):
    con = sqlite3.connect(DB_PATH)
    temporada_obj = temporada or _get_current_season(con)
    if not temporada_obj:
        con.close()
        raise RuntimeError("No hay temporada disponible en BD.")

    avg_home, avg_away, strengths = _team_strengths_multiseason(
        con,
        temporada_obj,
        max_seasons=3,
        decay=0.7,
        prior_k=5.0
    )

    home = norm_team(home_raw)
    away = norm_team(away_raw)

    def find_team(name):
        for t in strengths:
            if name.lower() in t.lower() or t.lower() in name.lower():
                return t
        return None

    h = find_team(home)
    a = find_team(away)

    if not h or not a:
        con.close()
        raise RuntimeError(f"No encuentro equipos en BD: {home_raw} vs {away_raw}")

    atk_h, def_h, atk_a, def_a = strengths[h]

    # Home advantage (+15%)
    home_adv = 1.15

    mu_home = max(0.05, avg_home * atk_h * def_a * home_adv)
    mu_away = max(0.05, avg_away * atk_a * def_h)

    grid = _grid_probs(mu_home, mu_away)

    con.close()
    return {
        "temporada": temporada_obj,
        "home": h,
        "away": a,
        "mu_home": mu_home,
        "mu_away": mu_away,
        "pred_score": f"{grid['best_score'][0]}-{grid['best_score'][1]}",
        "p_home": grid["p_home"],
        "p_draw": grid["p_draw"],
        "p_away": grid["p_away"],
    }