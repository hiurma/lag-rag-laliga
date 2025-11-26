# -*- coding: utf-8 -*-
from __future__ import annotations
import sqlite3, re
from pathlib import Path
from typing import Dict, Tuple, Optional
import math

DB_PATH = Path("data/laliga.sqlite")

# ---------- Normalización de equipos ----------
TEAM_ALIASES = {
    # LaLiga (ajusta o añade si tu BD usa otros nombres)
    "real madrid": "Real Madrid",
    "r. madrid": "Real Madrid",
    "realmadrid": "Real Madrid",
    "fc barcelona": "FC Barcelona",
    "barcelona": "FC Barcelona",
    "barça": "FC Barcelona",
    "atletico de madrid": "Atlético Madrid",
    "atlético de madrid": "Atlético Madrid",
    "atletico madrid": "Atlético Madrid",
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
    "celta": "Celta de Vigo",
    "celta de vigo": "Celta de Vigo",
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

def _get_current_season(con: sqlite3.Connection) -> Optional[str]:
    try:
        cur = con.cursor()
        # coge la mayor temporada alfabéticamente (funciona con "2025/2026")
        row = cur.execute("SELECT MAX(Temporada) FROM resultados").fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None

# ---------- Poisson util ----------
def _poisson_prob(lmbd: float, k: int) -> float:
    # P(X=k) para Poisson(λ)
    if lmbd <= 0:
        return 0.0 if k > 0 else 1.0
    return math.exp(-lmbd) * (lmbd ** k) / math.factorial(k)

def _grid_probs(mu_h: float, mu_a: float, max_goals: int = 6):
    grid = []
    best = (0, 0, 0.0)
    p_home = p_draw = p_away = 0.0
    for h in range(max_goals + 1):
        ph = _poisson_prob(mu_h, h)
        for a in range(max_goals + 1):
            pa = _poisson_prob(mu_a, a)
            p = ph * pa
            grid.append(((h, a), p))
            if p > best[2]:
                best = (h, a, p)
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
    return {
        "best_score": (best[0], best[1]),
        "best_prob": best[2],
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
    }

# ---------- Modelo: medias ataque/defensa ----------
def _team_strengths(con: sqlite3.Connection, temporada: str):
    cur = con.cursor()
    rows = cur.execute("""
        SELECT Local, Visitante, GolesLocal, GolesVisitante
        FROM resultados
        WHERE REPLACE(TRIM(Temporada), '-', '/') = REPLACE(TRIM(?), '-', '/')
    """, (temporada,)).fetchall()

    if not rows:
        return None

    teams = {}
    sum_home_goals = sum_away_goals = 0
    n_matches = 0

    for loc, vis, gl, gv in rows:
        n_matches += 1
        sum_home_goals += (gl or 0)
        sum_away_goals += (gv or 0)

        tloc = teams.setdefault(loc, {"home_scored":0, "home_conc":0, "home_played":0,
                                      "away_scored":0, "away_conc":0, "away_played":0})
        tvis = teams.setdefault(vis, {"home_scored":0, "home_conc":0, "home_played":0,
                                      "away_scored":0, "away_conc":0, "away_played":0})

        tloc["home_scored"] += (gl or 0)
        tloc["home_conc"]   += (gv or 0)
        tloc["home_played"] += 1

        tvis["away_scored"] += (gv or 0)
        tvis["away_conc"]   += (gl or 0)
        tvis["away_played"] += 1

    avg_home = sum_home_goals / max(n_matches,1)
    avg_away = sum_away_goals / max(n_matches,1)

    strengths = {}
    for team, st in teams.items():
        atk_h = (st["home_scored"]/st["home_played"]) / avg_home if st["home_played"] else 1.0
        def_h = (st["home_conc"]/st["home_played"])   / avg_away if st["home_played"] else 1.0
        atk_a = (st["away_scored"]/st["away_played"]) / avg_away if st["away_played"] else 1.0
        def_a = (st["away_conc"]/st["away_played"])   / avg_home if st["away_played"] else 1.0
        strengths[team] = (atk_h, def_h, atk_a, def_a)

    return avg_home, avg_away, strengths

def predict_match_poisson(home_raw: str, away_raw: str, temporada: Optional[str]=None):
    """
    Devuelve:
      {
        'temporada': '2025/2026',
        'home': 'Real Madrid',
        'away': 'FC Barcelona',
        'mu_home': 1.62, 'mu_away': 1.21,
        'pred_score': '2-1',
        'p_home': 0.45, 'p_draw': 0.27, 'p_away': 0.28
      }
    """
    con = sqlite3.connect(DB_PATH)
    if not temporada:
        temporada = _get_current_season(con)

    # fuerzas de equipos
    base = _team_strengths(con, temporada)
    if not base:
        con.close()
        raise RuntimeError(f"No hay resultados en BD para temporada {temporada}.")
    avg_home, avg_away, strengths = base

    # normalizamos nombres
    home = norm_team(home_raw)
    away = norm_team(away_raw)

    # intentar “match” tolerante si no están exactos
    def _closest(tname: str) -> Optional[str]:
        tname_n = tname.lower()
        # exacto
        if tname in strengths:
            return tname
        # coincide por “in”
        for k in strengths.keys():
            if tname_n in k.lower() or k.lower() in tname_n:
                return k
        return None

    h = _closest(home)
    a = _closest(away)
    if not h or not a:
        con.close()
        raise RuntimeError(f"No encuentro equipos en BD: '{home_raw}' vs '{away_raw}'")

    atk_h, def_h, atk_a, def_a = strengths[h]
    mu_home = max(0.05, avg_home * atk_h * def_a)
    mu_away = max(0.05, avg_away * atk_a * def_h)

    grid = _grid_probs(mu_home, mu_away, max_goals=6)
    s_h, s_a = grid["best_score"]

    out = {
        "temporada": temporada,
        "home": h,
        "away": a,
        "mu_home": mu_home,
        "mu_away": mu_away,
        "pred_score": f"{s_h}-{s_a}",
        "p_home": grid["p_home"],
        "p_draw": grid["p_draw"],
        "p_away": grid["p_away"],
    }
    con.close()
    return out