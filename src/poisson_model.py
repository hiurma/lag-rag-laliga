# -*- coding: utf-8 -*-
from __future__ import annotations
import sqlite3, re, math
from pathlib import Path
from typing import Optional, Dict, Tuple

DB_PATH = Path("data/laliga.sqlite")

TEAM_ALIASES = {
    "real madrid": "Real Madrid",
    "realmadrid": "Real Madrid",
    "r. madrid": "Real Madrid",
    "fc barcelona": "FC Barcelona",
    "barcelona": "FC Barcelona",
    "barça": "FC Barcelona",
    "girona": "Girona FC",
    "girona fc": "Girona FC",
    "sevilla": "Sevilla FC",
    "sevilla fc": "Sevilla FC",
    "atletico madrid": "Atlético Madrid",
    "atlético de madrid": "Atlético Madrid",
    "atletico de madrid": "Atlético Madrid",
}

def norm_team(n: str) -> str:
    s = n.lower().strip()
    return TEAM_ALIASES.get(s, n.strip())


def _poisson(l, k):
    return math.exp(-l) * (l ** k) / math.factorial(k)


def _grid(mu_h, mu_a, maxg=6):
    best = (0, 0, 0.0)
    p_h = p_d = p_a = 0

    for h in range(maxg+1):
        for a in range(maxg+1):
            p = _poisson(mu_h, h) * _poisson(mu_a, a)
            if p > best[2]:
                best = (h, a, p)
            if h > a: p_h += p
            elif h == a: p_d += p
            else: p_a += p

    return {
        "best": best,
        "p_home": p_h,
        "p_draw": p_d,
        "p_away": p_a,
    }


def _get_current_season(con):
    row = con.execute("SELECT MAX(Temporada) FROM resultados").fetchone()
    return row[0] if row else None


def _get_strengths(con, temporada):
    rows = con.execute("""
        SELECT Local, Visitante, GolesLocal, GolesVisitante
        FROM resultados
        WHERE Temporada = ?
    """, (temporada,)).fetchall()

    if not rows:
        return None

    teams = {}
    total_h = total_a = 0
    n = 0

    for loc, vis, gl, gv in rows:
        if gl is None or gv is None: continue
        gl, gv = float(gl), float(gv)

        n += 1
        total_h += gl
        total_a += gv

        tloc = teams.setdefault(loc, {"h_s":0,"h_c":0,"h_p":0, "a_s":0,"a_c":0,"a_p":0})
        tvis = teams.setdefault(vis, {"h_s":0,"h_c":0,"h_p":0, "a_s":0,"a_c":0,"a_p":0})

        tloc["h_s"] += gl; tloc["h_c"] += gv; tloc["h_p"] += 1
        tvis["a_s"] += gv; tvis["a_c"] += gl; tvis["a_p"] += 1

    avg_h = total_h / n
    avg_a = total_a / n

    strengths = {}
    for t, d in teams.items():
        atk_h = (d["h_s"]/d["h_p"])/avg_h if d["h_p"] else 1
        def_h = (d["h_c"]/d["h_p"])/avg_a if d["h_p"] else 1
        atk_a = (d["a_s"]/d["a_p"])/avg_a if d["a_p"] else 1
        def_a = (d["a_c"]/d["a_p"])/avg_h if d["a_p"] else 1
        strengths[t] = (atk_h, def_h, atk_a, def_a)

    return avg_h, avg_a, strengths


def _get_elo(con, equipo, temporada, fallback):
    row = con.execute("""
        SELECT Elo FROM elo
        WHERE Equipo = ? AND Temporada = ?
        ORDER BY Jornada DESC LIMIT 1
    """, (equipo, temporada)).fetchone()
    return float(row[0]) if row else fallback


def _league_avg_elo(con, temporada):
    row = con.execute("SELECT AVG(Elo) FROM elo WHERE Temporada = ?", (temporada,)).fetchone()
    return float(row[0]) if row else 1500


def _elo_fac(e, avg):
    f = 1 + (e - avg) / 400
    return max(0.6, min(1.4, f))


def predict_match_poisson(home_raw, away_raw, temporada=None):
    con = sqlite3.connect(DB_PATH)
    try:
        if not temporada:
            temporada = _get_current_season(con)

        base = _get_strengths(con, temporada)
        if not base:
            raise RuntimeError("No hay datos de resultados.")

        avg_h, avg_a, strengths = base

        home = norm_team(home_raw)
        away = norm_team(away_raw)

        def closest(n):
            n2 = n.lower()
            for t in strengths:
                if n2 in t.lower():
                    return t
            return None

        h = closest(home)
        a = closest(away)

        atk_h, def_h, atk_a, def_a = strengths[h]
        atk2_h, def2_h, atk2_a, def2_a = strengths[a]

        mu_h = avg_h * atk_h * def2_h
        mu_a = avg_a * atk2_a * def_h

        avg_elo = _league_avg_elo(con, temporada)
        elo_h = _get_elo(con, h, temporada, avg_elo)
        elo_a = _get_elo(con, a, temporada, avg_elo)

        fac_h = _elo_fac(elo_h, avg_elo)
        fac_a = _elo_fac(elo_a, avg_elo)

        mu_h *= fac_h
        mu_a *= fac_a

        grid = _grid(mu_h, mu_a)

        s_h, s_a, _p = grid["best"]

        return {
            "temporada": temporada,
            "home": h,
            "away": a,
            "pred_score": f"{s_h}-{s_a}",
            "p_home": grid["p_home"],
            "p_draw": grid["p_draw"],
            "p_away": grid["p_away"],
            "elo_home": elo_h,
            "elo_away": elo_a,
        }
    finally:
        con.close()