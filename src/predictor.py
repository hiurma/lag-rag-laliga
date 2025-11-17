# predictor.py
from pathlib import Path
from dataclasses import dataclass
from itertools import product
from collections import defaultdict
import sqlite3
import math
import numpy as np
import joblib

DB_PATH    = Path("data/laliga.sqlite")
MODEL_PATH = Path("models/predictor_v1.pkl")

# --- parámetros globales (ajustables) ---
LEAGUE_GF_AVG = 2.70                 # goles totales por partido en LaLiga aprox
BASE_LAMBDA   = LEAGUE_GF_AVG / 2.0  # ~1.35 por equipo
HOME_ADV_EPS  = 0.15                 # ventaja local en goles esperados
MIN_LAMBDA    = 0.65                 # recorte bajo
MAX_LAMBDA    = 2.60                 # recorte alto
MIX_ALPHA     = 0.45                 # mezcla modelo vs heurística (0.45 = 45% modelo)

def _clip(x, lo=MIN_LAMBDA, hi=MAX_LAMBDA):
    return float(max(lo, min(hi, float(x))))

def _safe(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

@dataclass
class PredOut:
    lambda_home: float
    lambda_away: float
    used_model: bool
    meta: dict

# ------------------------- SQL helpers -------------------------

def _avg_goals(con, team_id: str, temporada: str | None, side: str, last_n: int = 10):
    cur = con.cursor()
    if side == "home":
        sql = """
            SELECT GolesLocal, GolesVisitante
            FROM resultados
            WHERE Local = ? AND (? IS NULL OR REPLACE(Temporada,'-','/') = REPLACE(?,'-','/'))
            ORDER BY Jornada DESC
            LIMIT ?;
        """
    else:
        sql = """
            SELECT GolesVisitante, GolesLocal
            FROM resultados
            WHERE Visitante = ? AND (? IS NULL OR REPLACE(Temporada,'-','/') = REPLACE(?,'-','/'))
            ORDER BY Jornada DESC
            LIMIT ?;
        """
    rows = cur.execute(sql, (team_id, temporada, temporada, last_n)).fetchall()
    if not rows:
        return 1.0, 1.0  # fallback neutral
    gf = np.array([r[0] for r in rows], dtype=float)
    ga = np.array([r[1] for r in rows], dtype=float)
    return float(np.nanmean(gf)), float(np.nanmean(ga))

def _team_value(con, team_id: str, temporada: str | None):
    cur = con.cursor()
    sql = """
        SELECT AVG(Valor) FROM valor_clubes
        WHERE Club = ? AND (? IS NULL OR REPLACE(Temporada,'-','/') = REPLACE(?,'-','/'))
    """
    v = cur.execute(sql, (team_id, temporada, temporada)).fetchone()[0]
    return float(v or 0.0)

def _sum_signings(con, team_id: str, temporada: str | None):
    cur = con.cursor()
    sql = """
        SELECT SUM(coste) FROM fichajes
        WHERE Club = ? AND (? IS NULL OR REPLACE(Temporada,'-','/') = REPLACE(?,'-','/'))
    """
    v = cur.execute(sql, (team_id, temporada, temporada)).fetchone()[0]
    return float(v or 0.0)

def _league_avg_goals(con, temporada: str | None):
    cur = con.cursor()
    sql = """
        SELECT AVG(GolesLocal), AVG(GolesVisitante)
        FROM resultados WHERE (? IS NULL OR REPLACE(Temporada,'-','/') = REPLACE(?,'-','/'));
    """
    r = cur.execute(sql, (temporada, temporada)).fetchone()
    hl, aw = r if r else (1.35, 1.35)
    return float(hl or 1.35), float(aw or 1.35)

# ------------------------- señales de fuerza -------------------------

def _team_strength(con, team_id: str, temporada: str | None) -> float:
    """
    Fuerza ~ combinación simple de valor de club y gasto en fichajes,
    normalizados contra la media de la temporada.
    1.0 = media de la liga.
    """
    val = _team_value(con, team_id, temporada)
    spend = _sum_signings(con, team_id, temporada)

    # medias de liga para normalizar (evita explotar escalas absolutas)
    cur = con.cursor()
    if temporada:
        avg_val = cur.execute(
            "SELECT AVG(Valor) FROM valor_clubes WHERE REPLACE(Temporada,'-','/')=REPLACE(?,'-','/')",
            (temporada,)
        ).fetchone()[0] or 1.0
        avg_sp  = cur.execute(
            "SELECT AVG(coste) FROM fichajes WHERE REPLACE(Temporada,'-','/')=REPLACE(?,'-','/')",
            (temporada,)
        ).fetchone()[0] or 1.0
    else:
        avg_val = cur.execute("SELECT AVG(Valor) FROM valor_clubes").fetchone()[0] or 1.0
        avg_sp  = cur.execute("SELECT AVG(coste) FROM fichajes").fetchone()[0] or 1.0

    s_val = _safe(val)   / _safe(avg_val,1.0)
    s_sp  = _safe(spend) / _safe(avg_sp,1.0)

    # mezcla y límites
    strength = 0.7*s_val + 0.3*s_sp
    return max(0.7, min(1.6, strength if strength>0 else 1.0))

def _heuristic_lambdas(con, home: str, away: str, temporada: str | None):
    # medias recientes (ataque/defensa)
    h_gf, h_ga = _avg_goals(con, home, temporada, "home", last_n=10)
    a_gf, a_ga = _avg_goals(con, away, temporada, "away", last_n=10)
    lig_h, lig_a = _league_avg_goals(con, temporada)

    # prior + ventaja local
    prior_h = BASE_LAMBDA + HOME_ADV_EPS/2.0
    prior_a = BASE_LAMBDA - HOME_ADV_EPS/2.0

    # señal por perfiles (escala relativa a medias de liga)
    sig_h = prior_h * (_safe(h_gf, BASE_LAMBDA)/_safe(lig_h, BASE_LAMBDA)) * (_safe(a_ga, BASE_LAMBDA)/_safe(lig_a, BASE_LAMBDA))
    sig_a = prior_a * (_safe(a_gf, BASE_LAMBDA)/_safe(lig_a, BASE_LAMBDA)) * (_safe(h_ga, BASE_LAMBDA)/_safe(lig_h, BASE_LAMBDA))

    # empuje por “fuerzas” económico-deportivas
    s_home = _team_strength(con, home, temporada)
    sig_h = (1 + 0.25*(s_home - 1.0))
    sig_a = (1 - 0.15*(s_home - 1.0))

    lam_h = _clip(sig_h)
    lam_a = _clip(sig_a)

    return lam_h, lam_a, {
        "h_gf":h_gf, "h_ga":h_ga, "a_gf":a_gf, "a_ga":a_ga,
        "lig_h":lig_h, "lig_a":lig_a,
        "strength_home":s_home
    }

# ------------------------- pública -------------------------

def predict_match(home_id: str, away_id: str, temporada: str | None = None) -> PredOut:
    con = sqlite3.connect(DB_PATH)
    try:
        # 1) heurística robusta
        h_lam, a_lam, h_meta = _heuristic_lambdas(con, home_id, away_id, temporada)

        # 2) si existe, mezclamos con modelo ML (misma escala de salida: lambdas)
        used_model = False
        lam_h, lam_a = h_lam, a_lam

        if MODEL_PATH.exists():
            try:
                model = joblib.load(MODEL_PATH)  # admite dict con dos modelos o uno único
                if isinstance(model, dict) and "m_home" in model and "m_away" in model:
                    m_home, m_away = model["m_home"], model["m_away"]
                else:
                    m_home = m_away = model

                feats = {
                    "home_gf10": _avg_goals(con, home_id, temporada, "home", 10)[0],
                    "home_ga10": _avg_goals(con, home_id, temporada, "home", 10)[1],
                    "away_gf10": _avg_goals(con, away_id, temporada, "away", 10)[0],
                    "away_ga10": _avg_goals(con, away_id, temporada, "away", 10)[1],
                    "home_valor": _team_value(con, home_id, temporada),
                    "away_valor": _team_value(con, away_id, temporada),
                    "home_fichajes": _sum_signings(con, home_id, temporada),
                    "away_fichajes": _sum_signings(con, away_id, temporada),
                }
                X = np.array([[feats[k] for k in feats]], dtype=float)
                pred_h = _clip(m_home.predict(X)[0])
                pred_a = _clip(m_away.predict(X)[0])

                lam_h = _clip(MIX_ALPHA*pred_h + (1-MIX_ALPHA)*h_lam)
                lam_a = _clip(MIX_ALPHA*pred_a + (1-MIX_ALPHA)*a_lam)
                used_model = True
            except Exception:
                used_model = False
                lam_h, lam_a = h_lam, a_lam

        meta = {
            "heur": h_meta,
            "mix_alpha": MIX_ALPHA,
            "home_adv": HOME_ADV_EPS
        }
        return PredOut(lambda_home=lam_h, lambda_away=lam_a, used_model=used_model, meta=meta)
    finally:
        con.close()