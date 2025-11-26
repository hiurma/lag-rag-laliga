import sqlite3
from pathlib import Path
import math

DB_PATH = Path("data/laliga.sqlite")

def poisson_probability(lmbda, k):
    """Distribución de Poisson P(k; lambda)."""
    return (lmbda**k) * math.exp(-lmbda) / math.factorial(k)


def get_team_stats(team_name: str):
    """
    Recupera estadísticas agregadas de goles anotados y recibidos
    desde la tabla 'resultados'.
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # goles anotados jugando como local
    cur.execute("""
        SELECT SUM(CAST(SUBSTR(Marcador, 1, INSTR(Marcador, '-') - 1) AS INT)),
               COUNT(*)
        FROM resultados
        WHERE Local = ?
    """, (team_name,))
    g_local, p_local = cur.fetchone()

    # goles anotados jugando como visitante
    cur.execute("""
        SELECT SUM(CAST(SUBSTR(Marcador, INSTR(Marcador, '-') + 1) AS INT)),
               COUNT(*)
        FROM resultados
        WHERE Visitante = ?
    """, (team_name,))
    g_visitante, p_visitante = cur.fetchone()

    # goles recibidos como local
    cur.execute("""
        SELECT SUM(CAST(SUBSTR(Marcador, INSTR(Marcador, '-') + 1) AS INT))
        FROM resultados
        WHERE Local = ?
    """, (team_name,))
    g_rec_local = cur.fetchone()[0]

    # goles recibidos como visitante
    cur.execute("""
        SELECT SUM(CAST(SUBSTR(Marcador, 1, INSTR(Marcador, '-') - 1) AS INT))
        FROM resultados
        WHERE Visitante = ?
    """, (team_name,))
    g_rec_visitante = cur.fetchone()[0]

    con.close()

    # evitar divisiones por cero
    g_local = g_local or 0
    g_visitante = g_visitante or 0
    g_rec_local = g_rec_local or 0
    g_rec_visitante = g_rec_visitante or 0
    p_local = p_local or 1
    p_visitante = p_visitante or 1

    return {
        "gf": g_local + g_visitante,
        "gc": g_rec_local + g_rec_visitante,
        "pj": p_local + p_visitante,
    }


def predict_match_poisson(team_home: str, team_away: str):
    """
    Predice un marcador usando modelo Poisson.
    """

    home = get_team_stats(team_home)
    away = get_team_stats(team_away)

    # medias
    avg_gf_home = home["gf"] / home["pj"]
    avg_gf_away = away["gf"] / away["pj"]
    avg_gc_home = home["gc"] / home["pj"]
    avg_gc_away = away["gc"] / away["pj"]

    # intensidad
    home_attack = avg_gf_home
    home_defense = avg_gc_home
    away_attack = avg_gf_away
    away_defense = avg_gc_away

    home_advantage = 1.15  # típico en LaLiga

    # λ (goles esperados)
    lambda_home = home_attack * away_defense * home_advantage
    lambda_away = away_attack * home_defense

    # probabilidades de goles 0-5
    probs_home = [poisson_probability(lambda_home, i) for i in range(6)]
    probs_away = [poisson_probability(lambda_away, i) for i in range(6)]

    # matriz de probabilidades
    prob_matrix = []
    for i in range(6):  # goles local
        for j in range(6):  # goles visitante
            prob_matrix.append(((i, j), probs_home[i] * probs_away[j]))

    # score más probable
    best_score, best_prob = max(prob_matrix, key=lambda x: x[1])

    # probabilidades generales
    prob_local = sum(p for (i, j), p in prob_matrix if i > j)
    prob_empate = sum(p for (i, j), p in prob_matrix if i == j)
    prob_visitante = sum(p for (i, j), p in prob_matrix if i < j)

    return {
        "xg_local": lambda_home,
        "xg_visitante": lambda_away,
        "marcador_mas_probable": best_score,
        "probabilidad_marcador": best_prob,
        "prob_local": prob_local,
        "prob_empate": prob_empate,
        "prob_visitante": prob_visitante,
    }