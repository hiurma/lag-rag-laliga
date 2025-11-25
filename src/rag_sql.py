import os
import sqlite3
from pathlib import Path
import re
import pandas as pd

DB_PATH = Path("data/laliga.sqlite")

# ----------------------------------------------------
# SQL helper
# ----------------------------------------------------
def _run_sql(sql: str, params: tuple = ()):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(sql, params)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    con.close()
    return cols, rows

# ----------------------------------------------------
# Detecta temporada en texto
# ----------------------------------------------------
def _guess_temporada(text: str):
    m = re.search(r"(20\d{2})[\/-](20\d{2})", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None

def _norm_team(t):
    return (t or "").lower().replace("fc ", "").replace("cd ", "").strip()

# ----------------------------------------------------
# Motor de pronóstico
# ----------------------------------------------------
def _predict_match(local: str, visitante: str, temporada: str | None):
    """
    Genera predicción solo si la base de datos tiene datos de goles, forma,
    o resultados históricos entre ambos equipos.
    """
    local_n = _norm_team(local)
    vis_n = _norm_team(visitante)

    # Buscar historial entre ambos equipos
    sql = """
        SELECT Local, Visitante, Marcador
        FROM resultados
    """
    cols, rows = _run_sql(sql)

    historial = []
    for L, V, M in rows:
        if _norm_team(L) == local_n and _norm_team(V) == vis_n:
            historial.append(M)

    # Si no hay historial → probabilidad genérica
    if not historial:
        p_local = 0.40
        p_empate = 0.30
        p_visitante = 0.30
    else:
        # Extraer goles
        goles_local = []
        goles_visit = []
        for m in historial:
            parts = m.split("-")
            if len(parts) == 2:
                try:
                    gL = int(parts[0].strip())
                    gV = int(parts[1].strip())
                    goles_local.append(gL)
                    goles_visit.append(gV)
                except:
                    pass

        if len(goles_local) == 0:
            p_local = 0.40
            p_empate = 0.30
            p_visitante = 0.30
        else:
            import numpy as np
            avg_L = np.mean(goles_local)
            avg_V = np.mean(goles_visit)

            total = avg_L + avg_V + 1e-6

            p_local = (avg_L / total)
            p_visitante = (avg_V / total)
            p_empate = 1 - (p_local + p_visitante)

    # Marcador esperado
    gL = round(p_local * 3)
    gV = round(p_visitante * 3)

    marcador = f"{gL} - {gV}"

    return {
        "pronostico": marcador,
        "p_local": round(p_local * 100, 1),
        "p_empate": round(p_empate * 100, 1),
        "p_visitante": round(p_visitante * 100, 1),
    }

# ----------------------------------------------------
# Router principal SQL+Pronósticos
# ----------------------------------------------------
def ask_rag(pregunta: str):
    q = pregunta.lower()

    # Detecta partido: “girona vs sevilla”
    m = re.search(r"(\w[\w\s]+)\s+vs\.?\s+(\w[\w\s]+)", q)
    temporada = _guess_temporada(q)

    if m:
        t1 = m.group(1).strip()
        t2 = m.group(2).strip()

        pred = _predict_match(t1, t2, temporada)

        return {
            "ok": True,
            "tipo": "pronostico",
            "local": t1,
            "visitante": t2,
            "temporada": temporada,
            "pronostico": pred["pronostico"],
            "probabilidades": {
                "local": pred["p_local"],
                "empate": pred["p_empate"],
                "visitante": pred["p_visitante"],
            }
        }

    # Si no es un partido, devuelve None → lo procesa ChatAgent
    return None