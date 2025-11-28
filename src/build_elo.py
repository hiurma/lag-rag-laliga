from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Dict

DB_PATH = Path("data/laliga.sqlite")

BASE_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADV = 100.0


def _goal_factor(diff: int) -> float:
    d = abs(diff)
    if d <= 1: return 1.0
    if d == 2: return 1.5
    return 1.75


def build_elo():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS elo;
    CREATE TABLE elo (
        Equipo TEXT NOT NULL,
        Temporada TEXT NOT NULL,
        Jornada INTEGER NOT NULL,
        Elo REAL NOT NULL,
        PRIMARY KEY (Equipo, Temporada, Jornada)
    );
    """)

    temporadas = [
        row[0] for row in cur.execute(
            "SELECT DISTINCT Temporada FROM resultados ORDER BY Temporada"
        ).fetchall()
    ]

    for temporada in temporadas:
        print(f"🚀 Calculando Elo para {temporada}...")
        ratings: Dict[str, float] = {}

        partidos = cur.execute("""
            SELECT Jornada, Local, Visitante, GolesLocal, GolesVisitante
            FROM resultados
            WHERE Temporada = ?
            ORDER BY Jornada, rowid
        """, (temporada,)).fetchall()

        for j, loc, vis, gl, gv in partidos:
            if gl is None or gv is None:
                continue

            Ra = ratings.get(loc, BASE_ELO)
            Rb = ratings.get(vis, BASE_ELO)

            if gl > gv:
                Sa, Sb = 1, 0
            elif gl == gv:
                Sa, Sb = 0.5, 0.5
            else:
                Sa, Sb = 0, 1

            Ra_eff = Ra + HOME_ADV
            Ea = 1 / (1 + 10 ** ((Rb - Ra_eff) / 400))
            Eb = 1 - Ea

            G = _goal_factor(gl - gv)

            Ra_new = Ra + K_FACTOR * G * (Sa - Ea)
            Rb_new = Rb + K_FACTOR * G * (Sb - Eb)

            ratings[loc] = Ra_new
            ratings[vis] = Rb_new

            cur.execute("INSERT OR REPLACE INTO elo VALUES (?, ?, ?, ?)", (loc, temporada, j, Ra_new))
            cur.execute("INSERT OR REPLACE INTO elo VALUES (?, ?, ?, ?)", (vis, temporada, j, Rb_new))

        con.commit()

    con.close()
    print("✅ Elo generado correctamente.")


if __name__ == "__main__":
    build_elo()