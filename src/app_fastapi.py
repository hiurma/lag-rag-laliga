from __future__ import annotations

import os
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

import sqlite3
from pathlib import Path
import pandas as pd

# 👇 Backend de chat (SQL + Poisson + LLM)
try:
    # Cuando se ejecuta como paquete: `python -m src.app_fastapi`
    from .chat_agent import ChatAgent  # type: ignore[attr-defined]
except ImportError:
    # Cuando se ejecuta directamente: `python src/app_fastapi.py`
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent))
    from chat_agent import ChatAgent

# 👇 Para gráficos
import matplotlib
matplotlib.use("Agg")  # backend sin interfaz gráfica (para Render)
import matplotlib.pyplot as plt
import io

# ---------------------------------------------------------------------
# Configuración básica
# ---------------------------------------------------------------------

app = FastAPI(title="LaLiga RAG API", version="1.1")

# CORS – GitHub Pages + comodín
origins: List[str] = [
    "https://hiurma.github.io",
    "https://hiurma.github.io/",
    "https://hiurma.github.io/laliga-chat-web",
    "https://hiurma.github.io/laliga-chat-web/",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = ChatAgent()
DB_PATH = Path("data/laliga.sqlite")


# ---------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------

class ChatBody(BaseModel):
    pregunta: str


# ---------------------------------------------------------------------
# Endpoints básicos
# ---------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "LaLiga RAG API está viva.",
        "endpoints": [
            "/health",
            "/chat",
            "/plot/puntos_equipo",
            "/plot/top_goleadores",
        ],
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat_endpoint(body: ChatBody):
    q = (body.pregunta or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La 'pregunta' no puede estar vacía")

    try:
        return agent.chat_query(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# 📊 ENDPOINT DE VISUALIZACIÓN
# =====================================================
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

@app.get("/visual")
def generar_visual():
    try:
        # Conexión a la BD
        con = sqlite3.connect("data/laliga.sqlite")
        df = pd.read_sql_query("""
            SELECT Club, Puntos
            FROM clasificaciones
            ORDER BY Puntos DESC
            LIMIT 10
        """, con)
        con.close()

        # Crear gráfica
        plt.figure(figsize=(8,5))
        plt.bar(df["Club"], df["Puntos"])
        plt.xticks(rotation=45, ha="right")
        plt.title("Top Clasificación (Visualización Automática)")
        plt.tight_layout()

        # Convertir a base64
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {"ok": True, "image": img_b64}

    except Exception as e:
        return {"ok": False, "error": str(e)}
# ---------------------------------------------------------------------
# Helpers internos para leer de SQLite
# ---------------------------------------------------------------------

def _connect_db():
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail=f"No encuentro la BD en {DB_PATH}")
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------------------
# 1) Gráfico: evolución de puntos de un equipo
# ---------------------------------------------------------------------

@app.get("/plot/puntos_equipo")
def plot_puntos_equipo(
    equipo: str = Query(..., description="Nombre del equipo, p.ej. 'Real Madrid'"),
    temporada: Optional[str] = Query(None, description="Temporada '2025/2026' (opcional)"),
):
    """
    Devuelve un PNG con la evolución de puntos acumulados de un equipo
    en una temporada (calculado a partir de la tabla 'resultados').
    """
    con = _connect_db()
    cur = con.cursor()

    # Normalizamos temporada si viene tipo 2025-2026
    if temporada:
        temp_norm = temporada.replace("-", "/").strip()
        temp_filter = """
            REPLACE(TRIM(Temporada), '-', '/') = REPLACE(TRIM(?), '-', '/')
        """
        params = (temp_norm, equipo, equipo)
    else:
        # Si no hay temporada, cogemos la máxima disponible
        row = cur.execute("SELECT MAX(Temporada) FROM resultados").fetchone()
        if not row or not row[0]:
            con.close()
            raise HTTPException(status_code=500, detail="No hay temporadas en la tabla resultados")
        temp_norm = row[0]
        temp_filter = """
            REPLACE(TRIM(Temporada), '-', '/') = REPLACE(TRIM(?), '-', '/')
        """
        params = (temp_norm, equipo, equipo)

    sql = f"""
        SELECT Jornada, Local, Visitante, GolesLocal, GolesVisitante
        FROM resultados
        WHERE {temp_filter}
          AND (Local = ? OR Visitante = ?)
        ORDER BY Jornada ASC;
    """
    rows = cur.execute(sql, params).fetchall()
    con.close()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No encontré partidos de '{equipo}' en la temporada {temp_norm}",
        )

    jornadas = []
    puntos_acum = []
    puntos_total = 0

    for jornada, local, visitante, gl, gv in rows:
        gl = gl or 0
        gv = gv or 0
        # ¿juega como local o visitante?
        if local == equipo:
            if gl > gv:
                puntos_total += 3
            elif gl == gv:
                puntos_total += 1
        elif visitante == equipo:
            if gv > gl:
                puntos_total += 3
            elif gv == gl:
                puntos_total += 1

        jornadas.append(jornada)
        puntos_acum.append(puntos_total)

    # --- Matplotlib: generamos PNG en memoria ---
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(jornadas, puntos_acum, marker="o")
    ax.set_title(f"Puntos acumulados - {equipo} ({temp_norm})")
    ax.set_xlabel("Jornada")
    ax.set_ylabel("Puntos acumulados")
    ax.grid(True, linestyle="--", alpha=0.4)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------
# 2) Gráfico: Top 10 goleadores
# ---------------------------------------------------------------------

@app.get("/plot/top_goleadores")
def plot_top_goleadores(
    temporada: Optional[str] = Query(None, description="Temporada '2025/2026' (opcional)"),
):
    """
    Devuelve un PNG con un gráfico de barras de los 10 máximos goleadores
    en una temporada (tabla 'goleadores').
    """
    con = _connect_db()
    cur = con.cursor()

    if temporada:
        temp_norm = temporada.replace("-", "/").strip()
        where = "REPLACE(TRIM(Temporada), '-', '/') = REPLACE(TRIM(?), '-', '/')"
        params = (temp_norm,)
    else:
        row = cur.execute("SELECT MAX(Temporada) FROM goleadores").fetchone()
        if not row or not row[0]:
            con.close()
            raise HTTPException(status_code=500, detail="No hay datos en goleadores")
        temp_norm = row[0]
        where = "REPLACE(TRIM(Temporada), '-', '/') = REPLACE(TRIM(?), '-', '/')"
        params = (temp_norm,)

    sql = f"""
        SELECT Jugador, Club, SUM(Goles) AS Goles
        FROM goleadores
        WHERE {where}
        GROUP BY Jugador, Club
        ORDER BY Goles DESC
        LIMIT 10;
    """
    rows = cur.execute(sql, params).fetchall()
    con.close()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No encontré goleadores para la temporada {temp_norm}",
        )

    df = pd.DataFrame(rows, columns=["Jugador", "Club", "Goles"])
    # Etiqueta tipo "Jugador (Club)"
    df["Etiqueta"] = df["Jugador"] + " (" + df["Club"] + ")"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["Etiqueta"], df["Goles"])
    ax.invert_yaxis()
    ax.set_title(f"Top 10 goleadores - {temp_norm}")
    ax.set_xlabel("Goles")

    for i, v in enumerate(df["Goles"]):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")

