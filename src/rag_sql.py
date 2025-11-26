from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd

DB_PATH = Path("data/laliga.sqlite")


# ---------------------------------------------------------------------
# Helpers básicos
# ---------------------------------------------------------------------
def _run_sql(sql: str, params: tuple = ()) -> Tuple[List[str], List[tuple]]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(sql, params)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    con.close()
    return cols, rows[:10]


def _get_last_season() -> Optional[str]:
    """
    Devuelve la última temporada disponible en la BD (clasificaciones),
    por ejemplo '2025/2026'.
    """
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        row = cur.execute("SELECT MAX(Temporada) FROM clasificaciones").fetchone()
        con.close()
        if row and row[0]:
            return row[0]
    except Exception:
        return None
    return None


def _guess_temporada(q: str) -> Optional[str]:
    q_low = q.lower()

    # 1) Si viene explícita en el texto
    m = re.search(r"(20\d{2})\s*[-/]\s*(20\d{2})", q_low)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    # 2) Si dice "actual", "ahora", "esta temporada" → última temporada en BD
    if any(k in q_low for k in ["actual", "ahora", "esta temporada", "hoy"]):
        return _get_last_season()

    return None


def _clean_temporada_for_where(colname: str) -> str:
    return f"REPLACE(TRIM({colname}), '-', '/') = REPLACE(TRIM(?), '-', '/')"


# ---------------------------------------------------------------------
# Plantillas SQL
# ---------------------------------------------------------------------
def _sql_top_goleadores(temp: Optional[str]):
    where = "1=1"
    params: tuple = ()
    if temp:
        where = _clean_temporada_for_where("Temporada")
        params = (temp,)
    sql = f"""
        SELECT
            Jugador,
            Club,
            MAX(Goles) AS Goles
        FROM goleadores
        WHERE {where}
        GROUP BY Jugador, Club
        ORDER BY Goles DESC
        LIMIT 10;
    """
    return sql, params, f"Top goleadores {temp or '(todas las temporadas)'}"


def _sql_top_valor_clubes(temp: Optional[str]):
    where = "1=1"
    params: tuple = ()
    if temp:
        where = _clean_temporada_for_where("Temporada")
        params = (temp,)

    sql = f"""
        SELECT
            Club,
            MAX(Valor) AS Valor
        FROM valor_clubes
        WHERE {where}
        GROUP BY Club
        ORDER BY Valor DESC
        LIMIT 10;
    """
    return sql, params, f"Clubes con más valor {temp or '(todas las temporadas)'}"


def _sql_top_fichajes(temp: Optional[str]):
    where = "1=1"
    params: tuple = ()
    if temp:
        where = _clean_temporada_for_where("Temporada")
        params = (temp,)

    sql = f"""
        SELECT
            Club,
            fichaje AS Fichaje,
            coste AS Coste
        FROM fichajes
        WHERE {where}
        ORDER BY Coste DESC
        LIMIT 10;
    """
    return sql, params, f"Fichajes más caros {temp or '(todas las temporadas)'}"


def _sql_resultados(temp: Optional[str]):
    where = "1=1"
    params: tuple = ()
    if temp:
        where = _clean_temporada_for_where("Temporada")
        params = (temp,)

    sql = f"""
        SELECT
            Jornada,
            Local,
            Visitante,
            Marcador
        FROM resultados
        WHERE {where}
        ORDER BY Jornada, Local, Visitante
        LIMIT 10;
    """
    return sql, params, f"Ejemplos de resultados {temp or '(todas las temporadas)'}"


def _sql_tabla_clasificacion(temp: Optional[str]):
    where = "1=1"
    params: tuple = ()
    if temp:
        where = _clean_temporada_for_where("Temporada")
        params = (temp,)

    sql = f"""
        SELECT
            Club,
            Puntos,
            Ganados,
            Empatados,
            Perdidos
        FROM clasificaciones
        WHERE {where}
        ORDER BY Puntos DESC
        LIMIT 10;
    """
    return sql, params, f"Top de la clasificación {temp or '(todas las temporadas)'}"


# ---------------------------------------------------------------------
# Router de intención muy sencillo
# ---------------------------------------------------------------------
def _pick_intent(q: str):
    q_low = q.lower()
    temp = _guess_temporada(q_low)

    if "pichichi" in q_low or "goleador" in q_low or "goles" in q_low:
        return _sql_top_goleadores(temp)

    if "valor" in q_low or "clubes mas caros" in q_low or "clubes más caros" in q_low:
        return _sql_top_valor_clubes(temp)

    if "fichaje" in q_low or "traspaso" in q_low or "transfer" in q_low:
        return _sql_top_fichajes(temp)

    if "resultado" in q_low or "marcador" in q_low or "partido" in q_low:
        return _sql_resultados(temp)

    if "clasific" in q_low or "tabla" in q_low or "puntos" in q_low or "liga" in q_low:
        return _sql_tabla_clasificacion(temp)

    # fallback: clasificación (última temporada si podemos)
    return _sql_tabla_clasificacion(temp)


# ---------------------------------------------------------------------
# Punto de entrada RAG SQL
# ---------------------------------------------------------------------
def ask_rag(pregunta: str) -> Dict[str, Any]:
    try:
        sql, params, descripcion = _pick_intent(pregunta)
        cols, rows = _run_sql(sql, params)

        df = pd.DataFrame(rows, columns=cols)

        if df.empty:
            resumen = f"No encontré datos para esa consulta en la base local. ({descripcion})"
        else:
            first = df.iloc[0].to_dict()
            resumen = f"{descripcion}. Destaca: {first}"

        return {
            "ok": True,
            "pregunta": pregunta,
            "descripcion": descripcion,
            "consulta": sql,
            "parametros": params,
            "columnas": cols,
            "resultados": rows,
            "resumen": resumen,
        }

    except Exception as e:
        return {"ok": False, "pregunta": pregunta, "error": str(e)}