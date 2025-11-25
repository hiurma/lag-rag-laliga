# --- rag_sql.py --------------------------------
from __future__ import annotations

import sqlite3
from pathlib import Path
import re
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd

DB_PATH = Path("data/laliga.sqlite")

# ----------------- Helpers generales ----------------------------

def _run_sql(sql: str, params: tuple = ()) -> Tuple[List[str], List[tuple]]:
    """
    Ejecuta SQL sobre laliga.sqlite y devuelve (columnas, filas).
    Máximo 10 filas para no petar el response.
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(sql, params)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    con.close()
    return cols, rows[:10]


def _guess_temporada(q: str) -> Optional[str]:
    """
    Detecta la temporada en el texto.
    Busca patrones tipo 2024/2025 o 2025-2026.
    """
    m = re.search(r"(20\d{2})[/-](20\d{2})", q)
    if m:
        t1, t2 = m.group(1), m.group(2)
        return f"{t1}/{t2}"
    return None


def _clean_temporada_for_where(colname: str) -> str:
    """
    Devuelve una expresión SQL que iguala temporada ignorando '-' vs '/' y espacios.
    EJ: REPLACE(TRIM(Temporada), '-', '/') = REPLACE(TRIM(?), '-', '/')
    """
    return f"REPLACE(TRIM({colname}), '-', '/') = REPLACE(TRIM(?), '-', '/')"


def _simple_table_text(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "(sin filas)"
    try:
        return "\n" + df.to_string(index=False) + "\n"
    except Exception:
        return "\n" + "\n".join(str(r) for _, r in df.iterrows()) + "\n"


# ----------------- SQL "canónicos" para stats --------------------

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

    val_num = (
        "CAST(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(Valor,'€',''),'M€',''),"
        "'M',''),'m',''),' ','')"
        " AS REAL)"
    )

    sql = f"""
        SELECT
            Club,
            MAX({val_num}) AS Valor
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

    coste_num = (
        "CAST(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(coste,'€',''),'M€',''),"
        "'M',''),'m',''),' ','')"
        " AS REAL)"
    )

    sql = f"""
        SELECT
            Club,
            fichaje AS Fichaje,
            {coste_num} AS Coste
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
            Perdidos,
            GolesAF,
            GolesEC
        FROM clasificaciones
        WHERE {where}
        ORDER BY Puntos DESC
        LIMIT 10;
    """
    return sql, params, f"Top de la clasificación {temp or '(todas las temporadas)'}"


# ----------------- Helpers para PRONÓSTICOS ----------------------

def _parse_match_from_question(q_low: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Saca 'equipo local' y 'equipo visitante' de frases tipo:
    - 'pronostico real madrid vs fc barcelona 2025/2026 en el bernabeu'
    - 'pronostico girona contra real oviedo 2025/2026'
    MUY sencillo, pero suficiente para tu TFG.
    """
    sep = None
    if " vs " in q_low:
        sep = " vs "
    elif " contra " in q_low:
        sep = " contra "
    if not sep:
        return None, None

    left, right = q_low.split(sep, 1)

    # limpiamos palabras de la izquierda
    for token in ["pronostico del partido", "pronostico", "del partido", "partido", "de", "el", "la"]:
        left = left.replace(token, "")
    home = left.strip()

    # en la derecha, cortamos por temporada / estadio / etc
    cortes = [" en ", " para ", " de la temporada", " temporada "]
    cut_idx = len(right)
    for c in cortes:
        i = right.find(c)
        if i != -1 and i < cut_idx:
            cut_idx = i
    away = right[:cut_idx].strip()

    if not home or not away:
        return None, None
    return home, away


def _find_club_in_db(con: sqlite3.Connection, name_like: str) -> Optional[str]:
    """
    Busca un club cuyo nombre contenga name_like (case-insensitive) en
    clasificaciones y resultados. Devuelve el nombre tal cual está en la BD.
    """
    pat = f"%{name_like.strip()}%"
    cur = con.cursor()

    # clasificaciones
    for sql in [
        "SELECT DISTINCT Club FROM clasificaciones WHERE Club LIKE ? LIMIT 1",
        "SELECT DISTINCT Local AS Club FROM resultados WHERE Local LIKE ? LIMIT 1",
        "SELECT DISTINCT Visitante AS Club FROM resultados WHERE Visitante LIKE ? LIMIT 1",
    ]:
        cur.execute(sql, (pat,))
        row = cur.fetchone()
        if row and row[0]:
            return row[0]
    return None


def _team_season_stats(con: sqlite3.Connection, club: str, temporada: Optional[str]) -> Dict[str, Any]:
    """
    Devuelve un pequeño diccionario con stats de clasificaciones para ese club.
    """
    where = "Club = ?"
    params: List[Any] = [club]
    if temporada:
        where += f" AND {_clean_temporada_for_where('Temporada')}"
        params.append(temporada)

    df = pd.read_sql_query(f"SELECT * FROM clasificaciones WHERE {where}", con, params=params)
    if df.empty:
        return {"club": club}

    row = df.iloc[0]
    return {
        "club": club,
        "temporada": row.get("Temporada"),
        "puntos": int(row.get("Puntos", 0)),
        "ganados": int(row.get("Ganados", 0)),
        "empatados": int(row.get("Empatados", 0)),
        "perdidos": int(row.get("Perdidos", 0)),
        "goles_af": int(row.get("GolesAF", 0)),
        "goles_ec": int(row.get("GolesEC", 0)),
    }


def _build_prediction_context(pregunta: str) -> Dict[str, Any]:
    """
    Construye el contexto estadístico para un pronóstico:
    - identifica equipos
    - busca nombres reales en la BD
    - saca stats de clasificaciones
    """
    q_low = pregunta.lower()
    temporada = _guess_temporada(q_low)
    home_raw, away_raw = _parse_match_from_question(q_low)
    if not home_raw or not away_raw:
        return {
            "ok": False,
            "intent": "prediction",
            "error": "No pude extraer los equipos del texto.",
        }

    con = sqlite3.connect(DB_PATH)
    try:
        home_db = _find_club_in_db(con, home_raw) or home_raw.title()
        away_db = _find_club_in_db(con, away_raw) or away_raw.title()

        home_stats = _team_season_stats(con, home_db, temporada)
        away_stats = _team_season_stats(con, away_db, temporada)

        df = pd.DataFrame(
            [
                {
                    "Club": home_stats.get("club"),
                    "Temporada": home_stats.get("temporada", temporada or ""),
                    "Puntos": home_stats.get("puntos", ""),
                    "G": home_stats.get("ganados", ""),
                    "E": home_stats.get("empatados", ""),
                    "P": home_stats.get("perdidos", ""),
                    "GF": home_stats.get("goles_af", ""),
                    "GC": home_stats.get("goles_ec", ""),
                },
                {
                    "Club": away_stats.get("club"),
                    "Temporada": away_stats.get("temporada", temporada or ""),
                    "Puntos": away_stats.get("puntos", ""),
                    "G": away_stats.get("ganados", ""),
                    "E": away_stats.get("empatados", ""),
                    "P": away_stats.get("perdidos", ""),
                    "GF": away_stats.get("goles_af", ""),
                    "GC": away_stats.get("goles_ec", ""),
                },
            ]
        )

        context_md = _simple_table_text(df)
        context_txt = (
            f"Stats de la temporada {temporada or '(según datos disponibles)'}:\n"
            f"- {home_stats.get('club')} → {home_stats.get('puntos', '?')} puntos, "
            f"{home_stats.get('ganados', '?')}G-{home_stats.get('empatados', '?')}E-"
            f"{home_stats.get('perdidos', '?')}P, GF {home_stats.get('goles_af', '?')}, "
            f"GC {home_stats.get('goles_ec', '?')}.\n"
            f"- {away_stats.get('club')} → {away_stats.get('puntos', '?')} puntos, "
            f"{away_stats.get('ganados', '?')}G-{away_stats.get('empatados', '?')}E-"
            f"{away_stats.get('perdidos', '?')}P, GF {away_stats.get('goles_af', '?')}, "
            f"GC {away_stats.get('goles_ec', '?')}."
        )

        return {
            "ok": True,
            "intent": "prediction",
            "home_team": home_db,
            "away_team": away_db,
            "temporada": temporada,
            "context_markdown": context_md,
            "context_text": context_txt,
        }
    finally:
        con.close()


# ----------------- Router semántico básico ------------------------

def _pick_intent_for_stats(q: str):
    """
    Devuelve (sql, params, descripcion) según la intención
    para consultas ESTADÍSTICAS (no pronósticos).
    """
    q_low = q.lower()
    temp = _guess_temporada(q_low)

    if "goleador" in q_low or "goles" in q_low or "pichichi" in q_low:
        return _sql_top_goleadores(temp)

    if "valor" in q_low or "clubes más caros" in q_low or "club mas caro" in q_low:
        return _sql_top_valor_clubes(temp)

    if "fichaje" in q_low or "traspaso" in q_low or "transfer" in q_low:
        return _sql_top_fichajes(temp)

    if "resultado" in q_low or "marcador" in q_low or "partido" in q_low:
        return _sql_resultados(temp)

    if "clasific" in q_low or "tabla" in q_low or "puntos" in q_low or "liga" in q_low:
        return _sql_tabla_clasificacion(temp)

    # fallback: clasificación general
    return _sql_tabla_clasificacion(temp)


# ----------------- Función principal usada por FastAPI ------------

def ask_rag(pregunta: str) -> Dict[str, Any]:
    """
    Entrada única desde FastAPI / ChatAgent.

    - Si la pregunta es un PRONÓSTICO (contiene 'pronostic' y 'vs/contra'):
      -> devuelve intent='prediction' + contexto para el LLM.
    - Si es una consulta estadística:
      -> ejecuta SQL y devuelve tabla + resumen.
    """
    q = (pregunta or "").strip()
    q_low = q.lower()

    # 1) Pronósticos
    if ("pronostic" in q_low or "resultado probable" in q_low) and (
        " vs " in q_low or " contra " in q_low
    ):
        ctx = _build_prediction_context(q)
        # nos aseguramos de marcar el intent
        ctx.setdefault("intent", "prediction")
        return ctx

    # 2) Consultas estadísticas normales
    try:
        sql, params, descripcion = _pick_intent_for_stats(q)
        cols, rows = _run_sql(sql, params)

        df = pd.DataFrame(rows, columns=cols)

        if df.empty:
            resumen = f"No encontré datos para esa consulta. ({descripcion})"
        else:
            primera = df.iloc[0].to_dict()
            resumen = f"{descripcion}. Destaca: {primera}"

        return {
            "ok": True,
            "intent": "stats",
            "pregunta": pregunta,
            "descripcion": descripcion,
            "consulta": sql,
            "parametros": params,
            "columnas": cols,
            "resultados": rows,
            "resumen": resumen,
        }

    except Exception as e:
        return {
            "ok": False,
            "intent": "stats",
            "pregunta": pregunta,
            "error": str(e),
        }