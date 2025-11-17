import os
import sqlite3
from pathlib import Path
import re
import pandas as pd

DB_PATH = Path("data/laliga.sqlite")

# --- Helpers ------------------------

def _run_sql(sql: str, params: tuple = ()) -> tuple[list[str], list[tuple]]:
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


def _guess_temporada(q: str) -> str | None:
    """
    Heurística simple para detectar la temporada que pregunta el usuario.
    Busca patrones tipo 2024/2025 o 2025-2026 en el texto.
    Si no encuentra nada, devuelve None.
    """
    # 2024/2025 ó 2024-2025
    m = re.search(r"(20\d{2})[/-](20\d{2})", q)
    if m:
        t1, t2 = m.group(1), m.group(2)
        return f"{t1}/{t2}"
    # a veces dirás "2025/2026" literal con slash; lo pillamos arriba
    return None


def _clean_temporada_for_where(colname: str) -> str:
    """
    Devuelve una expresión SQL que iguala temporada ignorando '-' vs '/' y espacios.
    EJ: REPLACE(TRIM(Temporada), '-', '/') = REPLACE(TRIM(?), '-', '/')
    """
    return f"REPLACE(TRIM({colname}), '-', '/') = REPLACE(TRIM(?), '-', '/')"


# --- Plantillas de consultas "canónicas" ------------------------

def _sql_top_goleadores(temp: str | None):
    where = "1=1"
    params = ()
    if temp:
        where = _clean_temporada_for_where("Temporada")
        params = (temp,)
    # usamos MAX(Goles) por jugador por si hay varias filas
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


def _sql_top_valor_clubes(temp: str | None):
    where = "1=1"
    params = ()
    if temp:
        where = _clean_temporada_for_where("Temporada")
        params = (temp,)

    # Convertimos texto tipo "€120.5M" o "120,5" a número REAL
    val_num = (
        "CAST(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(Valor,'€',''),'M€',''),'M',''),'m',''),' ','')"
        ",',','.') AS REAL)"
    )

    sql = f"""
        SELECT
            Club,
            MAX({val_num}) AS Valor_Millones
        FROM valor_clubes
        WHERE {where}
        GROUP BY Club
        ORDER BY Valor_Millones DESC
        LIMIT 10;
    """
    return sql, params, f"Clubes con más valor {temp or '(todas las temporadas)'}"


def _sql_top_fichajes(temp: str | None):
    where = "1=1"
    params = ()
    if temp:
        where = _clean_temporada_for_where("Temporada")
        params = (temp,)

    coste_num = (
        "CAST(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(coste,'€',''),'M€',''),'M',''),'m',''),' ','')"
        ",',','.') AS REAL)"
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


def _sql_resultados(temp: str | None):
    where = "1=1"
    params = ()
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


def _sql_tabla_clasificacion(temp: str | None):
    where = "1=1"
    params = ()
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


# --- Router semántico muy simple ------------------------

def _pick_intent(q: str):
    """
    Devuelve qué tipo de info pide la pregunta.
    Miramos palabras clave muy básicas.
    """
    q_low = q.lower()

    temp = _guess_temporada(q_low)

    if "goleador" in q_low or "goles" in q_low:
        return _sql_top_goleadores(temp)

    if "valor" in q_low or "clubes más caros" in q_low or "club mas caro" in q_low:
        return _sql_top_valor_clubes(temp)

    if "fichaje" in q_low or "traspaso" in q_low or "transfer" in q_low:
        return _sql_top_fichajes(temp)

    if "resultado" in q_low or "marcador" in q_low or "partido" in q_low:
        return _sql_resultados(temp)

    if "clasific" in q_low or "tabla" in q_low or "puntos" in q_low or "liga" in q_low:
        return _sql_tabla_clasificacion(temp)

    # fallback: por defecto saco tabla de clasificación
    return _sql_tabla_clasificacion(temp)


# --- Respuesta principal usada por FastAPI ------------------------

def ask_rag(pregunta: str) -> dict:
    """
    Recibe la pregunta en texto.
    1. Detecta intención.
    2. Construye SQL.
    3. Ejecuta.
    4. Devuelve filas + mini resumen.
    """
    try:
        sql, params, descripcion = _pick_intent(pregunta)
        cols, rows = _run_sql(sql, params)

        # Creamos un pequeño dataframe para un preview bonito (markdown)
        df = pd.DataFrame(rows, columns=cols)

        # Mini resumen textual
        resumen = ""
        if df.empty:
            resumen = f"No encontré datos para esa consulta. ({descripcion})"
        else:
            # ejemplo rápido: primera fila
            primera = df.iloc[0].to_dict()
            resumen = f"{descripcion}. Destaca: {primera}"

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
        return {
            "ok": False,
            "pregunta": pregunta,
            "error": str(e),
        }




