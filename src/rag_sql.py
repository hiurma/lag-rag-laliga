import os
import sqlite3
from pathlib import Path
import re
import unicodedata
import pandas as pd
from typing import Optional, List, Tuple

DB_PATH = Path("data/laliga.sqlite")

# =========================
# Helpers generales SQL
# =========================

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


def _guess_temporada(q: str) -> Optional[str]:
    """
    Detecta temporada del estilo 2024/2025 o 2024-2025 en el texto.
    """
    m = re.search(r"(20\d{2})[/-](20\d{2})", q)
    if m:
        t1, t2 = m.group(1), m.group(2)
        return f"{t1}/{t2}"
    return None


def _clean_temporada_for_where(colname: str) -> str:
    """
    Expresión SQL que iguala temporada ignorando '-' vs '/' y espacios.
    EJ: REPLACE(TRIM(Temporada), '-', '/') = REPLACE(TRIM(?), '-', '/')
    """
    return f"REPLACE(TRIM({colname}), '-', '/') = REPLACE(TRIM(?), '-', '/')"


# =========================
# Consultas "canónicas"
# =========================

def _sql_top_goleadores(temp: Optional[str]):
    where = "1=1"
    params: tuple = ()
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


def _sql_top_valor_clubes(temp: Optional[str]):
    where = "1=1"
    params: tuple = ()
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


def _sql_top_fichajes(temp: Optional[str]):
    where = "1=1"
    params: tuple = ()
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


# ==========================================
# Soporte de PRONÓSTICO basado en histórico
# ==========================================

# Cache de clubes para mapear nombres escritos por el usuario
_CLUB_CACHE: Optional[list[tuple[str, str]]] = None  # (nombre_original, nombre_normalizado)


def _normalize_name(s: str) -> str:
    """
    Normaliza nombres de clubes / texto:
      - sin tildes
      - minúsculas
      - sin signos raros
      - quita palabras muy genéricas (fc, cf, club, de, la...)
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9ñ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    stop = {"fc", "cf", "sd", "cd", "ud", "club", "de", "la", "el", "club de futbol"}
    tokens = [t for t in s.split() if t not in stop]
    return " ".join(tokens) if tokens else s


def _load_club_cache() -> list[tuple[str, str]]:
    """
    Carga y cachea la lista de clubes distintos (clasificaciones + resultados)
    con su versión normalizada.
    """
    global _CLUB_CACHE
    if _CLUB_CACHE is not None:
        return _CLUB_CACHE

    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT DISTINCT Club AS nombre FROM clasificaciones
        UNION
        SELECT DISTINCT Local AS nombre FROM resultados
        UNION
        SELECT DISTINCT Visitante AS nombre FROM resultados
        """,
        con,
    )
    con.close()
    cache: list[tuple[str, str]] = []
    for raw in df["nombre"].dropna().unique():
        raw = str(raw)
        norm = _normalize_name(raw)
        if norm:
            cache.append((raw, norm))
    _CLUB_CACHE = cache
    return cache


def _find_teams_in_question(q: str) -> tuple[Optional[str], Optional[str]]:
    """
    Intenta detectar los dos equipos mencionados en la pregunta
    comparando contra la lista de clubes de la base de datos.
    """
    norm_q = _normalize_name(q)
    clubs = _load_club_cache()

    hits: list[tuple[str, str]] = []  # (nombre_original, norm)
    for orig, norm in clubs:
        if norm and norm in norm_q:
            hits.append((orig, norm))

    # si encontramos más de dos, nos quedamos con los de nombre normalizado más largo
    hits = sorted(hits, key=lambda x: len(x[1]), reverse=True)
    if len(hits) >= 2:
        return hits[0][0], hits[1][0]
    if len(hits) == 1:
        return hits[0][0], None
    return None, None


def _predict_match(pregunta: str, temporada: Optional[str]) -> dict:
    """
    Predicción tipo casa de apuestas basada en:
      - histórico de enfrentamientos Local vs Visitante

    Devuelve SOLO probabilidades 1-X-2 y un resumen textual.
    """
    q_low = pregunta.lower()
    local_name, visit_name = _find_teams_in_question(q_low)

    if not local_name or not visit_name:
        return {
            "ok": False,
            "modo": "pronostico",
            "pregunta": pregunta,
            "error": "No pude identificar claramente los dos equipos en la pregunta.",
        }

    # Construimos consulta de histórico
    where_parts = ["Local = ?", "Visitante = ?"]
    params: list = [local_name, visit_name]

    if temporada:
        where_parts.append(_clean_temporada_for_where("Temporada"))
        params.append(temporada)

    where_clause = " AND ".join(where_parts)

    sql = f"""
        SELECT
          SUM(CASE WHEN GolesLocal > GolesVisitante THEN 1 ELSE 0 END) AS victorias_local,
          SUM(CASE WHEN GolesLocal = GolesVisitante THEN 1 ELSE 0 END) AS empates,
          SUM(CASE WHEN GolesLocal < GolesVisitante THEN 1 ELSE 0 END) AS victorias_visitante,
          COUNT(*) AS total_partidos,
          AVG(GolesLocal) AS media_goles_local,
          AVG(GolesVisitante) AS media_goles_visitante
        FROM resultados
        WHERE {where_clause};
    """

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(sql, tuple(params))
    row = cur.fetchone()
    con.close()

    if not row:
        return {
            "ok": False,
            "modo": "pronostico",
            "pregunta": pregunta,
            "error": "No encontré histórico de enfrentamientos para ese partido.",
        }

    (
        vict_local,
        empates,
        vict_visitante,
        total_partidos,
        media_gl,
        media_gv,
    ) = row

    if not total_partidos or total_partidos == 0:
        return {
            "ok": False,
            "modo": "pronostico",
            "pregunta": pregunta,
            "error": "No hay suficientes partidos históricos entre esos equipos.",
        }

    vict_local = vict_local or 0
    empates = empates or 0
    vict_visitante = vict_visitante or 0

    p_local = vict_local / total_partidos
    p_empate = empates / total_partidos
    p_visitante = vict_visitante / total_partidos

    resumen = (
        f"Histórico {temporada or 'global'} en tu base local para {local_name} vs {visit_name}: "
        f"{int(vict_local)} victorias locales, {int(empates)} empates y "
        f"{int(vict_visitante)} victorias visitantes en {int(total_partidos)} partidos. "
        f"Probabilidades estimadas → 1 (local): {p_local:.2%}, "
        f"X (empate): {p_empate:.2%}, 2 (visitante): {p_visitante:.2%}."
    )

    return {
        "ok": True,
        "modo": "pronostico",
        "pregunta": pregunta,
        "partido": {
            "temporada": temporada or "histórico",
            "local": local_name,
            "visitante": visit_name,
        },
        "historial": {
            "total_partidos": int(total_partidos),
            "victorias_local": int(vict_local),
            "empates": int(empates),
            "victorias_visitante": int(vict_visitante),
            "media_goles_local": float(media_gl) if media_gl is not None else None,
            "media_goles_visitante": float(media_gv) if media_gv is not None else None,
        },
        "probabilidades": {
            "p_local": p_local,
            "p_empate": p_empate,
            "p_visitante": p_visitante,
        },
        "resumen": resumen,
        "sql_usada": sql,
        "parametros": params,
    }


# =========================
# Router semántico
# =========================

def _pick_intent(q: str):
    """
    Devuelve qué tipo de info pide la pregunta (para consultas SQL normales).
    Miramos palabras clave muy básicas.
    """
    q_low = q.lower()
    temp = _guess_temporada(q_low)

    if "goleador" in q_low or "goles" in q_low or "pichichi" in q_low:
        return _sql_top_goleadores(temp)

    if "valor" in q_low or "clubes más caros" in q_low or "club mas caro" in q_low:
        return _sql_top_valor_clubes(temp)

    if "fichaje" in q_low or "traspaso" in q_low or "transfer" in q_low:
        return _sql_top_fichajes(temp)

    if "resultado" in q_low or "marcador" in q_low:
        return _sql_resultados(temp)

    if "clasific" in q_low or "tabla" in q_low or "puntos" in q_low or "liga" in q_low:
        return _sql_tabla_clasificacion(temp)

    # fallback: por defecto saco tabla de clasificación
    return _sql_tabla_clasificacion(temp)


# =========================
# Entrada principal
# =========================

def ask_rag(pregunta: str) -> dict:
    """
    Recibe la pregunta en texto.
    1. Si detecta que es un PRONÓSTICO → usa histórico de resultados.
    2. Si no, detecta intención + construye SQL canónica.
    3. Ejecuta.
    4. Devuelve filas + mini resumen.
    """
    try:
        q_low = (pregunta or "").lower()

        # --- 1) Preguntas de pronóstico / probabilidad ---
        if any(
            w in q_low
            for w in [
                "pronostic",
                "probabilidad",
                "quien gana",
                "quién gana",
                "resultado probable",
                "apuesta",
            ]
        ):
            temp = _guess_temporada(q_low)
            return _predict_match(pregunta, temp)

        # --- 2) Consultas estándar tipo tabla / top / fichajes ---
        sql, params, descripcion = _pick_intent(q_low)
        cols, rows = _run_sql(sql, params)

        # DataFrame para preview bonito
        df = pd.DataFrame(rows, columns=cols)

        if df.empty:
            resumen = f"No encontré datos para esa consulta. ({descripcion})"
        else:
            primera = df.iloc[0].to_dict()
            resumen = f"{descripcion}. Destaca: {primera}"

        return {
            "ok": True,
            "modo": "sql",
            "pregunta": pregunta,
            "descripcion": descripcion,
            "consulta": sql,
            "parametros": params,
            "columnas": cols,
            "resultados": rows,
            "resumen": resumen,
            "tabla_md": df.head(20).to_markdown(index=False) if not df.empty else "",
        }

    except Exception as e:
        return {
            "ok": False,
            "pregunta": pregunta,
            "error": str(e),
        }