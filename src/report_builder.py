# src/report_builder.py
from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
import unicodedata, re

# ---------- paths ----------
DB_PATH = str((Path("data") / "laliga.sqlite").resolve())
REPORTS_DIR = Path("reports")

# ---------- util texto ----------
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def _connect():
    return sqlite3.connect(DB_PATH)

def _table_cols(table: str) -> dict[str,str]:
    """
    devuelve dict {real_col_name: normalized_col_name}
    ej: {"Goles_local": "goleslocal", "GolesVisitante":"golesvisitante", ...}
    """
    con = _connect()
    rows = con.execute(f'PRAGMA table_info("{table}")').fetchall()
    con.close()
    return {r[1]: _norm(r[1]) for r in rows}

def _pick(cols_map: dict[str,str], *wanted) -> str|None:
    """
    dame el nombre REAL de la primera columna cuyo normalizado
    coincida con cualquiera de wanted
    """
    wanted_norm = [_norm(w) for w in wanted]
    # match exact
    for real, normed in cols_map.items():
        if normed in wanted_norm:
            return real
    # contiene
    for real, normed in cols_map.items():
        if any(w in normed for w in wanted_norm):
            return real
    return None

def _season_filter(col_temp: str|None) -> str:
    if not col_temp:
        return "1=1"
    # igualamos "2024/2025" y "2024-2025"
    return f"REPLACE(TRIM({col_temp}),'-','/') = REPLACE(TRIM(?),'-','/')"

def _run_sql_df(sql: str, params: tuple=()) -> pd.DataFrame:
    con = _connect()
    try:
        df = pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()
    return df

def _preview_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return "_(sin filas)_"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)

import markdown

def md_to_html(md_text: str, title: str = "Informe") -> str:
    """
    Convierte un texto Markdown en HTML con un estilo simple.
    """
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"]
    )

    html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            background-color: #fafafa;
            color: #222;
        }}
        h1, h2, h3 {{
            color: #800000;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f0f0f0;
        }}
        hr {{
            border: none;
            border-top: 1px solid #ddd;
            margin: 40px 0;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    return html_template

#-------------------------------------------------
# 1. CLASIFICACIONES
#-------------------------------------------------
def _sec_clasificaciones(temporada: str) -> str:
    cols = _table_cols("clasificaciones")

    col_temp  = _pick(cols, "Temporada","season")
    col_club  = _pick(cols, "Club","equipo","clubid")
    col_pts   = _pick(cols, "Puntos","ptos","points")
    col_gan   = _pick(cols, "Ganados","ganado","wins")
    col_emp   = _pick(cols, "Empatados","empates","draws")
    col_per   = _pick(cols, "Perdidos","perdidos","losses")

    # seguridad mínima: necesitamos club y puntos
    if not (col_club and col_pts):
        return "_(sin filas)_"

    where = _season_filter(col_temp)

    sql = f"""
        SELECT
            {col_club}   AS Club,
            {col_pts}    AS Puntos,
            {col_gan or 'NULL'} AS Ganados,
            {col_emp or 'NULL'} AS Empatados,
            {col_per or 'NULL'} AS Perdidos
        FROM clasificaciones
        WHERE {where}
        GROUP BY {col_club}
        ORDER BY CAST({col_pts} AS INTEGER) DESC
        LIMIT 10
    """
    params = (temporada,) if col_temp else ()
    df = _run_sql_df(sql, params)
    return _preview_markdown(df)

#-------------------------------------------------
# 2. GOLEADORES
#-------------------------------------------------
def _sec_goleadores(temporada: str) -> str:
    """
    Devuelve el máximo goleador de la temporada (1 fila).
    Si en el futuro metemos más filas por temporada en 'goleadores',
    esta query ya está preparada para pillar el TOP 10 ordenado por goles.
    """
    con = _connect()
    cur = con.cursor()

    # Miramos columnas reales y hacemos mapping flexible
    cols = [r[1] for r in cur.execute("PRAGMA table_info(goleadores);").fetchall()]
    have = {c.lower(): c for c in cols}

    col_temp   = have.get("temporada")
    col_jugador= have.get("jugador")
    col_club   = have.get("club")
    col_goles  = have.get("goles")

    if not (col_temp and col_jugador and col_goles):
        con.close()
        return "_(sin filas)_"

    # Filtro temporada flexible tipo "2025" -> "2025/2026"
    where_temp = f"REPLACE(TRIM({col_temp}), '-', '/') = REPLACE(TRIM(?), '-', '/')"

    sql = f"""
        SELECT
            {col_jugador} AS Jugador,
            {col_club}    AS Club,
            {col_goles}   AS Goles
        FROM goleadores
        WHERE {where_temp}
        ORDER BY {col_goles} DESC
        LIMIT 10;
    """

    rows = cur.execute(sql, (temporada,)).fetchall()
    con.close()

    df = pd.DataFrame(rows, columns=["Jugador", "Club", "Goles"])

    titulo = "## 2. Top goleadores\n\n"
    if df.empty:
        return titulo + "_(sin filas)_\n"

    return titulo + df.to_markdown(index=False) + "\n"


#-------------------------------------------------
# 3. VALOR CLUBES
#-------------------------------------------------
def _sec_valor_clubes(temporada: str) -> str:
    cols = _table_cols("valor_clubes")

    col_temp = _pick(cols, "Temporada", "season", "anio", "año", "year")
    col_club = _pick(cols, "Club", "equipo", "team")
    col_val  = _pick(cols, "Valor", "marketvalue", "valorclub", "valor_millones", "valor_en_millones")

    if not (col_club and col_val):
        return "_(sin filas)_"

    # ✅ versión corregida del limpiador: más simple, sin exceso de comas ni paréntesis
    clean_val = f"""
        CAST(
            REPLACE(
                REPLACE(
                    REPLACE(
                        REPLACE(
                            REPLACE({col_val}, '€', ''),
                        'M€', ''),
                    'M', ''),
                'm', ''),
            ',', '.') AS REAL
        )
    """

    where = _season_filter(col_temp)

    sql = f"""
        SELECT
            {col_club} AS Club,
            MAX({clean_val}) AS Valor_Millones
        FROM valor_clubes
        WHERE {where}
        GROUP BY {col_club}
        ORDER BY Valor_Millones DESC
        LIMIT 10
    """
    params = (temporada,) if col_temp else ()
    df = _run_sql_df(sql, params)
    return _preview_markdown(df)


#-------------------------------------------------
# 4. RESULTADOS (usa Marcador si existe, si no lo calcula)
#-------------------------------------------------
def _sec_resultados(temporada: str) -> str:
    cols = _table_cols("resultados")

    col_temp    = _pick(cols,"Temporada","season")
    col_jornada = _pick(cols,"Jornada","matchday")
    col_local   = _pick(cols,"Local")
    col_visit   = _pick(cols,"Visitante","Visitor","Away","visitante")
    col_gl      = _pick(cols,"GolesLocal","Goles_local","goleslocal","goles_local")
    col_gv      = _pick(cols,"GolesVisitante","Goles_visitante","golesvisitante","goles_visitante")
    col_marc    = _pick(cols,"Marcador","ResultadoNumerico","Score","MarcadorPartido")

    if not (col_local and col_visit):
        return "_(sin filas)_"

    where = _season_filter(col_temp)

    if col_marc:
        marcador_expr = col_marc
    else:
        # si no hay marcador en tabla, lo fabricamos con printf
        if col_gl and col_gv:
            marcador_expr = f"printf('%d-%d',{col_gl},{col_gv})"
        else:
            marcador_expr = "'-'"

    campos = []
    if col_jornada: campos.append(f"{col_jornada} AS Jornada")
    if col_local:   campos.append(f"{col_local} AS Local")
    if col_visit:   campos.append(f"{col_visit} AS Visitante")
    campos.append(f"{marcador_expr} AS Marcador")

    order_bits = []
    if col_jornada: order_bits.append(col_jornada)
    if col_local:   order_bits.append(col_local)
    if col_visit:   order_bits.append(col_visit)
    order_by = ", ".join(order_bits) if order_bits else "1"

    sql = f"""
        SELECT
            {", ".join(campos)}
        FROM resultados
        WHERE {where}
        ORDER BY {order_by}
        LIMIT 20
    """
    params = (temporada,) if col_temp else ()
    df = _run_sql_df(sql, params)
    return _preview_markdown(df)

#-------------------------------------------------
# 5. FICHAJES TOP
#-------------------------------------------------
def _sec_fichajes_top(temporada: str) -> str:
    """
    Top fichajes ordenados por 'coste' ya numérico en la tabla fichajes.
    No intentamos limpiar el texto porque ya lo normalizamos al cargar.
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # inspeccionamos columnas reales de la tabla
    cols = [r[1] for r in cur.execute("PRAGMA table_info(fichajes);").fetchall()]
    col_temp  = "Temporada" if "Temporada" in cols else None
    col_club  = "Club" if "Club" in cols else None
    col_fich  = "fichaje" if "fichaje" in cols else None
    col_coste = "coste" if "coste" in cols else None

    if not (col_club and col_fich and col_coste):
        con.close()
        return "_(sin filas)_"

    # filtro por temporada solo si hay columna Temporada
    if col_temp:
        where_clause = f"REPLACE(TRIM({col_temp}),'-','/') = REPLACE(TRIM(?),'-','/')"
        params = (temporada,)
    else:
        where_clause = "1=1"
        params = ()

    sql = f"""
        SELECT
            {col_club}  AS Club,
            {col_fich}  AS Fichaje,
            {col_coste} AS Coste
        FROM fichajes
        WHERE {where_clause}
        ORDER BY {col_coste} DESC
        LIMIT 10
    """

    rows = cur.execute(sql, params).fetchall()
    con.close()

    df = pd.DataFrame(rows, columns=["Club", "Fichaje", "Coste"])

    # formato bonito: si Coste es float grande en euros, lo mostramos sin .0 feo
    if not df.empty:
        df["Coste"] = df["Coste"].apply(
            lambda x: (f"{int(x):,}".replace(",", ".")) if pd.notna(x) and float(x).is_integer()
            else (f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            if pd.notna(x) else x
        )

    titulo = "## 5. Top fichajes\n\n"
    return titulo + (df.to_markdown(index=False) if not df.empty else "_(sin filas)_") + "\n"


#-------------------------------------------------
# render MD + HTML
#-------------------------------------------------
_HTML_CSS = """
<style>
  :root{color-scheme: light dark;}
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:2rem;line-height:1.5}
  h1,h2,h3{color:#b30000;margin-top:1.6rem}
  .meta{color:#666;font-size:.95rem;margin-bottom:1rem}
  table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.9rem}
  th,td{border:1px solid #ddd;padding:.4rem .5rem;text-align:left}
  th{background:#f2f2f2}
  code{background:#f6f8fa;padding:.1rem .3rem;border-radius:4px}
  hr{border:none;border-top:1px dashed #ccc;margin:1.5rem 0}
</style>
"""

try:
    import markdown as _md
except Exception:
    _md = None

def _md_to_html(md_text: str, title: str) -> str:
    if _md is None:
        safe = (md_text
                .replace("&","&amp;")
                .replace("<","&lt;"))
        body = f"<pre>{safe}</pre>"
    else:
        body = _md.markdown(md_text, extensions=["tables","fenced_code"])
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{title}</title>
{_HTML_CSS}
</head>
<body>
<div class="meta">Generado: {datetime.now():%Y-%m-%d %H:%M}</div>
{body}
</body></html>"""

def build_report(temporada: str, preguntas: list[str]):
    REPORTS_DIR.mkdir(exist_ok=True)

    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = f"report_{temporada.replace('/','-')}_{stamp}"

    md_path = REPORTS_DIR / f"{base}.md"
    html_path = REPORTS_DIR / f"{base}.html"

    parts: list[str] = []
    parts.append(f"# Informe LaLiga – {temporada}\n")
    parts.append(f"_Generado: {datetime.now():%Y-%m-%d %H:%M}_\n\n")

    parts.append("## 1. Top 10 equipos por Puntos\n\n")
    parts.append(_sec_clasificaciones(temporada) + "\n\n---\n")

    parts.append("## 2. Top 10 goleadores\n\n")
    parts.append(_sec_goleadores(temporada) + "\n\n---\n")

    parts.append("## 3. Top 10 clubes por Valor\n\n")
    parts.append(_sec_valor_clubes(temporada) + "\n\n---\n")

    parts.append("## 4. Muestra de Resultados\n\n")
    parts.append(_sec_resultados(temporada) + "\n\n---\n")

    parts.append("## 5. Top fichajes\n\n")
    parts.append(_sec_fichajes_top(temporada) + "\n")

    md_text = "".join(parts)
    md_path.write_text(md_text, encoding="utf-8")

    # convertir markdown → html con estilos
    html_text = md_to_html(md_text, title=f"Informe LaLiga — {temporada}")
    html_path.write_text(html_text, encoding="utf-8")

    return str(md_path), str(html_path), md_text, html_text
