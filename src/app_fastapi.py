from __future__ import annotations

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))  # asegura que 'src' esté en el path
import pathlib
import sqlite3
import unicodedata
import re
import pandas as pd
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from chat_agent import ChatAgent
from dotenv import load_dotenv
load_dotenv()  # carga .env si existe

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chat_agent import ChatAgent   # <- si tu archivo se llama igual

app = FastAPI(title="LaLiga RAG API")

import re
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- CORS ---
# Puedes dejar "*" o limitar a tu página de GitHub Pages
origins: List[str] = [
    "https://hiurma.github.io",
    "https://hiurma.github.io/",
    "https://hiurma.github.io/laliga-chat-web"
    "https://hiurma.github.io/laliga-chat-web/",
    "*",  # si quieres ir a lo fácil
]

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hiurma.github.io",
        "https://hiurma.github.io/laliga-chat-web",
        "https://hiurma.github.io/laliga-chat-web/",
        "https://hiurma.github.io/laliga-chat-web/index.html"
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos ---
class ChatRequest(BaseModel):
    pregunta: str


agent = ChatAgent()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(body: ChatRequest):
    try:
        result = agent.chat_query(body.pregunta)
        # result ya es un dict con "mode", "respuesta", "tabla", etc.
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

agent = ChatAgent()

class ChatRequest(BaseModel):
    pregunta: str

@app.post("/chat")
def chat_endpoint(body: ChatRequest):
    return agent.chat_query(body.pregunta)


# --- importa ChatAgent desde el paquete 'src' con fallback ---
try:
    from src.chat_agent import ChatAgent
except ImportError:
    import os, sys
    ROOT = os.path.dirname(__file__)      # -> /ruta/.../src
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)          # metemos /src, no el padre
    from chat_agent import ChatAgent       # ahora sí lo encuentra
agent= ChatAgent()

DATA = pathlib.Path("data")
DB   = DATA / "laliga.sqlite"
DB.parent.mkdir(parents=True, exist_ok=True)

def _norm_text(s):
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = s.strip().lower()
    s = re.sub(r"\s+"," ", s)
    return s

def _best_match(colnames, *candidates):
    cols_n = { _norm_text(c): c for c in colnames }
    for cand in candidates:
        cand_n = _norm_text(cand)
        for k,orig in cols_n.items():
            if cand_n == k or cand_n in k or k in cand_n:
                return orig
    return None

def _read_csv_try(path):
    # intenta diferentes separadores/codificaciones
    for sep in [",",";","|","\t"]:
        for enc in ["utf-8","utf-8-sig","latin-1","cp1252"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python", on_bad_lines="skip")
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
    # fallback
    return pd.read_csv(path)

# ------------------
# 1. Crear / limpiar tablas base una vez
# ------------------
con = sqlite3.connect(DB)
cur = con.cursor()

cur.executescript("""
DROP TABLE IF EXISTS clasificaciones;
CREATE TABLE clasificaciones (
  Temporada TEXT NOT NULL,
  Jornada   INTEGER NOT NULL,
  Posicion  INTEGER,
  Club      TEXT NOT NULL,
  Puntos    INTEGER,
  Ganados   INTEGER,
  Empatados INTEGER,
  Perdidos  INTEGER,
  GolesAF   INTEGER,
  GolesEC   INTEGER,
  DifGoles  INTEGER,
  PRIMARY KEY (Temporada, Jornada, Club)
);

DROP TABLE IF EXISTS resultados;
CREATE TABLE resultados (
  Temporada      TEXT    NOT NULL,
  Jornada        INTEGER NOT NULL,
  Local          TEXT    NOT NULL,
  Visitante      TEXT    NOT NULL,
  GolesLocal     INTEGER,
  GolesVisitante INTEGER,
  Marcador       TEXT,
  PRIMARY KEY (Temporada, Jornada, Local, Visitante)
);

DROP TABLE IF EXISTS goleadores;
CREATE TABLE goleadores (
  Temporada TEXT,
  Jornada   INTEGER,
  Jugador   TEXT,
  Club      TEXT,
  Goles     INTEGER,
  Partidos  INTEGER,
  Minutos   INTEGER
);

DROP TABLE IF EXISTS valor_clubes;
CREATE TABLE valor_clubes (
  Temporada TEXT,
  Club      TEXT,
  Valor     REAL,
  Moneda    TEXT
);

DROP TABLE IF EXISTS fichajes;
CREATE TABLE fichajes (
  Temporada TEXT    NOT NULL,
  Club_ID   TEXT,
  Club      TEXT    NOT NULL,
  fichaje   TEXT    NOT NULL,
  coste     REAL,
  valor     REAL
);
""")
con.commit()

# ------------------
# 2. Cargar CLASIFICACIONES
# ------------------
f_clasif = DATA / "La liga BD - La Liga clasificaciones (3).csv"
if not f_clasif.exists():
    alts = list(DATA.glob("*clasific*.*"))
    if alts:
        f_clasif = alts[0]

if f_clasif.exists():
    df = _read_csv_try(f_clasif)
    cols = list(df.columns)

    c_temp = _best_match(cols, "temporada","season")
    c_pos  = _best_match(cols, "clasificacion","clasificación","posicion","rank")
    c_club = _best_match(cols, "club","equipo","club_id")
    c_pts  = _best_match(cols, "ptos","puntos","points")
    c_gan  = _best_match(cols, "ganado","ganados")
    c_emp  = _best_match(cols, "empatado","empates","empatados")
    c_per  = _best_match(cols, "perdido","perdidos")
    c_gf   = _best_match(cols, "goles","goles a favor","gf")
    c_gc   = _best_match(cols, "goles encajados","goles en contra","gc")
    c_dif  = _best_match(cols, "diferencia goles","diferencia_goles","difgoles","dg")

    out = pd.DataFrame({
        "Temporada": df[c_temp].astype(str) if c_temp else "",
        "Jornada":   0,
        "Posicion":  pd.to_numeric(df[c_pos], errors="coerce").fillna(0).astype(int) if c_pos else 0,
        "Club":      df[c_club].astype(str) if c_club else "",
        "Puntos":    pd.to_numeric(df[c_pts], errors="coerce").fillna(0).astype(int) if c_pts else 0,
        "Ganados":   pd.to_numeric(df[c_gan], errors="coerce").fillna(0).astype(int) if c_gan else 0,
        "Empatados": pd.to_numeric(df[c_emp], errors="coerce").fillna(0).astype(int) if c_emp else 0,
        "Perdidos":  pd.to_numeric(df[c_per], errors="coerce").fillna(0).astype(int) if c_per else 0,
        "GolesAF":   pd.to_numeric(df[c_gf], errors="coerce").fillna(0).astype(int) if c_gf else 0,
        "GolesEC":   pd.to_numeric(df[c_gc], errors="coerce").fillna(0).astype(int) if c_gc else 0,
        "DifGoles":  pd.to_numeric(df[c_dif], errors="coerce").fillna(0).astype(int) if c_dif else 0,
    })

    out.to_sql("clasificaciones", con, if_exists="append", index=False)
    con.commit()
    print(f"✅ CLASIFICACIONES: {len(out)} filas de '{f_clasif.name}'")
else:
    print("⚠️ No encontré CSV de clasificaciones")

# ------------------
# 3. Cargar GOLEADORES
# ------------------
goleadores_csv = DATA / "La liga BD - max goleadores (2).csv"
if not goleadores_csv.exists():
    alts = list(DATA.glob("*goleador*.*")) + list(DATA.glob("*max*gol*.*"))
    if alts:
        goleadores_csv = alts[0]

if goleadores_csv.exists():
    df = _read_csv_try(goleadores_csv)
    cols = list(df.columns)

    c_temp = _best_match(cols, "temporada","season","año","anio")
    c_j    = _best_match(cols, "jornada","matchday")
    c_jug  = _best_match(cols, "jugador","player","nombre")
    c_club = _best_match(cols, "club","equipo","team")
    c_g    = _best_match(cols, "goles","goals")
    c_pj   = _best_match(cols, "partidos","pj","apps")
    c_min  = _best_match(cols, "minutos","mins","minutes")

    out = pd.DataFrame({
        "Temporada": df[c_temp].astype(str) if c_temp else "",
        "Jornada":   pd.to_numeric(df[c_j], errors="coerce").fillna(0).astype(int) if c_j else 0,
        "Jugador":   df[c_jug].astype(str) if c_jug else "",
        "Club":      df[c_club].astype(str) if c_club else "",
        "Goles":     pd.to_numeric(df[c_g], errors="coerce").fillna(0).astype(int) if c_g else 0,
        "Partidos":  pd.to_numeric(df[c_pj], errors="coerce").fillna(0).astype(int) if c_pj else 0,
        "Minutos":   pd.to_numeric(df[c_min], errors="coerce").fillna(0).astype(int) if c_min else 0,
    })

    out.to_sql("goleadores", con, if_exists="append", index=False)
    con.commit()
    print(f"✅ GOLEADORES: {len(out)} filas de '{goleadores_csv.name}'")
else:
    print("⚠️ No encontré CSV de goleadores")

# ------------------
# 4. Cargar VALOR_CLUBES
# ------------------
valor_csv = DATA / "La liga BD - Valor Clubes (8).csv"
if not valor_csv.exists():
    alts = list(DATA.glob("*Valor*Club*.*")) + list(DATA.glob("*market*value*.*"))
    if alts:
        valor_csv = alts[0]

if valor_csv.exists():
    df = _read_csv_try(valor_csv)
    cols = list(df.columns)

    c_temp = _best_match(cols, "temporada","season","año","anio","year")
    c_club = _best_match(cols, "club","equipo","team")
    c_val  = _best_match(cols, "valor","market value","value","importe")
    c_mon  = _best_match(cols, "moneda","currency")

    # normalizamos el valor numérico a float (quitando símbolos €/M/etc)
    val_series = df[c_val].astype(str)
    val_series = (
        val_series
        .str.replace(r"[€]", "", regex=True)
        .str.replace("M€","", regex=False)
        .str.replace("M","", regex=False)
        .str.replace("m","", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(".","", regex=False)  # quita miles tipo 12.500.000
        .str.replace(",",".", regex=False) # coma → punto decimal
    )
    val_series = pd.to_numeric(val_series, errors="coerce")

    out = pd.DataFrame({
        "Temporada": df[c_temp].astype(str) if c_temp else "",
        "Club":      df[c_club].astype(str) if c_club else "",
        "Valor":     val_series.fillna(0).astype(float),
        "Moneda":    df[c_mon].astype(str) if c_mon else "EUR",
    })

    out.to_sql("valor_clubes", con, if_exists="append", index=False)
    con.commit()
    print(f"✅ VALOR_CLUBES: {len(out)} filas de '{valor_csv.name}'")
else:
    print("⚠️ No encontré CSV de valor_clubes")

# ------------------
# 5. Cargar RESULTADOS (incluye Marcador)
# ------------------
res_csv = DATA / "La liga BD - Resultados (3).csv"
if not res_csv.exists():
    alts = list(DATA.glob("*Resultado*.*"))
    if alts:
        res_csv = alts[0]

if res_csv.exists():
    df = _read_csv_try(res_csv)
    cols = [c.strip() for c in df.columns]

    # helper local
    def pick(*names):
        for want in names:
            for col in cols:
                if _norm_text(col) == _norm_text(want):
                    return col
        return None

    c_temp  = pick("Temporada")
    c_jorn  = pick("Jornada")
    c_local = pick("Local")
    c_vis   = pick("Visitante")
    c_gl    = pick("Goles_local","Goles local","GolesLocal")
    c_gv    = pick("Goles_visitante","Goles visitante","GolesVisitante")
    c_marc  = pick("Marcador")

    out = pd.DataFrame({
        "Temporada": df[c_temp].astype(str).str.strip() if c_temp else "",
        "Jornada":   pd.to_numeric(df[c_jorn], errors="coerce").fillna(0).astype(int) if c_jorn else 0,
        "Local":     df[c_local].astype(str).str.strip() if c_local else "",
        "Visitante": df[c_vis].astype(str).str.strip() if c_vis else "",
        "GolesLocal":     pd.to_numeric(df[c_gl], errors="coerce").fillna(0).astype(int) if c_gl else 0,
        "GolesVisitante": pd.to_numeric(df[c_gv], errors="coerce").fillna(0).astype(int) if c_gv else 0,
        "Marcador":  df[c_marc].astype(str).str.strip() if c_marc else None,
    })

    # si no hay marcador en CSV, lo fabricamos
    out["Marcador"] = out.apply(
        lambda r: f"{int(r['GolesLocal'])}-{int(r['GolesVisitante'])}"
        if (not r["Marcador"] or str(r["Marcador"]).lower() in ("nan","none",""))
        else r["Marcador"],
        axis=1
    )

    out.to_sql("resultados", con, if_exists="append", index=False)
    con.commit()
    print(f"✅ RESULTADOS: {len(out)} filas de '{res_csv.name}'")
else:
    print("⚠️ No encontré CSV de resultados")

# ------------------
# 6. Cargar FICHAJES TOP
# ------------------
# ====== 5) FICHAJES TOP ======
import sys, re, sqlite3, pathlib
import pandas as pd

DATA_DIR = pathlib.Path("data")
DB_PATH  = pathlib.Path("data/laliga.sqlite")
TABLE    = "fichajes"  # <- nombre de tabla destino
import re

_num_re = re.compile(r"[-+]?\d*\.?\d+")

def to_float(v):
    """
    Convierte strings tipo:
    - '75.000.000'
    - '75,0'
    - '€ 120,5M'
    - '120 M€'
    - '90m'
    - '1,2 K'
    en un float con valor absoluto en euros (no millones).
    """

    if v is None:
        return None

    s = str(v).strip()
    if s == "" or s.lower() in ("nan","none","null"):
        return None

    # quitamos espacios alrededor
    s = s.replace("€", "").replace("eur", "").replace("€", "")
    s = s.replace("m€", "m").replace("M€", "m").replace("millones", "m").replace("Millones", "m")
    s = s.replace("k€", "k").replace("K€", "k")

    # normalizamos separadores:
    #   '75.000.000' -> '75000000'
    #   '120,5'     -> '120.5'
    # pero ojo: si es '75.000.000' no queremos transformarlo en '75.000.000'→'75.000.000'→'75.000.000'
    # estrategia:
    #   1) quita puntos de miles
    #   2) cambia comas decimales a punto
    s = s.replace(".", "").replace(",", ".")

    # detecta sufijos K / M (m = millones, k = miles)
    mult = 1.0
    if s.lower().endswith("k"):
        mult = 1_000
        s = s[:-1].strip()
    elif s.lower().endswith("m"):
        mult = 1_000_000
        s = s[:-1].strip()

    # ahora extraemos el primer número que veamos
    m = _num_re.search(s)
    if not m:
        return None

    try:
        base = float(m.group(0))
    except:
        return None

    return base * mult


# ---------- Helpers específicos fichajes ----------

def list_data_dir():
    print("📁 Buscando CSV en:", DATA_DIR.resolve())
    for p in sorted(DATA_DIR.glob("*")):
        print(" -", p.name)

def pick_csv():
    """
    Elige el CSV más reciente que contenga 'fichaj' en el nombre.
    Ej.: 'La liga BD - fichajes top (4).csv'
    """
    cands = [p for p in DATA_DIR.glob("*.csv") if re.search(r"fichaj", p.name, re.I)]
    if not cands:
        print("❌ No encontré un CSV de fichajes en /data.")
        list_data_dir()
        sys.exit(1)

    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

def try_read_csv(path: pathlib.Path):
    """Lee el CSV probando separadores/codificaciones hasta que salga algo con >1 columna."""
    seps = [",",";","|","\t"]
    encs = ["utf-8","utf-8-sig","latin-1","cp1252"]
    last_err = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding=enc,
                    engine="python",
                    on_bad_lines="skip"
                )
                if df.shape[1] >= 2:
                    print(f"✅ Leído '{path.name}' con sep='{sep}', encoding='{enc}' → {df.shape[0]} filas, {df.shape[1]} columnas")
                    return df
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"No pude leer {path.name}. Último error: {last_err}")

def _norm_col(s: str) -> str:
    """Para mapear columnas: minúsculas, sin tildes, sin espacios/guiones."""
    if s is None:
        return ""
    rep = {
        "á":"a","é":"e","í":"i","ó":"o","ú":"u","ñ":"n",
        "Á":"a","É":"e","Í":"i","Ó":"o","Ú":"u","Ñ":"n",
    }
    out = []
    for ch in str(s):
        out.append(rep.get(ch, ch))
    s = "".join(out).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    return s

def map_columns(cols_raw):
    """
    Devuelve un dict con las columnas estándar que necesitamos:
      Temporada, Club_ID, Club, fichaje, coste, valor
    usando heurísticas sobre los nombres reales del CSV.
    """
    normed = { _norm_col(c): c for c in cols_raw }

    def pick(*cands):
        for cand in cands:
            cand_n = _norm_col(cand)
            # match exacto
            if cand_n in normed:
                return normed[cand_n]
        # match 'empieza por'
        for cand in cands:
            cand_n = _norm_col(cand)
            for k, orig in normed.items():
                if k.startswith(cand_n):
                    return orig
        return None

    return {
        "Temporada": pick("temporada","season"),
        "Club_ID"  : pick("club_id","idclub","id_club","clubid"),
        "Club"     : pick("club","equipo","club_destino","clubdestino","destino"),
        "fichaje"  : pick("fichaje","jugador","player","nombre_jugador","nombrejugador"),
        "coste"    : pick("coste","precio","importe","fee","monto","valor_millones","valor traspaso","precio_traspaso"),
        "valor"    : pick("valor","valoractual","valor_actual","marketvalue","market_value","valor_jugador"),
    }

def reload_fichajes():
    csv_path = pick_csv()
    print("📄 CSV detectado:", csv_path.name)

    df = try_read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    print("🧭 Columnas detectadas:", list(df.columns))

    colmap = map_columns(df.columns)

    # Campos obligatorios
    obligatorias = ["Temporada","Club","fichaje","coste"]
    faltan = [k for k in obligatorias if not colmap.get(k)]
    if faltan:
        print("❌ Faltan columnas obligatorias en tu CSV:", faltan)
        print("   Mapeo detectado:", colmap)
        sys.exit(1)

    # Construimos dataframe normalizado que meteremos en SQLite
    norm_df = pd.DataFrame()
    norm_df["Temporada"] = df[colmap["Temporada"]].astype(str).str.strip()
    norm_df["Club_ID"]   = df[colmap["Club_ID"]].astype(str).str.strip() if colmap.get("Club_ID") else ""
    norm_df["Club"]      = df[colmap["Club"]].astype(str).str.strip()
    norm_df["fichaje"]   = df[colmap["fichaje"]].astype(str).str.strip()

    # coste -> número en euros
    norm_df["coste"]     = df[colmap["coste"]].apply(to_float)

    # valor (opcional, valor de mercado jugador, etc.)
    if colmap.get("valor"):
        norm_df["valor"] = df[colmap["valor"]].apply(to_float)
    else:
        norm_df["valor"] = None

    # Quitamos filas donde no haya club o jugador
    norm_df = norm_df[(norm_df["Club"].astype(bool)) & (norm_df["fichaje"].astype(bool))]
    if norm_df.empty:
        print("❌ Tras normalizar no quedan filas válidas.")
        sys.exit(1)

    print("\n🔎 Vista previa normalizada:")
    print(norm_df.head(10).to_string(index=False))

    # Reescribimos completamente la tabla fichajes con estos datos limpios
    with sqlite3.connect(DB_PATH) as con:
        con.executescript(f"""
        DROP TABLE IF EXISTS {TABLE};
        CREATE TABLE {TABLE} (
            Temporada TEXT    NOT NULL,
            Club_ID   TEXT,
            Club      TEXT    NOT NULL,
            fichaje   TEXT    NOT NULL,
            coste     REAL,
            valor     REAL
        );
        """)
        norm_df.to_sql(TABLE, con, if_exists="append", index=False)

        # Comprobación
        cur = con.cursor()
        cnt = cur.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        ejemplo = cur.execute(f"SELECT Club,fichaje,coste FROM {TABLE} ORDER BY coste DESC LIMIT 5;").fetchall()

    print(f"\n✅ Cargado OK → {DB_PATH} :: tabla '{TABLE}' con {cnt} filas.")
    print("💰 Top 5 por coste después de guardar:")
    for row in ejemplo:
        print(" ", row)

# Llama a la recarga al arrancar (puedes comentar esto si no quieres que se ejecute cada vez)
reload_fichajes()





import os, sys, sqlite3, pathlib, time
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Asegura que 'src' esté en el path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.rag_sql import ask_rag
except Exception:
    from .rag_sql import ask_rag

try:
    from src.report_builder import build_report
except Exception:
    from .report_builder import build_report

app = FastAPI(title="LaLiga RAG SQL API", version="1.0")

class AskBody(BaseModel):
    pregunta: Optional[str] = None
    question: Optional[str] = None

@app.get("/")
def root():
    return {"mensaje": "API activa"}

@app.get("/health")
def health():
    return {"ok": True}

from pydantic import BaseModel
from typing import Optional, List

class ReportBody(BaseModel):
    temporada: Optional[str] = None
    preguntas: Optional[List[str]] = None
    formato: Optional[str] = "md"   # "md" | "html" | "both"

class ChatBody(BaseModel):
    pregunta: str
from pydantic import BaseModel
from fastapi import HTTPException

class ChatBody(BaseModel):
    pregunta: str

@app.post("/chat")
def chat_endpoint(body: ChatBody):
    q = (body.pregunta or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La 'pregunta' no puede estar vacía")
    return agent.chat_query(q)



from fastapi.responses import HTMLResponse, PlainTextResponse
import os
@app.post("/report")
def generar_report(body: ReportBody):
    """
    Genera el informe de la temporada.
    formato:
      - "md":   responde con el markdown en texto plano
      - "html": responde con el HTML listo para ver en navegador
      - "both": responde JSON con rutas guardadas y ambos textos
    """
    temporada = body.temporada
    if not temporada:
        raise HTTPException(status_code=400, detail="Falta 'temporada' en el body")

    # si no pasan preguntas, usamos batería
    preguntas = body.preguntas or _leer_bateria(temporada)
    if not preguntas:
        raise HTTPException(
            status_code=400,
            detail="No hay preguntas. Pasa 'preguntas' o crea data/prompts/bateria.md"
        )

    # generamos informe usando la nueva build_report
    md_path, html_path, md_text, html_text = build_report(temporada, preguntas)

    # Qué quiere el cliente
    formato = (body.formato or "md").lower()

    if formato == "html":
        # devolvemos HTML directamente con content-type text/html
        return HTMLResponse(
            content=html_text,
            status_code=200
        )

    if formato == "md":
        # devolvemos el markdown plano
        return PlainTextResponse(
            content=md_text,
            status_code=200
        )

    # formato "both" (o cualquier otra cosa rara)
    return {
    "ok": "ok",
    "total": len(preguntas),
    "report_md": md_path,
    "report_html": html_path,
    "url_html": f"/reports/{os.path.basename(html_path)}",
    "preguntas_usadas": preguntas
}

    


@app.get("/diag/sql")
def diag_sql(sql: str = Query(..., description="Ejecuta una consulta SQL manual (solo SELECT)")):
    try:
        con = sqlite3.connect(os.getenv("DB_PATH", "data/laliga.sqlite"))
        cur = con.cursor()
        cur.execute(sql)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
        con.close()
        return {"ok": True, "columnas": cols, "resultados": rows[:10]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

class ReportBody(BaseModel):
    temporada: Optional[str] = None
    preguntas: Optional[List[str]] = None
    formato: Optional[str] = "md"   # "md" | "html" | "both"

def _leer_bateria(temporada: str | None):
    p = pathlib.Path("data/prompts/bateria.md")
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("-"): 
            continue
        q = line.lstrip("-").strip().strip('"').strip("'")
        if temporada and "{temporada}" in q:
            q = q.replace("{temporada}", temporada)
        out.append(q)
    return out
import re

def _normalize_temporada(t_raw: str | None) -> str | None:
    """
    Acepta cosas como:
      - "2025/2026"
      - "2025-2026"
      - "2025"
    y devuelve siempre "2025/2026" cuando se puede inferir.
    Si no puede, devuelve lo que venga tal cual.
    """
    if not t_raw:
        return None

    t = t_raw.strip()
    # Caso exacto tipo "2025/2026" o "2025-2026"
    m = re.match(r"^(\d{4})\s*[-/]\s*(\d{4})$", t)
    if m:
        y1 = m.group(1)
        y2 = m.group(2)
        return f"{y1}/{y2}"

    # Caso solo "2025" -> asumimos temporada "2025/2026"
    m2 = re.match(r"^(\d{4})$", t)
    if m2:
        y1 = int(m2.group(1))
        y2 = y1 + 1
        return f"{y1}/{y2}"

    # fallback: lo que haya
    return t

@app.post("/report")
def generar_report(body: ReportBody):
    # 1. normalizamos la temporada que llega del usuario
    temporada_norm = _normalize_temporada(body.temporada)

    # 2. si no te pasan preguntas explícitas, cargamos batería.md
    preguntas = body.preguntas or _leer_bateria(temporada_norm)
    if not preguntas:
        raise HTTPException(status_code=400,
                            detail="No hay preguntas. Pasa 'preguntas' o crea data/prompts/bateria.md")

    # 3. construimos el informe con la temporada normalizada
    md_path, html_path, md_text, html_text = build_report(temporada_norm, preguntas)

    return {
        "ok": True,
        "temporada_usada": temporada_norm,
        "report_md": md_path,
        "report_html": html_path,
        "url_html": f"/reports/{os.path.basename(html_path)}",
        "preview_markdown": md_text[:500]  # mini snippet
    }


@app.get("/reports/list")
def listar_reports():
    """Lista los informes disponibles en /reports (md y html)."""
    base = pathlib.Path(__file__).resolve().parents[1] / "reports"
    base.mkdir(exist_ok=True)
    files = []
    for f in sorted(base.glob("report_*.*")):
        st = f.stat()
        files.append({
            "name": f.name,
            "path": str(f),
            "size_bytes": st.st_size,
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
        })
    return {"count": len(files), "items": files}

# Monta la carpeta /reports como estático
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

