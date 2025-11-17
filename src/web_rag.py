# src/web_rag.py
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import httpx
import pandas as pd

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
UA = "Mozilla/5.0 (codespaces: lag-rag-laliga)"
TIMEOUT = 20
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# ESPN (clasificación)
ESPN_STANDINGS_EN = "https://www.espn.com/soccer/standings/_/league/esp.1"
ESPN_STANDINGS_ES = "https://espndeportes.espn.com/futbol/posiciones/_/liga/esp.1"  # suele tener tabla en español

# Wikipedia (títulos por club)
WIKI_LALIGA_ES = "https://es.wikipedia.org/wiki/Primera_División_de_España#Títulos_por_club"
WIKI_LALIGA_EN = "https://en.wikipedia.org/wiki/La_Liga#By_club"

# --------------------------------------------------------------------------------------
# Cache muy simple (archivo json con TTL)
# --------------------------------------------------------------------------------------
def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{re.sub(r'[^a-zA-Z0-9_.-]+','_', key)}.json"

def _cache_get(key: str, ttl_sec: int) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if time.time() - data.get("_ts", 0) <= ttl_sec:
            return data.get("payload")
    except Exception:
        return None
    return None

def _cache_set(key: str, payload: dict) -> None:
    p = _cache_path(key)
    p.write_text(json.dumps({"_ts": time.time(), "payload": payload}, ensure_ascii=False), encoding="utf-8")

# --------------------------------------------------------------------------------------
# Utilidades HTTP/HTML
# --------------------------------------------------------------------------------------
def _get_text(url: str) -> str:
    headers = {"User-Agent": UA}
    with httpx.Client(timeout=TIMEOUT, headers=headers, follow_redirects=True) as cli:
        r = cli.get(url)
        r.raise_for_status()
        return r.text

def _read_html_tables(html_text: str, match: str | None = None) -> list[pd.DataFrame]:
    # pandas usa lxml bajo el capó
    return pd.read_html(html_text, match=match)

# --------------------------------------------------------------------------------------
# Limpieza/normalización: CLASIFICACIÓN (ESPN)
# --------------------------------------------------------------------------------------
def _normalize_standings_df(df: pd.DataFrame) -> pd.DataFrame:
    # Detecta columnas típicas y normaliza nombres
    cols_l = {c: str(c).strip().lower() for c in df.columns}

    # Posición / Equipo
    c_pos = next((k for k, v in cols_l.items() if v in ("pos", "position", "puesto", "#")), None)
    c_team = next((k for k, v in cols_l.items() if any(t in v for t in ["equipo", "team", "club"])), None)

    # Partidos y métricas
    c_pj = next((k for k, v in cols_l.items() if v in ("pj", "gp", "jug", "p")), None)
    c_g  = next((k for k, v in cols_l.items() if v in ("g", "w", "gan", "wins")), None)
    c_e  = next((k for k, v in cols_l.items() if v in ("e", "d", "emp", "draws")), None)
    c_p  = next((k for k, v in cols_l.items() if v in ("p", "l", "per", "losses")), None)
    c_gf = next((k for k, v in cols_l.items() if v in ("gf", "f", "goles a favor", "gf.", "gf ")), None)
    c_gc = next((k for k, v in cols_l.items() if v in ("gc", "c", "goles en contra", "ga", "ga ")), None)
    c_dg = next((k for k, v in cols_l.items() if "dif" in v or "gd" in v), None)
    c_pts= next((k for k, v in cols_l.items() if v in ("pts", "puntos", "points")), None)

    out = pd.DataFrame()

    # Rellena lo que encuentre
    if c_pos is not None: out["Pos"] = pd.to_numeric(df[c_pos], errors="coerce")
    if c_team is not None: out["Club"] = df[c_team].astype(str).str.strip()

    if c_pj is not None: out["PJ"] = pd.to_numeric(df[c_pj], errors="coerce")
    if c_g  is not None: out["G"]  = pd.to_numeric(df[c_g], errors="coerce")
    if c_e  is not None: out["E"]  = pd.to_numeric(df[c_e], errors="coerce")
    if c_p  is not None: out["P"]  = pd.to_numeric(df[c_p], errors="coerce")

    if c_gf is not None: out["GF"] = pd.to_numeric(df[c_gf], errors="coerce")
    if c_gc is not None: out["GC"] = pd.to_numeric(df[c_gc], errors="coerce")
    if c_dg is not None:
        out["DG"] = pd.to_numeric(df[c_dg], errors="coerce")
    elif "GF" in out.columns and "GC" in out.columns:
        out["DG"] = out["GF"] - out["GC"]

    if c_pts is not None: out["Pts"] = pd.to_numeric(df[c_pts], errors="coerce")

    # Filtra filas vacías y ordena por posición/puntos
    if "Club" in out.columns:
        out = out[out["Club"].astype(str).str.len() > 0]
    if "Pos" in out.columns:
        out = out.sort_values(["Pos"]).reset_index(drop=True)
    elif "Pts" in out.columns:
        out = out.sort_values(["Pts", "DG"], ascending=[False, False]).reset_index(drop=True)

    # Reordena columnas útiles si existen
    cols_pref = [c for c in ["Pos","Club","PJ","G","E","P","GF","GC","DG","Pts"] if c in out.columns]
    return out[cols_pref] if cols_pref else out

def fetch_standings_espn() -> tuple[pd.DataFrame, dict]:
    import pandas as pd, requests
    from io import StringIO

    ESPN_URL = "https://espndeportes.espn.com/futbol/posiciones/_/liga/esp.1"
    try:
        r = requests.get(ESPN_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        df = tables[0]
        meta = {"source": "espn_es", "url": ESPN_URL}
        return df, meta
    except Exception as e:
        raise RuntimeError(f"Fallo al obtener ESPN: {e}")

    # 2) Inglés
    try:
        html = _get_text(ESPN_STANDINGS_EN)
        tables = _read_html_tables(html, match="Team|PTS|Points|Pos|GP|W|D|L")
        for t in tables:
            df = _normalize_standings_df(t)
            if not df.empty and "Club" in df.columns and len(df) >= 10:
                meta = {"source": "espn_en", "url": ESPN_STANDINGS_EN}
                _cache_set(cache_key, {"rows": df.to_dict(orient="records"), "meta": meta})
                return df, meta
    except Exception as e:
        print(f"[WARN] ESPN EN falló: {e}")

    raise RuntimeError("Fallo al obtener ESPN: no se pudo extraer una tabla válida de posiciones.")

# --------------------------------------------------------------------------------------
# Limpieza/normalización: TÍTULOS (Wikipedia)
# --------------------------------------------------------------------------------------
def _clean_titles_df_es(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: str(c).strip().lower() for c in df.columns}
    # Candidatas de columnas
    club_cols  = [k for k, v in cols.items() if "club" in v or "equipo" in v]
    title_cols = [k for k, v in cols.items() if any(x in v for x in ["título", "títulos", "liga", "títulos de liga", "titulos"])]
    last_cols  = [k for k, v in cols.items() if any(x in v for x in ["último", "ultimo", "season", "año"])]

    c_club  = club_cols[0]  if club_cols  else None
    c_ligas = title_cols[0] if title_cols else None
    c_last  = last_cols[0]  if last_cols  else None

    out = pd.DataFrame()
    if c_club is not None:
        out["Club"] = df[c_club].astype(str).str.strip()
    if c_ligas is not None:
        out["Ligas"] = pd.to_numeric(df[c_ligas], errors="coerce").fillna(0).astype(int)
    else:
        out["Ligas"] = 0
    if c_last is not None:
        out["Último título"] = df[c_last].astype(str).str.replace(r"[^\d/–-]", "", regex=True)
    else:
        out["Último título"] = ""

    out = out[out["Club"].ne("")].sort_values("Ligas", ascending=False).reset_index(drop=True)
    return out

def _clean_titles_df_en(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: str(c).strip().lower() for c in df.columns}
    c_club  = next((k for k, v in cols.items() if "club" in v or "team" in v), None)
    c_titles= next((k for k, v in cols.items() if "title" in v), None)
    c_last  = next((k for k, v in cols.items() if "last" in v or "season" in v), None)

    out = pd.DataFrame()
    if c_club is not None:
        out["Club"] = df[c_club].astype(str).str.strip()
    if c_titles is not None:
        out["Ligas"] = pd.to_numeric(df[c_titles], errors="coerce").fillna(0).astype(int)
    else:
        out["Ligas"] = 0
    out["Último título"] = df[c_last].astype(str).str.strip() if c_last else ""

    out = out[out["Club"].ne("")].sort_values("Ligas", ascending=False).reset_index(drop=True)
    return out

def fetch_laliga_titles_wikipedia() -> tuple[pd.DataFrame, dict]:
    """
    Obtiene la tabla de títulos de LaLiga desde Wikipedia (EN/ES)
    """
    import pandas as pd, requests
    from bs4 import BeautifulSoup

    urls = [
        "https://en.wikipedia.org/wiki/La_Liga",  # ✅ página actual con la tabla correcta
        "https://es.wikipedia.org/wiki/Primera_Divisi%C3%B3n_de_Espa%C3%B1a"
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            tables = pd.read_html(str(soup))
            # Buscar una tabla que contenga nombres de clubes y títulos
            for t in tables:
                cols = [str(c).lower() for c in t.columns]
                if any("club" in c or "equipo" in c for c in cols) and any("title" in c or "título" in c for c in cols):
                    df = t.rename(columns=lambda c: str(c).strip())
                    meta = {"source": "wikipedia", "url": url}
                    print(f"[OK] Wikipedia table found at {url}")
                    return df, meta
        except Exception as e:
            print(f"[WARN] Wikipedia falló para {url}: {e}")

    # Fallback: crea una tabla básica si no se puede obtener de Wikipedia
    data = [
        {"Club": "Real Madrid", "Títulos": 36},
        {"Club": "FC Barcelona", "Títulos": 27},
        {"Club": "Atlético de Madrid", "Títulos": 11},
        {"Club": "Athletic Club", "Títulos": 8},
        {"Club": "Valencia CF", "Títulos": 6},
        {"Club": "Real Sociedad", "Títulos": 2},
        {"Club": "Deportivo de La Coruña", "Títulos": 1},
        {"Club": "Sevilla FC", "Títulos": 1},
        {"Club": "Real Betis", "Títulos": 1},
    ]
    df = pd.DataFrame(data)
    meta = {"source": "local-fallback", "url": "https://en.wikipedia.org/wiki/La_Liga"}
    print("[INFO] Usando fallback local de títulos.")
    return df, meta