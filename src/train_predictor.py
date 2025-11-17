# src/train_predictor.py
from __future__ import annotations
import sqlite3, pathlib, pandas as pd, numpy as np, joblib
from sklearn.linear_model import PoissonRegressor

DB = pathlib.Path("data/laliga.sqlite")
MODEL_PATH = pathlib.Path("data/predictor_poisson_plus.pkl")

def _read(sql: str, params: tuple = ()):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(sql, con, params=params)
    con.close()
    return df

def _diccionario_alias() -> dict[str, str]:
    """
    Devuelve {alias_normalizado -> TeamID} si existe el CSV de diccionario.
    Si no existe, retorna {} y usaremos el nombre tal cual.
    """
    try:
        csv = pathlib.Path("data/La liga BD - Diccionario_equipos.csv")
        if not csv.exists():
            return {}
        df = pd.read_csv(csv)
        # columnas esperadas: Alias, Canonical Name, Team ID
        cols = {c.lower().strip(): c for c in df.columns}
        a = cols.get("alias")
        tid = cols.get("team id") or cols.get("team_id") or cols.get("id")
        if not (a and tid):
            return {}
        def norm(s: str) -> str:
            return str(s).strip().lower()
        return {norm(r[a]): str(r[tid]).strip() for _, r in df.iterrows()}
    except Exception:
        return {}

_ALIAS2ID = _diccionario_alias()

def _canon_team(s: str) -> str:
    n = str(s).strip().lower()
    return _ALIAS2ID.get(n, str(s).strip())

def load_matches() -> pd.DataFrame:
    df = _read("""
        SELECT Temporada, Jornada,
               Local, Visitante,
               COALESCE(GolesLocal, "Goles_local") AS GolesLocal,
               COALESCE(GolesVisitante, "Goles_visitante") AS GolesVisitante
        FROM resultados
        WHERE GolesLocal IS NOT NULL OR "Goles_local" IS NOT NULL
    """)
    # Normaliza columnas de goles (por si vienen duplicadas)
    if "Goles_local" in df.columns and df["GolesLocal"].isna().any():
        df["GolesLocal"] = df["GolesLocal"].fillna(df["Goles_local"])
    if "Goles_visitante" in df.columns and df["GolesVisitante"].isna().any():
        df["GolesVisitante"] = df["GolesVisitante"].fillna(df["Goles_visitante"])

    # Canonicaliza equipos a TeamID cuando sea posible
    df["home_id"] = df["Local"].map(_canon_team)
    df["away_id"] = df["Visitante"].map(_canon_team)
    # Limpieza
    df = df.dropna(subset=["GolesLocal","GolesVisitante"])
    df["GolesLocal"] = pd.to_numeric(df["GolesLocal"], errors="coerce").fillna(0).astype(int)
    df["GolesVisitante"] = pd.to_numeric(df["GolesVisitante"], errors="coerce").fillna(0).astype(int)
    return df

def features_valor(temporada: str) -> pd.DataFrame:
    df = _read("""
        SELECT Temporada, Club, Valor
        FROM valor_clubes
        WHERE REPLACE(TRIM(Temporada),'-','/') = REPLACE(TRIM(?),'-','/')
    """, (temporada,))
    if df.empty:
        return pd.DataFrame(columns=["Club","valor"])
    out = (
        df.groupby("Club", as_index=False)["Valor"]
          .max()
          .rename(columns={"Club":"club_id","Valor":"valor"})
    )
    return out

def features_fichajes(temporada: str) -> pd.DataFrame:
    df = _read("""
        SELECT Temporada, Club, coste
        FROM fichajes
        WHERE REPLACE(TRIM(Temporada),'-','/') = REPLACE(TRIM(?),'-','/')
    """, (temporada,))
    if df.empty:
        return pd.DataFrame(columns=["Club","coste_fichajes"])
    out = (
        df.groupby("Club", as_index=False)["coste"]
          .sum()
          .rename(columns={"Club":"club_id","coste":"coste_fichajes"})
    )
    return out

def features_clasif(temporada: str) -> pd.DataFrame:
    df = _read("""
        SELECT Temporada, Club, Puntos
        FROM clasificaciones
        WHERE REPLACE(TRIM(Temporada),'-','/') = REPLACE(TRIM(?),'-','/')
    """, (temporada,))
    if df.empty:
        return pd.DataFrame(columns=["Club","Puntos"])
    out = df[["Club","Puntos"]].rename(columns={"Club":"club_id"})
    return out

def build_training() -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    matches = load_matches()
    # para cada temporada, unimos features de esa temporada
    rows = []
    for temporada, df in matches.groupby("Temporada"):
        f_val = features_valor(temporada)
        f_fic = features_fichajes(temporada)
        f_pts = features_clasif(temporada)

        aux = df.copy()
        # join por club_id
        aux = aux.merge(f_val, how="left", left_on="home_id", right_on="club_id").drop(columns=["club_id"])
        aux = aux.rename(columns={"valor":"home_valor"})
        aux = aux.merge(f_val, how="left", left_on="away_id", right_on="club_id").drop(columns=["club_id"])
        aux = aux.rename(columns={"valor":"away_valor"})

        aux = aux.merge(f_fic, how="left", left_on="home_id", right_on="club_id").drop(columns=["club_id"])
        aux = aux.rename(columns={"coste_fichajes":"home_coste"})
        aux = aux.merge(f_fic, how="left", left_on="away_id", right_on="club_id").drop(columns=["club_id"])
        aux = aux.rename(columns={"coste_fichajes":"away_coste"})

        aux = aux.merge(f_pts, how="left", left_on="home_id", right_on="club_id").drop(columns=["club_id"])
        aux = aux.rename(columns={"Puntos":"home_puntos"})
        aux = aux.merge(f_pts, how="left", left_on="away_id", right_on="club_id").drop(columns=["club_id"])
        aux = aux.rename(columns={"Puntos":"away_puntos"})

        rows.append(aux)

    full = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if full.empty:
        raise RuntimeError("No hay datos de partidos en 'resultados' para entrenar.")

    # Relleno sencillo
    for c in ["home_valor","away_valor","home_coste","away_coste","home_puntos","away_puntos"]:
        full[c] = pd.to_numeric(full[c], errors="coerce").fillna(full[c].median())

    # One-hot de IDs de equipo (mejor que usar IDs numéricos crudos)
    X_cat = pd.get_dummies(full[["home_id","away_id"]], prefix=["H","A"])
    X_num = full[["home_valor","away_valor","home_coste","away_coste","home_puntos","away_puntos"]]
    X = pd.concat([X_num, X_cat], axis=1)

    y_home = full["GolesLocal"].astype(int)
    y_away = full["GolesVisitante"].astype(int)

    # Guardamos también metadata necesaria
    metadata = {
        "feature_columns": list(X.columns),
        "seen_home_ids": sorted(full["home_id"].unique()),
        "seen_away_ids": sorted(full["away_id"].unique()),
        "medians": {c: float(X_num[c].median()) for c in X_num.columns}
    }
    return X, y_home, y_away, metadata

def train():
    print("📦 Cargando datos y construyendo features…")
    X, y_home, y_away, meta = build_training()

    print(f"➡️  {X.shape[0]} partidos | {X.shape[1]} features")
    m_home = PoissonRegressor(alpha=0.1, max_iter=500)
    m_away = PoissonRegressor(alpha=0.1, max_iter=500)
    X = X.fillna(0)
    X.replace([float('inf'), float('-inf')], 0, inplace=True)
    y_home = y_home.fillna(0)
    y_away = y_away.fillna(0)
    m_home.fit(X, y_home)
    m_away.fit(X, y_away)

    joblib.dump({
        "model_home": m_home,
        "model_away": m_away,
        "feature_columns": meta["feature_columns"],
        "medians": meta["medians"],
        "seen_home_ids": meta["seen_home_ids"],
        "seen_away_ids": meta["seen_away_ids"],
    }, MODEL_PATH)
    print(f"✅ Modelo avanzado guardado en {MODEL_PATH}")

if __name__ == "__main__":
    train()
