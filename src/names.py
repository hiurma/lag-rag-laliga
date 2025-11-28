# -*- coding: utf-8 -*-
# Normalizador simple de nombres de equipos
from __future__ import annotations
import unicodedata, re

# Amplía este diccionario cuando veas variantes nuevas
_ALIASES = {
    # ➤ Real Madrid
    "real madrid": "Real Madrid",
    "r. madrid": "Real Madrid",
    "madrid": "Real Madrid",
    "realmadrid": "Real Madrid",

    # ➤ FC Barcelona
    "fc barcelona": "FC Barcelona",
    "barcelona": "FC Barcelona",
    "barça": "FC Barcelona",
    "fcb": "FC Barcelona",

    # ➤ Atlético de Madrid
    "atletico de madrid": "Atletico de Madrid",
    "atlético de madrid": "Atletico de Madrid",
    "atletico madrid": "Atletico de Madrid",
    "atlético madrid": "Atletico de Madrid",
    "atm": "Atletico de Madrid",

    # ➤ Sevilla FC
    "sevilla": "Sevilla FC",
    "sevilla fc": "Sevilla FC",

    # ➤ Real Sociedad
    "real sociedad": "Real Sociedad",
    "sociedad": "Real Sociedad",

    # ➤ RCD Mallorca
    "mallorca": "RCD Mallorca",
    "rcd mallorca": "RCD Mallorca",

    # ➤ Osasuna
    "osasuna": "Osasuna",

    # ➤ Levante UD
    "levante": "Levante UD",
    "levante ud": "Levante UD",

    # ➤ Elche CF
    "elche": "Elche CF",
    "elche cf": "Elche CF",

    # ➤ Real Betis
    "betis": "Betis",
    "real betis": "Betis",

    # ➤ Getafe
    "getafe": "Getafe",

    # ➤ Celta de Vigo
    "celta": "Celta de Vigo",
    "celta de vigo": "Celta de Vigo",

    # ➤ Villarreal
    "villarreal": "Villarreal",

    # ➤ Valencia
    "valencia": "Valencia",
    "valencia cf": "Valencia",

    # ➤ Rayo Vallecano
    "rayo": "Rayo Vallecano",
    "rayo vallecano": "Rayo Vallecano",

    # ➤ Athletic Club
    "athletic": "Athletic Club",
    "athletic club": "Athletic Club",

    # ➤ Girona
    "girona": "Girona",
    "girona fc": "Girona",
}

def _strip(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

def canon_team(name: str) -> str:
    n = _strip(name)
    for canon, variants in _ALIASES.items():
        if n == canon or n in {_strip(v) for v in variants}:
            return canon
    return n  # si no está en el diccionario, devolvemos normalizado