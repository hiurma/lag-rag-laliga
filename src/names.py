# -*- coding: utf-8 -*-
# Normalizador simple de nombres de equipos
from __future__ import annotations
import unicodedata, re

# Amplía este diccionario cuando veas variantes nuevas
_ALIASES = {
    "real madrid": {"real madrid", "real madrid cf", "rma", "esp_rma", "rm"},
    "fc barcelona": {"barcelona", "fc barcelona", "fcb", "esp_bar"},
    "atletico de madrid": {"atlético de madrid", "atletico de madrid", "atlético madrid", "atletico madrid", "atm", "esp_atm"},
    "girona fc": {"girona", "girona fc", "esp_gir"},
    "sevilla fc": {"sevilla", "sevilla fc", "esp_sev"},
    "rcd espanyol": {"espanyol", "rcd espanyol", "esp_esp"},
    "osasuna": {"osasuna", "ca osasuna", "esp_osa"},
    "real sociedad": {"real sociedad", "esp_rso", "rso"},
    "villarreal": {"villarreal", "villarreal cf", "esp_vil"},
    "real betis": {"betis", "real betis", "esp_bet"},
    "athletic club": {"athletic", "athletic club", "ath bilbao", "esp_ath"},
    "celta de vigo": {"celta", "celta vigo", "rc celta", "esp_cel"},
    "rayo vallecano": {"rayo", "rayo vallecano", "esp_ray"},
    "valencia": {"valencia", "valencia cf", "esp_val"},
    "mallorca": {"mallorca", "rcd mallorca", "esp_mal"},
    "alaves": {"alaves", "deportivo alaves", "esp_ala"},
    "getafe": {"getafe", "getafe cf", "esp_get"},
    "elche": {"elche", "elche cf", "esp_elc"},
    "osasuna": {"osasuna", "ca osasuna", "esp_osa"},
    "real oviedo": {"real oviedo", "oviedo", "esp_ovi"},
    "girona": {"girona", "girona fc", "esp_gir"},
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