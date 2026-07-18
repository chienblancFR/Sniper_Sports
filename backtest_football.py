"""
backtest_football.py — Back-test complet du modèle Dixon-Coles (multi-saisons)
=============================================================================
Usage :
  python backtest_football.py --collect    # Phase 1 : télécharge les données
  python backtest_football.py --simulate   # Phase 2 : simule les paris
  python backtest_football.py --report     # Phase 3 : génère le rapport
  python backtest_football.py              # Les 3 phases d'un coup

Reset (avant de relancer le backtest / dashboard) :
  python backtest_football.py --reset
      → Vide bt_signaux + supprime backtest_results.csv
      → Garde backtest_data.db (fixtures, cotes, xG) — pas d'appel API
      → Puis : python backtest_football.py --simulate --report

  python backtest_football.py --reset-full
      → Supprime backtest_data.db + backtest_results.csv (tout effacer)
      → Puis : python backtest_football.py --collect --simulate --report
      → Consomme le quota API

  python backtest_football.py --reset --simulate --report
      → Reset soft + regénération en une commande

  python backtest_football.py --collect --ligue Championship --odds-only
      → Recollecte cotes historiques pour une seule ligue (fixtures/xG déjà en base)

  python backtest_football.py --collect --saisons 2025 --europe-only
      → Ajoute saison 2025-26 (API season=2025) : fixtures + xG + cotes, ligues EU only

  python backtest_football.py --collect --saisons 2021,2022
      → Ajoute saisons 2021-22 et 2022-23 (fixtures + xG + cotes Pinnacle H-24/clôture)

  python backtest_football.py --collect --saisons 2025 --europe-only --odds-only
      → Cotes 2025-26 uniquement (fixtures/xG déjà en base)

  python backtest_football.py --tune --tune-metric clv
      → Calibration walk-forward orientee CLV AH (H-24 vs cloture Pinnacle)
      → Grille inclut dc_xg_blend (poids DC vs xG, plus le 50/50 fixe)

  python backtest_football.py --tune --tune-metric blend --tune-clv-weight 0.6

  python backtest_football.py --explore-ev
      → Compare volume / CLV / ROI par seuil ev_max (post-hoc sur signaux en base)

  python backtest_football.py --simulate --ev-max-spreads 0.09 --report
      → Backtest avec plafond EV AH 9% (totaux inchanges)

  python backtest_football.py --simulate --p1-ah --ev-max-totals 0.12 --calibrer-ah platt --report
      → P1 + plafond totaux + calibration walk-forward AH

  python backtest_football.py --fit-calibration platt
      → Fit offline bt_signaux → foot_calibration_ah.json (live : FOOT_CALIB_AH=true)

Dashboard Streamlit : après --report, menu ⋮ → Clear cache → Rerun
Si le CSV distant PythonAnywhere est utilisé, uploader le nouveau backtest_results.csv.

Résultats dans backtest_results.csv et imprimés dans la console.
"""

import argparse
import asyncio
import aiohttp
import aiosqlite
import numpy as np
import csv
import os
import sys
import unicodedata
from scipy.stats import poisson
from scipy.optimize import minimize_scalar, minimize
from thefuzz import process
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from config_env import env_files_hint, load_project_env
from foot_params import (
    N_PRIOR_DEFAULT,
    RHO_DEFAULT,
    get_dc_half_life_days,
    get_dc_xg_blend,
    get_n_prior,
    get_rho_fallback,
    get_xg_half_life_days,
    save_tuned_params,
    xg_decay_rate,
)
from odds_devig import cote_fair_2way
from utils import (
    construire_tableau_objectifs_clv_ah,
    formater_objectifs_clv_ah_texte,
    get_ev_min_spreads_ligue,
    EV_MIN_SPREADS_TIER,
    CalibrateurWalkForwardAH,
    prob_implicite_ah,
    ajuster_ev_proportionnel,
    shrink_proba_vers_marche,
    outcome_binaire_ah,
    fit_calibration_ah_par_ligue,
    save_calibration_ah,
    brier_score_prob,
    CALIB_AH_MIN_SAMPLES_PLATT,
    CALIB_AH_MIN_SAMPLES_ISOTONIC,
)

load_project_env("foot")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
API_ODDS_KEY     = os.getenv("API_ODDS_KEY")

URL_FOOTBALL = "https://v3.football.api-sports.io"

def _env_bool_bt(key: str, default: bool) -> bool:
    v = os.environ.get(key, str(default)).strip().lower()
    return v in ("1", "true", "yes", "on")


# Fatigue AH (parité live sniper_bot_foot — coupe UEFA absente de bt_fixtures)
FOOT_FATIGUE_AH_ACTIF = _env_bool_bt("FOOT_FATIGUE_AH_ACTIF", True)
FOOT_FATIGUE_FENETRE_J = float(os.environ.get("FOOT_FATIGUE_FENETRE_J", "7"))
FOOT_FATIGUE_MAX_MATCHS = int(os.environ.get("FOOT_FATIGUE_MAX_MATCHS", "3"))
FOOT_FATIGUE_MIN_REPOS_J = float(os.environ.get("FOOT_FATIGUE_MIN_REPOS_J", "3"))
FOOT_FATIGUE_AH_MODE = os.environ.get("FOOT_FATIGUE_AH_MODE", "either").strip().lower()

# Shrink AH model ↔ no-vig Pinnacle (même clés que sniper_bot_foot)
FOOT_AH_SHRINK_ACTIF = _env_bool_bt("FOOT_AH_SHRINK_ACTIF", True)
FOOT_AH_SHRINK_W = float(os.environ.get("FOOT_AH_SHRINK_W", "0.70"))


def _headers_football():
    return {"x-apisports-key": API_FOOTBALL_KEY, "v": "3"}


def verifier_cles_api():
    """Bloque la collecte si les clés API ne sont pas chargées."""
    missing = [k for k, v in (
        ("API_FOOTBALL_KEY", API_FOOTBALL_KEY),
        ("API_ODDS_KEY", API_ODDS_KEY),
    ) if not v]
    if not missing:
        return True
    print("\n❌ Clés API manquantes : " + ", ".join(missing))
    print("   Créez un fichier .env à la racine du projet avec :")
    print("   API_FOOTBALL_KEY=votre_cle")
    print("   API_ODDS_KEY=votre_cle")
    print("\n   Configurez vos clés dans : " + env_files_hint("foot"))
    return False
DB_PATH      = "backtest_data.db"


def verifier_fichier_db(path=DB_PATH):
    """Vérifie que la base SQLite est lisible (évite 'file is not a database')."""
    import sqlite3
    if not os.path.exists(path):
        print(f"\n❌ Fichier absent : {path}")
        return False
    size = os.path.getsize(path)
    if size < 100_000:
        print(f"\n❌ {path} trop petit ({size:,} octets) — collecte probablement incomplète.")
        return False
    try:
        conn = sqlite3.connect(path)
        n_fix = conn.execute("SELECT COUNT(*) FROM bt_fixtures").fetchone()[0]
        n_odds = conn.execute("SELECT COUNT(*) FROM bt_odds_h24").fetchone()[0]
        conn.close()
        print(f"  ✅ Base OK — {n_fix:,} fixtures, {n_odds:,} cotes H-24")
        return True
    except sqlite3.Error as e:
        print(f"\n❌ {path} illisible ou corrompu ({e})")
        print("\n   → Sur PythonAnywhere (Bash), créez une copie propre puis retéléchargez :")
        print("     cd ~")
        print("     python3 -c \"import sqlite3; s=sqlite3.connect('backtest_data.db');")
        print("     d=sqlite3.connect('backtest_export.db'); s.backup(d); d.close(); s.close();")
        print("     print('OK', s.execute('select count(*) from bt_fixtures').fetchone()[0])\"")
        print("\n   → Files : téléchargez backtest_export.db")
        print("     Renommez-le backtest_data.db sur votre PC (remplace l'ancien).")
        return False

# ─────────────────────────────────────────────────────────────
# ⚙️  CONFIGURATION
# ─────────────────────────────────────────────────────────────
SAISONS_BACKTEST = [2021, 2022, 2023, 2024]   # 2021 = 2021-22 (ligues hivernales)
SAISONS_EUROPE_EXTRA = [2025]       # 2025 = 2025-26 — collectée en plus pour l'Europe

# Ligues calendrier estival (saison = année civile) — hors lot EU 2025-26 par défaut
LIGUES_ESTIVALES = {71, 113, 253, 103}  # Brésil, Allsvenskan, MLS, Eliteserien

# Championnats européens (calendrier hivernal) — reçoivent SAISONS_EUROPE_EXTRA
LIGUES_EUROPEENNES = {
    140, 78, 88, 135, 94, 203, 61, 141, 39, 40, 144, 136,
}

# Preset P1 — réduction volume AH (live + backtest : EV max 9% + tier ; cap optionnel via --max-ah-ligue-saison)
# Steam AH : live only (FOOT_STEAM_* dans sniper) — pas de snapshots intradayout en bt_odds_h24
P1_AH_EV_MAX_SPREADS = 0.09
P1_AH_MAX_LIGUE_SAISON = 45  # exploration only — plus dans --p1-ah (parité live)

CHAMPIONNATS = [
    {"nom": "La Liga",          "id": 140, "key": "soccer_spain_la_liga",            "c1": 4,  "euro": 6,  "rel": 18, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Bundesliga",       "id": 78,  "key": "soccer_germany_bundesliga",        "c1": 4,  "euro": 6,  "rel": 16, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Eredivisie",       "id": 88,  "key": "soccer_netherlands_eredivisie",    "c1": 2,  "euro": 5,  "rel": 16, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Serie A",          "id": 135, "key": "soccer_italy_serie_a",             "c1": 4,  "euro": 6,  "rel": 18, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Primeira Liga",    "id": 94,  "key": "soccer_portugal_primeira_liga",    "c1": 2,  "euro": 5,  "rel": 16, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Süper Lig",        "id": 203, "key": "soccer_turkey_super_league",       "c1": 2,  "euro": 4,  "rel": 17, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Allsvenskan",      "id": 113, "key": "soccer_sweden_allsvenskan",        "c1": 3,  "euro": 3,  "rel": 14, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Série A Brésil",   "id": 71,  "key": "soccer_brazil_campeonato",         "c1": 6,  "euro": 6,  "rel": 17, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Ligue 1",          "id": 61,  "key": "soccer_france_ligue_one",          "c1": 4,  "euro": 6,  "rel": 16, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "LaLiga 2",         "id": 141, "key": "soccer_spain_segunda_division",    "c1": 2,  "euro": 6,  "rel": 19, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Premier League",   "id": 39,  "key": "soccer_epl",                       "c1": 4,  "euro": 6,  "rel": 18, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Championship",     "id": 40,  "key": "soccer_efl_champ",                 "c1": 2,  "euro": 6,  "rel": 22, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "MLS",              "id": 253, "key": "soccer_usa_mls",                   "c1": 7,  "euro": 9,  "rel": 99, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Eliteserien",      "id": 103, "key": "soccer_norway_eliteserien",        "c1": 2,  "euro": 4,  "rel": 14, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Jupiler Pro",      "id": 144, "key": "soccer_belgium_first_div",         "c1": 6,  "euro": 12, "rel": 13, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Serie B",          "id": 136, "key": "soccer_italy_serie_b",             "c1": 2,  "euro": 8,  "rel": 16, "ev_min": 0.05, "ev_max": 0.15},
]

KELLY_FRAC = 0.05
MIN_COTE = 1.70
# Cotes backtest collectées à H-24 (table bt_odds_h24).
# Live (sniper_bot_foot) : bande FOOT_SCAN_HEURES_MIN/MAX (défaut 18–30) autour de ce snapshot.
H_ODDS_BACKTEST = 24.0


def _semaine_dc(date_utc):
    """Bucket hebdomadaire pour cache DC (1 MLE/semaine/ligue, pas 1/match)."""
    try:
        dt = datetime.fromisoformat(date_utc.replace('Z', '+00:00'))
        y, w, _ = dt.isocalendar()
        return (y, w)
    except Exception:
        return date_utc[:10]


def _jour_cache(date_utc):
    return date_utc[:10] if date_utc else ""


def calculer_poids_dyn(hr: float) -> float:
    """
    Poids du modèle dans le blend EV — identique au bot live (sniper_bot_foot.py).
    Courbe ancrée [0.9h, 36h] : 10% → 30%. Backtest appelle avec hr=H_ODDS_BACKTEST (24).
    Live ne prend des paris que dans FOOT_SCAN_* (≈ H-30→H-18), donc proche de ce poids.
    """
    return 0.10 + 0.20 * min(1.0, (hr - 0.9) / (36.0 - 0.9))

NAME_MAPPING = {
    # 🇫🇷 LIGUE 1
    "Paris Saint-Germain": "Paris Saint Germain",
    "Olympique Lyonnais": "Lyon", "Olympique de Marseille": "Marseille",
    "Stade Rennais FC": "Rennes", "Stade Rennais": "Rennes",
    "Stade de Reims": "Reims", "AS Monaco": "Monaco", "OGC Nice": "Nice",
    "RC Lens": "Lens", "Lille OSC": "Lille", "FC Nantes": "Nantes",
    "RC Strasbourg Alsace": "Strasbourg", "RC Strasbourg": "Strasbourg",
    "Montpellier HSC": "Montpellier", "Stade Brestois 29": "Brest",
    "FC Lorient": "Lorient", "AJ Auxerre": "Auxerre", "Le Havre AC": "Le Havre",
    "Toulouse FC": "Toulouse", "Angers SCO": "Angers",
    "AS Saint-Étienne": "Saint-Etienne", "AS Saint-Etienne": "Saint-Etienne",
    "Girondins de Bordeaux": "Bordeaux",
    # 🇪🇸 LA LIGA
    "Athletic Club": "Athletic Bilbao", "Athletic Club de Bilbao": "Athletic Bilbao",
    "Atlético Madrid": "Atletico Madrid", "Deportivo Alavés": "Alaves",
    "Cádiz CF": "Cadiz", "RC Celta": "Celta Vigo", "RCD Espanyol": "Espanyol",
    "RCD Mallorca": "Mallorca", "Getafe CF": "Getafe", "CA Osasuna": "Osasuna",
    "Real Betis Balompié": "Real Betis", "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia", "Real Valladolid": "Valladolid",
    "Girona FC": "Girona", "UD Las Palmas": "Las Palmas",
    "CD Leganés": "Leganes", "Villarreal CF": "Villarreal",
    "UD Almería": "Almeria", "Granada CF": "Granada",
    # 🇪🇸 LALIGA 2
    "SD Huesca": "Huesca", "Real Oviedo": "Oviedo",
    "Sporting Gijón": "Sporting Gijon", "Sporting de Gijón": "Sporting Gijon",
    "Real Zaragoza": "Zaragoza", "SD Eibar": "Eibar", "Málaga CF": "Malaga",
    "Racing Club de Santander": "Racing Santander", "Burgos CF": "Burgos",
    "Elche CF": "Elche", "Levante UD": "Levante",
    "Albacete Balompié": "Albacete", "FC Cartagena": "Cartagena",
    "CD Tenerife": "Tenerife", "Córdoba CF": "Cordoba", "CD Eldense": "Eldense",
    # 🇩🇪 BUNDESLIGA
    "FC Bayern München": "Bayern Munich", "Bayern München": "Bayern Munich",
    "1. FC Köln": "FC Koeln", "Borussia Mönchengladbach": "Borussia Monchengladbach",
    "TSG Hoffenheim": "Hoffenheim", "TSG 1899 Hoffenheim": "Hoffenheim",
    "SC Freiburg": "Freiburg", "VfB Stuttgart": "Stuttgart",
    "1. FSV Mainz 05": "Mainz", "FC Augsburg": "Augsburg",
    "SV Werder Bremen": "Werder Bremen", "VfL Wolfsburg": "Wolfsburg",
    "VfL Bochum 1848": "Bochum", "Hertha BSC": "Hertha Berlin",
    "1. FC Union Berlin": "Union Berlin", "FC Union Berlin": "Union Berlin",
    "1. FC Heidenheim 1846": "Heidenheim", "1. FC Heidenheim": "Heidenheim",
    "FC St. Pauli": "St. Pauli", "SV Darmstadt 98": "Darmstadt 98",
    "Holstein Kiel": "Holstein Kiel",
    # 🇮🇹 SERIE A & SERIE B
    "Inter": "Inter Milan", "AC Milan": "AC Milan", "AS Roma": "Roma",
    "SS Lazio": "Lazio", "ACF Fiorentina": "Fiorentina",
    "Atalanta BC": "Atalanta", "Hellas Verona": "Verona",
    "Torino FC": "Torino", "Bologna FC 1909": "Bologna",
    "Genoa CFC": "Genoa", "US Sassuolo Calcio": "Sassuolo",
    "Udinese Calcio": "Udinese", "Cagliari Calcio": "Cagliari",
    "Empoli FC": "Empoli", "US Lecce": "Lecce",
    "Parma Calcio 1913": "Parma", "Como 1907": "Como",
    "Venezia FC": "Venezia", "US Salernitana 1919": "Salernitana",
    "Frosinone Calcio": "Frosinone", "US Cremonese": "Cremonese",
    "AC Pisa 1909": "Pisa", "Brescia Calcio": "Brescia",
    "Spezia Calcio": "Spezia", "SSC Bari": "Bari",
    "UC Sampdoria": "Sampdoria", "Modena FC 2018": "Modena",
    # 🏴󠁧󠁢󠁥󠁮󠁧󠁿 PREMIER LEAGUE & CHAMPIONSHIP
    "Manchester United": "Manchester United", "Manchester City": "Manchester City",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "Wolverhampton Wanderers": "Wolverhampton",
    "Brighton & Hove Albion": "Brighton", "West Ham United": "West Ham",
    "Newcastle United": "Newcastle", "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich", "Southampton": "Southampton",
    "Sheffield United": "Sheffield United", "Sheff Utd": "Sheffield United",
    "Sheffield Wednesday": "Sheffield Wednesday", "Sheff Wed": "Sheffield Wednesday",
    "Queens Park Rangers": "QPR", "Leeds United": "Leeds",
    "West Bromwich Albion": "West Brom", "Swansea City": "Swansea",
    "Luton Town": "Luton", "Hull City": "Hull", "Middlesbrough": "Middlesbrough",
    "Coventry City": "Coventry", "Sunderland": "Sunderland",
    "Plymouth Argyle": "Plymouth", "Bristol City": "Bristol City",
    "Watford": "Watford", "Norwich City": "Norwich", "Cardiff City": "Cardiff",
    "Stoke City": "Stoke", "Blackburn Rovers": "Blackburn",
    "Preston North End": "Preston", "Burnley FC": "Burnley",
    "Burnley": "Burnley", "Millwall": "Millwall",
    "Huddersfield Town": "Huddersfield", "Birmingham City": "Birmingham",
    "Rotherham United": "Rotherham", "Derby County": "Derby",
    "Portsmouth": "Portsmouth", "Oxford United": "Oxford Utd",
    "Nottingham Forest": "Nottingham Forest", "Nottm Forest": "Nottingham Forest",
    # 🇳🇱 EREDIVISIE
    "AFC Ajax": "Ajax", "PSV Eindhoven": "PSV", "AZ Alkmaar": "AZ",
    "FC Utrecht": "Utrecht", "FC Twente": "Twente",
    "SC Heerenveen": "Heerenveen", "PEC Zwolle": "PEC Zwolle",
    "Almere City FC": "Almere City", "FC Groningen": "Groningen",
    "Sparta Rotterdam": "Sparta Rotterdam", "RKC Waalwijk": "RKC Waalwijk",
    "NEC Nijmegen": "NEC",
    # 🇵🇹 PRIMEIRA LIGA
    "SL Benfica": "Benfica", "FC Porto": "Porto", "Sporting CP": "Sporting CP",
    "SC Braga": "Braga", "Vitória SC": "Vitoria Guimaraes",
    "Vitoria SC": "Vitoria Guimaraes", "Gil Vicente FC": "Gil Vicente",
    "Boavista FC": "Boavista", "Moreirense FC": "Moreirense",
    "GD Estoril Praia": "Estoril", "GD Chaves": "Chaves",
    "Casa Pia AC": "Casa Pia", "CD Famalicão": "Famalicao",
    "Rio Ave FC": "Rio Ave", "SC Farense": "Farense",
    "CF Arouca": "Arouca", "CD Nacional": "Nacional",
    # 🇹🇷 SÜPER LIG
    "Galatasaray SK": "Galatasaray", "Fenerbahçe SK": "Fenerbahce",
    "Fenerbahce SK": "Fenerbahce", "Beşiktaş JK": "Besiktas",
    "Besiktas JK": "Besiktas", "Kasımpaşa SK": "Kasimpasa",
    "İstanbul Başakşehir FK": "Basaksehir", "Istanbul Basaksehir FK": "Basaksehir",
    "Göztepe SK": "Goztepe", "Yılport Samsunspor": "Samsunspor",
    "Fatih Karagümrük SK": "Karagumruk", "Sivasspor": "Sivasspor",
    "Alanyaspor": "Alanyaspor", "Konyaspor": "Konyaspor",
    "Antalyaspor": "Antalyaspor", "Kayserispor": "Kayserispor",
    "Gaziantep FK": "Gaziantep", "MKE Ankaragücü": "Ankaragucu",
    "Adana Demirspor": "Adana Demirspor",
    # 🇧🇷 SÉRIE A BRÉSIL
    "Athletico Paranaense": "Athletico-PR", "Atlético Paranaense": "Athletico-PR",
    "Atlético-MG": "Atletico Mineiro", "Atlético Mineiro": "Atletico Mineiro",
    "Bragantino": "Red Bull Bragantino", "Grêmio": "Gremio",
    "São Paulo FC": "Sao Paulo", "Sao Paulo": "Sao Paulo",
    "Sport Club Corinthians Paulista": "Corinthians",
    "Sociedade Esportiva Palmeiras": "Palmeiras",
    "Club de Regatas do Flamengo": "Flamengo",
    "Fluminense FC": "Fluminense", "Botafogo FR": "Botafogo",
    "CR Vasco da Gama": "Vasco da Gama", "EC Bahia": "Bahia",
    "Fortaleza EC": "Fortaleza", "Sport Club Internacional": "Internacional",
    "Cruzeiro EC": "Cruzeiro", "EC Juventude": "Juventude",
    "Criciúma EC": "Criciuma", "Santos FC": "Santos",
    "Ceará SC": "Ceara", "Coritiba FC": "Coritiba",
    # 🇸🇪 ALLSVENSKAN
    "AIK Fotboll": "AIK", "Malmö FF": "Malmo FF",
    "Djurgårdens IF": "Djurgarden", "IFK Göteborg": "IFK Goteborg",
    "BK Häcken": "BK Hacken", "IFK Norrköping": "IFK Norrkoping",
    "Mjällby AIF": "Mjallby", "Halmstads BK": "Halmstad",
    "IK Sirius FK": "Sirius", "Kalmar FF": "Kalmar",
    "Degerfors IF": "Degerfors", "GIF Sundsvall": "Sundsvall",
    # 🇳🇴 ELITESERIEN — canonique = libellés Odds API / Pinnacle (historical)
    "FK Bodø/Glimt": "Bodø/Glimt", "Bodø/Glimt": "Bodø/Glimt", "Bodo/Glimt": "Bodø/Glimt",
    "Molde FK": "Molde", "Molde": "Molde",
    "Rosenborg BK": "Rosenborg", "Rosenborg": "Rosenborg",
    "Brann": "SK Brann", "SK Brann": "SK Brann",
    "Viking": "Viking FK", "Viking FK": "Viking FK",
    "Tromsø IL": "Tromso", "IL Tromso": "Tromso", "Tromso": "Tromso",
    "Stabæk Fotball": "Stabaek", "Stabaek": "Stabaek",
    "Strømsgodset IF": "Stromsgodset", "Stromsgodset": "Stromsgodset",
    "FK Haugesund": "Haugesund", "Haugesund": "Haugesund",
    "Odd BK": "Odds BK", "ODD Ballklubb": "Odds BK", "Odds BK": "Odds BK",
    "Sandefjord Fotball": "Sandefjord", "Sandefjord": "Sandefjord",
    "Lillestrøm SK": "Lillestrom", "Lillestrom": "Lillestrom",
    "Aalesunds FK": "Aalesund", "Aalesund": "Aalesund",
    "Kristiansund BK": "Kristiansund BK",
    "Sarpsborg 08 FF": "Sarpsborg FK", "Sarpsborg FK": "Sarpsborg FK",
    "Valerenga": "Vålerenga", "Vålerenga": "Vålerenga", "Vålerenga IF": "Vålerenga",
    "Mjondalen": "Mjøndalen", "Mjøndalen IF": "Mjøndalen", "Mjøndalen": "Mjøndalen",
    "Ham-Kam": "HamKam", "HamKam": "HamKam",
    "FK Jerv": "Jerv", "jerv": "Jerv", "Jerv": "Jerv",
    "Kongsvinger IL": "Kongsvinger", "Kongsvinger": "Kongsvinger",
    # 🇧🇪 JUPILER PRO LEAGUE — canonique = libellés Odds API / Pinnacle
    # Noms API-Football (bt_fixtures)
    "Antwerp": "Royal Antwerp",
    "Union St. Gilloise": "Union Saint-Gilloise",
    "OH Leuven": "Leuven",
    "Cercle Brugge": "Cercle Brugge KSV",
    "Zulte Waregem": "SV Zulte-Waregem",
    "St. Truiden": "Sint Truiden",
    "KVC Westerlo": "Westerlo",
    "Club Brugge KV": "Club Brugge",
    "KV Mechelen": "KV Mechelen",
    "Anderlecht": "Anderlecht",
    "Genk": "Genk",
    "Gent": "Gent",
    "Charleroi": "Charleroi",
    "Standard Liege": "Standard Liege",
    "Standard Liège": "Standard Liege",
    "Dender": "Dender",
    "AS Eupen": "Eupen",
    "Kortrijk": "Kortrijk",
    "Beerschot VA": "Beerschot",
    "RWDM": "RWDM",
    "RAAL La Louvière": "RAAL La Louvière",
    "RAAL La Louviere": "RAAL La Louvière",
    "Liège": "RFC Liege",
    "Liege": "RFC Liege",
    # Variantes Odds API / bookmakers
    "RSC Anderlecht": "Anderlecht",
    "KAA Gent": "Gent",
    "Standard de Liège": "Standard Liege",
    "KRC Genk": "Genk",
    "Royal Antwerp FC": "Royal Antwerp",
    "Royale Union Saint-Gilloise": "Union Saint-Gilloise",
    "Union Saint Gilloise": "Union Saint-Gilloise",
    "R. Charleroi SC": "Charleroi",
    "Sporting Charleroi": "Charleroi",
    "Cercle Brugge KSV": "Cercle Brugge KSV",
    "Sint-Truidense VV": "Sint Truiden",
    "KV Kortrijk": "Kortrijk",
    "K. Beerschot VA": "Beerschot",
    "KAS Eupen": "Eupen",
    "Oud-Heverlee Leuven": "Leuven",
    "OHL Leuven": "Leuven",
    "Westerlo": "Westerlo",
    "RWDM Brussels FC": "RWDM",
    "SV Zulte-Waregem": "SV Zulte-Waregem",
    "Sint Truiden": "Sint Truiden",
    "FC Verbroedering Dender Eendracht Halle": "Dender",
    "FCV Dender EH": "Dender",
    "K. Beerschot AC": "Beerschot",
    "Lommel United": "Lommel United",
    "Patro Eisden": "Patro Eisden",
    # 🇺🇸 MLS
    "Inter Miami CF": "Inter Miami", "LA Galaxy": "Los Angeles Galaxy",
    "LAFC": "Los Angeles FC", "New York Red Bulls": "NY Red Bulls",
    "New York City FC": "New York City FC", "Seattle Sounders FC": "Seattle Sounders",
    "Atlanta United FC": "Atlanta United", "D.C. United": "DC United",
    "Colorado Rapids": "Colorado Rapids", "Houston Dynamo FC": "Houston Dynamo",
    "Minnesota United FC": "Minnesota United", "CF Montréal": "CF Montreal",
    "Chicago Fire FC": "Chicago Fire", "Vancouver Whitecaps FC": "Vancouver Whitecaps",
    "St. Louis City SC": "St. Louis City", "Austin FC": "Austin FC",
}

# ─────────────────────────────────────────────────────────────
# 🗄️  BASE DE DONNÉES
# ─────────────────────────────────────────────────────────────
async def configure_sqlite(conn):
    """WAL + busy_timeout — évite 'database is locked' sur PythonAnywhere."""
    await conn.execute("PRAGMA busy_timeout=120000")
    try:
        await conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    await conn.execute("PRAGMA synchronous=NORMAL")


async def persister_signaux(pending):
    """
    Écrit bt_signaux via une connexion dédiée (courte durée).
    Retourne le nombre de lignes écrites, ou -1 en cas d'échec.
    """
    if not pending:
        await _vider_signaux()
        print("\n  💾 bt_signaux vidée (0 signaux)")
        return 0

    sql = "INSERT OR REPLACE INTO bt_signaux VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    for attempt in range(8):
        try:
            async with aiosqlite.connect(DB_PATH, timeout=120.0) as conn:
                await configure_sqlite(conn)
                await conn.execute("DELETE FROM bt_signaux")
                await conn.executemany(sql, pending)
                await conn.commit()
            print(f"\n  💾 {len(pending)} signaux enregistrés en base")
            return len(pending)
        except Exception as e:
            locked = "locked" in str(e).lower()
            if locked and attempt < 7:
                wait = min(30, 2 ** attempt)
                print(f"  ⏳ DB verrouillée, nouvel essai dans {wait}s ({attempt + 1}/8)...")
                await asyncio.sleep(wait)
                continue
            print(f"\n  ❌ Impossible d'écrire bt_signaux : {e}")
            if locked:
                print("  💡 Sur PythonAnywhere :")
                print("     ps aux | grep python")
                print("     kill <PID>  # collect ou autre simulate en cours")
                print("     rm -f backtest_data.db-wal backtest_data.db-shm  # si aucun process actif")
            return -1
    return -1


async def _vider_signaux():
    for attempt in range(5):
        try:
            async with aiosqlite.connect(DB_PATH, timeout=120.0) as conn:
                await configure_sqlite(conn)
                await conn.execute("DELETE FROM bt_signaux")
                await conn.commit()
            return
        except Exception as e:
            if "locked" in str(e).lower() and attempt < 4:
                await asyncio.sleep(2 ** attempt)
                continue
            raise


async def init_db(conn):
    await conn.executescript("""
        CREATE TABLE IF NOT EXISTS bt_fixtures (
            id          INTEGER PRIMARY KEY,
            ligue_id    INTEGER,
            saison      INTEGER,
            date_utc    TEXT,
            home_id     INTEGER,
            away_id     INTEGER,
            home_name   TEXT,
            away_name   TEXT,
            gh          INTEGER,
            ga          INTEGER
        );
        CREATE TABLE IF NOT EXISTS bt_xg (
            fixture_id  INTEGER,
            team_id     INTEGER,
            xg_p        REAL,
            xg_c        REAL,
            PRIMARY KEY (fixture_id, team_id)
        );
        CREATE TABLE IF NOT EXISTS bt_odds_h24 (
            fixture_id  INTEGER,
            market      TEXT,
            outcome     TEXT,
            h_val       REAL,
            cote        REAL,
            PRIMARY KEY (fixture_id, market, outcome, h_val)
        );
        CREATE TABLE IF NOT EXISTS bt_odds_cloture (
            fixture_id  INTEGER,
            market      TEXT,
            outcome     TEXT,
            h_val       REAL,
            cote        REAL,
            PRIMARY KEY (fixture_id, market, outcome, h_val)
        );
        CREATE TABLE IF NOT EXISTS bt_signaux (
            fixture_id  INTEGER,
            ligue_id    INTEGER,
            saison      INTEGER,
            market      TEXT,
            outcome     TEXT,
            h_val       REAL,
            cote_h24    REAL,
            cote_cloture REAL,
            ev_modele   REAL,
            kelly       REAL,
            mise        REAL,
            gh          INTEGER,
            ga          INTEGER,
            resultat    REAL,
            clv         REAL,
            p_modele    REAL,
            p_cal       REAL,
            PRIMARY KEY (fixture_id, market, outcome, h_val)
        );
    """)
    for migration in (
        "ALTER TABLE bt_signaux ADD COLUMN p_modele REAL DEFAULT NULL",
        "ALTER TABLE bt_signaux ADD COLUMN p_cal REAL DEFAULT NULL",
    ):
        try:
            await conn.execute(migration)
        except Exception:
            pass
    await conn.commit()


# ─────────────────────────────────────────────────────────────
# 🌐  HTTP HELPERS
# ─────────────────────────────────────────────────────────────
semaphore = asyncio.Semaphore(3)

async def fetch(session, url, headers=None, params=None):
    async with semaphore:
        try:
            async with session.get(url, headers=headers, params=params, timeout=20) as r:
                if r.status == 200:
                    return await r.json()
                if r.status == 429:
                    print("⏳ Rate limit — pause 30s")
                    await asyncio.sleep(30)
                else:
                    print(f"  ⚠️ HTTP {r.status} — {url.split('?')[0][-60:]}")
                return None
        except Exception as e:
            print(f"  ⚠️ fetch error: {e}")
            return None


def filtrer_ligues(nom: str | None):
    """Filtre CHAMPIONNATS par nom partiel (insensible à la casse)."""
    if not nom:
        return list(CHAMPIONNATS)
    nom_l = nom.lower().strip()
    matches = [l for l in CHAMPIONNATS if nom_l in l['nom'].lower()]
    if not matches:
        print(f"\n❌ Ligue « {nom} » introuvable. Ligues disponibles :")
        for l in CHAMPIONNATS:
            print(f"   • {l['nom']} (key: {l['key']})")
        sys.exit(1)
    return matches


def saison_pour_ligue(ligue_id, annee):
    """Ligues estivales : saison = année civile. Ligues hivernales : saison = année de début."""
    return annee


def saisons_pour_ligue(ligue_id, saisons_override=None):
    """Saisons à collecter / simuler pour une ligue."""
    if saisons_override is not None:
        return sorted({int(s) for s in saisons_override})
    saisons = list(SAISONS_BACKTEST)
    if ligue_id in LIGUES_EUROPEENNES:
        for y in SAISONS_EUROPE_EXTRA:
            if y not in saisons:
                saisons.append(y)
    return sorted(saisons)


def filtrer_ligues_europe(ligues=None):
    """Restreint aux championnats européens (calendrier hivernal)."""
    src = ligues or CHAMPIONNATS
    return [l for l in src if l['id'] in LIGUES_EUROPEENNES]


def parser_saisons_cli(texte: str) -> list[int]:
    """Parse --saisons 2023,2024,2025"""
    out = []
    for part in texte.split(','):
        part = part.strip()
        if not part:
            continue
        y = int(part)
        if y < 2000 or y > 2100:
            raise ValueError(f"saison invalide : {y}")
        out.append(y)
    if not out:
        raise ValueError("aucune saison dans --saisons")
    return sorted(set(out))


# ─────────────────────────────────────────────────────────────
# 📐  MODÈLE MATHÉMATIQUE (copie des fonctions du bot principal)
# ─────────────────────────────────────────────────────────────
def generer_matrice(l_dom, l_ext, rho=-0.12):
    """
    Matrice Dixon-Coles — taille dynamique alignée sur generer_matrice_dixon() du bot live.
    Couvre 99.8% de la masse Poisson (min 10, max 15) pour éviter la troncature
    sur les ligues à fort volume de buts (λ ≥ 3.5).
    """
    max_goals = max(10, min(int(np.ceil(poisson.ppf(0.998, max(l_dom, l_ext)))) + 1, 15))
    p_d = [poisson.pmf(i, l_dom) for i in range(max_goals)]
    p_e = [poisson.pmf(i, l_ext) for i in range(max_goals)]
    m = np.outer(p_d, p_e).astype(float)
    m[0, 0] *= max(0, 1 - l_dom * l_ext * rho)
    m[1, 0] *= max(0, 1 + l_ext * rho)
    m[0, 1] *= max(0, 1 + l_dom * rho)
    m[1, 1] *= max(0, 1 - rho)
    return m / np.sum(m)


_EPS = 1e-6   # tolérance pour comparaisons float (quarts de handicap)

def _payout_ah(res_net, cote):
    """5 issues Asian Handicap : full win / half win / push / half loss / full loss."""
    if res_net > 0.25 + _EPS:           return cote            # full win
    if abs(res_net - 0.25) < _EPS:      return 1.0 + (cote - 1.0) / 2  # half win
    if abs(res_net) < _EPS:             return 1.0             # push
    if abs(res_net + 0.25) < _EPS:      return 0.5             # half loss
    return 0.0                                                  # full loss

def _x_kelly_ah(res_net, cote):
    """Gain net pour Kelly mean-variance."""
    if res_net > 0.25 + _EPS:           return cote - 1.0
    if abs(res_net - 0.25) < _EPS:      return (cote - 1.0) / 2.0
    if abs(res_net) < _EPS:             return 0.0
    if abs(res_net + 0.25) < _EPS:      return -0.5
    return -1.0

def _payout_total(res_net, cote):
    """5 issues Asian Total : même logique que AH."""
    return _payout_ah(res_net, cote)

def _x_kelly_total(res_net, cote):
    return _x_kelly_ah(res_net, cote)


def ev_ah(mat, h, is_home, cote):
    """EV Asian Handicap — signe et 5 issues identiques au bot principal."""
    esp = 0.0
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            prob = mat[i, j]
            if prob < 0.0001:
                continue
            diff = (i - j) if is_home else (j - i)
            res_net = diff + h          # ← signe correct (même convention que le bot)
            esp += prob * _payout_ah(res_net, cote)
    return esp - 1.0


def ev_total(mat, h, is_over, cote):
    """EV Total Asiatique — 5 issues."""
    esp = 0.0
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            prob = mat[i, j]
            if prob < 0.0001:
                continue
            tot = i + j
            res_net = (tot - h) if is_over else (h - tot)
            esp += prob * _payout_total(res_net, cote)
    return esp - 1.0


def kelly_ah(mat, h, is_home, cote):
    """Kelly mean-variance AH — 5 issues, signe correct."""
    e1, e2 = 0.0, 0.0
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            prob = mat[i, j]
            if prob < 0.0001:
                continue
            diff = (i - j) if is_home else (j - i)
            res_net = diff + h
            x = _x_kelly_ah(res_net, cote)
            e1 += prob * x
            e2 += prob * x * x
    return (e1 / e2) if e2 > 1e-9 else 0.0


def kelly_total(mat, h, is_over, cote):
    """Kelly mean-variance Total — 5 issues."""
    e1, e2 = 0.0, 0.0
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            prob = mat[i, j]
            if prob < 0.0001:
                continue
            tot = i + j
            res_net = (tot - h) if is_over else (h - tot)
            x = _x_kelly_total(res_net, cote)
            e1 += prob * x
            e2 += prob * x * x
    return (e1 / e2) if e2 > 1e-9 else 0.0


def resultat_ah(gh, ga, h, is_home):
    """Résultat réel AH — même convention de signe que le bot."""
    diff = (gh - ga) if is_home else (ga - gh)
    res_net = diff + h
    if res_net > 0.25 + _EPS:           return 1.0      # full win
    if abs(res_net - 0.25) < _EPS:      return 0.5      # half win
    if abs(res_net) < _EPS:             return 0.0      # push
    if abs(res_net + 0.25) < _EPS:      return -0.5     # half loss
    return -1.0                                          # full loss


def resultat_total(gh, ga, h, is_over):
    """Résultat réel Total Asiatique."""
    tot = gh + ga
    res_net = (tot - h) if is_over else (h - tot)
    if res_net > 0.25 + _EPS:           return 1.0
    if abs(res_net - 0.25) < _EPS:      return 0.5
    if abs(res_net) < _EPS:             return 0.0
    if abs(res_net + 0.25) < _EPS:      return -0.5
    return -1.0


# ─────────────────────────────────────────────────────────────
# 📥  PHASE 1 — COLLECTE DES DONNÉES
# ─────────────────────────────────────────────────────────────
async def collecter_fixtures(conn, session, ligue, saison):
    """Télécharge tous les matchs terminés d'une ligue/saison."""
    url = f"{URL_FOOTBALL}/fixtures?league={ligue['id']}&season={saison}&status=FT"
    data = await fetch(session, url, _headers_football())
    if not data or not data.get('response'):
        return 0

    rows = []
    for f in data['response']:
        rows.append((
            f['fixture']['id'], ligue['id'], saison,
            f['fixture']['date'],
            f['teams']['home']['id'], f['teams']['away']['id'],
            f['teams']['home']['name'], f['teams']['away']['name'],
            f['goals']['home'], f['goals']['away']
        ))

    await conn.executemany(
        "INSERT OR IGNORE INTO bt_fixtures VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    await conn.commit()
    print(f"  ✅ {ligue['nom']} {saison} : {len(rows)} matchs")
    return len(rows)


async def collecter_xg(conn, session, fixture_id, home_id, away_id):
    """Télécharge les xG réels d'un match (ou fait fallback sur les buts)."""
    # Vérifier le cache d'abord
    async with conn.execute(
        "SELECT 1 FROM bt_xg WHERE fixture_id=? AND team_id=?", (fixture_id, home_id)
    ) as cur:
        if await cur.fetchone():
            return

    url = f"{URL_FOOTBALL}/fixtures/statistics?fixture={fixture_id}"
    data = await fetch(session, url, _headers_football())

    xg = {home_id: None, away_id: None}
    if data and data.get('response'):
        for team_stat in data['response']:
            t_id = team_stat['team']['id']
            raw = next(
                (s['value'] for s in team_stat['statistics'] if s['type'] == 'expected_goals'),
                None
            )
            try:
                xg[t_id] = float(raw) if raw not in (None, 'null', '') else None
            except (TypeError, ValueError):
                xg[t_id] = None

    # Fallback sur buts si xG indisponible
    async with conn.execute(
        "SELECT home_id, away_id, gh, ga FROM bt_fixtures WHERE id=?", (fixture_id,)
    ) as cur:
        row = await cur.fetchone()

    if row:
        h_id, a_id, gh, ga = row
        if xg.get(h_id) is None: xg[h_id] = float(gh or 0)
        if xg.get(a_id) is None: xg[a_id] = float(ga or 0)
        # xg_c = xG encaissé = xG de l'adversaire
        rows = [
            (fixture_id, h_id, xg.get(h_id, 0), xg.get(a_id, 0)),
            (fixture_id, a_id, xg.get(a_id, 0), xg.get(h_id, 0)),
        ]
        await conn.executemany("INSERT OR IGNORE INTO bt_xg VALUES (?,?,?,?)", rows)
        await conn.commit()


async def collecter_odds_historiques(conn, session, ligue, date_utc, table, fixture_ids_date):
    """
    Télécharge les cotes Pinnacle à un instant donné pour une ligue entière.
    date_utc : datetime UTC (ex: fixture_date - 24h pour H-24, ou - 5min pour closing)
    Un seul appel API couvre tous les matchs de la ligue à cette date.
    Retourne le nombre de lignes insérées.
    """
    date_str = date_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    url = (f"https://api.the-odds-api.com/v4/historical/sports/{ligue['key']}/odds"
           f"?apiKey={API_ODDS_KEY}&regions=eu&markets=spreads,totals"
           f"&oddsFormat=decimal&bookmakers=pinnacle&date={date_str}")

    raw = await fetch(session, url)
    if not raw:
        return 0
    # L'endpoint /v4/historical/ enveloppe les résultats dans {"timestamp":..., "data":[...]}
    data = raw.get('data', raw) if isinstance(raw, dict) else raw
    if not isinstance(data, list) or not data:
        return 0

    inserted = 0
    for event in data:
        # Trouver le fixture correspondant par fuzzy-match sur les noms d'équipes
        match = trouver_fixture(event['home_team'], event['away_team'],
                                event['commence_time'], fixture_ids_date)
        if not match:
            continue

        fixture_id = match
        pinnacle = next((b for b in event.get('bookmakers', []) if b['key'] == 'pinnacle'), None)
        if not pinnacle:
            continue

        rows = []
        for market in pinnacle['markets']:
            for out in market['outcomes']:
                rows.append((
                    fixture_id, market['key'], out['name'],
                    float(out.get('point', 0)), float(out['price'])
                ))

        if rows:
            await conn.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?,?,?,?,?)", rows)
            inserted += len(rows)
    if inserted:
        await conn.commit()
    return inserted


async def collecter_odds_ligue_saison(conn, session, ligue, saison):
    """Collecte cotes H-24 + clôture pour une ligue/saison (fixtures déjà en base)."""
    async with conn.execute(
        "SELECT id, home_id, away_id, date_utc FROM bt_fixtures WHERE ligue_id=? AND saison=?",
        (ligue['id'], saison)
    ) as cur:
        fixtures = await cur.fetchall()

    if not fixtures:
        print(f"  ⚠️ Aucun fixture en base pour {ligue['nom']} {saison}")
        return 0

    # Ré-association propre (évite cotes orphelines après fix mapping / date)
    await conn.execute(
        "DELETE FROM bt_odds_h24 WHERE fixture_id IN "
        "(SELECT id FROM bt_fixtures WHERE ligue_id=? AND saison=?)",
        (ligue['id'], saison),
    )
    await conn.execute(
        "DELETE FROM bt_odds_cloture WHERE fixture_id IN "
        "(SELECT id FROM bt_fixtures WHERE ligue_id=? AND saison=?)",
        (ligue['id'], saison),
    )
    await conn.commit()

    par_date = defaultdict(dict)
    for fid, hid, aid, date_str in fixtures:
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            date_key = dt.strftime('%Y-%m-%d')
            par_date[date_key][(fid, hid, aid)] = dt
        except Exception:
            pass

    fixture_name_map = {}
    for fid, hid, aid, date_str in fixtures:
        async with conn.execute(
            "SELECT home_name, away_name FROM bt_fixtures WHERE id=?", (fid,)
        ) as cur:
            row = await cur.fetchone()
        if row:
            jour = date_str[:10]
            fixture_name_map[(row[0], row[1], jour)] = fid

    print(f"  📈 Collecte cotes ({ligue['key']}) — {len(par_date)} journées, {len(fixtures)} matchs…")
    total = 0
    for i, (date_key, fixtures_du_jour) in enumerate(par_date.items(), 1):
        sample_dt = next(iter(fixtures_du_jour.values()))
        dt_h24 = sample_dt - timedelta(hours=24)
        dt_close = sample_dt - timedelta(minutes=5)
        total += await collecter_odds_historiques(
            conn, session, ligue, dt_h24, "bt_odds_h24", fixture_name_map
        )
        total += await collecter_odds_historiques(
            conn, session, ligue, dt_close, "bt_odds_cloture", fixture_name_map
        )
        if i % 20 == 0:
            print(f"     … {i}/{len(par_date)} journées", flush=True)
        await asyncio.sleep(0.5)

    async with conn.execute(
        "SELECT COUNT(DISTINCT o.fixture_id) FROM bt_odds_h24 o "
        "JOIN bt_fixtures f ON f.id=o.fixture_id WHERE f.ligue_id=? AND f.saison=?",
        (ligue['id'], saison)
    ) as cur:
        n_fix_odds = (await cur.fetchone())[0]
    print(f"  ✅ {ligue['nom']} {saison} : {total} lignes cotes | {n_fix_odds}/{len(fixtures)} matchs couverts")
    return n_fix_odds


def _canonical_equipe(name: str) -> str:
    """Normalise un nom d'équipe (API-Football ou Odds API) vers libellé canonique."""
    if not name:
        return name
    mapped = NAME_MAPPING.get(name, name)
    nf = unicodedata.normalize('NFKD', mapped)
    return ''.join(c for c in nf if not unicodedata.combining(c))


def trouver_fixture(home_odds, away_odds, commence_time, fixture_map):
    """
    Trouve l'ID du fixture API-Football correspondant à un event Odds API.
    fixture_map : {(home_name, away_name, date_YYYY-MM-DD): fixture_id}
    """
    event_day = commence_time[:10] if commence_time else None
    h_odds = _canonical_equipe(home_odds)
    a_odds = _canonical_equipe(away_odds)
    best_id, best_score = None, 0
    for key, fid in fixture_map.items():
        h_name, a_name, fix_day = key
        if event_day and fix_day != event_day:
            continue
        h_mapped = _canonical_equipe(h_name)
        a_mapped = _canonical_equipe(a_name)
        score_h = process.extractOne(h_odds, [h_mapped])[1]
        score_a = process.extractOne(a_odds, [a_mapped])[1]
        score = (score_h + score_a) / 2
        if score > best_score and score > 75:
            best_score = score
            best_id = fid
        # même paire inversée (rare)
        score_h2 = process.extractOne(h_odds, [a_mapped])[1]
        score_a2 = process.extractOne(a_odds, [h_mapped])[1]
        score2 = (score_h2 + score_a2) / 2
        if score2 > best_score and score2 > 75:
            best_score = score2
            best_id = fid
    return best_id


def _libelle_saison(ligue_id: int, saison: int) -> str:
    """Libellé lisible (ex. 2025 → 2025-26 pour ligues hivernales)."""
    if ligue_id in LIGUES_ESTIVALES:
        return str(saison)
    return f"{saison}-{str(saison + 1)[-2:]}"


async def phase_collecte(conn, session, ligues=None, odds_only=False,
                         saisons_override=None, europe_only=False):
    print("\n" + "="*60)
    print("📥  PHASE 1 — COLLECTE DES DONNÉES HISTORIQUES")
    if odds_only:
        print("  (mode --odds-only : fixtures/xG ignorés, cotes uniquement)")
    if europe_only:
        print("  (filtre : championnats européens uniquement)")
    if saisons_override:
        print(f"  (saisons : {', '.join(str(s) for s in saisons_override)})")
    if ligues and len(ligues) < len(CHAMPIONNATS):
        print(f"  (ligues : {', '.join(l['nom'] for l in ligues)})")
    print("="*60)

    if not verifier_cles_api():
        return

    ligues = ligues or CHAMPIONNATS
    if europe_only:
        ligues = filtrer_ligues_europe(ligues)
        if not ligues:
            print("\n❌ Aucune ligue européenne dans le filtre.")
            return

    for ligue in ligues:
        saisons_l = saisons_pour_ligue(ligue['id'], saisons_override)
        for saison in saisons_l:
            print(f"\n🔄 {ligue['nom']} — Saison {saison} ({_libelle_saison(ligue['id'], saison)})")

            if not odds_only:
                # 1. Fixtures
                n = await collecter_fixtures(conn, session, ligue, saison)
                if n == 0:
                    await collecter_fixtures(conn, session, ligue, saison - 1)

                # 2. xG par fixture
                async with conn.execute(
                    "SELECT id, home_id, away_id, date_utc FROM bt_fixtures WHERE ligue_id=? AND saison=?",
                    (ligue['id'], saison)
                ) as cur:
                    fixtures = await cur.fetchall()

                print(f"  📊 Collecte xG pour {len(fixtures)} matchs...")
                for fid, hid, aid, date_str in fixtures:
                    await collecter_xg(conn, session, fid, hid, aid)

            # 3. Odds historiques H-24 et clôture H-5min
            await collecter_odds_ligue_saison(conn, session, ligue, saison)

    print("\n✅ Phase 1 terminée.")


# ─────────────────────────────────────────────────────────────
# 🔬  PHASE 2 — SIMULATION DU MODÈLE
# ─────────────────────────────────────────────────────────────
async def calculer_ligue_avg(conn, ligue_id, saison, avant_date):
    """Moyenne de buts par équipe par match dans la ligue/saison AVANT avant_date."""
    async with conn.execute("""
        SELECT AVG((CAST(gh AS REAL) + CAST(ga AS REAL)) / 2.0)
        FROM bt_fixtures
        WHERE ligue_id=? AND saison=? AND date_utc < ?
          AND gh IS NOT NULL AND ga IS NOT NULL
    """, (ligue_id, saison, avant_date)) as cur:
        row = await cur.fetchone()
    return max(0.8, row[0]) if row and row[0] else 1.3


async def calculer_moyennes_venue(conn, ligue_id, saison, avant_date):
    """
    Moyennes de buts domicile / extérieur AVANT avant_date.
    m_dom_l = AVG(gh), m_ext_l = AVG(ga) — aligné sur actualiser_stats_ligue du bot.
    """
    async with conn.execute("""
        SELECT AVG(CAST(gh AS REAL)), AVG(CAST(ga AS REAL))
        FROM bt_fixtures
        WHERE ligue_id=? AND saison=? AND date_utc < ?
          AND gh IS NOT NULL AND ga IS NOT NULL
    """, (ligue_id, saison, avant_date)) as cur:
        row = await cur.fetchone()
    if row and row[0] and row[1]:
        return float(row[0]), float(row[1])
    return 1.4, 1.1


async def calculer_sos_maps(conn, ligue_id, saison, avant_date):
    """
    Force défensive/offensive adverse par équipe — aligné actualiser_stats_ligue().
    sos_map[tid]        = buts concédés / match (normalise notre attaque)
    sos_attack_map[tid] = buts marqués / match (normalise notre défense)
    """
    async with conn.execute("""
        SELECT home_id, away_id, gh, ga
        FROM bt_fixtures
        WHERE ligue_id=? AND saison=? AND date_utc < ?
          AND gh IS NOT NULL AND ga IS NOT NULL
    """, (ligue_id, saison, avant_date)) as cur:
        rows = await cur.fetchall()

    if not rows:
        return {}, {}

    teams: dict[int, dict] = {}
    for home_id, away_id, gh, ga in rows:
        for tid, gf, ga_team in ((home_id, gh, ga), (away_id, ga, gh)):
            if tid not in teams:
                teams[tid] = {'gf': 0, 'ga': 0, 'played': 0}
            teams[tid]['gf'] += gf
            teams[tid]['ga'] += ga_team
            teams[tid]['played'] += 1

    sos, sos_attack = {}, {}
    for tid, stats in teams.items():
        j = stats['played'] or 1
        sos[tid] = stats['ga'] / j
        sos_attack[tid] = stats['gf'] / j
    return sos, sos_attack


def _xg_shrinkage_targets(venue, m_dom_l, m_ext_l):
    """Cibles shrinkage off/def selon venue — aligné obtenir_xg_moyenne_async()."""
    ligue_avg_all = (m_dom_l + m_ext_l) / 2
    if venue == 'home':
        return m_dom_l, m_ext_l
    if venue == 'away':
        return m_ext_l, m_dom_l
    return ligue_avg_all, ligue_avg_all


async def calculer_mot_luck_bt(conn, ligue_cfg, saison, avant_date):
    """
    Reconstruit mot_map et luck_map à partir des matchs joués AVANT avant_date.
    Réplique actualiser_stats_ligue() du bot live (enjeux c1 + euro + rel, PDO proxy).
    Pas de lookahead : uniquement les fixtures terminées avec date_utc < avant_date.
    """
    async with conn.execute("""
        SELECT home_id, away_id, gh, ga
        FROM bt_fixtures
        WHERE ligue_id=? AND saison=? AND date_utc < ?
          AND gh IS NOT NULL AND ga IS NOT NULL
    """, (ligue_cfg['id'], saison, avant_date)) as cur:
        rows = await cur.fetchall()

    if not rows:
        return {}, {}

    teams: dict[int, dict] = {}
    for home_id, away_id, gh, ga in rows:
        for tid, gf, ga_team, pts in (
            (home_id, gh, ga, 3 if gh > ga else (1 if gh == ga else 0)),
            (away_id, ga, gh, 3 if ga > gh else (1 if gh == ga else 0)),
        ):
            if tid not in teams:
                teams[tid] = {'pts': 0, 'gf': 0, 'ga': 0, 'played': 0}
            teams[tid]['pts'] += pts
            teams[tid]['gf'] += gf
            teams[tid]['ga'] += ga_team
            teams[tid]['played'] += 1

    if len(teams) < 10:
        neutral = {tid: 1.0 for tid in teams}
        return neutral, dict(neutral)

    standings = sorted(teams.items(), key=lambda x: x[1]['pts'], reverse=True)
    n = len(standings)

    pts_c1 = standings[min(ligue_cfg['c1'] - 1, n - 1)][1]['pts']
    pts_rel = standings[min(ligue_cfg['rel'] - 1, n - 1)][1]['pts']
    pts_euro = None
    if ligue_cfg.get('euro'):
        pts_euro = standings[min(ligue_cfg['euro'] - 1, n - 1)][1]['pts']

    total_played = sum(t['played'] for _, t in standings) or 1
    league_avg_gf = sum(t['gf'] for _, t in standings) / total_played
    league_avg_ga = sum(t['ga'] for _, t in standings) / total_played

    mot, luck = {}, {}
    for tid, stats in teams.items():
        j = stats['played'] or 1
        enjeux = [pts_c1, pts_rel] + ([pts_euro] if pts_euro is not None else [])
        d = min(abs(stats['pts'] - p) for p in enjeux)
        mot[tid] = 1.0 + (0.10 * (1 / (d + 1))) if d <= 4 else (0.95 if d > 12 else 1.0)
        gf_ratio = (stats['gf'] / j) / (league_avg_gf or 1)
        ga_ratio = (stats['ga'] / j) / (league_avg_ga or 1)
        pdo = (gf_ratio + (2.0 - ga_ratio)) / 2.0
        luck[tid] = 1.0 - (pdo - 1.0) * 0.30

    return mot, luck


async def estimer_parametres_dc_bt(conn, ligue_id, saison, avant_date, mu_h, mu_a,
                                   dc_half_life_days=None):
    """
    MLE joint Dixon-Coles (α, β, γ, ρ) par équipe — réplique estimer_parametres_dc_complet().
    Utilise UNIQUEMENT les matchs avec date_utc < avant_date (pas de lookahead).
    Pondération temporelle (demi-vie configurable) relative à avant_date.
    """
    async with conn.execute("""
        SELECT gh, ga, home_id, away_id, date_utc
        FROM bt_fixtures
        WHERE ligue_id=? AND saison=? AND date_utc < ?
          AND gh IS NOT NULL AND ga IS NOT NULL
    """, (ligue_id, saison, avant_date)) as cur:
        rows_curr = await cur.fetchall()

    async with conn.execute("""
        SELECT gh, ga, home_id, away_id, date_utc
        FROM bt_fixtures
        WHERE ligue_id=? AND saison=? AND date_utc < ?
          AND gh IS NOT NULL AND ga IS NOT NULL
    """, (ligue_id, saison - 1, avant_date)) as cur:
        rows_prev = await cur.fetchall()

    valid_curr = [(d, e, h, a, md) for d, e, h, a, md in rows_curr if h and a]
    valid_prev = [(d, e, h, a, md) for d, e, h, a, md in rows_prev if h and a]

    if len(valid_curr) < 40:
        return None

    team_counts: dict[int, int] = {}
    for _, _, h, a, _ in valid_curr:
        team_counts[h] = team_counts.get(h, 0) + 1
        team_counts[a] = team_counts.get(a, 0) + 1
    eligible = {t for t, c in team_counts.items() if c >= 4}
    if len(eligible) < 8:
        return None

    teams = sorted(eligible)
    N = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    try:
        ref_ts = datetime.fromisoformat(avant_date.replace('Z', '+00:00'))
        if ref_ts.tzinfo is None:
            ref_ts = ref_ts.replace(tzinfo=timezone.utc)
    except Exception:
        ref_ts = datetime.now(timezone.utc)

    HALF_LIFE_DAYS = float(dc_half_life_days or get_dc_half_life_days(ligue_id))
    decay = np.log(2) / HALF_LIFE_DAYS

    def weight(match_date_str, base_w):
        if not match_date_str:
            return base_w
        try:
            dt = datetime.fromisoformat(match_date_str.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            days_ago = max(0, (ref_ts - dt).total_seconds() / 86400)
            return base_w * np.exp(-decay * days_ago)
        except Exception:
            return base_w

    all_matches = []
    for d, e, h, a, md in valid_curr:
        if h in idx and a in idx:
            all_matches.append((int(d), int(e), idx[h], idx[a], weight(md, 1.0)))
    for d, e, h, a, md in valid_prev:
        if h in idx and a in idx:
            all_matches.append((int(d), int(e), idx[h], idx[a], weight(md, 0.5)))

    if len(all_matches) < 40:
        return None

    max_g = max(max(d, e) for d, e, _, _, _ in all_matches)
    log_fact = np.zeros(max_g + 2)
    for k in range(1, max_g + 2):
        log_fact[k] = log_fact[k - 1] + np.log(k)

    def neg_ll(x):
        log_a = x[:N]
        log_d = x[N:2 * N]
        log_g = x[2 * N]
        rho = float(np.clip(x[2 * N + 1], -0.40, -0.001))
        pen = (float(np.sum(log_a))) ** 2 * 15.0

        ll = 0.0
        for gh, ga, hi, ai, w in all_matches:
            lh = float(np.clip(np.exp(log_a[hi] + log_d[ai] + log_g), 0.1, 12.0))
            la = float(np.clip(np.exp(log_a[ai] + log_d[hi]), 0.1, 12.0))
            ll += w * (gh * np.log(lh) - lh - float(log_fact[gh]))
            ll += w * (ga * np.log(la) - la - float(log_fact[ga]))
            if gh == 0 and ga == 0:
                tau = max(1e-9, 1.0 - lh * la * rho)
            elif gh == 1 and ga == 0:
                tau = max(1e-9, 1.0 + la * rho)
            elif gh == 0 and ga == 1:
                tau = max(1e-9, 1.0 + lh * rho)
            elif gh == 1 and ga == 1:
                tau = max(1e-9, 1.0 - rho)
            else:
                tau = 1.0
            ll += w * np.log(tau)
        return -(ll - pen)

    x0 = np.zeros(2 * N + 2)
    x0[2 * N] = np.log(max(0.5, mu_h))
    x0[2 * N + 1] = -0.13

    bounds = ([(-2.5, 2.5)] * N +
              [(-2.5, 2.5)] * N +
              [(-0.5, 0.5)] +
              [(-0.40, -0.001)])

    try:
        res = await asyncio.to_thread(
            minimize, neg_ll, x0,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 1500, 'ftol': 1e-8},
        )
    except Exception:
        return None

    if not res.success:
        return None

    x = res.x
    log_a = x[:N]
    log_d = x[N:2 * N]
    gamma = float(np.exp(x[2 * N]))
    rho = float(np.clip(x[2 * N + 1], -0.40, -0.001))

    mean_log_d = float(np.mean(log_d))
    log_d -= mean_log_d
    gamma *= np.exp(mean_log_d)

    if gamma > 0:
        scale = mu_h / gamma
        log_a += np.log(max(scale, 1e-6))
        gamma *= scale

    return {
        'gamma': gamma,
        'rho': rho,
        'teams': {
            team_id: {
                'attack': float(np.exp(log_a[i])),
                'defense': float(np.exp(log_d[i])),
            }
            for i, team_id in enumerate(teams)
        },
    }


def calculer_lambda_blend(dc, h_id, a_id, xg_off_d, xg_def_d, xg_off_e, xg_def_e, m_dom_l, m_ext_l,
                          ligue_id=None, mot_map=None, luck_map=None, rho_fallback=None,
                          dc_weight=None):
    """
    Calcule L_A / L_B avec la même logique que analyser_un_match() du bot live :
      λ_xg : formule venue-normalisée (xg_off × xg_def_adverse / moyenne ligue)
      motivation + PDO (mot_map / luck_map) appliqués sur l'attaque
      λ_dc : attack × defense × γ
      blend DC/xG (dc_weight depuis foot_params_tuned, défaut 0.50).
    """
    if not m_dom_l or not m_ext_l:
        m_dom_l, m_ext_l = max(m_dom_l or 1.4, 0.1), max(m_ext_l or 1.1, 0.1)

    mot_map = mot_map or {}
    luck_map = luck_map or {}
    d_d = (mot_map.get(h_id, 1.0) - 1) + (luck_map.get(h_id, 1.0) - 1)
    d_e = (mot_map.get(a_id, 1.0) - 1) + (luck_map.get(a_id, 1.0) - 1)
    m_d = 1.0 + max(-0.25, min(0.25, d_d))
    m_e = 1.0 + max(-0.25, min(0.25, d_e))

    L_A_xg = (xg_off_d * m_d * xg_def_e) / m_dom_l
    L_B_xg = (xg_off_e * m_e * xg_def_d) / m_ext_l
    if rho_fallback is not None:
        rho = rho_fallback
    else:
        rho = get_rho_fallback(ligue_id) if ligue_id else RHO_DEFAULT

    if dc and h_id in dc['teams'] and a_id in dc['teams']:
        td = dc['teams'][h_id]
        te = dc['teams'][a_id]
        L_A_dc = td['attack'] * te['defense'] * dc['gamma']
        L_B_dc = te['attack'] * td['defense']
        if dc_weight is None:
            dc_weight = get_dc_xg_blend(ligue_id) if ligue_id is not None else 0.50
        w_dc = max(0.0, min(1.0, float(dc_weight)))
        w_xg = 1.0 - w_dc
        L_A = w_dc * L_A_dc + w_xg * L_A_xg
        L_B = w_dc * L_B_dc + w_xg * L_B_xg
        rho = dc['rho']
    else:
        L_A, L_B = L_A_xg, L_B_xg

    return max(0.4, L_A), max(0.4, L_B), rho


async def reconstruire_xg_equipe(conn, team_id, ligue_id, avant_date, saison, venue='all',
                                 ligue_avg=1.3, ligue_avg_def=None, n_prior=None,
                                 xg_half_life_days=None, sos_map=None, sos_attack_map=None):
    """
    Calcule le xG moyen de l'équipe en utilisant UNIQUEMENT les matchs
    joués AVANT avant_date. Réplique la logique du bot principal :
    - split home/away (venue='home'|'away'|'all')
    - normalisation SOS (force adverse)
    - decay exponentiel (demi-vie configurable, défaut foot_params)
    - shrinkage bayésien adaptatif (n_prior, cibles off/def séparées)
    - fallback promu si < 5 matchs (0.85× / 1.25× moyenne ligue)
    Retourne (xg_off, xg_def, n_matchs).
    """
    if ligue_avg_def is None:
        ligue_avg_def = ligue_avg
    if sos_map is None:
        sos_map = {}
    if sos_attack_map is None:
        sos_attack_map = sos_map
    if n_prior is None:
        n_prior = get_n_prior(ligue_id)
    decay = np.log(2) / max(float(xg_half_life_days or get_xg_half_life_days(ligue_id)), 1.0)
    if venue == 'home':
        venue_filter = "AND f.home_id = ?"
        params = (team_id, ligue_id, saison, saison - 1, avant_date, team_id)
    elif venue == 'away':
        venue_filter = "AND f.away_id = ?"
        params = (team_id, ligue_id, saison, saison - 1, avant_date, team_id)
    else:
        venue_filter = ""
        params = (team_id, ligue_id, saison, saison - 1, avant_date)

    async with conn.execute(f"""
        SELECT f.id, f.date_utc, x.xg_p, x.xg_c, f.home_id, f.away_id, f.saison
        FROM bt_fixtures f
        JOIN bt_xg x ON x.fixture_id = f.id AND x.team_id = ?
        WHERE f.ligue_id = ?
          AND f.saison IN (?, ?)
          AND f.date_utc < ?
          AND f.gh IS NOT NULL
          {venue_filter}
        ORDER BY f.date_utc DESC
        LIMIT 15
    """, params) as cur:
        rows = await cur.fetchall()

    # Fallback sur toutes les venues si < 5 matchs dans le venue demandé
    if len(rows) < 5 and venue != 'all':
        la_all = (ligue_avg + ligue_avg_def) / 2
        return await reconstruire_xg_equipe(
            conn, team_id, ligue_id, avant_date, saison, venue='all',
            ligue_avg=la_all, ligue_avg_def=la_all, n_prior=n_prior,
            xg_half_life_days=xg_half_life_days, sos_map=sos_map,
            sos_attack_map=sos_attack_map,
        )

    if len(rows) < 5:
        return ligue_avg * 0.85, (ligue_avg_def or ligue_avg) * 1.25, len(rows)

    now = datetime.fromisoformat(avant_date.replace('Z', '+00:00'))
    tp = tc = tw = 0.0
    for fid, date_str, xg_p, xg_c, home_id, away_id, saison_m in rows:
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            jours = max(0, (now - dt).days)
        except Exception:
            jours = 30
        w = np.exp(-decay * jours)
        if saison_m < saison:
            w *= 0.80
        opp_id = away_id if home_id == team_id else home_id
        ratio_def = max(0.6, min(1.6, sos_map.get(opp_id, ligue_avg) / (ligue_avg or 1)))
        ratio_att = max(0.6, min(1.6, sos_attack_map.get(opp_id, ligue_avg) / (ligue_avg or 1)))
        tp += (xg_p / ratio_def) * w
        tc += (xg_c / ratio_att) * w
        tw += w

    xg_off_brut = tp / tw
    xg_def_brut = tc / tw

    n = len(rows)
    w_eq = n / (n + n_prior)
    xg_off = w_eq * xg_off_brut + (1 - w_eq) * ligue_avg
    xg_def = w_eq * xg_def_brut + (1 - w_eq) * ligue_avg_def
    return xg_off, xg_def, n


async def _blend_xg_equipe(conn, team_id, ligue_id, saison, date_utc, side,
                           m_dom_l, m_ext_l, n_prior, xg_hl, sos_map, sos_attack_map, cache):
    """
    Blend venue-spécifique + global — aligné analyser_un_match().
    side: 'home' | 'away' (venue de l'équipe dans le match simulé).
    """
    venue_sp = 'home' if side == 'home' else 'away'
    la_sp, lad_sp = _xg_shrinkage_targets(venue_sp, m_dom_l, m_ext_l)
    la_all, lad_all = _xg_shrinkage_targets('all', m_dom_l, m_ext_l)

    async def _fetch(venue, la, lad):
        key = (team_id, ligue_id, date_utc, saison, venue, n_prior, xg_hl, round(la, 4), round(lad, 4))
        if key not in cache:
            cache[key] = await reconstruire_xg_equipe(
                conn, team_id, ligue_id, date_utc, saison, venue=venue,
                ligue_avg=la, ligue_avg_def=lad, n_prior=n_prior,
                xg_half_life_days=xg_hl, sos_map=sos_map, sos_attack_map=sos_attack_map,
            )
        return cache[key]

    xg_off_sp, xg_def_sp, n = await _fetch(venue_sp, la_sp, lad_sp)
    xg_off_gl, xg_def_gl, _ = await _fetch('all', la_all, lad_all)
    w = min(0.80, (n / 10.0) * 0.80)
    return (
        xg_off_sp * w + xg_off_gl * (1 - w),
        xg_def_sp * w + xg_def_gl * (1 - w),
        n,
    )


# ─────────────────────────────────────────────────────────────
# 🔧  CALIBRATION WALK-FORWARD (n_prior / rho / demi-vies)
# ─────────────────────────────────────────────────────────────
TUNE_GRID_N_PRIOR = [6, 8, 11]
TUNE_GRID_RHO = [-0.14, -0.11, -0.08]
TUNE_GRID_XG_HL = [35, 46, 58]
TUNE_GRID_DC_HL = [75, 90, 120]
TUNE_GRID_DC_XG_BLEND = [0.30, 0.40, 0.50, 0.60, 0.70]
TUNE_MIN_MATCHS = 80
TUNE_MIN_CLV_SIGNALS = 30
TUNE_METRICS = ('loglik', 'clv', 'blend')


async def _lambdas_pour_match(conn, ligue, saison, date_utc, h_id, a_id,
                              n_prior, xg_half_life, dc_half_life, rho_fallback, caches,
                              dc_xg_blend=None):
    """λ_home / λ_away / rho — données strictement avant date_utc."""
    ligue_id = ligue['id']
    jour = _jour_cache(date_utc)

    venue_key = (ligue_id, saison, jour)
    if venue_key not in caches['venue']:
        caches['venue'][venue_key] = await calculer_moyennes_venue(conn, ligue_id, saison, date_utc)
    m_dom_l, m_ext_l = caches['venue'][venue_key]

    if 'sos' not in caches:
        caches['sos'] = {}
    sos_key = (ligue_id, saison, jour)
    if sos_key not in caches['sos']:
        caches['sos'][sos_key] = await calculer_sos_maps(conn, ligue_id, saison, date_utc)
    sos_map, sos_attack_map = caches['sos'][sos_key]

    if 'xg' not in caches:
        caches['xg'] = {}
    xg_off_d, xg_def_d, _ = await _blend_xg_equipe(
        conn, h_id, ligue_id, saison, date_utc, 'home', m_dom_l, m_ext_l,
        n_prior, xg_half_life, sos_map, sos_attack_map, caches['xg'],
    )
    xg_off_e, xg_def_e, _ = await _blend_xg_equipe(
        conn, a_id, ligue_id, saison, date_utc, 'away', m_dom_l, m_ext_l,
        n_prior, xg_half_life, sos_map, sos_attack_map, caches['xg'],
    )

    mot_luck_key = (ligue_id, saison, jour)
    if mot_luck_key not in caches['mot_luck']:
        caches['mot_luck'][mot_luck_key] = await calculer_mot_luck_bt(conn, ligue, saison, date_utc)
    mot_map, luck_map = caches['mot_luck'][mot_luck_key]

    dc_key = (ligue_id, saison, _semaine_dc(date_utc), dc_half_life)
    if dc_key not in caches['dc']:
        caches['dc'][dc_key] = await estimer_parametres_dc_bt(
            conn, ligue_id, saison, date_utc, m_dom_l, m_ext_l,
            dc_half_life_days=dc_half_life,
        )
    dc = caches['dc'][dc_key]

    return calculer_lambda_blend(
        dc, h_id, a_id, xg_off_d, xg_def_d, xg_off_e, xg_def_e, m_dom_l, m_ext_l, ligue_id,
        mot_map=mot_map, luck_map=luck_map, rho_fallback=rho_fallback,
        dc_weight=dc_xg_blend,
    )


def _ev_min_pour_marche(market, ligue_id, ev_min_l, ev_min_by_market=None,
                        ev_min_spreads_tier=False):
    if ev_min_by_market and market in ev_min_by_market:
        return ev_min_by_market[market]
    if market == 'spreads' and ev_min_spreads_tier and ligue_id is not None:
        return get_ev_min_spreads_ligue(ligue_id, ev_min_l)
    return ev_min_l


def _parse_dt_bt(raw) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    else:
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def _preload_calendrier_fatigue(conn) -> dict[int, list[datetime]]:
    """team_id → dates de match FT triées (toutes ligues en base)."""
    cal: dict[int, list[datetime]] = defaultdict(list)
    async with conn.execute(
        "SELECT home_id, away_id, date_utc FROM bt_fixtures WHERE gh IS NOT NULL"
    ) as cur:
        rows = await cur.fetchall()
    for hid, aid, date_utc in rows:
        dt = _parse_dt_bt(date_utc)
        if dt is None:
            continue
        if hid is not None:
            cal[int(hid)].append(dt)
        if aid is not None:
            cal[int(aid)].append(dt)
    for tid in cal:
        cal[tid].sort()
    return cal


def _equipe_fatigue_bt(
    dates: list[datetime], kickoff_raw, fenetre_j: float, max_matchs: int, min_repos_j: float
) -> bool:
    ko = _parse_dt_bt(kickoff_raw)
    if ko is None or not dates:
        return False
    fenetre = max(1.0, fenetre_j)
    debut = ko - timedelta(days=fenetre)
    n_avant = sum(1 for d in dates if debut <= d < ko)
    if n_avant >= max(1, max_matchs) - 1:
        return True
    if min_repos_j > 0:
        precedents = [d for d in dates if d < ko]
        if precedents:
            repos_j = (ko - precedents[-1]).total_seconds() / 86400.0
            if repos_j < min_repos_j:
                return True
    return False


def _match_fatigue_ah_bt(cal: dict, h_id, a_id, date_utc) -> bool:
    if not FOOT_FATIGUE_AH_ACTIF:
        return False
    mode = FOOT_FATIGUE_AH_MODE if FOOT_FATIGUE_AH_MODE in ("either", "home", "away") else "either"
    checks = []
    if mode in ("either", "home") and h_id is not None:
        checks.append(int(h_id))
    if mode in ("either", "away") and a_id is not None:
        checks.append(int(a_id))
    for tid in checks:
        if _equipe_fatigue_bt(
            cal.get(tid, []),
            date_utc,
            FOOT_FATIGUE_FENETRE_J,
            FOOT_FATIGUE_MAX_MATCHS,
            FOOT_FATIGUE_MIN_REPOS_J,
        ):
            return True
    return False


def _candidats_pour_match(mat, odds_h24, home_name_odds, ev_min_l, ev_max_l,
                          markets=('spreads', 'totals'), poids_dyn=None,
                          ev_max_by_market=None, ev_min_by_market=None,
                          ev_min_spreads_tier=False, ligue_id=None,
                          calibrateur=None):
    """
    Candidats EV/Kelly pour un match — logique alignée simuler_paris() / bot live.
    Retourne [(ev_final, market, outcome, h_val, cote_h24, k, mise, side_flag), ...]
    """
    if poids_dyn is None:
        poids_dyn = calculer_poids_dyn(H_ODDS_BACKTEST)

    spreads_partner: dict = {}
    totals_partner: dict = {}
    for mk, out, hv, c in odds_h24:
        if mk == 'spreads':
            spreads_partner[(mk, hv)] = c
        elif mk == 'totals':
            totals_partner[(hv, out.lower())] = c

    candidats = []
    for market, outcome, h_val, cote_h24 in odds_h24:
        if market not in markets:
            continue
        if cote_h24 < MIN_COTE:
            continue

        if market == 'spreads':
            is_home = (outcome == home_name_odds) or (
                process.extractOne(outcome, [home_name_odds])[1] > 85
            )
            ev_modele = ev_ah(mat, h_val, is_home, cote_h24)
            k = kelly_ah(mat, h_val, is_home, cote_h24)
            cote_partner = spreads_partner.get((market, -h_val))
            cote_novig = None
            if cote_partner and cote_partner > 1.0:
                cote_novig = cote_fair_2way(cote_h24, cote_partner) or cote_h24
                ev_pinnacle = ev_ah(mat, h_val, is_home, cote_novig)
            else:
                ev_pinnacle = ev_modele
            side_flag = is_home
        else:
            is_over = outcome.lower() == 'over'
            ev_modele = ev_total(mat, h_val, is_over, cote_h24)
            k = kelly_total(mat, h_val, is_over, cote_h24)
            partner_side = 'under' if is_over else 'over'
            cote_partner = totals_partner.get((h_val, partner_side))
            cote_novig = None
            if cote_partner and cote_partner > 1.0:
                cote_novig = cote_fair_2way(cote_h24, cote_partner) or cote_h24
                ev_pinnacle = ev_total(mat, h_val, is_over, cote_novig)
            else:
                ev_pinnacle = ev_modele
            side_flag = is_over

        p_modele, p_cal = None, None
        if market == 'spreads':
            p_modele = prob_implicite_ah(mat, h_val, is_home)
            p_cal = p_modele
            if calibrateur is not None and ligue_id is not None:
                p_cal = calibrateur.apply(ligue_id, p_modele)
                ev_modele = ajuster_ev_proportionnel(ev_modele, p_modele, p_cal)
            if FOOT_AH_SHRINK_ACTIF and cote_novig is not None and p_cal is not None:
                p_shrunk = shrink_proba_vers_marche(p_cal, cote_novig, FOOT_AH_SHRINK_W)
                if abs(p_shrunk - p_cal) > 1e-9:
                    ev_modele = ajuster_ev_proportionnel(ev_modele, p_cal, p_shrunk)
                    k = k * (p_shrunk / p_cal) if p_cal > 1e-9 else k
                    p_cal = p_shrunk
                # Shrink remplace le blend EV poids_dyn (sinon double-amortissement)
                ev_final = ev_modele
            else:
                ev_final = ev_modele * poids_dyn + ev_pinnacle * (1.0 - poids_dyn)
        else:
            ev_final = ev_modele * poids_dyn + ev_pinnacle * (1.0 - poids_dyn)

        ev_cap = (ev_max_by_market or {}).get(market, ev_max_l)
        ev_floor = _ev_min_pour_marche(
            market, ligue_id, ev_min_l, ev_min_by_market, ev_min_spreads_tier,
        )
        if not (ev_floor <= ev_final <= ev_cap):
            continue
        mise = min(round(k * 100 * KELLY_FRAC, 2), 5.0)
        if mise < 0.1:
            continue
        candidats.append(
            (ev_final, market, outcome, h_val, cote_h24, k, mise, side_flag, p_modele, p_cal)
        )
    return candidats


async def _preload_odds_tune(conn, ligue_id):
    """Pré-charge cotes H-24 et clôture pour le tuning CLV (1 requête/ligue)."""
    odds_by_fid: dict = defaultdict(list)
    async with conn.execute(
        "SELECT o.fixture_id, o.market, o.outcome, o.h_val, o.cote "
        "FROM bt_odds_h24 o JOIN bt_fixtures f ON f.id=o.fixture_id "
        "WHERE f.ligue_id=?",
        (ligue_id,),
    ) as cur:
        for fid, mk, out, hv, c in await cur.fetchall():
            odds_by_fid[fid].append((mk, out, hv, c))

    close_by_key: dict = {}
    async with conn.execute(
        "SELECT o.fixture_id, o.market, o.outcome, o.h_val, o.cote "
        "FROM bt_odds_cloture o JOIN bt_fixtures f ON f.id=o.fixture_id "
        "WHERE f.ligue_id=?",
        (ligue_id,),
    ) as cur:
        for fid, mk, out, hv, c in await cur.fetchall():
            close_by_key[(fid, mk, out, hv)] = c
    return odds_by_fid, close_by_key


def _composite_tune_score(mean_clv: float, mean_ll: float, clv_weight: float) -> float:
    """Combine CLV (basis points) et log P(score) pour --tune-metric blend."""
    clv_scaled = mean_clv * 10000.0
    ll_scaled = mean_ll * 10.0
    return clv_weight * clv_scaled + (1.0 - clv_weight) * ll_scaled


async def _eval_params_tune(
    conn, ligue, fixtures, params, start_idx,
    metric='loglik', clv_weight=0.5,
    odds_by_fid=None, close_by_key=None, clv_markets=('spreads',),
):
    """
    Évalue un jeu d'hyperparamètres en walk-forward.
    metric : loglik | clv | blend
    clv : CLV moyen AH (spreads) sur les signaux simulés (H-24 vs clôture Pinnacle).
    """
    use_ll = metric in ('loglik', 'blend')
    use_clv = metric in ('clv', 'blend')
    caches = {'xg': {}, 'venue': {}, 'sos': {}, 'mot_luck': {}, 'dc': {}}
    ll_total, ll_n = 0.0, 0
    clv_sum, clv_n, sig_n = 0.0, 0, 0
    ev_min_l = ligue.get('ev_min', 0.05)
    ev_max_l = ligue.get('ev_max', 0.15)

    for row in fixtures[start_idx:]:
        fid, saison, date_utc, h_id, a_id, h_name, a_name, gh, ga = row
        if gh is None or ga is None:
            continue

        L_A, L_B, rho = await _lambdas_pour_match(
            conn, ligue, saison, date_utc, h_id, a_id,
            params['n_prior'], params['xg_half_life_days'], params['dc_half_life_days'],
            params['rho'], caches,
            dc_xg_blend=params.get('dc_xg_blend'),
        )
        mat = generer_matrice(L_A, L_B, rho)

        if use_ll:
            gi = min(int(gh), mat.shape[0] - 1)
            ga_i = min(int(ga), mat.shape[1] - 1)
            ll_total += float(np.log(max(float(mat[gi, ga_i]), 1e-12)))
            ll_n += 1

        if use_clv and odds_by_fid is not None and close_by_key is not None:
            odds_h24 = odds_by_fid.get(fid)
            if not odds_h24:
                continue
            home_name_odds = NAME_MAPPING.get(h_name, h_name)
            candidats = _candidats_pour_match(
                mat, odds_h24, home_name_odds, ev_min_l, ev_max_l, markets=clv_markets,
            )
            if not candidats:
                continue
            candidats.sort(key=lambda x: x[0], reverse=True)
            _, market, outcome, h_val, cote_h24, *_ = candidats[0]
            sig_n += 1
            cote_cloture = close_by_key.get((fid, market, outcome, h_val))
            if cote_cloture and cote_cloture > 1.0:
                clv_sum += (cote_h24 / cote_cloture) - 1.0
                clv_n += 1

    mean_ll = ll_total / ll_n if ll_n else float('-inf')
    mean_clv = clv_sum / clv_n if clv_n >= TUNE_MIN_CLV_SIGNALS else float('-inf')
    stats = {
        'mean_log_score': round(mean_ll, 5) if ll_n else None,
        'mean_clv': round(mean_clv, 6) if clv_n >= TUNE_MIN_CLV_SIGNALS else None,
        'n_clv': clv_n,
        'n_signals': sig_n,
    }

    if metric == 'loglik':
        return mean_ll, stats
    if metric == 'clv':
        return mean_clv, stats
    if mean_ll == float('-inf') or mean_clv == float('-inf'):
        return float('-inf'), stats
    return _composite_tune_score(mean_clv, mean_ll, clv_weight), stats


async def _eval_params_walkforward(conn, ligue, fixtures, params, start_idx=0):
    score, _ = await _eval_params_tune(
        conn, ligue, fixtures, params, start_idx, metric='loglik',
    )
    return score


def _format_tune_score(metric, score, stats):
    if metric == 'loglik':
        return f"log P={score:.4f}"
    if metric == 'clv':
        return f"CLV={score:+.4f} (n={stats.get('n_clv', 0)})"
    ll = stats.get('mean_log_score')
    clv = stats.get('mean_clv')
    ll_s = f"{ll:.4f}" if ll is not None else "N/A"
    clv_s = f"{clv:+.4f}" if clv is not None else "N/A"
    return f"blend={score:.4f} (CLV={clv_s}, logP={ll_s})"


async def tune_ligue_walkforward(conn, ligue, fixtures, metric='loglik', clv_weight=0.5):
    if len(fixtures) < TUNE_MIN_MATCHS:
        print(f"  ⚠️ {ligue['nom']} : {len(fixtures)} matchs (< {TUNE_MIN_MATCHS}) — ignorée", flush=True)
        return None

    odds_by_fid, close_by_key = None, None
    if metric in ('clv', 'blend'):
        odds_by_fid, close_by_key = await _preload_odds_tune(conn, ligue['id'])
        if not odds_by_fid:
            print(f"  ⚠️ {ligue['nom']} : pas de cotes H-24 — tuning CLV impossible", flush=True)
            return None

    n_eval = (
        len(TUNE_GRID_N_PRIOR) * len(TUNE_GRID_XG_HL)
        + len(TUNE_GRID_RHO) * len(TUNE_GRID_DC_HL)
        + len(TUNE_GRID_DC_XG_BLEND)
    )
    start = max(TUNE_MIN_MATCHS // 2, len(fixtures) // 4)
    n_scored = len(fixtures) - start
    metric_label = {'loglik': 'log P(score)', 'clv': 'CLV AH moyen', 'blend': f'blend CLV/logP ({clv_weight:.0%} CLV)'}
    print(
        f"  ▶ {ligue['nom']} : {len(fixtures)} matchs, {n_scored} scorés (burn-in {start}), "
        f"{n_eval} combinaisons, métrique={metric_label.get(metric, metric)}…",
        flush=True,
    )

    best = {
        'n_prior': get_n_prior(ligue['id']),
        'rho': get_rho_fallback(ligue['id']),
        'xg_half_life_days': get_xg_half_life_days(ligue['id']),
        'dc_half_life_days': get_dc_half_life_days(ligue['id']),
        'dc_xg_blend': get_dc_xg_blend(ligue['id']),
    }
    best_score = float('-inf')
    best_stats: dict = {}
    step = 0

    async def _try_params(p, label):
        nonlocal best_score, best, best_stats
        score, stats = await _eval_params_tune(
            conn, ligue, fixtures, p, start,
            metric=metric, clv_weight=clv_weight,
            odds_by_fid=odds_by_fid, close_by_key=close_by_key,
        )
        print(f"     {label} → {_format_tune_score(metric, score, stats)}", flush=True)
        if score > best_score:
            best_score, best, best_stats = score, dict(p), dict(stats)

    for np_ in TUNE_GRID_N_PRIOR:
        for xg_hl in TUNE_GRID_XG_HL:
            step += 1
            p = {**best, 'n_prior': np_, 'xg_half_life_days': xg_hl}
            await _try_params(p, f"[{step}/{n_eval}] n_prior={np_} xg_hl={xg_hl}j")

    for rho in TUNE_GRID_RHO:
        for dc_hl in TUNE_GRID_DC_HL:
            step += 1
            p = {**best, 'rho': rho, 'dc_half_life_days': dc_hl}
            await _try_params(p, f"[{step}/{n_eval}] rho={rho:.2f} dc_hl={dc_hl}j")

    for dc_w in TUNE_GRID_DC_XG_BLEND:
        step += 1
        p = {**best, 'dc_xg_blend': dc_w}
        await _try_params(p, f"[{step}/{n_eval}] dc_xg_blend={dc_w:.2f}")

    if best_stats.get('mean_log_score') is not None:
        best['mean_log_score'] = best_stats['mean_log_score']
    if best_stats.get('mean_clv') is not None:
        best['mean_clv'] = best_stats['mean_clv']
    if best_stats.get('n_clv') is not None:
        best['n_clv'] = best_stats['n_clv']

    print(
        f"  ✅ {ligue['nom']} : n_prior={best['n_prior']} rho={best['rho']:.2f} "
        f"xg_hl={best['xg_half_life_days']:.0f}j dc_hl={best['dc_half_life_days']:.0f}j "
        f"dc_blend={best['dc_xg_blend']:.2f} "
        f"({_format_tune_score(metric, best_score, best_stats)})",
        flush=True,
    )
    return best


async def tune_hyperparams_walkforward(conn, ligues=None, metric='loglik', clv_weight=0.5):
    print("\n" + "=" * 60)
    print("🔧  CALIBRATION WALK-FORWARD (n_prior / rho / demi-vies)")
    print("=" * 60)
    if metric == 'loglik':
        print("  Metrique : log P(score reel | modele) — pas de lookahead, pas de ROI", flush=True)
    elif metric == 'clv':
        print("  Metrique : CLV AH moyen (H-24 vs cloture Pinnacle) sur signaux simules", flush=True)
        print(f"  Min signaux CLV : {TUNE_MIN_CLV_SIGNALS} par combinaison", flush=True)
    else:
        print(f"  Metrique : blend {clv_weight:.0%} CLV + {1-clv_weight:.0%} log P(score)", flush=True)
    print(f"  Grilles : n_prior{TUNE_GRID_N_PRIOR}, rho{TUNE_GRID_RHO}, "
          f"xg_hl{TUNE_GRID_XG_HL}, dc_hl{TUNE_GRID_DC_HL}, "
          f"dc_blend{TUNE_GRID_DC_XG_BLEND}", flush=True)
    print("  (1ere combinaison lente : DC MLE + xG par match — ~5–15 min/ligue)", flush=True)

    ligues = ligues or CHAMPIONNATS
    results = {}
    for ligue in ligues:
        async with conn.execute(
            "SELECT id, saison, date_utc, home_id, away_id, home_name, away_name, gh, ga "
            "FROM bt_fixtures WHERE ligue_id=? AND gh IS NOT NULL ORDER BY date_utc",
            (ligue['id'],),
        ) as cur:
            fixtures = await cur.fetchall()
        tuned = await tune_ligue_walkforward(
            conn, ligue, fixtures, metric=metric, clv_weight=clv_weight,
        )
        if tuned:
            results[ligue['id']] = tuned

    if not results:
        print("\n❌ Aucune ligue calibrée (données insuffisantes).")
        return

    meta_metric = {
        'loglik': 'mean_log_score_prob',
        'clv': 'mean_clv_ah_spreads',
        'blend': f'blend_clv_{int(clv_weight * 100)}_loglik',
    }.get(metric, metric)
    path = save_tuned_params(results, metric=meta_metric)
    print(f"\n✅ Paramètres enregistrés → {path}")
    print("   Bot live + backtest : rechargement auto via foot_params.py")


async def simuler_paris(conn, ligues=None, ev_max_by_market=None, ev_min_by_market=None,
                        ev_min_spreads_tier=False, max_ah_ligue_saison=None,
                        calibrer_ah=None):
    print("\n" + "="*60)
    print("🔬  PHASE 2 — SIMULATION DU MODÈLE")
    print("="*60)

    # Diagnostics rapides pour détecter les problèmes de collecte
    async with conn.execute("SELECT COUNT(*) FROM bt_fixtures WHERE gh IS NOT NULL") as cur:
        n_fix = (await cur.fetchone())[0]
    async with conn.execute("SELECT COUNT(*) FROM bt_odds_h24") as cur:
        n_odds = (await cur.fetchone())[0]
    async with conn.execute("SELECT COUNT(*) FROM bt_xg") as cur:
        n_xg = (await cur.fetchone())[0]
    async with conn.execute("SELECT COUNT(*) FROM bt_signaux") as cur:
        n_old = (await cur.fetchone())[0]
    print(f"  📋 Fixtures avec résultat : {n_fix}")
    print(f"  📊 Cotes H-24 en base     : {n_odds}")
    print(f"  🎯 Entrées xG en base     : {n_xg}")
    if n_old:
        print(f"  📦 Signaux existants      : {n_old} (seront remplacés si écriture OK)")
    print("  🧮 Modèle : blend DC + xG (poids/ligue via foot_params_tuned), poids_dyn dynamique (formule bot live)")
    if ev_max_by_market:
        parts = [f"{mk}≤{cap:.0%}" for mk, cap in sorted(ev_max_by_market.items())]
        print(f"  🎚️  Plafonds EV backtest : {', '.join(parts)}")
    if ev_min_by_market:
        parts = [f"{mk}≥{floor:.0%}" for mk, floor in sorted(ev_min_by_market.items())]
        print(f"  🎚️  Seuils EV min backtest : {', '.join(parts)}")
    if ev_min_spreads_tier:
        tiers = ", ".join(f"{t}={v:.0%}" for t, v in sorted(EV_MIN_SPREADS_TIER.items()))
        print(f"  🎚️  EV min AH par tier : {tiers}")
    if max_ah_ligue_saison:
        print(f"  🎚️  Cap volume AH : {max_ah_ligue_saison} signaux max / ligue / saison (meilleurs EV)")
    if calibrer_ah:
        min_c = CALIB_AH_MIN_SAMPLES_ISOTONIC if calibrer_ah == "isotonic" else CALIB_AH_MIN_SAMPLES_PLATT
        print(f"  📐 Calibration AH walk-forward : {calibrer_ah} (min {min_c} paris/ligue)")
    if FOOT_AH_SHRINK_ACTIF:
        print(
            f"  ⚖️ Shrink AH : p = {FOOT_AH_SHRINK_W:.0%}·modèle + "
            f"{1.0 - FOOT_AH_SHRINK_W:.0%}·no-vig Pinnacle (remplace blend poids_dyn sur AH)"
        )
    if FOOT_FATIGUE_AH_ACTIF:
        repos = (
            f"repos≥{FOOT_FATIGUE_MIN_REPOS_J:g}j"
            if FOOT_FATIGUE_MIN_REPOS_J > 0 else "repos off"
        )
        print(
            f"  😴 Fatigue AH : ≤{FOOT_FATIGUE_MAX_MATCHS - 1} matchs "
            f"/{FOOT_FATIGUE_FENETRE_J:.0f}j avant, {repos} "
            f"(mode {FOOT_FATIGUE_AH_MODE}; coupe UEFA N/A en BT)"
        )

    if n_odds == 0:
        print("\n  ⚠️  AUCUNE cote H-24 trouvée — la collecte d'odds a échoué.")
        print("  💡 Vérifiez API_ODDS_KEY et que votre plan inclut l'endpoint /v4/historical/")
        print("\n✅ Phase 2 terminée (0 signaux).")
        return -1

    all_pending = []
    ligues = ligues or CHAMPIONNATS
    cal_fatigue = await _preload_calendrier_fatigue(conn) if FOOT_FATIGUE_AH_ACTIF else {}
    n_skip_fatigue = 0

    for ligue in ligues:
        async with conn.execute(
            "SELECT id, saison, date_utc, home_id, away_id, home_name, away_name, gh, ga "
            "FROM bt_fixtures WHERE ligue_id=? ORDER BY date_utc",
            (ligue['id'],)
        ) as cur:
            fixtures = await cur.fetchall()

        n_fixtures = len(fixtures)
        print(f"\n  ▶ {ligue['nom']} — {n_fixtures} matchs…", flush=True)

        # Pré-charge toutes les cotes H-24 de la ligue (évite 1 requête SQL/match)
        odds_by_fid = defaultdict(list)
        async with conn.execute(
            "SELECT o.fixture_id, o.market, o.outcome, o.h_val, o.cote "
            "FROM bt_odds_h24 o "
            "JOIN bt_fixtures f ON f.id = o.fixture_id "
            "WHERE f.ligue_id=?",
            (ligue['id'],),
        ) as cur:
            for fid, mk, out, hv, c in await cur.fetchall():
                odds_by_fid[fid].append((mk, out, hv, c))

        signaux = 0
        pending = []
        calibrateur = CalibrateurWalkForwardAH(calibrer_ah) if calibrer_ah else None
        dc_cache = {}
        venue_cache = {}
        sos_cache = {}
        xg_cache = {}
        mot_luck_cache = {}

        for idx, (fid, saison, date_utc, h_id, a_id, h_name, a_name, gh, ga) in enumerate(fixtures, 1):
            if idx % 200 == 0 or idx == n_fixtures:
                print(f"     … {idx}/{n_fixtures} matchs", flush=True)

            if gh is None or ga is None:
                continue

            if calibrateur:
                calibrateur.refit_ligue(ligue['id'])

            odds_h24 = odds_by_fid.get(fid)
            if not odds_h24:
                continue

            jour = _jour_cache(date_utc)

            venue_key = (ligue['id'], saison, jour)
            if venue_key not in venue_cache:
                venue_cache[venue_key] = await calculer_moyennes_venue(
                    conn, ligue['id'], saison, date_utc
                )
            m_dom_l, m_ext_l = venue_cache[venue_key]

            sos_key = (ligue['id'], saison, jour)
            if sos_key not in sos_cache:
                sos_cache[sos_key] = await calculer_sos_maps(
                    conn, ligue['id'], saison, date_utc
                )
            sos_map, sos_attack_map = sos_cache[sos_key]

            n_prior_l = get_n_prior(ligue['id'])
            xg_hl = get_xg_half_life_days(ligue['id'])
            dc_hl = get_dc_half_life_days(ligue['id'])

            # Reconstituer xG AVANT ce match — SOS + shrinkage off/def + blend venue (bot live)
            xg_off_d, xg_def_d, _ = await _blend_xg_equipe(
                conn, h_id, ligue['id'], saison, date_utc, 'home', m_dom_l, m_ext_l,
                n_prior_l, xg_hl, sos_map, sos_attack_map, xg_cache,
            )
            xg_off_e, xg_def_e, _ = await _blend_xg_equipe(
                conn, a_id, ligue['id'], saison, date_utc, 'away', m_dom_l, m_ext_l,
                n_prior_l, xg_hl, sos_map, sos_attack_map, xg_cache,
            )

            mot_luck_key = (ligue['id'], saison, jour)
            if mot_luck_key not in mot_luck_cache:
                mot_luck_cache[mot_luck_key] = await calculer_mot_luck_bt(
                    conn, ligue, saison, date_utc
                )
            mot_map, luck_map = mot_luck_cache[mot_luck_key]

            dc_key = (ligue['id'], saison, _semaine_dc(date_utc), dc_hl)
            if dc_key not in dc_cache:
                dc_cache[dc_key] = await estimer_parametres_dc_bt(
                    conn, ligue['id'], saison, date_utc, m_dom_l, m_ext_l,
                    dc_half_life_days=dc_hl,
                )
            dc = dc_cache[dc_key]

            # λ blend DC + xG + motivation/PDO — aligné sur analyser_un_match() du bot live
            L_A, L_B, rho = calculer_lambda_blend(
                dc, h_id, a_id, xg_off_d, xg_def_d, xg_off_e, xg_def_e, m_dom_l, m_ext_l, ligue['id'],
                mot_map=mot_map, luck_map=luck_map,
                dc_weight=get_dc_xg_blend(ligue['id']),
            )

            mat = generer_matrice(L_A, L_B, rho)

            home_name_odds = NAME_MAPPING.get(h_name, h_name)
            ev_min_l = ligue.get('ev_min', 0.05)
            ev_max_l = ligue.get('ev_max', 0.15)
            candidats = _candidats_pour_match(
                mat, odds_h24, home_name_odds, ev_min_l, ev_max_l,
                ev_max_by_market=ev_max_by_market,
                ev_min_by_market=ev_min_by_market,
                ev_min_spreads_tier=ev_min_spreads_tier,
                ligue_id=ligue['id'],
                calibrateur=calibrateur,
            )

            if FOOT_FATIGUE_AH_ACTIF and _match_fatigue_ah_bt(
                cal_fatigue, h_id, a_id, date_utc
            ):
                n_ah_avant = sum(1 for c in candidats if c[1] == 'spreads')
                if n_ah_avant:
                    n_skip_fatigue += n_ah_avant
                    candidats = [c for c in candidats if c[1] != 'spreads']

            if not candidats:
                continue

            # Meilleur signal par marché (AH + totaux indépendants sur le même match)
            candidats.sort(key=lambda x: x[0], reverse=True)
            by_market: dict = {}
            for row in candidats:
                mk = row[1]
                if mk not in by_market:
                    by_market[mk] = row

            for row in by_market.values():
                ev_final, market, outcome, h_val, cote_h24, k, mise, flag = row[:8]
                p_modele = row[8] if len(row) > 8 else None
                p_cal = row[9] if len(row) > 9 else None
                # Cote de clôture
                async with conn.execute(
                    "SELECT cote FROM bt_odds_cloture WHERE fixture_id=? AND market=? AND outcome=? AND h_val=?",
                    (fid, market, outcome, h_val)
                ) as cur:
                    row = await cur.fetchone()
                cote_cloture = row[0] if row else None

                # Résultat réel
                if market == 'spreads':
                    res = resultat_ah(gh, ga, h_val, flag)
                else:
                    res = resultat_total(gh, ga, h_val, flag)

                clv = round((cote_h24 / cote_cloture) - 1, 4) if cote_cloture else None

                pending.append((
                    fid, ligue['id'], saison, market, outcome, h_val,
                    cote_h24, cote_cloture, round(ev_final, 4), round(k, 4),
                    mise, gh, ga, res, clv,
                    round(p_modele, 4) if p_modele is not None else None,
                    round(p_cal, 4) if p_cal is not None else None,
                ))
                signaux += 1

                if calibrateur and market == 'spreads' and p_modele is not None and res not in (None, 0.0):
                    calibrateur.observe(
                        ligue['id'], p_modele, outcome_binaire_ah(res),
                    )

        if max_ah_ligue_saison:
            n_avant = len(pending)
            pending = _cap_signaux_ah_ligue_saison(pending, max_ah_ligue_saison)
            signaux -= n_avant - len(pending)

        all_pending.extend(pending)
        print(f"  {ligue['nom']} : {signaux} signaux générés")

    if FOOT_FATIGUE_AH_ACTIF and n_skip_fatigue:
        print(f"  😴 AH skippés (fatigue domestique) : {n_skip_fatigue}")

    print(f"\n  📝 Total calculé : {len(all_pending)} signaux — écriture en base...")
    n_saved = await persister_signaux(all_pending)
    if n_saved >= 0:
        print("\n✅ Phase 2 terminée.")
    else:
        print("\n❌ Phase 2 échouée (données en base inchangées — rapport = anciens signaux).")
    return n_saved


def _t_stat_clv(clv_vals: list) -> float:
    if len(clv_vals) < 2:
        return float('nan')
    m = float(np.mean(clv_vals))
    s = float(np.std(clv_vals, ddof=1))
    if s == 0:
        return float('nan')
    return m / (s / np.sqrt(len(clv_vals)))


def _resume_signaux(rows, idx_res=13, idx_mise=10, idx_clv=14):
    """Stats CLV / ROI / t-stat pour un sous-ensemble de signaux bt_signaux."""
    if not rows:
        return None
    res = [(r[idx_res], r[idx_mise]) for r in rows if r[idx_res] is not None]
    clv = [r[idx_clv] for r in rows if r[idx_clv] is not None]
    if not res:
        return None
    pnl = sum(r * m for r, m in res)
    mises = sum(m for _, m in res)
    return {
        'n': len(rows),
        'clv': float(np.mean(clv)) if clv else 0.0,
        'n_clv': len(clv),
        't_clv': _t_stat_clv(clv),
        'roi': pnl / mises if mises else 0.0,
        'pnl': pnl,
        'mises': mises,
        'win_rate': sum(1 for r, _ in res if r > 0) / len(res),
    }


def _print_ligne_marche(label: str, st: dict | None, indent: str = "  "):
    if not st:
        print(f"{indent}{label:<22} — aucun signal")
        return
    t_s = f"{st['t_clv']:+.2f}" if st['n_clv'] >= 2 and not np.isnan(st['t_clv']) else "n/a"
    print(
        f"{indent}{label:<22} n={st['n']:>5}  "
        f"CLV={st['clv']:>+7.2%} (t={t_s:>5})  "
        f"ROI={st['roi']:>+7.2%}  P&L={st['pnl']:>+7.1f}u  WR={st['win_rate']:.1%}"
    )


TRANCHES_EV = [(0.05, 0.07), (0.07, 0.09), (0.09, 0.12), (0.12, 0.20)]


def _print_clv_ah_par_ev(signaux, tranches=None):
    """Segmente CLV / ROI AH par tranche d'EV modèle (filtre spreads uniquement)."""
    tranches = tranches or TRANCHES_EV
    ah = [s for s in signaux if s[3] == 'spreads']
    print(f"\n{'─'*50}")
    print(f"  CLV AH PAR TRANCHE EV (Handicap Asiatique)")
    print(f"{'─'*50}")
    print(
        f"  {'EV':<14} {'N':>5} {'CLV':>8} {'t':>6} {'ROI':>8} "
        f"{'P&L':>8} {'WR':>6} {'EV_moy':>8}"
    )
    print(f"  {'─'*14} {'─'*5} {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*8}")
    for lo, hi in tranches:
        rows = [s for s in ah if lo <= s[8] < hi]
        st = _resume_signaux(rows)
        if not st:
            continue
        t_s = f"{st['t_clv']:+.1f}" if st['n_clv'] >= 2 and not np.isnan(st['t_clv']) else "n/a"
        ev_moy = float(np.mean([s[8] for s in rows]))
        print(
            f"  [{lo:.0%}-{hi:.0%}]{'':<4} {st['n']:>5} "
            f"{st['clv']:>+7.2%} {t_s:>6} {st['roi']:>+7.1%} "
            f"{st['pnl']:>+7.1f}u {st['win_rate']:>5.1%} {ev_moy:>+7.2%}"
        )
    st_all = _resume_signaux(ah)
    if st_all:
        t_s = f"{st_all['t_clv']:+.1f}" if st_all['n_clv'] >= 2 and not np.isnan(st_all['t_clv']) else "n/a"
        ev_moy = float(np.mean([s[8] for s in ah]))
        print(f"  {'─'*14} {'─'*5} {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*8}")
        print(
            f"  {'TOTAL AH':<14} {st_all['n']:>5} "
            f"{st_all['clv']:>+7.2%} {t_s:>6} {st_all['roi']:>+7.1%} "
            f"{st_all['pnl']:>+7.1f}u {st_all['win_rate']:>5.1%} {ev_moy:>+7.2%}"
        )
        best = max(
            ((lo, hi, _resume_signaux([s for s in ah if lo <= s[8] < hi]))
             for lo, hi in tranches),
            key=lambda x: x[2]['clv'] if x[2] else float('-inf'),
        )
        if best[2] and best[2]['clv'] > st_all['clv']:
            print(
                f"  → Meilleure tranche CLV : [{best[0]:.0%}-{best[1]:.0%}] "
                f"({best[2]['clv']:+.2%}, n={best[2]['n']})"
            )


def _cap_signaux_ah_ligue_saison(rows, max_per):
    """Garde les max_per meilleurs signaux AH par (ligue_id, saison) — tri EV décroissant."""
    if not max_per or max_per <= 0:
        return rows
    others = [r for r in rows if r[3] != 'spreads']
    by_key = defaultdict(list)
    for r in rows:
        if r[3] == 'spreads':
            by_key[(r[1], r[2])].append(r)
    capped = []
    for ah_rows in by_key.values():
        ah_rows.sort(key=lambda x: x[8], reverse=True)
        capped.extend(ah_rows[:max_per])
    return others + capped


def _cap_signaux_posthoc(signaux, max_ah_ligue_saison):
    """Applique le cap AH sur signaux chargés (explore-ev post-hoc)."""
    if not max_ah_ligue_saison:
        return signaux
    others = [s for s in signaux if s[3] != 'spreads']
    by_key = defaultdict(list)
    for s in signaux:
        if s[3] == 'spreads':
            by_key[(s[1], s[2])].append(s)
    capped = []
    for ah_rows in by_key.values():
        ah_rows.sort(key=lambda x: x[8], reverse=True)
        capped.extend(ah_rows[:max_ah_ligue_saison])
    return others + capped


def _filtrer_signaux_ev(signaux, ev_max_spreads=None, ev_max_totals=None, ev_max_all=None,
                        ev_min_spreads=None, ev_min_spreads_tier=False):
    """Sous-ensemble post-hoc selon plafonds / seuils EV par marché."""
    out = []
    for s in signaux:
        mk, ev, lid = s[3], s[8], s[1]
        if ev_max_all is not None and ev > ev_max_all:
            continue
        if mk == 'spreads' and ev_max_spreads is not None and ev > ev_max_spreads:
            continue
        if mk == 'totals' and ev_max_totals is not None and ev > ev_max_totals:
            continue
        if mk == 'spreads':
            if ev_min_spreads is not None and ev < ev_min_spreads:
                continue
            if ev_min_spreads_tier and ev < get_ev_min_spreads_ligue(lid):
                continue
        out.append(s)
    return out


def explore_ev_filtres(signaux):
    """Compare volume / CLV / ROI pour plusieurs plafonds ev_max (backtest only)."""
    if not signaux:
        print("\n⚠️  Aucun signal — lancez --simulate d'abord.")
        return
    n_lig = len({s[1] for s in signaux})
    n_saisons = len({s[2] for s in signaux})
    n_tot = len(signaux)
    per_base = n_tot / max(n_lig * n_saisons, 1)

    print(f"\n{'='*60}")
    print(f"  EXPLORATION EV — {n_tot} signaux actuels ({per_base:.0f}/ligue/saison)")
    print(f"  Ligues: {n_lig} | Saisons: {n_saisons} | AH: {sum(1 for s in signaux if s[3]=='spreads')}")
    print(f"{'='*60}")

    def _ligne(label, sub):
        if not sub:
            print(f"  {label:<28} — aucun signal")
            return
        st = _resume_signaux(sub)
        st_ah = _resume_signaux([s for s in sub if s[3] == 'spreads'])
        per = len(sub) / max(n_lig * n_saisons, 1)
        clv_ah = st_ah['clv'] if st_ah else float('nan')
        print(
            f"  {label:<28} n={st['n']:>5} ({per:.0f}/lig/s)  "
            f"AH={st_ah['n'] if st_ah else 0:>4}  CLV_AH={clv_ah:+.2%}  "
            f"ROI={st['roi']:+.2%}  P&L={st['pnl']:+.1f}u"
        )

    print("\n  ev_max AH seulement (totaux inchangés) :")
    for cap in (0.15, 0.12, 0.10, 0.09, 0.07):
        sub = _filtrer_signaux_ev(signaux, ev_max_spreads=cap)
        _ligne(f"AH ≤ {cap:.0%}", sub)

    print("\n  ev_max global (AH + totaux) :")
    for cap in (0.15, 0.12, 0.10, 0.09):
        sub = _filtrer_signaux_ev(signaux, ev_max_all=cap)
        _ligne(f"tous ≤ {cap:.0%}", sub)

    print("\n  AH seulement (qualité vs volume) :")
    for cap in (0.15, 0.12, 0.10, 0.09, 0.07):
        ah = _filtrer_signaux_ev(signaux, ev_max_spreads=cap)
        ah = [s for s in ah if s[3] == 'spreads']
        st = _resume_signaux(ah)
        if not st:
            continue
        per = len(ah) / max(n_lig * n_saisons, 1)
        print(
            f"  AH ≤ {cap:.0%}  n={st['n']:>4} ({per:.0f}/lig/s)  "
            f"CLV={st['clv']:+.2%}  ROI={st['roi']:+.2%}  P&L={st['pnl']:+.1f}u"
        )

    print("\n  P1 volume AH — EV min par tier (post-hoc, totaux inchangés) :")
    for label, kwargs in (
        ("tier Top7/Mid6/Niche5", {"ev_max_spreads": 0.09, "ev_min_spreads_tier": True}),
        ("EV min 6% flat", {"ev_max_spreads": 0.09, "ev_min_spreads": 0.06}),
        ("EV min 7% flat", {"ev_max_spreads": 0.09, "ev_min_spreads": 0.07}),
    ):
        sub = _filtrer_signaux_ev(signaux, **kwargs)
        _ligne(label, sub)

    print("\n  P1 volume AH — cap / ligue / saison (post-hoc, EV AH ≤ 9%) :")
    base_ah_cap = _filtrer_signaux_ev(signaux, ev_max_spreads=0.09)
    for cap in (60, 45, 30, 20):
        sub = _cap_signaux_posthoc(base_ah_cap, cap)
        _ligne(f"cap {cap} AH/lig/s", sub)

    print("\n  Preset --p1-ah (tier + EV AH ≤ 9%, sans cap — parité live) :")
    p1_sub = _filtrer_signaux_ev(
        signaux, ev_max_spreads=0.09, ev_min_spreads_tier=True,
    )
    _ligne("p1-ah", p1_sub)
    st_p1_ah = _resume_signaux([s for s in p1_sub if s[3] == 'spreads'])
    if st_p1_ah:
        per_ah = st_p1_ah['n'] / max(n_lig * n_saisons, 1)
        print(
            f"\n  → Commande : python backtest_football.py --simulate --p1-ah --report"
            f"\n     AH seul : n={st_p1_ah['n']} (~{per_ah:.0f}/lig/s), "
            f"CLV={st_p1_ah['clv']:+.2%}, ROI={st_p1_ah['roi']:+.2%}, "
            f"P&L={st_p1_ah['pnl']:+.1f}u"
        )
        print(
            f"     Cap optionnel (exploration) : --max-ah-ligue-saison {P1_AH_MAX_LIGUE_SAISON}"
        )

    # Recommandation synthétique
    best_ah = max(
        ((cap, _resume_signaux(_filtrer_signaux_ev(
            [s for s in signaux if s[3] == 'spreads'], ev_max_spreads=cap)))
         for cap in (0.12, 0.10, 0.09)),
        key=lambda x: x[1]['pnl'] if x[1] else float('-inf'),
    )
    if best_ah[1]:
        print(
            f"\n  → Piste backtest AH : --ev-max-spreads {best_ah[0]:.2f} "
            f"(P&L AH {best_ah[1]['pnl']:+.1f}u, ROI {best_ah[1]['roi']:+.2%}, "
            f"n={best_ah[1]['n']}, ~{best_ah[1]['n']/(n_lig*n_saisons):.0f}/lig/s)"
        )
        print("     Live inchangé — tester d'abord en --simulate --report")


# Tranches de p_modele pour fiabilité AH (couverture)
TRANCHES_P_AH = [
    (0.40, 0.48), (0.48, 0.52), (0.52, 0.56), (0.56, 0.62), (0.62, 0.75),
]


def _print_calibration_ah(signaux):
    """
    Diagnostic calib AH only (pas de mélange totaux).
    1) ROI réalisé vs EV annoncé par tranche EV
    2) Brier global + fiabilité par tranche de p_modele
    """
    ah = [s for s in signaux if s[3] == 'spreads' and s[13] is not None]
    print(f"\n{'─'*50}")
    print(f"  CALIBRATION AH — ROI réalisé vs EV annoncé")
    print(f"{'─'*50}")
    print("  (AH seul — l'ancien bloc mélangeait AH+totaux et n'utilisait pas p_modele)")
    if not ah:
        print("  Aucun signal AH.")
        return

    print(
        f"  {'EV':<14} {'N':>5} {'EV_moy':>8} {'ROI':>8} {'écart':>8} {'WR':>6} {'P&L':>8}"
    )
    print(f"  {'─'*14} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*8}")
    for lo, hi in TRANCHES_EV:
        rows = [s for s in ah if lo <= s[8] < hi]
        if not rows:
            continue
        mises = sum(s[10] for s in rows)
        pnl = sum(s[13] * s[10] for s in rows)
        roi = pnl / mises if mises else 0.0
        ev_moy = float(np.mean([s[8] for s in rows]))
        # Même convention que _stats_from_res : wins / tous (push inclus)
        wr = sum(1 for s in rows if s[13] > 0) / len(rows)
        print(
            f"  [{lo:.0%}-{hi:.0%}]  {len(rows):>5}  {ev_moy:>+7.2%}  {roi:>+7.2%}  "
            f"{roi - ev_moy:>+7.2%}  {wr:>5.1%}  {pnl:>+7.1f}u"
        )

    ah_all_ev = ah
    mises_a = sum(s[10] for s in ah_all_ev)
    pnl_a = sum(s[13] * s[10] for s in ah_all_ev)
    roi_a = pnl_a / mises_a if mises_a else 0.0
    ev_a = float(np.mean([s[8] for s in ah_all_ev]))
    print(
        f"  {'TOTAL AH':<14} {len(ah_all_ev):>5}  {ev_a:>+7.2%}  {roi_a:>+7.2%}  "
        f"{roi_a - ev_a:>+7.2%}  {'':>6}  {pnl_a:>+7.1f}u"
    )
    print("  écart = ROI − EV_moy  (négatif ≈ sur-confiance du modèle)")

    ah_cal = [
        s for s in ah
        if len(s) > 16 and s[16] is not None and s[13] not in (None, 0.0)
    ]
    print(f"\n{'─'*50}")
    print(f"  CALIBRATION P(couverture) AH — Brier / fiabilité")
    print(f"{'─'*50}")
    if not ah_cal:
        print("  Pas de p_modele AH exploitable.")
        return

    y = [outcome_binaire_ah(s[13]) for s in ah_cal]
    p_raw = [float(s[16]) for s in ah_cal]
    p_cal = [
        float(s[17]) if len(s) > 17 and s[17] is not None else float(s[16])
        for s in ah_cal
    ]
    b_raw = brier_score_prob(p_raw, y)
    b_cal = brier_score_prob(p_cal, y)
    if b_raw is not None:
        print(f"  Brier P brut (p_modele)     : {b_raw:.4f}  (n={len(ah_cal)})")
    if b_cal is not None:
        print(f"  Brier P calibré (p_cal)     : {b_cal:.4f}")
        if b_raw is not None:
            delta = b_raw - b_cal
            if delta > 1e-6:
                print(f"  → Gain Brier : {delta:+.4f}")
            else:
                print(f"  → Gain Brier : {delta:+.4f} (calib WF quasi neutre)")

    print(f"\n  Fiabilité par tranche de p_modele (fréquence réelle de couverture) :")
    print(
        f"  {'p':<14} {'N':>5} {'p_moy':>7} {'f_réel':>7} {'écart':>8} {'Brier':>7}"
    )
    print(f"  {'─'*14} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*7}")
    for lo, hi in TRANCHES_P_AH:
        idx = [i for i, p in enumerate(p_raw) if lo <= p < hi]
        if len(idx) < 20:
            continue
        pr = [p_raw[i] for i in idx]
        yr = [y[i] for i in idx]
        p_m = float(np.mean(pr))
        f_r = float(np.mean(yr))
        b_bin = brier_score_prob(pr, yr)
        b_s = f"{b_bin:.4f}" if b_bin is not None else "  n/a"
        print(
            f"  [{lo:.0%}-{hi:.0%}]  {len(idx):>5}  {p_m:>6.1%}  {f_r:>6.1%}  "
            f"{f_r - p_m:>+7.1%}  {b_s:>7}"
        )


# ─────────────────────────────────────────────────────────────
# 📊  PHASE 3 — RAPPORT D'ANALYSE
# ─────────────────────────────────────────────────────────────
async def charger_signaux(conn):
    nom_par_id = {c['id']: c['nom'] for c in CHAMPIONNATS}
    async with conn.execute("""
        SELECT s.fixture_id, s.ligue_id, s.saison, s.market, s.outcome,
               s.h_val, s.cote_h24, s.cote_cloture, s.ev_modele, s.kelly,
               s.mise, s.gh, s.ga, s.resultat, s.clv, f.date_utc,
               s.p_modele, s.p_cal
        FROM bt_signaux s
        JOIN bt_fixtures f ON f.id = s.fixture_id
        WHERE s.resultat IS NOT NULL
        ORDER BY f.date_utc
    """) as cur:
        rows = await cur.fetchall()
    return [(*r, nom_par_id.get(r[1], str(r[1]))) for r in rows]


async def fit_calibration_ah_db(conn, method='platt', min_samples=None, out_path=None):
    """Fit Platt/isotonic par ligue sur bt_signaux (AH) → foot_calibration_ah.json."""
    if min_samples is None:
        min_samples = (
            CALIB_AH_MIN_SAMPLES_ISOTONIC if method == "isotonic"
            else CALIB_AH_MIN_SAMPLES_PLATT
        )
    try:
        async with conn.execute(
            "SELECT ligue_id, market, p_modele, resultat FROM bt_signaux "
            "WHERE p_modele IS NOT NULL AND resultat IS NOT NULL"
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        print("\n  ❌ Colonne p_modele absente — relancez d'abord :")
        print("     python backtest_football.py --simulate --p1-ah --report")
        return None

    if not rows:
        print("\n  ❌ Aucun p_modele en base — relancez --simulate (stocke p_modele sur chaque AH).")
        return None

    fit_rows = []
    for lid, mk, p_mod, res in rows:
        if mk != 'spreads' or p_mod is None or res in (None, 0.0):
            continue
        fit_rows.append((lid, p_mod, outcome_binaire_ah(res)))

    if not fit_rows:
        print("\n  ❌ Aucun pari AH avec p_modele exploitable.")
        return None

    data = fit_calibration_ah_par_ligue(fit_rows, method=method, min_samples=min_samples)
    path = save_calibration_ah(data, path=out_path)
    n_lig = len(data.get("ligues", {}))
    print(f"\n  📐 Calibration AH ({method}) : {n_lig} ligue(s) fitées → {path}")
    for lid_s, m in sorted(data.get("ligues", {}).items(), key=lambda x: int(x[0])):
        nom = next((c['nom'] for c in CHAMPIONNATS if c['id'] == int(lid_s)), lid_s)
        brier = m.get("brier", float('nan'))
        print(f"     {nom:<18} n={m.get('n_fit', 0):>4}  Brier={brier:.4f}")
    return path


async def generer_rapport(conn):
    print("\n" + "="*60)
    print("📊  PHASE 3 — RAPPORT D'ANALYSE")
    print("="*60)

    signaux = await charger_signaux(conn)

    if not signaux:
        print("⚠️  Aucun signal trouvé. Lancez d'abord --collect et --simulate.")
        return

    # ── Rapport global ──────────────────────────────────────
    total = len(signaux)
    clv_vals  = [s[14] for s in signaux if s[14] is not None]
    res_vals  = [(s[13], s[10]) for s in signaux if s[13] is not None]  # (resultat, mise)
    pnl_total = sum(r * m for r, m in res_vals)
    mises_tot = sum(m for _, m in res_vals)
    clv_moy   = np.mean(clv_vals) if clv_vals else 0
    win_rate  = sum(1 for r, _ in res_vals if r > 0) / len(res_vals) if res_vals else 0

    print(f"\n{'─'*50}")
    print(f"  RÉSULTATS GLOBAUX ({total} signaux)")
    print(f"{'─'*50}")
    print(f"  CLV moyen           : {clv_moy:+.2%}")
    print(f"  P&L total           : {pnl_total:+.1f} u")
    print(f"  Mises totales       : {mises_tot:.1f} u")
    print(f"  ROI                 : {pnl_total/mises_tot:+.2%}" if mises_tot else "  ROI : N/A")
    print(f"  Win rate            : {win_rate:.1%}")
    print(f"  Signaux avec CLV    : {len(clv_vals)}/{total}")
    n_lig_r = len({s[1] for s in signaux})
    n_sais_r = len({s[2] for s in signaux})
    n_ah_r = sum(1 for s in signaux if s[3] == 'spreads')
    if n_lig_r and n_sais_r:
        print(f"  Densité AH            : {n_ah_r} (~{n_ah_r / (n_lig_r * n_sais_r):.0f}/ligue/saison)")

    # ── AH vs Totaux (CLV + ROI séparés) ─────────────────────
    print(f"\n{'─'*50}")
    print(f"  PAR TYPE DE MARCHÉ — CLV vs Pinnacle + ROI")
    print(f"{'─'*50}")
    st_ah = _resume_signaux([s for s in signaux if s[3] == 'spreads'])
    st_tot = _resume_signaux([s for s in signaux if s[3] == 'totals'])
    _print_ligne_marche("Handicap Asiatique", st_ah)
    _print_ligne_marche("Totaux (O/U)", st_tot)
    if st_ah and st_tot:
        delta_clv = st_ah['clv'] - st_tot['clv']
        print(f"\n  Écart CLV AH − Totaux : {delta_clv:+.2%}")
        if st_ah['clv'] > 0 and st_tot['clv'] <= 0:
            print("  → Edge vs Pinnacle sur l'AH ; totaux sans CLV structurel dans ce backtest.")

    _print_clv_ah_par_ev(signaux)

    # Drawdown maximum
    bankroll = 100.0
    peak = bankroll
    max_dd = 0.0
    for r, m in res_vals:
        bankroll += r * m
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak
        if dd > max_dd:
            max_dd = dd
    print(f"  Drawdown max        : {max_dd:.1%}")

    # ── Rapport par saison ──────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  PAR SAISON")
    print(f"{'─'*50}")
    print(f"  {'Saison':<8} {'N':>5} {'CLV':>8} {'ROI':>8} {'P&L':>8} {'Drawdown':>10}")
    print(f"  {'─'*8} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

    par_saison = defaultdict(list)
    for s in signaux:
        par_saison[s[2]].append(s)  # s[2] = saison

    for saison, rows_s in sorted(par_saison.items()):
        clv_s  = [r[14] for r in rows_s if r[14] is not None]
        res_s  = [(r[13], r[10]) for r in rows_s if r[13] is not None]
        pnl_s  = sum(r * m for r, m in res_s)
        mis_s  = sum(m for _, m in res_s)
        roi_s  = pnl_s / mis_s if mis_s else 0
        clv_s_moy = np.mean(clv_s) if clv_s else 0
        # Drawdown par saison
        bk, pk, dd_s = 100.0, 100.0, 0.0
        for r, m in res_s:
            bk += r * m
            if bk > pk: pk = bk
            dd_s = max(dd_s, (pk - bk) / pk)
        marker = "✅" if roi_s > 0 else "❌"
        print(f"  {marker} {saison:<6} {len(rows_s):>5} {clv_s_moy:>+7.1%} {roi_s:>+7.1%} {pnl_s:>+7.1f}u {dd_s:>9.1%}")

    # ── Rapport par ligue ───────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  PAR LIGUE")
    print(f"{'─'*50}")
    print(f"  {'Ligue':<20} {'N':>5} {'CLV':>8} {'ROI':>8} {'P&L':>8}")
    print(f"  {'─'*20} {'─'*5} {'─'*8} {'─'*8} {'─'*8}")

    par_ligue = defaultdict(list)
    for s in signaux:
        par_ligue[s[-1]].append(s)  # s[-1] = ligue_nom

    for nom, rows in sorted(par_ligue.items()):
        n = len(rows)
        clv_l = [r[14] for r in rows if r[14] is not None]
        res_l = [(r[13], r[10]) for r in rows if r[13] is not None]
        pnl_l = sum(r * m for r, m in res_l)
        mis_l = sum(m for _, m in res_l)
        clv_l_moy = np.mean(clv_l) if clv_l else 0
        roi_l = pnl_l / mis_l if mis_l else 0
        marker = "✅" if roi_l > 0 else "❌"
        print(f"  {marker} {nom:<18} {n:>5} {clv_l_moy:>+7.1%} {roi_l:>+7.1%} {pnl_l:>+7.1f}u")

    # ── Rapport par marché ──────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  PAR MARCHÉ (détail)")
    print(f"{'─'*50}")
    for market, label in [('spreads', 'Handicap Asiatique'), ('totals', 'Totaux (O/U)')]:
        rows_m = [s for s in signaux if s[3] == market]
        st_m = _resume_signaux(rows_m)
        _print_ligne_marche(label, st_m)

    # ── Par ligue × marché ───────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  PAR LIGUE × MARCHÉ (CLV / ROI)")
    print(f"{'─'*50}")
    print(f"  {'Ligue':<18} {'Marché':<8} {'N':>5} {'CLV':>8} {'t':>6} {'ROI':>8}")
    print(f"  {'─'*18} {'─'*8} {'─'*5} {'─'*8} {'─'*6} {'─'*8}")
    par_ligue_id = defaultdict(lambda: defaultdict(list))
    for s in signaux:
        par_ligue_id[s[-1]][s[3]].append(s)
    for nom in sorted(par_ligue_id.keys()):
        for market, mlabel in [('spreads', 'AH'), ('totals', 'Tot')]:
            st_lm = _resume_signaux(par_ligue_id[nom][market])
            if not st_lm or st_lm['n'] < 10:
                continue
            t_s = f"{st_lm['t_clv']:+.1f}" if st_lm['n_clv'] >= 2 and not np.isnan(st_lm['t_clv']) else "n/a"
            print(
                f"  {nom:<18} {mlabel:<8} {st_lm['n']:>5} "
                f"{st_lm['clv']:>+7.2%} {t_s:>6} {st_lm['roi']:>+7.1%}"
            )

    # ── Objectifs CLV AH par ligue (vs Pinnacle) ─────────────
    nom_par_id = {c['id']: c['nom'] for c in CHAMPIONNATS}
    df_obj_clv = construire_tableau_objectifs_clv_ah(signaux=signaux, noms_ligue=nom_par_id)
    print(f"\n{'─'*50}")
    print(f"  OBJECTIFS CLV AH PAR LIGUE — battre Pinnacle (long terme)")
    print(f"{'─'*50}")
    print("  Critère ✅ : CLV ≥ cible ligue, t ≥ 2.0, n ≥ min paris AH")
    print(formater_objectifs_clv_ah_texte(df_obj_clv))
    if not df_obj_clv.empty:
        obj_csv = "clv_objectifs_ligue.csv"
        df_obj_clv.assign(
            clv_ah_pct=(df_obj_clv["clv_ah"] * 100).round(2),
            cible_clv_pct=(df_obj_clv["cible_clv"] * 100).round(2),
            clv_min_sig_pct=(df_obj_clv["clv_min_sig"] * 100).round(2),
            t_stat=df_obj_clv["t_stat"].round(2),
        ).to_csv(obj_csv, index=False, encoding="utf-8")
        print(f"\n  📄 Objectifs exportés : {obj_csv}")

    _print_calibration_ah(signaux)

    # ── Export CSV ──────────────────────────────────────────
    csv_path = "backtest_results.csv"
    async with conn.execute("""
        SELECT s.fixture_id, s.ligue_id, s.saison, s.market, s.outcome,
               s.h_val, s.cote_h24, s.cote_cloture, s.ev_modele, s.kelly,
               s.mise, s.gh, s.ga, s.resultat, s.clv, f.date_utc,
               s.p_modele, s.p_cal
        FROM bt_signaux s
        JOIN bt_fixtures f ON f.id = s.fixture_id
        WHERE s.resultat IS NOT NULL
        ORDER BY f.date_utc
    """) as cur:
        rows_csv = await cur.fetchall()

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['fixture_id', 'ligue_id', 'saison', 'market', 'outcome',
                    'h_val', 'cote_h24', 'cote_cloture', 'ev_modele', 'kelly',
                    'mise', 'gh', 'ga', 'resultat', 'clv', 'date_utc',
                    'p_modele', 'p_cal'])
        w.writerows(rows_csv)

    print(f"\n📄 Résultats détaillés exportés dans {csv_path}")
    print("\n✅ Phase 3 terminée.")


async def reset_backtest(full: bool = False):
    """
    Reset des résultats backtest pour le dashboard Streamlit.

    full=False (--reset)       : vide bt_signaux + supprime backtest_results.csv
    full=True  (--reset-full)  : supprime aussi backtest_data.db (recollecte requise)
    """
    print("\n" + "=" * 60)
    print("RESET BACKTEST")
    print("=" * 60)

    csv_path = "backtest_results.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"  [OK] {csv_path} supprime")
    else:
        print(f"  [--] {csv_path} absent (OK)")

    if full:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
            print(f"  [OK] {DB_PATH} supprimee (fixtures, cotes H-24, xG effaces)")
        else:
            print(f"  [--] {DB_PATH} absente (OK)")
        print("\n  >> Reset COMPLET — prochaine etape (consomme le quota API) :")
        print("      python backtest_football.py --collect --simulate --report")
        return

    if not os.path.exists(DB_PATH):
        print(f"  [--] {DB_PATH} absente — rien a vider en DB")
        print("\n  >> Prochaine etape : python backtest_football.py --collect --simulate --report")
        return

    async with aiosqlite.connect(DB_PATH, timeout=120.0) as conn:
        await configure_sqlite(conn)
        async with conn.execute("SELECT COUNT(*) FROM bt_signaux") as cur:
            n_before = (await cur.fetchone())[0]
        await conn.execute("DELETE FROM bt_signaux")
        await conn.commit()

    print(f"  [OK] bt_signaux videe ({n_before} signaux supprimes)")
    print("\n  >> Prochaine etape (sans API) :")
    print("      python backtest_football.py --simulate --report")
    print("  >> Dashboard : menu Clear cache puis Rerun")


# ─────────────────────────────────────────────────────────────
# 🚀  POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Back-test Dixon-Coles Football")
    parser.add_argument('--collect',    action='store_true', help='Phase 1 : collecte données')
    parser.add_argument('--simulate',   action='store_true', help='Phase 2 : simulation')
    parser.add_argument('--report',     action='store_true', help='Phase 3 : rapport + export CSV')
    parser.add_argument('--reset',      action='store_true',
                        help='Vide bt_signaux + supprime backtest_results.csv (garde la DB)')
    parser.add_argument('--reset-full', action='store_true',
                        help='Supprime backtest_data.db + CSV (recollecte API requise)')
    parser.add_argument('--tune',       action='store_true',
                        help='Calibration walk-forward n_prior / rho / demi-vies -> foot_params_tuned.json')
    parser.add_argument('--tune-metric', type=str, default='loglik',
                        choices=TUNE_METRICS,
                        help='Metrique de tuning : loglik (defaut), clv (CLV AH), blend (mixte)')
    parser.add_argument('--tune-clv-weight', type=float, default=0.5,
                        help='Poids CLV dans --tune-metric blend (0.0-1.0, defaut 0.5)')
    parser.add_argument('--ligue',      type=str, default=None,
                        help='Filtrer une ligue (nom partiel, ex: Championship, "Ligue 1")')
    parser.add_argument('--odds-only',  action='store_true',
                        help='Avec --collect : recollecte uniquement les cotes (fixtures/xG déjà en base)')
    parser.add_argument('--explore-ev',  action='store_true',
                        help='Compare scénarios ev_max (volume / CLV / ROI) sur signaux en base')
    parser.add_argument('--ev-max-spreads', type=float, default=None,
                        help='Plafond EV AH en simulation (ex: 0.09). Live inchangé.')
    parser.add_argument('--ev-max-totals', type=float, default=None,
                        help='Plafond EV totaux en simulation (ex: 0.12). Live inchangé.')
    parser.add_argument('--ev-min-spreads', type=float, default=None,
                        help='Seuil EV min AH en simulation (ex: 0.06). Live inchangé.')
    parser.add_argument('--ev-min-spreads-tier', action='store_true',
                        help='EV min AH par tier ligue (Top 7%%, Mid 6%%, Niche 5%%). Backtest only.')
    parser.add_argument('--max-ah-ligue-saison', type=int, default=None,
                        help='Cap signaux AH / ligue / saison (meilleurs EV). Backtest only.')
    parser.add_argument('--p1-ah', action='store_true',
                        help='Preset volume AH : ev-max-spreads 9%% + ev-min tier (sans cap ; parité live)')
    parser.add_argument('--calibrer-ah', type=str, default=None, choices=('platt', 'isotonic'),
                        help='Calibration walk-forward P(couverture) AH par ligue (backtest only)')
    parser.add_argument('--fit-calibration', type=str, default=None, choices=('platt', 'isotonic'),
                        help='Fit offline bt_signaux → foot_calibration_ah.json')
    parser.add_argument('--fit-cal-min', type=int, default=None,
                        help='Min paris AH/ligue pour --fit-calibration (defaut 80 platt / 100 isotonic)')
    parser.add_argument('--saisons', type=str, default=None,
                        help='Saisons API à collecter (ex: 2025 ou 2023,2024,2025). Défaut : config ligue.')
    parser.add_argument('--europe-only', action='store_true',
                        help='Avec --collect : ligues européennes uniquement (12 championnats)')
    args = parser.parse_args()

    if args.tune_metric not in TUNE_METRICS:
        print(f"\n❌ --tune-metric invalide : {args.tune_metric}")
        return
    if not 0.0 <= args.tune_clv_weight <= 1.0:
        print(f"\n❌ --tune-clv-weight doit etre entre 0 et 1")
        return

    if args.odds_only and not args.collect:
        print("\n❌ --odds-only requiert --collect")
        return

    saisons_override = None
    if args.saisons:
        try:
            saisons_override = parser_saisons_cli(args.saisons)
        except ValueError as e:
            print(f"\n❌ --saisons : {e}")
            return

    ligues_filtrees = filtrer_ligues(args.ligue) if args.ligue else None
    if args.europe_only and ligues_filtrees is not None:
        ligues_filtrees = filtrer_ligues_europe(ligues_filtrees)
        if not ligues_filtrees:
            print("\n❌ Aucune ligue européenne ne correspond au filtre --ligue")
            return

    ev_max_by_market = {}
    if args.ev_max_spreads is not None:
        ev_max_by_market['spreads'] = args.ev_max_spreads
    if args.ev_max_totals is not None:
        ev_max_by_market['totals'] = args.ev_max_totals

    ev_min_by_market = {}
    ev_min_spreads_tier = args.ev_min_spreads_tier
    max_ah_ligue_saison = args.max_ah_ligue_saison

    if args.p1_ah:
        if args.ev_max_spreads is None:
            ev_max_by_market['spreads'] = P1_AH_EV_MAX_SPREADS
        if args.ev_min_spreads is None and not args.ev_min_spreads_tier:
            ev_min_spreads_tier = True
        # Pas de cap auto : FIFO live ≠ top-EV ; utiliser --max-ah-ligue-saison N si besoin

    if args.ev_min_spreads is not None:
        ev_min_by_market['spreads'] = args.ev_min_spreads

    ev_max_by_market = ev_max_by_market or None
    ev_min_by_market = ev_min_by_market or None

    if args.reset_full:
        await reset_backtest(full=True)
    elif args.reset:
        await reset_backtest(full=False)

    run_phases = (
        args.collect or args.simulate or args.report or args.tune or args.explore_ev
        or args.fit_calibration
    )
    all_phases = not run_phases and not args.reset and not args.reset_full

    if not run_phases and not all_phases:
        return

    needs_db = (
        args.simulate or args.report or args.tune or args.explore_ev
        or args.fit_calibration or all_phases
    )
    if needs_db and os.path.exists(DB_PATH) and not verifier_fichier_db():
        return

    async with aiosqlite.connect(DB_PATH, timeout=120.0) as conn:
        await configure_sqlite(conn)
        await init_db(conn)

        async with aiohttp.ClientSession() as session:
            sim_ok = True
            if args.collect or all_phases:
                await phase_collecte(
                    conn, session,
                    ligues=ligues_filtrees,
                    odds_only=args.odds_only,
                    saisons_override=saisons_override,
                    europe_only=args.europe_only,
                )
            if args.tune:
                await tune_hyperparams_walkforward(
                    conn, ligues=ligues_filtrees,
                    metric=args.tune_metric, clv_weight=args.tune_clv_weight,
                )
            if args.simulate or all_phases:
                async with aiosqlite.connect(
                    f"file:{DB_PATH}?mode=ro", uri=True, timeout=120.0
                ) as conn_ro:
                    sim_ok = (await simuler_paris(
                        conn_ro, ligues=ligues_filtrees, ev_max_by_market=ev_max_by_market,
                        ev_min_by_market=ev_min_by_market,
                        ev_min_spreads_tier=ev_min_spreads_tier,
                        max_ah_ligue_saison=max_ah_ligue_saison,
                        calibrer_ah=args.calibrer_ah,
                    )) >= 0
            if args.fit_calibration:
                min_fit = args.fit_cal_min
                if min_fit is None:
                    min_fit = (
                        CALIB_AH_MIN_SAMPLES_ISOTONIC if args.fit_calibration == "isotonic"
                        else CALIB_AH_MIN_SAMPLES_PLATT
                    )
                await fit_calibration_ah_db(
                    conn, method=args.fit_calibration, min_samples=min_fit,
                )
            if args.explore_ev:
                signaux = await charger_signaux(conn)
                explore_ev_filtres(signaux)
            if args.report or all_phases:
                if (args.simulate or all_phases) and not sim_ok:
                    print("\n⚠️  Rapport ignoré — la simulation n'a pas pu écrire en base.")
                    print("    Corrigez le verrou DB puis relancez --simulate --report")
                else:
                    await generer_rapport(conn)


if __name__ == "__main__":
    asyncio.run(main())
