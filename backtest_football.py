"""
backtest_football.py — Back-test complet du modèle Dixon-Coles sur 2 saisons
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
    get_n_prior,
    get_rho_fallback,
    get_xg_half_life_days,
    save_tuned_params,
    xg_decay_rate,
)
from odds_devig import cote_fair_2way

load_project_env("foot")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
API_ODDS_KEY     = os.getenv("API_ODDS_KEY")

URL_FOOTBALL = "https://v3.football.api-sports.io"


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
SAISONS_BACKTEST = [2023, 2024]  # 2023 = saison 2023-24 pour ligues hivernales

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
# Cotes backtest collectées à H-24 (table bt_odds_h24)
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
    Fenêtre [0.9h, 36h] : 10% (marché sharp) → 30% (signal modèle plus libre).
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
    # 🇳🇴 ELITESERIEN
    "FK Bodø/Glimt": "Bodo/Glimt", "Bodø/Glimt": "Bodo/Glimt",
    "Molde FK": "Molde", "Rosenborg BK": "Rosenborg", "SK Brann": "Brann",
    "Viking FK": "Viking", "Tromsø IL": "Tromso", "IL Tromso": "Tromso",
    "Stabæk Fotball": "Stabek", "Strømsgodset IF": "Stromsgodset",
    "FK Haugesund": "Haugesund", "Odd BK": "Odd",
    "Sandefjord Fotball": "Sandefjord", "Lillestrøm SK": "Lillestrom",
    # 🇧🇪 JUPILER PRO LEAGUE
    "RSC Anderlecht": "Anderlecht", "Club Brugge KV": "Club Brugge",
    "KAA Gent": "Gent", "Standard Liège": "Standard Liege",
    "Standard de Liège": "Standard Liege", "KRC Genk": "Genk",
    "Royal Antwerp FC": "Antwerp",
    "Royale Union Saint-Gilloise": "Union Saint Gilloise",
    "R. Charleroi SC": "Charleroi", "Cercle Brugge KSV": "Cercle Brugge",
    "Sint-Truidense VV": "Sint-Truiden", "KV Mechelen": "Mechelen",
    "KV Kortrijk": "Kortrijk", "K. Beerschot VA": "Beerschot",
    "KAS Eupen": "Eupen", "OH Leuven": "OHL Leuven",
    "Westerlo": "Westerlo", "RWDM Brussels FC": "RWDM",
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

    sql = "INSERT OR REPLACE INTO bt_signaux VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
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
            PRIMARY KEY (fixture_id, market, outcome, h_val)
        );
    """)
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
    estivales = {71, 113, 253, 103}
    return annee if ligue_id in estivales else annee


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
            fixture_name_map[(row[0], row[1])] = fid

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


def trouver_fixture(home_odds, away_odds, commence_time, fixture_map):
    """
    Trouve l'ID du fixture API-Football correspondant à un event Odds API.
    fixture_map : {(home_name, away_name, date_str): fixture_id}
    """
    best_id, best_score = None, 0
    for (h_name, a_name), fid in fixture_map.items():
        h_mapped = NAME_MAPPING.get(h_name, h_name)
        a_mapped = NAME_MAPPING.get(a_name, a_name)
        score_h = process.extractOne(home_odds, [h_mapped])[1]
        score_a = process.extractOne(away_odds, [a_mapped])[1]
        score = (score_h + score_a) / 2
        if score > best_score and score > 75:
            best_score = score
            best_id = fid
    return best_id


async def phase_collecte(conn, session, ligues=None, odds_only=False):
    print("\n" + "="*60)
    print("📥  PHASE 1 — COLLECTE DES DONNÉES HISTORIQUES")
    if odds_only:
        print("  (mode --odds-only : fixtures/xG ignorés, cotes uniquement)")
    if ligues and len(ligues) < len(CHAMPIONNATS):
        print(f"  (filtre : {', '.join(l['nom'] for l in ligues)})")
    print("="*60)

    if not verifier_cles_api():
        return

    ligues = ligues or CHAMPIONNATS

    for ligue in ligues:
        for saison in SAISONS_BACKTEST:
            print(f"\n🔄 {ligue['nom']} — Saison {saison}")

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
                          ligue_id=None, mot_map=None, luck_map=None, rho_fallback=None):
    """
    Calcule L_A / L_B avec la même logique que analyser_un_match() du bot live :
      λ_xg : formule venue-normalisée (xg_off × xg_def_adverse / moyenne ligue)
      motivation + PDO (mot_map / luck_map) appliqués sur l'attaque
      λ_dc : attack × defense × γ
      blend 50/50 si DC disponible pour les deux équipes, sinon xG pur.
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
        L_A = 0.50 * L_A_dc + 0.50 * L_A_xg
        L_B = 0.50 * L_B_dc + 0.50 * L_B_xg
        rho = dc['rho']
    else:
        L_A, L_B = L_A_xg, L_B_xg

    return max(0.4, L_A), max(0.4, L_B), rho


async def reconstruire_xg_equipe(conn, team_id, ligue_id, avant_date, saison, venue='all',
                                 ligue_avg=1.3, n_prior=None, xg_half_life_days=None):
    """
    Calcule le xG moyen de l'équipe en utilisant UNIQUEMENT les matchs
    joués AVANT avant_date. Réplique la logique du bot principal :
    - split home/away (venue='home'|'away'|'all')
    - decay exponentiel (demi-vie configurable, défaut foot_params)
    - shrinkage bayésien adaptatif (n_prior)
    - fallback saison précédente si < 10 matchs
    Retourne (xg_off, xg_def, n_matchs).
    """
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
        return await reconstruire_xg_equipe(
            conn, team_id, ligue_id, avant_date, saison, venue='all',
            ligue_avg=ligue_avg, n_prior=n_prior, xg_half_life_days=xg_half_life_days,
        )

    if len(rows) < 5:
        return 1.3, 1.1, len(rows)  # Promu / données insuffisantes

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
        tp += xg_p * w
        tc += xg_c * w
        tw += w

    xg_off_brut = tp / tw
    xg_def_brut = tc / tw

    n = len(rows)
    w_eq = n / (n + n_prior)
    xg_off = w_eq * xg_off_brut + (1 - w_eq) * ligue_avg
    xg_def = w_eq * xg_def_brut + (1 - w_eq) * ligue_avg
    return xg_off, xg_def, n


# ─────────────────────────────────────────────────────────────
# 🔧  CALIBRATION WALK-FORWARD (n_prior / rho / demi-vies)
# ─────────────────────────────────────────────────────────────
TUNE_GRID_N_PRIOR = [6, 8, 11]
TUNE_GRID_RHO = [-0.14, -0.11, -0.08]
TUNE_GRID_XG_HL = [35, 46, 58]
TUNE_GRID_DC_HL = [75, 90, 120]
TUNE_MIN_MATCHS = 80


async def _lambdas_pour_match(conn, ligue, saison, date_utc, h_id, a_id,
                              n_prior, xg_half_life, dc_half_life, rho_fallback, caches):
    """λ_home / λ_away / rho — données strictement avant date_utc."""
    ligue_id = ligue['id']
    jour = _jour_cache(date_utc)

    avg_key = (ligue_id, saison, jour, n_prior, xg_half_life)
    if avg_key not in caches['avg']:
        caches['avg'][avg_key] = await calculer_ligue_avg(conn, ligue_id, saison, date_utc)
    avg_ligue = caches['avg'][avg_key]

    async def xg(team, venue):
        k = (team, ligue_id, date_utc, saison, venue, n_prior, xg_half_life)
        if k not in caches['xg']:
            caches['xg'][k] = await reconstruire_xg_equipe(
                conn, team, ligue_id, date_utc, saison, venue=venue,
                ligue_avg=avg_ligue, n_prior=n_prior, xg_half_life_days=xg_half_life,
            )
        return caches['xg'][k]

    xg_off_d_sp, xg_def_d_sp, n_d = await xg(h_id, 'home')
    xg_off_e_sp, xg_def_e_sp, n_e = await xg(a_id, 'away')
    xg_off_d_gl, xg_def_d_gl, _ = await xg(h_id, 'all')
    xg_off_e_gl, xg_def_e_gl, _ = await xg(a_id, 'all')

    def w_venue(n_spec, max_w=0.80):
        return min(max_w, (n_spec / 10.0) * max_w)

    wd, we = w_venue(n_d), w_venue(n_e)
    xg_off_d = xg_off_d_sp * wd + xg_off_d_gl * (1 - wd)
    xg_def_d = xg_def_d_sp * wd + xg_def_d_gl * (1 - wd)
    xg_off_e = xg_off_e_sp * we + xg_off_e_gl * (1 - we)
    xg_def_e = xg_def_e_sp * we + xg_def_e_gl * (1 - we)

    venue_key = (ligue_id, saison, jour)
    if venue_key not in caches['venue']:
        caches['venue'][venue_key] = await calculer_moyennes_venue(conn, ligue_id, saison, date_utc)
    m_dom_l, m_ext_l = caches['venue'][venue_key]

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
    )


async def _log_prob_score(conn, ligue, match_row, params, caches):
    _, saison, date_utc, h_id, a_id, _, _, gh, ga = match_row
    if gh is None or ga is None:
        return None
    L_A, L_B, rho = await _lambdas_pour_match(
        conn, ligue, saison, date_utc, h_id, a_id,
        params['n_prior'], params['xg_half_life_days'], params['dc_half_life_days'],
        params['rho'], caches,
    )
    mat = generer_matrice(L_A, L_B, rho)
    gi = min(int(gh), mat.shape[0] - 1)
    ga_i = min(int(ga), mat.shape[1] - 1)
    return float(np.log(max(float(mat[gi, ga_i]), 1e-12)))


async def _eval_params_walkforward(conn, ligue, fixtures, params, start_idx=0):
    caches = {'avg': {}, 'xg': {}, 'venue': {}, 'mot_luck': {}, 'dc': {}}
    total, n = 0.0, 0
    for row in fixtures[start_idx:]:
        ll = await _log_prob_score(conn, ligue, row, params, caches)
        if ll is not None:
            total += ll
            n += 1
    return total / n if n else float('-inf')


async def tune_ligue_walkforward(conn, ligue, fixtures):
    if len(fixtures) < TUNE_MIN_MATCHS:
        print(f"  ⚠️ {ligue['nom']} : {len(fixtures)} matchs (< {TUNE_MIN_MATCHS}) — ignorée", flush=True)
        return None

    n_eval = len(TUNE_GRID_N_PRIOR) * len(TUNE_GRID_XG_HL) + len(TUNE_GRID_RHO) * len(TUNE_GRID_DC_HL)
    start = max(TUNE_MIN_MATCHS // 2, len(fixtures) // 4)
    n_scored = len(fixtures) - start
    print(
        f"  ▶ {ligue['nom']} : {len(fixtures)} matchs, {n_scored} scorés (burn-in {start}), "
        f"{n_eval} combinaisons…",
        flush=True,
    )

    best = {
        'n_prior': get_n_prior(ligue['id']),
        'rho': get_rho_fallback(ligue['id']),
        'xg_half_life_days': get_xg_half_life_days(ligue['id']),
        'dc_half_life_days': get_dc_half_life_days(ligue['id']),
    }
    best_ll = float('-inf')
    step = 0

    for np_ in TUNE_GRID_N_PRIOR:
        for xg_hl in TUNE_GRID_XG_HL:
            step += 1
            p = {**best, 'n_prior': np_, 'xg_half_life_days': xg_hl}
            ll = await _eval_params_walkforward(conn, ligue, fixtures, p, start)
            print(
                f"     [{step}/{n_eval}] n_prior={np_} xg_hl={xg_hl}j → log P={ll:.4f}",
                flush=True,
            )
            if ll > best_ll:
                best_ll, best = ll, p

    for rho in TUNE_GRID_RHO:
        for dc_hl in TUNE_GRID_DC_HL:
            step += 1
            p = {**best, 'rho': rho, 'dc_half_life_days': dc_hl}
            ll = await _eval_params_walkforward(conn, ligue, fixtures, p, start)
            print(
                f"     [{step}/{n_eval}] rho={rho:.2f} dc_hl={dc_hl}j → log P={ll:.4f}",
                flush=True,
            )
            if ll > best_ll:
                best_ll, best = ll, p

    best['mean_log_score'] = round(best_ll, 5)
    print(
        f"  ✅ {ligue['nom']} : n_prior={best['n_prior']} rho={best['rho']:.2f} "
        f"xg_hl={best['xg_half_life_days']:.0f}j dc_hl={best['dc_half_life_days']:.0f}j "
        f"(log P score={best_ll:.4f})",
        flush=True,
    )
    return best


async def tune_hyperparams_walkforward(conn, ligues=None):
    print("\n" + "=" * 60)
    print("🔧  CALIBRATION WALK-FORWARD (n_prior / rho / demi-vies)")
    print("=" * 60)
    print("  Métrique : log P(score réel | modèle) — pas de lookahead, pas de ROI", flush=True)
    print(f"  Grilles : n_prior{TUNE_GRID_N_PRIOR}, rho{TUNE_GRID_RHO}, "
          f"xg_hl{TUNE_GRID_XG_HL}, dc_hl{TUNE_GRID_DC_HL}", flush=True)
    print("  (1ère combinaison lente : DC MLE + xG par match — ~5–15 min/ligue)", flush=True)

    ligues = ligues or CHAMPIONNATS
    results = {}
    for ligue in ligues:
        async with conn.execute(
            "SELECT id, saison, date_utc, home_id, away_id, home_name, away_name, gh, ga "
            "FROM bt_fixtures WHERE ligue_id=? AND gh IS NOT NULL ORDER BY date_utc",
            (ligue['id'],),
        ) as cur:
            fixtures = await cur.fetchall()
        tuned = await tune_ligue_walkforward(conn, ligue, fixtures)
        if tuned:
            results[ligue['id']] = tuned

    if not results:
        print("\n❌ Aucune ligue calibrée (données insuffisantes).")
        return

    path = save_tuned_params(results)
    print(f"\n✅ Paramètres enregistrés → {path}")
    print("   Bot live + backtest : rechargement auto via foot_params.py")


async def simuler_paris(conn, ligues=None):
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
    print("  🧮 Modèle : blend 50/50 DC + xG, poids_dyn dynamique (formule bot live)")

    if n_odds == 0:
        print("\n  ⚠️  AUCUNE cote H-24 trouvée — la collecte d'odds a échoué.")
        print("  💡 Vérifiez API_ODDS_KEY et que votre plan inclut l'endpoint /v4/historical/")
        print("\n✅ Phase 2 terminée (0 signaux).")
        return -1

    all_pending = []
    ligues = ligues or CHAMPIONNATS

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
        dc_cache = {}
        avg_cache = {}
        venue_cache = {}
        mot_luck_cache = {}

        for idx, (fid, saison, date_utc, h_id, a_id, h_name, a_name, gh, ga) in enumerate(fixtures, 1):
            if idx % 200 == 0 or idx == n_fixtures:
                print(f"     … {idx}/{n_fixtures} matchs", flush=True)

            if gh is None or ga is None:
                continue

            odds_h24 = odds_by_fid.get(fid)
            if not odds_h24:
                continue

            jour = _jour_cache(date_utc)
            avg_key = (ligue['id'], saison, jour)
            if avg_key not in avg_cache:
                avg_cache[avg_key] = await calculer_ligue_avg(
                    conn, ligue['id'], saison, date_utc
                )
            avg_ligue = avg_cache[avg_key]
            n_prior_l = get_n_prior(ligue['id'])
            xg_hl = get_xg_half_life_days(ligue['id'])
            dc_hl = get_dc_half_life_days(ligue['id'])

            # Reconstituer xG AVANT ce match — split home/away + global comme le bot principal
            xg_off_d_sp, xg_def_d_sp, n_d = await reconstruire_xg_equipe(
                conn, h_id, ligue['id'], date_utc, saison, venue='home',
                ligue_avg=avg_ligue, n_prior=n_prior_l, xg_half_life_days=xg_hl,
            )
            xg_off_e_sp, xg_def_e_sp, n_e = await reconstruire_xg_equipe(
                conn, a_id, ligue['id'], date_utc, saison, venue='away',
                ligue_avg=avg_ligue, n_prior=n_prior_l, xg_half_life_days=xg_hl,
            )

            # Filtre : ignorer si l'une des équipes manque d'historique suffisant
            if n_d < 8 or n_e < 8:
                continue

            # Venue blending adaptatif : réplique w_venue() du bot
            # Moins de matchs venue-spécifiques → on se fie davantage aux stats globales
            xg_off_d_gl, xg_def_d_gl, _ = await reconstruire_xg_equipe(
                conn, h_id, ligue['id'], date_utc, saison, venue='all',
                ligue_avg=avg_ligue, n_prior=n_prior_l, xg_half_life_days=xg_hl,
            )
            xg_off_e_gl, xg_def_e_gl, _ = await reconstruire_xg_equipe(
                conn, a_id, ligue['id'], date_utc, saison, venue='all',
                ligue_avg=avg_ligue, n_prior=n_prior_l, xg_half_life_days=xg_hl,
            )

            def w_venue(n_spec, max_w=0.80):
                return min(max_w, (n_spec / 10.0) * max_w)

            wd = w_venue(n_d)
            we = w_venue(n_e)
            xg_off_d = xg_off_d_sp * wd + xg_off_d_gl * (1.0 - wd)
            xg_def_d = xg_def_d_sp * wd + xg_def_d_gl * (1.0 - wd)
            xg_off_e = xg_off_e_sp * we + xg_off_e_gl * (1.0 - we)
            xg_def_e = xg_def_e_sp * we + xg_def_e_gl * (1.0 - we)

            # Moyennes venue ligue + estimation DC (cache hebdomadaire, cron ~bot live)
            venue_key = (ligue['id'], saison, jour)
            if venue_key not in venue_cache:
                venue_cache[venue_key] = await calculer_moyennes_venue(
                    conn, ligue['id'], saison, date_utc
                )
            m_dom_l, m_ext_l = venue_cache[venue_key]

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

            # λ blend 50/50 DC + xG + motivation/PDO — aligné sur analyser_un_match() du bot live
            L_A, L_B, rho = calculer_lambda_blend(
                dc, h_id, a_id, xg_off_d, xg_def_d, xg_off_e, xg_def_e, m_dom_l, m_ext_l, ligue['id'],
                mot_map=mot_map, luck_map=luck_map,
            )

            mat = generer_matrice(L_A, L_B, rho)

            # Parcourir les marchés — collecter tous les signaux valides pour ce fixture
            home_name_odds = NAME_MAPPING.get(h_name, h_name)
            candidats = []  # (ev_final, market, outcome, h_val, cote_h24, k, mise, side_flag)

            ev_min_l = ligue.get('ev_min', 0.05)
            ev_max_l = ligue.get('ev_max', 0.15)

            # Index cotes partenaires pour no-vig (Shin) — aligné sniper_bot_foot.py
            # spreads : partenaire à handicap opposé (-h)
            # totals  : partenaire à même ligne (Over ↔ Under)
            spreads_partner: dict = {}
            totals_partner: dict = {}
            for mk, out, hv, c in odds_h24:
                if mk == 'spreads':
                    spreads_partner[(mk, hv)] = c
                elif mk == 'totals':
                    totals_partner[(hv, out.lower())] = c

            # Blend EV dynamique — à H-24 : poids_dyn ≈ 23.2% (formule bot recalibrée)
            poids_dyn = calculer_poids_dyn(H_ODDS_BACKTEST)

            for market, outcome, h_val, cote_h24 in odds_h24:
                if market not in ('spreads', 'totals'):
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
                    if cote_partner and cote_partner > 1.0:
                        cote_novig = cote_fair_2way(cote_h24, cote_partner) or cote_h24
                        ev_pinnacle = ev_total(mat, h_val, is_over, cote_novig)
                    else:
                        ev_pinnacle = ev_modele
                    side_flag = is_over

                ev_final = ev_modele * poids_dyn + ev_pinnacle * (1.0 - poids_dyn)

                if not (ev_min_l <= ev_final <= ev_max_l):
                    continue

                mise = min(round(k * 100 * KELLY_FRAC, 2), 5.0)
                if mise < 0.1:
                    continue

                candidats.append((ev_final, market, outcome, h_val, cote_h24, k, mise, side_flag))

            if not candidats:
                continue

            # 🔒 FILTRE 1 PARI/MATCH : garder uniquement le signal avec le meilleur EV final
            candidats.sort(key=lambda x: x[0], reverse=True)
            ev_final, market, outcome, h_val, cote_h24, k, mise, flag = candidats[0]

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
            ))
            signaux += 1

        all_pending.extend(pending)
        print(f"  {ligue['nom']} : {signaux} signaux générés")

    print(f"\n  📝 Total calculé : {len(all_pending)} signaux — écriture en base...")
    n_saved = await persister_signaux(all_pending)
    if n_saved >= 0:
        print("\n✅ Phase 2 terminée.")
    else:
        print("\n❌ Phase 2 échouée (données en base inchangées — rapport = anciens signaux).")
    return n_saved


# ─────────────────────────────────────────────────────────────
# 📊  PHASE 3 — RAPPORT D'ANALYSE
# ─────────────────────────────────────────────────────────────
async def generer_rapport(conn):
    print("\n" + "="*60)
    print("📊  PHASE 3 — RAPPORT D'ANALYSE")
    print("="*60)

    nom_par_id = {c['id']: c['nom'] for c in CHAMPIONNATS}

    async with conn.execute("""
        SELECT s.fixture_id, s.ligue_id, s.saison, s.market, s.outcome,
               s.h_val, s.cote_h24, s.cote_cloture, s.ev_modele, s.kelly,
               s.mise, s.gh, s.ga, s.resultat, s.clv, f.date_utc
        FROM bt_signaux s
        JOIN bt_fixtures f ON f.id = s.fixture_id
        WHERE s.resultat IS NOT NULL
        ORDER BY f.date_utc
    """) as cur:
        rows = await cur.fetchall()

    # Ajouter le nom de la ligue en fin de tuple (compatible avec toutes versions SQLite)
    signaux = [(*r, nom_par_id.get(r[1], str(r[1]))) for r in rows]

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
    print(f"  PAR MARCHÉ")
    print(f"{'─'*50}")
    for market in ['spreads', 'totals']:
        rows_m = [s for s in signaux if s[3] == market and s[13] is not None]
        if not rows_m:
            continue
        pnl_m = sum(r[13] * r[10] for r in rows_m)
        mis_m = sum(r[10] for r in rows_m)
        roi_m = pnl_m / mis_m if mis_m else 0
        label = "Handicap Asiatique" if market == 'spreads' else "Totaux"
        print(f"  {label:<22} {len(rows_m):>5} signaux → ROI {roi_m:+.2%} | P&L {pnl_m:+.1f}u")

    # ── Courbe de calibration ───────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  CALIBRATION DU MODÈLE (P_modèle vs fréquence réelle)")
    print(f"{'─'*50}")
    # Discrétiser les EV en tranches
    tranches = [(0.05, 0.07), (0.07, 0.09), (0.09, 0.12), (0.12, 0.20)]
    for lo, hi in tranches:
        rows_t = [s for s in signaux if lo <= s[8] < hi and s[13] is not None]
        if not rows_t:
            continue
        win = sum(1 for s in rows_t if s[13] > 0)
        push = sum(1 for s in rows_t if s[13] == 0)
        n_t = len(rows_t)
        wr = win / (n_t - push) if (n_t - push) > 0 else 0
        ev_moy = np.mean([s[8] for s in rows_t])
        print(f"  EV [{lo:.0%}-{hi:.0%}]  n={n_t:>4}  Win={wr:.1%}  EV_moy={ev_moy:+.2%}")

    # ── Export CSV ──────────────────────────────────────────
    csv_path = "backtest_results.csv"
    async with conn.execute("""
        SELECT s.fixture_id, s.ligue_id, s.saison, s.market, s.outcome,
               s.h_val, s.cote_h24, s.cote_cloture, s.ev_modele, s.kelly,
               s.mise, s.gh, s.ga, s.resultat, s.clv, f.date_utc
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
                    'mise', 'gh', 'ga', 'resultat', 'clv', 'date_utc'])
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
                        help='Calibration walk-forward n_prior / rho / demi-vies → foot_params_tuned.json')
    parser.add_argument('--ligue',      type=str, default=None,
                        help='Filtrer une ligue (nom partiel, ex: Championship, "Ligue 1")')
    parser.add_argument('--odds-only',  action='store_true',
                        help='Avec --collect : recollecte uniquement les cotes (fixtures/xG déjà en base)')
    args = parser.parse_args()

    if args.odds_only and not args.collect:
        print("\n❌ --odds-only requiert --collect")
        return

    ligues_filtrees = filtrer_ligues(args.ligue) if args.ligue else None

    if args.reset_full:
        await reset_backtest(full=True)
    elif args.reset:
        await reset_backtest(full=False)

    run_phases = args.collect or args.simulate or args.report or args.tune
    all_phases = not run_phases and not args.reset and not args.reset_full

    if not run_phases and not all_phases:
        return

    needs_db = args.simulate or args.report or args.tune or all_phases
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
                )
            if args.tune:
                await tune_hyperparams_walkforward(conn, ligues=ligues_filtrees)
            if args.simulate or all_phases:
                async with aiosqlite.connect(
                    f"file:{DB_PATH}?mode=ro", uri=True, timeout=120.0
                ) as conn_ro:
                    sim_ok = (await simuler_paris(conn_ro, ligues=ligues_filtrees)) >= 0
            if args.report or all_phases:
                if (args.simulate or all_phases) and not sim_ok:
                    print("\n⚠️  Rapport ignoré — la simulation n'a pas pu écrire en base.")
                    print("    Corrigez le verrou DB puis relancez --simulate --report")
                else:
                    await generer_rapport(conn)


if __name__ == "__main__":
    asyncio.run(main())
