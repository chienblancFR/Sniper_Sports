import copy
import requests
import csv
import math
import time
import os
import logging
import ftplib
import json
from datetime import datetime, timedelta, timezone
from io import StringIO
import traceback
from scipy.optimize import minimize
from config_env import env_files_hint, load_project_env

load_project_env("nhl")

# ==========================================
# ⚙️ CONFIGURATION GLOBALE
# ==========================================
logging.basicConfig(
    filename="nhl_sniper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)


def _env_bool(key, default=False):
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes", "on")


def log_nhl(msg, level="info"):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))
    getattr(logging, level, logging.info)(msg)


TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ODDS_API_KEY = os.environ.get("API_ODDS_KEY", "")
FICHIER_MEMOIRE = "alertes_nhl_envoyees.txt"
JOURNAL_NOM = "journal_trading_nhl_SEC2026xOmG.csv"
PA_DATA_DIR = "/home/chienblanc/data"
FICHIER_JOURNAL = (
    os.path.join(PA_DATA_DIR, JOURNAL_NOM)
    if os.path.isdir(PA_DATA_DIR)
    else JOURNAL_NOM
)
BANKROLL_INITIALE = float(os.environ.get("NHL_BANKROLL", "1000.0"))
NHL_SEASON = int(os.environ.get("NHL_SEASON", "2026"))
EDGE_MINIMUM = float(os.environ.get("NHL_EDGE_MIN", "0.02"))
# Edge min plus élevé tant que les gardiens ne sont pas confirmés (sinon on attend)
NHL_EDGE_MIN_PROBABLE = float(os.environ.get("NHL_EDGE_MIN_PROBABLE", "0.04"))
KELLY_FRACTION = float(os.environ.get("NHL_KELLY_FRACTION", "0.25"))
KELLY_FRACTION_GARDIEN_INCERTAIN = float(os.environ.get("NHL_KELLY_GARDIEN", "0.125"))
NHL_KELLY_DYNAMIQUE_ACTIF = _env_bool("NHL_KELLY_DYNAMIQUE_ACTIF", True)
NHL_KELLY_BRIER_FENETRE = int(os.environ.get("NHL_KELLY_BRIER_FENETRE", "40"))
NHL_KELLY_BRIER_MIN_PARIS = int(os.environ.get("NHL_KELLY_BRIER_MIN_PARIS", "20"))
NHL_KELLY_BSS_SENSIBILITE = float(os.environ.get("NHL_KELLY_BSS_SENSIBILITE", "0.30"))
NHL_KELLY_MULT_MIN = float(os.environ.get("NHL_KELLY_MULT_MIN", "0.4"))
NHL_KELLY_CLV_ACTIF = _env_bool("NHL_KELLY_CLV_ACTIF", True)
NHL_KELLY_CLV_FENETRE = int(os.environ.get("NHL_KELLY_CLV_FENETRE", "40"))
NHL_KELLY_CLV_MIN_PARIS = int(os.environ.get("NHL_KELLY_CLV_MIN_PARIS", "15"))
NHL_KELLY_CLV_SENSIBILITE = float(os.environ.get("NHL_KELLY_CLV_SENSIBILITE", "0.04"))
NHL_KELLY_CLV_MULT_MIN = float(os.environ.get("NHL_KELLY_CLV_MULT_MIN", "0.5"))
NHL_LINE_MOVE_ACTIF = _env_bool("NHL_LINE_MOVE_ACTIF", True)
NHL_LINE_MIN_AGE_MIN = int(os.environ.get("NHL_LINE_MIN_AGE_MIN", "45"))
NHL_LINE_MAX_SNAPSHOTS = int(os.environ.get("NHL_LINE_MAX_SNAPSHOTS", "48"))
NHL_LINE_STEAM_WARN_PCT = float(os.environ.get("NHL_LINE_STEAM_WARN_PCT", "0.025"))
NHL_LINE_STEAM_BLOCK_PCT = float(os.environ.get("NHL_LINE_STEAM_BLOCK_PCT", "0.05"))
NHL_LINE_STEAM_EDGE_EXTRA = float(os.environ.get("NHL_LINE_STEAM_EDGE_EXTRA", "0.015"))
NHL_MISE_MAX_PCT = float(os.environ.get("NHL_MISE_MAX_PCT", "2"))
NHL_DRY_RUN = _env_bool("NHL_DRY_RUN", False)
NHL_PARIS_JOUR_MAX = int(os.environ.get("NHL_PARIS_JOUR_MAX", "0"))
NHL_ODDS_QUOTA_ALERT = int(os.environ.get("NHL_ODDS_QUOTA_ALERT", "100"))
NHL_BLEND_GP_PLEIN = float(os.environ.get("NHL_BLEND_GP_PLEIN", "20"))
NHL_PP_PK_SHRINK_GP = float(os.environ.get("NHL_PP_PK_SHRINK_GP", "20"))
# MoneyPuck fournit nativement des CSV "forme récente" (10 ou 20 derniers matchs)
NHL_GSAX_RECENT_WINDOW = int(os.environ.get("NHL_GSAX_RECENT_WINDOW", "10"))
NHL_GSAX_RECENT_GP_PLEIN = float(os.environ.get("NHL_GSAX_RECENT_GP_PLEIN", "8"))
NHL_TEAM_RECENT_WINDOW = int(os.environ.get("NHL_TEAM_RECENT_WINDOW", "10"))
NHL_TEAM_RECENT_GP_PLEIN = float(os.environ.get("NHL_TEAM_RECENT_GP_PLEIN", "10"))
NHL_MARCHE_SHRINK_ACTIF = _env_bool("NHL_MARCHE_SHRINK_ACTIF", True)
NHL_MODEL_TRUST_MIN = float(os.environ.get("NHL_MODEL_TRUST_MIN", "0.70"))
NHL_MODEL_TRUST_MAX = float(os.environ.get("NHL_MODEL_TRUST_MAX", "0.90"))
NHL_MODEL_TRUST_GP_PLEIN = float(os.environ.get("NHL_MODEL_TRUST_GP_PLEIN", "20"))
NHL_RHO_MIN_MATCHS = int(os.environ.get("NHL_RHO_MIN_MATCHS", "30"))
NHL_RHO_INTERVAL_MATCHS = int(os.environ.get("NHL_RHO_INTERVAL_MATCHS", "20"))
NHL_EMPTY_NET_MIN_MATCHS = int(os.environ.get("NHL_EMPTY_NET_MIN_MATCHS", "60"))
# Calibration MLE élargie à tous les matchs ligue (pas seulement les paris placés)
NHL_LIGUE_CALIB_ACTIF = _env_bool("NHL_LIGUE_CALIB_ACTIF", True)
NHL_LIGUE_CALIB_SCAN_JOURS = int(os.environ.get("NHL_LIGUE_CALIB_SCAN_JOURS", "3"))
NHL_LIGUE_CALIB_LOOKBACK_JOURS = int(os.environ.get("NHL_LIGUE_CALIB_LOOKBACK_JOURS", "30"))
NHL_LIGUE_CALIB_MIN_MATCHS = int(os.environ.get("NHL_LIGUE_CALIB_MIN_MATCHS", "200"))
NHL_LIGUE_CALIB_MAX_GAMES = int(os.environ.get("NHL_LIGUE_CALIB_MAX_GAMES", "1400"))
# Pondération temporelle MLE : les matchs récents comptent plus (demi-vie en jours)
NHL_MLE_RECENCY_ACTIF = _env_bool("NHL_MLE_RECENCY_ACTIF", True)
NHL_MLE_RECENCY_HALFLIFE_JOURS = float(os.environ.get("NHL_MLE_RECENCY_HALFLIFE_JOURS", "28"))
NHL_MARCHE_COHERENCE_ACTIF = _env_bool("NHL_MARCHE_COHERENCE_ACTIF", True)
NHL_REF_SHRINK_ACTIF = _env_bool("NHL_REF_SHRINK_ACTIF", True)
NHL_OT_HOME_ADVANTAGE = float(os.environ.get("NHL_OT_HOME_ADVANTAGE", "0.52"))
NHL_HIA_DEFAULT = float(os.environ.get("NHL_HIA_DEFAULT", "0.05"))
NHL_HIA_PAR_EQUIPE_ACTIF = _env_bool("NHL_HIA_PAR_EQUIPE_ACTIF", True)
NHL_HIA_TEAM_GP_PLEIN = float(os.environ.get("NHL_HIA_TEAM_GP_PLEIN", "15"))
NHL_HIA_TEAM_MIN_GAMES = int(os.environ.get("NHL_HIA_TEAM_MIN_GAMES", "3"))
HIA_REF_CALIBRATION = 0.05  # HIA utilisé lors du calcul des lambdas historiques du journal
NHL_MARCHES_ACTIFS = {m.strip().upper() for m in os.environ.get("NHL_MARCHES_ACTIFS", "ML,PL,OU").split(",") if m.strip()}
RHO_META_FILE = "rho_calibrage_meta.json"
REFEREE_META_FILE = "referee_calibrage_meta.json"
ODDS_HISTORY_FILE = "nhl_odds_history.json"
LEAGUE_CALIB_FILE = "nhl_league_calib.json"
HIA_TEAM_META_FILE = "hia_equipes_meta.json"
# Préférence colonnes xG MoneyPuck (score/venue > flurry > brut)
XG_FOR_COLONNES = (
    "flurryScoreVenueAdjustedxGoalsFor",
    "scoreVenueAdjustedxGoalsFor",
    "flurryAdjustedxGoalsFor",
    "xGoalsFor",
)
XG_AGAINST_COLONNES = (
    "flurryScoreVenueAdjustedxGoalsAgainst",
    "scoreVenueAdjustedxGoalsAgainst",
    "flurryAdjustedxGoalsAgainst",
    "xGoalsAgainst",
)
_xg_colonnes_actives = {"for": None, "against": None}
_pp_pk_shrink_logue = False
_gsax_recent_logue = False
_team_recent_logue = False
_travel_fatigue_logue = False
_faceoff_adj_logue = False
_ref_adj_logue = False
_line_move_logue = False
_marche_coherence_logue = False
_ref_shrink_logue = False
_hia_equipe_logue = False
FLOAT_TOL = 0.01
# Part du temps de jeu gardien (~54 min) pour convertir GSAx/60 → impact/match
GSAX_MINUTES_PAR_MATCH = float(os.environ.get("NHL_GSAX_MINUTES", "54.0"))
NHL_SCAN_HEURES_AVANCE = float(os.environ.get("NHL_SCAN_HEURES_AVANCE", "18"))
# Minutes avant le puck drop où l'on arrête de chercher de nouveaux paris
NHL_MINUTES_AVANT_PUCK = float(os.environ.get("NHL_MINUTES_AVANT_PUCK", "5"))
NHL_TRAVEL_FATIGUE_ACTIF = _env_bool("NHL_TRAVEL_FATIGUE_ACTIF", True)
NHL_TRAVEL_LOOKBACK_JOURS = int(os.environ.get("NHL_TRAVEL_LOOKBACK_JOURS", "5"))
NHL_TRAVEL_MILES_REF = float(os.environ.get("NHL_TRAVEL_MILES_REF", "1500"))
NHL_TRAVEL_B2B_ATK_PCT = float(os.environ.get("NHL_TRAVEL_B2B_ATK_PCT", "0.04"))
NHL_TRAVEL_B2B_DEF_PCT = float(os.environ.get("NHL_TRAVEL_B2B_DEF_PCT", "0.06"))
NHL_TRAVEL_SOLO_ATK_PCT = float(os.environ.get("NHL_TRAVEL_SOLO_ATK_PCT", "0.015"))
NHL_TRAVEL_SOLO_DEF_PCT = float(os.environ.get("NHL_TRAVEL_SOLO_DEF_PCT", "0.02"))
NHL_TRAVEL_LONG_MILES = float(os.environ.get("NHL_TRAVEL_LONG_MILES", "1000"))
NHL_FACEOFF_ADJ_ACTIF = _env_bool("NHL_FACEOFF_ADJ_ACTIF", True)
NHL_FACEOFF_SENSIBILITE = float(os.environ.get("NHL_FACEOFF_SENSIBILITE", "0.25"))
NHL_REF_ADJ_ACTIF = _env_bool("NHL_REF_ADJ_ACTIF", True)
NHL_REF_SENSIBILITE = float(os.environ.get("NHL_REF_SENSIBILITE", "0.20"))
NHL_REF_LOOKBACK_JOURS = int(os.environ.get("NHL_REF_LOOKBACK_JOURS", "30"))
NHL_REF_SCAN_JOURS = int(os.environ.get("NHL_REF_SCAN_JOURS", "3"))
NHL_REF_MIN_MATCHS = int(os.environ.get("NHL_REF_MIN_MATCHS", "12"))

ETATS_MATCH_EXCLUS = {"FINAL", "OFF", "OFFICIAL", "POSTPONED", "PPD", "LIVE", "CRIT"}
ETATS_MATCH_PRIORITAIRES = {"FUT", "PRE"}

# Mappage des abréviations (NHL API) vers noms complets (The-Odds-API)
NHL_TEAMS_MAPPING = {
    "ANA": "Anaheim Ducks", "UTA": "Utah Hockey Club",
    "BOS": "Boston Bruins", "BUF": "Buffalo Sabres", "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars", "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers", "FLA": "Florida Panthers", "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens", "NSH": "Nashville Predators",
    "NJD": "New Jersey Devils", "NYI": "New York Islanders", "NYR": "New York Rangers",
    "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins",
    "SJS": "San Jose Sharks", "SEA": "Seattle Kraken", "STL": "St Louis Blues",
    "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs", "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights", "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}

# Alias The-Odds-API (noms alternatifs Pinnacle → clé interne)
ODDS_API_ALIASES = {
    "Montreal Canadiens": ["Montréal Canadiens"],
    "Utah Hockey Club": ["Utah Mammoth"],
}
# Index inverse : alias → nom primaire
_ODDS_NOM_PRIMAIRE = {}
for _primaire, _alias_liste in ODDS_API_ALIASES.items():
    for _alias in _alias_liste:
        _ODDS_NOM_PRIMAIRE[_alias] = _primaire

# Carte des fuseaux horaires (0 = EST, 1 = CENTRAL, 2 = MOUNTAIN, 3 = PACIFIC)
NHL_TIMEZONES = {
    "BOS": 0, "BUF": 0, "CAR": 0, "CBJ": 0, "DET": 0, "FLA": 0, "MTL": 0, "NJD": 0,
    "NYI": 0, "NYR": 0, "OTT": 0, "PHI": 0, "PIT": 0, "TBL": 0, "TOR": 0, "WSH": 0,
    "CHI": 1, "DAL": 1, "MIN": 1, "NSH": 1, "STL": 1, "WPG": 1,
    "COL": 2, "UTA": 2, "CGY": 2, "EDM": 2,
    "ANA": 3, "LAK": 3, "SJS": 3, "SEA": 3, "VAN": 3, "VGK": 3
}

# Coordonnées arènes domicile (lat, lon) — distance réelle vs simple décalage fuseau
NHL_ARENA_COORDS = {
    "ANA": (33.8078, -117.8765), "BOS": (42.3662, -71.0621), "BUF": (42.8750, -78.8764),
    "CGY": (51.0374, -114.0517), "CAR": (35.8033, -78.7219), "CHI": (41.8807, -87.6742),
    "COL": (39.7487, -105.0077), "CBJ": (39.9692, -83.0061), "DAL": (32.7905, -96.8103),
    "DET": (42.3411, -83.0553), "EDM": (53.5469, -113.4978), "FLA": (26.1584, -80.3257),
    "LAK": (34.0430, -118.2673), "MIN": (44.9448, -93.1010), "MTL": (45.4961, -73.5693),
    "NSH": (36.1592, -86.7785), "NJD": (40.7335, -74.1711), "NYI": (40.7229, -73.5904),
    "NYR": (40.7505, -73.9934), "OTT": (45.2969, -75.9271), "PHI": (39.9012, -75.1720),
    "PIT": (40.4394, -79.9892), "SJS": (37.3327, -121.9010), "SEA": (47.6220, -122.3540),
    "STL": (38.6268, -90.2027), "TBL": (27.9428, -82.4519), "TOR": (43.6435, -79.3791),
    "UTA": (40.7683, -111.9011), "VAN": (49.2778, -123.1089), "VGK": (36.1029, -115.1784),
    "WSH": (38.8981, -77.0209), "WPG": (49.8928, -97.1436),
}

JOURNAL_COLONNES = [
    "Date", "ID_Match", "Visiteur", "Local", "Pari",
    "Vraie_Cote_Bot", "Cote_Prise", "Cote_CLV",
    "Lam_Ext", "Lam_Dom", "Score_Ext", "Score_Dom",
    "Edge(%)", "Risque(%)", "Mise_€", "Statut", "P&L",
    "Gardien_Ext", "Gardien_Dom", "Gardiens_Confirmes", "B2B_Home", "B2B_Away", "Rho", "Hia",
]

_odds_quota_state = {"derniere_alerte": None}

# ==========================================
# 1. ASPIRATEURS DE DONNÉES (MoneyPuck)
# ==========================================
def _float_proche(a, b, tol=FLOAT_TOL):
    return abs(float(a) - float(b)) < tol


def _arrondir_cut(point):
    """Arrondit une ligne O/U au demi-point le plus proche (5.5, 6.0…)."""
    return round(float(point) * 2) / 2


def _trouver_cle_float(dictionnaire, cible):
    """Retourne (cle, valeur) en tolérant les imprécisions float."""
    for cle, val in dictionnaire.items():
        if _float_proche(cle, cible):
            return cle, val
    return None, None


def _fetch_moneypuck_csv(kind, season):
    """Télécharge un CSV MoneyPuck seasonSummary (saison N, puis N-1 si échec)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    for s in (season, season - 1):
        url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{s}/regular/{kind}.csv"
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200 and response.text.strip():
                return response.text, s
            print(f"⚠️ MoneyPuck {kind} saison {s} : HTTP {response.status_code}")
        except Exception as e:
            print(f"⚠️ MoneyPuck {kind} saison {s} : {e}")
    return None, None




def _resoudre_colonnes_xg(fieldnames):
    """Choisit la meilleure paire de colonnes xG disponible dans le CSV MoneyPuck."""
    if not fieldnames:
        return "xGoalsFor", "xGoalsAgainst"
    cols = set(fieldnames)
    col_for = next((c for c in XG_FOR_COLONNES if c in cols), "xGoalsFor")
    col_against = next((c for c in XG_AGAINST_COLONNES if c in cols), "xGoalsAgainst")
    return col_for, col_against


def _log_colonnes_xg_si_nouveau(col_for, col_against):
    global _xg_colonnes_actives
    if _xg_colonnes_actives["for"] == col_for and _xg_colonnes_actives["against"] == col_against:
        return
    _xg_colonnes_actives["for"] = col_for
    _xg_colonnes_actives["against"] = col_against
    if col_for == "xGoalsFor":
        log_nhl("ℹ️ xG 5v5 : colonnes score-adjusted absentes — repli sur xGoalsFor brut", level="warning")
    else:
        log_nhl(f"📐 xG 5v5 : colonnes actives → {col_for} / {col_against}")


def _parser_team_stats_csv(texte):
    csv_reader = csv.DictReader(StringIO(texte))
    col_for, col_against = _resoudre_colonnes_xg(csv_reader.fieldnames)
    _log_colonnes_xg_si_nouveau(col_for, col_against)
    teams_dict = {}
    for row in csv_reader:
        team, sit = row["team"], row.get("situation")
        if team not in teams_dict:
            teams_dict[team] = {
                "team": team, "xGF_per_game": 0.0, "xGA_per_game": 0.0,
                "xGF_PP": 0.0, "xGA_PK": 0.0, "games_played": 0,
                "fo_pct": 0.5,
            }
        gp = max(float(row.get("games_played", 1)), 1)
        if sit == "5on5":
            teams_dict[team]["xGF_per_game"] = round(float(row.get(col_for, 0) or 0) / gp, 3)
            teams_dict[team]["xGA_per_game"] = round(float(row.get(col_against, 0) or 0) / gp, 3)
            teams_dict[team]["games_played"] = int(gp)
            fo_won = float(row.get("faceOffsWonFor", 0) or 0)
            fo_lost = float(row.get("faceOffsWonAgainst", 0) or 0)
            fo_total = fo_won + fo_lost
            if fo_total > 0:
                teams_dict[team]["fo_pct"] = round(fo_won / fo_total, 4)
        elif sit == "5on4":
            teams_dict[team]["xGF_PP"] = round(float(row.get(col_for, 0) or 0) / gp, 3)
        elif sit == "4on5":
            teams_dict[team]["xGA_PK"] = round(float(row.get(col_against, 0) or 0) / gp, 3)
    return list(teams_dict.values())


def _shrink_special_teams(teams):
    """
    Régularise PP/PK vers la moyenne ligue quand peu de matchs joués.
    shrink = w * valeur_equipe + (1-w) * moyenne_ligue, w = min(1, GP / NHL_PP_PK_SHRINK_GP)
    """
    global _pp_pk_shrink_logue
    if NHL_PP_PK_SHRINK_GP <= 0 or not teams:
        return teams

    pp_vals = [t["xGF_PP"] for t in teams if t.get("xGF_PP", 0) > 0]
    pk_vals = [t["xGA_PK"] for t in teams if t.get("xGA_PK", 0) > 0]
    league_pp = sum(pp_vals) / len(pp_vals) if pp_vals else 0.0
    league_pk = sum(pk_vals) / len(pk_vals) if pk_vals else 0.0

    nb_shrink = 0
    for team in teams:
        gp = max(team.get("games_played", 0), 0)
        w = min(1.0, gp / NHL_PP_PK_SHRINK_GP)
        if w < 1.0:
            nb_shrink += 1
        raw_pp = team.get("xGF_PP", league_pp)
        raw_pk = team.get("xGA_PK", league_pk)
        team["xGF_PP"] = round(w * raw_pp + (1.0 - w) * league_pp, 3)
        team["xGA_PK"] = round(w * raw_pk + (1.0 - w) * league_pk, 3)

    if not _pp_pk_shrink_logue:
        _pp_pk_shrink_logue = True
        log_nhl(
            f"📉 Shrinkage PP/PK actif — confiance pleine à {NHL_PP_PK_SHRINK_GP:.0f} GP "
            f"(ligue PP≈{league_pp:.3f} PK≈{league_pk:.3f}, {nb_shrink}/{len(teams)} équipes partiellement rétrécies)"
        )
    return teams


def _blend_team_stats(teams_courant, teams_precedent, poids_courant):
    prev_map = {t["team"]: t for t in teams_precedent}
    blended = []
    for team in teams_courant:
        prev = prev_map.get(team["team"])
        if not prev:
            blended.append(team)
            continue
        merged = {"team": team["team"], "games_played": team.get("games_played", 0)}
        for key in ("xGF_per_game", "xGA_per_game", "xGF_PP", "xGA_PK", "fo_pct"):
            v_n = team.get(key, 0.0)
            v_n1 = prev.get(key, v_n)
            prec = 4 if key == "fo_pct" else 3
            merged[key] = round(poids_courant * v_n + (1 - poids_courant) * v_n1, prec)
        blended.append(merged)
    return blended


def _blend_team_recent_form(teams, teams_recent):
    """
    Blend base (saison, déjà shrink + blend N-1 si besoin) + forme récente
    (teams_{N}.csv MoneyPuck) : w = min(1, GP_fenêtre / GP_plein).
    Remplace le momentum L10 (point %) par un signal xG directement actionnable.
    """
    global _team_recent_logue
    if NHL_TEAM_RECENT_WINDOW <= 0 or not teams or not teams_recent:
        return teams

    recent_map = {t["team"]: t for t in teams_recent}
    nb_blend = 0
    for team in teams:
        recent = recent_map.get(team["team"])
        gp_fenetre = recent.get("games_played", 0) if recent else 0
        if not recent or gp_fenetre <= 0:
            continue
        w = min(1.0, gp_fenetre / NHL_TEAM_RECENT_GP_PLEIN) if NHL_TEAM_RECENT_GP_PLEIN > 0 else 1.0
        for key in ("xGF_per_game", "xGA_per_game", "xGF_PP", "xGA_PK", "fo_pct"):
            base_val = team.get(key, 0.0)
            recent_val = recent.get(key, base_val)
            team[key] = round(w * recent_val + (1.0 - w) * base_val, 3 if key != "fo_pct" else 4)
        if w < 1.0:
            nb_blend += 1

    if not _team_recent_logue:
        _team_recent_logue = True
        log_nhl(
            f"📈 Forme récente équipes actif — fenêtre {NHL_TEAM_RECENT_WINDOW} derniers matchs "
            f"(teams_{NHL_TEAM_RECENT_WINDOW}.csv), confiance pleine à {NHL_TEAM_RECENT_GP_PLEIN:.0f} GP "
            f"({nb_blend}/{len(teams)} équipes partiellement blendées)"
        )
    return teams


def get_team_stats(season=None, blend=True):
    if season is None:
        season = NHL_SEASON
    """Aspire les xG score/venue-adjusted (5v5 + PP/PK), blend N/N-1 en début de saison,
    puis blend de forme récente (teams_{N}.csv) par-dessus."""
    try:
        texte, saison_utilisee = _fetch_moneypuck_csv("teams", season)
        if not texte:
            return []
        if saison_utilisee != season:
            log_nhl(f"ℹ️ MoneyPuck équipes : repli sur la saison {saison_utilisee}")
        teams = _shrink_special_teams(_parser_team_stats_csv(texte))
        if not teams:
            return []

        if blend and NHL_BLEND_GP_PLEIN > 0:
            gp_values = [t["games_played"] for t in teams if t.get("games_played", 0) > 0]
            gp_moyen = sum(gp_values) / len(gp_values) if gp_values else NHL_BLEND_GP_PLEIN
            poids_n = min(1.0, gp_moyen / NHL_BLEND_GP_PLEIN)
            if poids_n < 1.0:
                texte_n1, _ = _fetch_moneypuck_csv("teams", season - 1)
                teams_n1 = _shrink_special_teams(_parser_team_stats_csv(texte_n1)) if texte_n1 else None
                if teams_n1:
                    log_nhl(
                        f"🔀 Blend MoneyPuck : {round(poids_n * 100)}% saison {season} / "
                        f"{round((1 - poids_n) * 100)}% {season - 1} (GP moyen {gp_moyen:.1f})"
                    )
                    teams = _blend_team_stats(teams, teams_n1, poids_n)

        if NHL_TEAM_RECENT_WINDOW > 0:
            kind_recent = f"teams_{NHL_TEAM_RECENT_WINDOW}"
            texte_recent, saison_recent = _fetch_moneypuck_csv(kind_recent, season)
            if texte_recent:
                if saison_recent != season:
                    log_nhl(f"ℹ️ MoneyPuck équipes forme récente : repli saison {saison_recent}")
                    texte_recent = None
            if texte_recent:
                teams_recent = _shrink_special_teams(_parser_team_stats_csv(texte_recent))
                teams = _blend_team_recent_form(teams, teams_recent)
            else:
                log_nhl("ℹ️ Forme récente équipes indisponible — saison seule", level="warning")

        return teams
    except Exception as e:
        log_nhl(f"⚠️ Erreur extracteur équipes: {e}", level="warning")
        return []


def _gsax_per_60_de_ligne(row):
    """GSAx/60 sur une ligne MoneyPuck (match ou cumul saison)."""
    icetime = float(row.get("icetime", 0) or 0)
    if icetime < 600:
        return None
    heures = icetime / 3600.0
    return (float(row.get("xGoals", 0) or 0) - float(row.get("goals", 0) or 0)) / heures


def _parser_goalie_season_csv(texte):
    goalies = []
    csv_reader = csv.DictReader(StringIO(texte))
    for row in csv_reader:
        if row.get("situation") != "all":
            continue
        if float(row.get("games_played", 0) or 0) < 5:
            continue
        gsax_per_60 = _gsax_per_60_de_ligne(row)
        if gsax_per_60 is None:
            continue
        goalies.append({
            "name": row["name"],
            "gsax_per_60": round(gsax_per_60, 3),
            "gsax_saison": round(gsax_per_60, 3),
        })
    return goalies


def _parser_goalie_recent_csv(texte):
    """
    Parse goalies_10.csv / goalies_20.csv (même schéma que goalies.csv,
    mais calculé par MoneyPuck sur les 10/20 derniers matchs).
    """
    recent = {}
    csv_reader = csv.DictReader(StringIO(texte))
    for row in csv_reader:
        if row.get("situation") != "all":
            continue
        gsax_per_60 = _gsax_per_60_de_ligne(row)
        if gsax_per_60 is None:
            continue
        gp_fenetre = float(row.get("games_played", 0) or 0)
        if gp_fenetre <= 0:
            continue
        recent[row["name"]] = {"gsax_recent": round(gsax_per_60, 3), "gp_fenetre": gp_fenetre}
    return recent


def _appliquer_blend_gsax_recent(goalies, recent_map):
    """Blend GSAx saison + forme récente (goalies_{N}.csv) : w = min(1, gp_fenetre / GP_plein)."""
    global _gsax_recent_logue
    if NHL_GSAX_RECENT_WINDOW <= 0 or not goalies or not recent_map:
        return goalies

    idx = {}
    for name, info in recent_map.items():
        idx[name] = info
        idx[normaliser_nom_joueur(name)] = info

    nb_blend = 0
    for gardien in goalies:
        info = idx.get(gardien["name"]) or idx.get(normaliser_nom_joueur(gardien["name"]))
        if not info:
            continue
        gsax_saison = gardien.get("gsax_saison", gardien["gsax_per_60"])
        w = min(1.0, info["gp_fenetre"] / NHL_GSAX_RECENT_GP_PLEIN) if NHL_GSAX_RECENT_GP_PLEIN > 0 else 1.0
        gardien["gsax_recent"] = info["gsax_recent"]
        gardien["gsax_per_60"] = round(w * info["gsax_recent"] + (1.0 - w) * gsax_saison, 3)
        if w < 1.0:
            nb_blend += 1

    if not _gsax_recent_logue:
        _gsax_recent_logue = True
        log_nhl(
            f"🥅 GSAx forme récente actif — fenêtre {NHL_GSAX_RECENT_WINDOW} derniers matchs "
            f"(goalies_{NHL_GSAX_RECENT_WINDOW}.csv), confiance pleine à {NHL_GSAX_RECENT_GP_PLEIN:.0f} GP "
            f"({nb_blend}/{len(goalies)} gardiens partiellement blendés)"
        )
    return goalies


def get_goalie_stats(season=None):
    if season is None:
        season = NHL_SEASON
    """Aspire GSAx gardiens (saison + blend forme récente goalies_{N}.csv si activé)."""
    try:
        texte, saison_utilisee = _fetch_moneypuck_csv("goalies", season)
        if not texte:
            return []
        if saison_utilisee != season:
            print(f"ℹ️ MoneyPuck gardiens : repli sur la saison {saison_utilisee}")
        goalies = _parser_goalie_season_csv(texte)
        if not goalies:
            return []

        if NHL_GSAX_RECENT_WINDOW > 0:
            kind_recent = f"goalies_{int(NHL_GSAX_RECENT_WINDOW)}"
            texte_recent, saison_recent = _fetch_moneypuck_csv(kind_recent, season)
            if texte_recent:
                if saison_recent != season:
                    log_nhl(f"ℹ️ MoneyPuck gardiens forme récente : repli saison {saison_recent}")
                recent_map = _parser_goalie_recent_csv(texte_recent)
                goalies = _appliquer_blend_gsax_recent(goalies, recent_map)
            else:
                log_nhl("ℹ️ GSAx forme récente indisponible — saison seule", level="warning")

        return goalies
    except Exception as e:
        print(f"⚠️ Erreur extracteur gardiens: {e}")
        return []


def get_stars_impact(season=None):
    if season is None:
        season = NHL_SEASON
    """Génère le Top 30 des patineurs les plus cruciaux de la ligue."""
    try:
        texte, saison_utilisee = _fetch_moneypuck_csv("skaters", season)
        if not texte:
            return {}
        if saison_utilisee != season:
            print(f"ℹ️ MoneyPuck patineurs : repli sur la saison {saison_utilisee}")
        csv_reader = csv.DictReader(StringIO(texte))
        skaters = []
        for row in csv_reader:
            if row.get("situation") == "5on5":
                gp = float(row.get("games_played", 0))
                if gp < 10:
                    continue
                skaters.append({
                    "name": row["name"], "team": row["team"],
                    "game_score_per_game": float(row.get("gameScore", 0)) / gp,
                })
        skaters_tries = sorted(skaters, key=lambda x: x["game_score_per_game"], reverse=True)[:30]
        stars_impact = {}
        for rank, player in enumerate(skaters_tries):
            if rank < 5:
                xgf_pen, xga_pen = 0.12, 0.05
            elif rank < 15:
                xgf_pen, xga_pen = 0.08, 0.03
            else:
                xgf_pen, xga_pen = 0.05, 0.02
            stars_impact[player["name"]] = {
                "team": player["team"], "xgf_penalty": xgf_pen, "xga_penalty": xga_pen,
            }
        return stars_impact
    except Exception as e:
        print(f"⚠️ Erreur extracteur stars: {e}")
        return {}

# ==========================================
# 2. RADAR API OFFICIEL NHL & MOMENTUM
# ==========================================
def _parse_utc(iso_str):
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _match_eligible_pour_scan(game):
    """True si le match n'a pas commencé et reste dans la fenêtre de scan."""
    etat = game.get("gameState", "")
    if etat in ETATS_MATCH_EXCLUS:
        return False

    now = datetime.now(timezone.utc)
    debut = _parse_utc(game.get("startTimeUTC"))

    if debut and now >= debut:
        return False
    if debut and (debut - now).total_seconds() > NHL_SCAN_HEURES_AVANCE * 3600:
        return False
    if debut and (debut - now).total_seconds() < NHL_MINUTES_AVANT_PUCK * 60:
        return False

    if etat in ETATS_MATCH_PRIORITAIRES:
        return True
    return debut is not None


def get_nhl_games_today():
    """Matchs du jour éligibles au scan (FUT/PRE + fenêtre horaire, hors LIVE/FINAL)."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    url = f"https://api-web.nhle.com/v1/score/{date_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        if "games" not in data:
            return []

        matchs = []
        for game in data["games"]:
            if not _match_eligible_pour_scan(game):
                continue
            matchs.append({
                "game_id": game["id"],
                "away_team": game["awayTeam"]["abbrev"],
                "home_team": game["homeTeam"]["abbrev"],
                "game_state": game.get("gameState", "?"),
                "start_utc": game.get("startTimeUTC"),
            })
        return matchs
    except Exception as e:
        print(f"⚠️ Erreur calendrier NHL : {e}")
        return []


def get_active_rosters(game_id):
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, None, [], []
        data = response.json()
        away_goalies = data.get("playerByGameStats", {}).get("awayTeam", {}).get("goalies", [])
        home_goalies = data.get("playerByGameStats", {}).get("homeTeam", {}).get("goalies", [])
        away_starter = away_goalies[0]["name"]["default"] if away_goalies else None
        home_starter = home_goalies[0]["name"]["default"] if home_goalies else None
        away_skaters, home_skaters = [], []
        for pos in ["forwards", "defense"]:
            for player in data.get("playerByGameStats", {}).get("awayTeam", {}).get(pos, []):
                away_skaters.append(player["name"]["default"])
            for player in data.get("playerByGameStats", {}).get("homeTeam", {}).get(pos, []):
                home_skaters.append(player["name"]["default"])
        return away_starter, home_starter, away_skaters, home_skaters
    except Exception:
        return None, None, [], []


def _extraire_nom_gardien(probable_entry):
    if not probable_entry:
        return None
    name = probable_entry.get("name")
    if isinstance(name, dict):
        return name.get("default")
    return name


def get_rosters_avec_fallback(game_id):
    """
    Boxscore NHL en priorité ; repli sur les gardiens probables (landing)
    si le boxscore n'est pas encore publié.
    Retourne (g_ext, g_dom, skaters_ext, skaters_dom, source).
    """
    g_ext, g_dom, sk_ext, sk_dom = get_active_rosters(game_id)
    if g_ext and g_dom:
        return g_ext, g_dom, sk_ext, sk_dom, "boxscore"

    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/landing"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return g_ext, g_dom, sk_ext, sk_dom, "indisponible"
        data = response.json()
        away_list = data.get("awayTeam", {}).get("probableGoalies", [])
        home_list = data.get("homeTeam", {}).get("probableGoalies", [])
        if not g_ext and away_list:
            g_ext = _extraire_nom_gardien(away_list[0])
        if not g_dom and home_list:
            g_dom = _extraire_nom_gardien(home_list[0])
        if g_ext and g_dom:
            return g_ext, g_dom, sk_ext, sk_dom, "landing_probable"
    except Exception:
        pass

    return g_ext, g_dom, sk_ext, sk_dom, "indisponible"

def get_goalie_confirmation_status(game_id):
    """
    Explore le Landing Page du match pour détecter si les gardiens
    sont validés ou simplement pressentis.
    Retourner un dictionnaire avec le statut de confirmation.
    """
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/landing"
    headers = {'User-Agent': 'Mozilla/5.0'}
    status = {"away_confirmed": False, "home_team_confirmed": False}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return status
        data = response.json()

        # On fouille l'aperçu du match (Match Preview / Summary)
        game_info = data.get("gameInfo", {})

        # L'API NHL utilise le drapeau 'isStarter' ou place le gardien dans
        # la section 'confirmedStarters' lorsqu'un journaliste officiel valide l'info.
        away_goalis_list = data.get("awayTeam", {}).get("probableGoalies", [])
        home_goalis_list = data.get("homeTeam", {}).get("probableGoalies", [])

        if away_goalis_list and away_goalis_list[0].get("status") == "CONFIRMED":
            status["away_confirmed"] = True
        if home_goalis_list and home_goalis_list[0].get("status") == "CONFIRMED":
            status["home_team_confirmed"] = True

        return status
    except:
        return status

def get_teams_played_yesterday():
    """Récupère la liste des abréviations des équipes qui ont joué hier."""
    hier = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    url = f"https://api-web.nhle.com/v1/score/{hier}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    teams_played = set()

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return teams_played
        data = response.json()
        if "games" not in data: return teams_played

        for game in data["games"]:
            teams_played.add(game["awayTeam"]["abbrev"])
            teams_played.add(game["homeTeam"]["abbrev"])
        return teams_played
    except:
        return teams_played


def _haversine_miles(lat1, lon1, lat2, lon2):
    """Distance à vol d'oiseau entre deux arènes (miles)."""
    r_miles = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return r_miles * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _distance_entre_arènes(venue_a, venue_b):
    """Miles entre deux abréviations d'équipe (arène domicile)."""
    if venue_a == venue_b:
        return 0.0
    coords_a = NHL_ARENA_COORDS.get(venue_a)
    coords_b = NHL_ARENA_COORDS.get(venue_b)
    if not coords_a or not coords_b:
        return 0.0
    return _haversine_miles(coords_a[0], coords_a[1], coords_b[0], coords_b[1])


def get_team_last_game_venues(days_back=None):
    """
    Lieu du dernier match terminé par équipe : abréviation de l'équipe à domicile
    ce jour-là (= ville/arène où le match s'est joué).
    """
    if days_back is None:
        days_back = NHL_TRAVEL_LOOKBACK_JOURS
    headers = {"User-Agent": "Mozilla/5.0"}
    etats_finaux = {"FINAL", "OFF", "OFFICIAL"}
    venues = {}

    for offset in range(1, days_back + 1):
        date_str = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
        try:
            response = requests.get(
                f"https://api-web.nhle.com/v1/score/{date_str}",
                headers=headers,
                timeout=10,
            )
            if response.status_code != 200:
                continue
            for game in response.json().get("games", []):
                if game.get("gameState") not in etats_finaux:
                    continue
                home = game["homeTeam"]["abbrev"]
                away = game["awayTeam"]["abbrev"]
                for team in (home, away):
                    if team not in venues:
                        venues[team] = home
        except Exception:
            continue
    return venues


def get_travel_miles_for_team(team_abbr, game_home_team, last_venues):
    """Miles parcourus depuis le dernier match jusqu'à l'arène du match du jour."""
    if not NHL_TRAVEL_FATIGUE_ACTIF or not last_venues:
        return 0.0
    last_venue = last_venues.get(team_abbr)
    if not last_venue:
        return 0.0
    return round(_distance_entre_arènes(last_venue, game_home_team), 1)


def _normaliser_nom_arbitre(nom):
    return str(nom).strip().lower()


def _extraire_nom_officiel(entry):
    if not entry:
        return None
    if isinstance(entry, dict):
        return entry.get("default")
    return str(entry)


def lire_referee_meta():
    default = {
        "referees": {},
        "league_penalties_pg": 10.0,
        "total_games": 0,
        "league_penalties_total": 0,
        "scanned_game_ids": [],
    }
    if not os.path.exists(REFEREE_META_FILE):
        return default
    try:
        with open(REFEREE_META_FILE, "r", encoding="utf-8") as f:
            return {**default, **json.load(f)}
    except Exception:
        return default


def get_game_referees(game_id):
    """Arbitres assignés au match (NHL right-rail → gameInfo.referees)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(
            f"https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail",
            headers=headers,
            timeout=10,
        )
        if response.status_code != 200:
            return []
        data = response.json()
        raw = data.get("gameInfo", {}).get("referees", [])
        return [n for n in (_extraire_nom_officiel(r) for r in raw) if n]
    except Exception:
        return []


def compter_penalites_match(game_id):
    """Nombre total de pénalités sifflées (landing summary)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(
            f"https://api-web.nhle.com/v1/gamecenter/{game_id}/landing",
            headers=headers,
            timeout=10,
        )
        if response.status_code != 200:
            return None
        blocs = response.json().get("summary", {}).get("penalties", [])
        total = 0
        for bloc in blocs:
            plist = bloc.get("penalties", [])
            if isinstance(plist, list):
                total += len(plist)
        return total
    except Exception:
        return None


def actualiser_stats_arbitres():
    """
    Scan incrémental des matchs terminés : pénalités/match par arbitre.
    Alimente referee_calibrage_meta.json pour l'ajustement PP/PK live.
    """
    if not NHL_REF_ADJ_ACTIF:
        return

    meta = lire_referee_meta()
    refs_db = meta.setdefault("referees", {})
    scanned = set(meta.get("scanned_game_ids", []))
    league_total_pens = int(meta.get("league_penalties_total", 0))
    league_total_games = int(meta.get("total_games", 0))
    etats_finaux = {"FINAL", "OFF", "OFFICIAL"}
    headers = {"User-Agent": "Mozilla/5.0"}

    lookback = NHL_REF_LOOKBACK_JOURS if league_total_games >= 50 else NHL_REF_LOOKBACK_JOURS
    scan_jours = NHL_REF_SCAN_JOURS if league_total_games >= 50 else lookback
    nouveaux = 0

    for offset in range(1, scan_jours + 1):
        date_str = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
        try:
            response = requests.get(
                f"https://api-web.nhle.com/v1/score/{date_str}",
                headers=headers,
                timeout=10,
            )
            if response.status_code != 200:
                continue
            for game in response.json().get("games", []):
                if game.get("gameState") not in etats_finaux:
                    continue
                gid = str(game["id"])
                if gid in scanned:
                    continue
                ref_names = get_game_referees(game["id"])
                if not ref_names:
                    continue
                pen_count = compter_penalites_match(game["id"])
                if pen_count is None:
                    continue
                for name in ref_names:
                    key = _normaliser_nom_arbitre(name)
                    if key not in refs_db:
                        refs_db[key] = {"name": name, "games": 0, "penalties": 0}
                    refs_db[key]["games"] += 1
                    refs_db[key]["penalties"] += pen_count
                league_total_pens += pen_count
                league_total_games += 1
                scanned.add(gid)
                nouveaux += 1
        except Exception:
            continue

    if nouveaux <= 0:
        return

    scanned_list = list(scanned)[-800:]
    league_ppg = league_total_pens / league_total_games if league_total_games else 10.0
    meta.update({
        "referees": refs_db,
        "total_games": league_total_games,
        "league_penalties_total": league_total_pens,
        "league_penalties_pg": round(league_ppg, 2),
        "scanned_game_ids": scanned_list,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    with open(REFEREE_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log_nhl(
        f"👨‍⚖️ Stats arbitres mises à jour — +{nouveaux} match(s), "
        f"ligue ≈{league_ppg:.1f} pén./match, {len(refs_db)} arbitres indexés"
    )


def compute_referee_pp_multiplier(referee_names):
    """
    Multiplicateur PP/PK selon la tendance pénalités du crew vs moyenne ligue.
    Retourne (mult, info_dict ou None).
    """
    global _ref_adj_logue, _ref_shrink_logue
    if not NHL_REF_ADJ_ACTIF or not referee_names:
        return 1.0, None

    meta = lire_referee_meta()
    league_ppg = float(meta.get("league_penalties_pg", 10.0))
    if league_ppg <= 0:
        return 1.0, None

    ppg_samples = []
    poids_samples = []
    known_refs = []
    for name in referee_names:
        key = _normaliser_nom_arbitre(name)
        ref_data = meta.get("referees", {}).get(key)
        if not ref_data or ref_data.get("games", 0) < 1:
            continue
        games = ref_data["games"]
        if NHL_REF_SHRINK_ACTIF:
            if games < NHL_REF_MIN_MATCHS:
                poids = min(1.0, games / NHL_REF_MIN_MATCHS)
            else:
                poids = 1.0
        else:
            if games < NHL_REF_MIN_MATCHS:
                continue
            poids = 1.0
        ppg_samples.append(ref_data["penalties"] / games)
        poids_samples.append(poids)
        known_refs.append(ref_data.get("name", name))

    if not ppg_samples:
        return 1.0, None

    if NHL_REF_SHRINK_ACTIF:
        poids_total = sum(poids_samples)
        if poids_total < 0.15:
            return 1.0, None
        crew_ppg = sum(p * w for p, w in zip(ppg_samples, poids_samples)) / poids_total
        if not _ref_shrink_logue:
            _ref_shrink_logue = True
            log_nhl(
                f"👨‍⚖️ Shrinkage arbitral progressif — confiance pleine à {NHL_REF_MIN_MATCHS} matchs/arbitre"
            )
    else:
        crew_ppg = sum(ppg_samples) / len(ppg_samples)
    rel = (crew_ppg - league_ppg) / league_ppg
    rel = max(min(rel, 0.25), -0.25)
    mult = round(max(min(1.0 + NHL_REF_SENSIBILITE * rel, 1.12), 0.88), 3)

    if not _ref_adj_logue:
        _ref_adj_logue = True
        log_nhl(
            f"👨‍⚖️ Ajustement arbitral actif — sensibilité {NHL_REF_SENSIBILITE:.2f}, "
            f"min {NHL_REF_MIN_MATCHS} matchs/arbitre, ligue ≈{league_ppg:.1f} pén./match"
        )

    return mult, {
        "crew_ppg": round(crew_ppg, 2),
        "league_ppg": round(league_ppg, 2),
        "refs": known_refs,
        "mult": mult,
    }

# ==========================================
# 3. UTILITAIRES D'IDENTIFICATION
# ==========================================
def formater_nom(nom_complet):
    parties = nom_complet.split(" ", 1)
    if len(parties) == 2:
        return f"{parties[0][0]}. {parties[1]}"
    return nom_complet


def normaliser_nom_joueur(nom):
    """Clé de comparaison MoneyPuck (nom complet) <-> NHL API (F. Nom)."""
    nom = str(nom).strip()
    parties = nom.split(" ", 1)
    if len(parties) == 2 and parties[0].endswith(".") and len(parties[0]) <= 3:
        return f"{parties[0][0].lower()}.{parties[1].lower()}"
    if len(parties) == 2:
        return f"{parties[0][0].lower()}.{parties[1].lower()}"
    return nom.lower()


def joueur_present(nom_reference, roster_nhl):
    cle = normaliser_nom_joueur(nom_reference)
    return any(normaliser_nom_joueur(n) == cle for n in roster_nhl)


def detecter_stars_absentes(team_abbr, roster_nhl, stars_vip):
    return [
        nom for nom, info in stars_vip.items()
        if info["team"] == team_abbr and not joueur_present(nom, roster_nhl)
    ]


def trouver_gsax(nom_nhl, goalies_data):
    for g in goalies_data:
        if formater_nom(g['name']) == nom_nhl or joueur_present(g['name'], [nom_nhl]):
            return g['gsax_per_60']
    return 0.0


def gsax_per_60_vers_lambda(gsax_per_60):
    """
    Convertit GSAx/60 MoneyPuck en ajustement lambda buts/match.
    GSAx > 0 = gardien performant → réduit les buts encaissés par l'équipe adverse.
    """
    return gsax_per_60 * (GSAX_MINUTES_PAR_MATCH / 60.0)

def apply_star_absence_penalty(team_name, base_xgf, base_xga, missing_players_list, stars_dict):
    adjusted_xgf, adjusted_xga = base_xgf, base_xga
    for player in missing_players_list:
        if player in stars_dict and stars_dict[player]["team"] == team_name:
            print(f"🚨 ABSENCE DE MARQUE : {player} est forfait pour {team_name} !")
            adjusted_xgf *= (1.0 - stars_dict[player]["xgf_penalty"])
            adjusted_xga *= (1.0 + stars_dict[player]["xga_penalty"])
    return round(adjusted_xgf, 3), round(adjusted_xga, 3)

def calculate_circadian_fatigue(team_abbrev, opponent_abbrev, is_b2b, is_home):
    """
    Modificateur fatigue fuseau horaire + B2B (sans distance).
    Retourne (modificateur_attaque, modificateur_defense).
    """
    mod_atk, mod_def = (0.95, 1.10) if is_b2b else (1.0, 1.0)

    if is_home:
        return mod_atk, mod_def

    tz_team = NHL_TIMEZONES.get(team_abbrev, 0)
    tz_opp = NHL_TIMEZONES.get(opponent_abbrev, 0)
    decalage = abs(tz_team - tz_opp)

    if decalage >= 2:
        if is_b2b:
            mod_atk *= 0.93
            mod_def *= 1.15
        else:
            mod_atk *= 0.98
            mod_def *= 1.04
    elif decalage == 1 and is_b2b:
        mod_atk *= 0.97
        mod_def *= 1.05

    return round(mod_atk, 3), round(mod_def, 3)


def calculate_schedule_fatigue(team_abbrev, opponent_abbrev, is_b2b, is_home, travel_miles=0.0):
    """
    Fatigue calendrier : B2B + fuseau horaire + miles parcourus depuis le dernier match.
    """
    global _travel_fatigue_logue
    mod_atk, mod_def = calculate_circadian_fatigue(team_abbrev, opponent_abbrev, is_b2b, is_home)

    if NHL_TRAVEL_FATIGUE_ACTIF and travel_miles > 0 and NHL_TRAVEL_MILES_REF > 0:
        scale = min(travel_miles / NHL_TRAVEL_MILES_REF, 2.0)
        if is_b2b:
            mod_atk *= max(1.0 - NHL_TRAVEL_B2B_ATK_PCT * scale, 0.88)
            mod_def *= min(1.0 + NHL_TRAVEL_B2B_DEF_PCT * scale, 1.20)
        elif travel_miles >= NHL_TRAVEL_LONG_MILES:
            mod_atk *= max(1.0 - NHL_TRAVEL_SOLO_ATK_PCT * scale, 0.95)
            mod_def *= min(1.0 + NHL_TRAVEL_SOLO_DEF_PCT * scale, 1.10)

    mod_atk = round(mod_atk, 3)
    mod_def = round(mod_def, 3)

    if not _travel_fatigue_logue and NHL_TRAVEL_FATIGUE_ACTIF:
        _travel_fatigue_logue = True
        log_nhl(
            f"✈️ Fatigue voyage actif — lookback {NHL_TRAVEL_LOOKBACK_JOURS}j, "
            f"ref {NHL_TRAVEL_MILES_REF:.0f} mi (B2B atk-{NHL_TRAVEL_B2B_ATK_PCT:.0%}/def+{NHL_TRAVEL_B2B_DEF_PCT:.0%} par ref)"
        )
    return mod_atk, mod_def


def _apply_faceoff_possession_adj(xgf, xga, fo_pct, league_fo_pct):
    """
    Ajuste légèrement xGF/xGA selon l'écart de % faceoffs vs moyenne ligue
    (proxy possession 5v5, complémentaire au xG MoneyPuck).
    """
    global _faceoff_adj_logue
    if not NHL_FACEOFF_ADJ_ACTIF or league_fo_pct <= 0:
        return xgf, xga

    delta = (fo_pct - league_fo_pct) / max(league_fo_pct, 0.01)
    delta = max(min(delta, 0.15), -0.15)
    sens = NHL_FACEOFF_SENSIBILITE
    atk_mult = 1.0 + sens * delta
    def_mult = 1.0 - sens * delta * 0.5

    if not _faceoff_adj_logue:
        _faceoff_adj_logue = True
        log_nhl(
            f"🏒 Ajustement faceoffs actif — sensibilité {sens:.2f} "
            f"(ligue FO≈{league_fo_pct:.1%})"
        )
    return round(xgf * atk_mult, 3), round(xga * def_mult, 3)

# ==========================================
# 4. MOTEUR MATHÉMATIQUE & INTELLIGENCE
# ==========================================
def poisson(lam, k):
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def tau_dixon_coles(lam_home, lam_away, h, a, rho=-0.12):
    if h == 0 and a == 0: return max(0, 1 - (lam_home * lam_away * rho))
    elif h == 1 and a == 0: return max(0, 1 + (lam_home * rho))
    elif h == 0 and a == 1: return max(0, 1 + (lam_away * rho))
    elif h == 1 and a == 1: return max(0, 1 - rho)
    return 1.0

def _ajuster_lambdas_pour_hia(lam_h, lam_a, hia, hia_ref=HIA_REF_CALIBRATION):
    """
    Approximation : les lambdas journalisés intègrent (1+hia_ref) attaque / (1-hia_ref) défense domicile.
    Recalibre vers un nouveau HIA sans recalculer tout le pipeline xG.
    """
    if abs(hia - hia_ref) < 1e-6:
        return lam_h, lam_a
    lam_h_adj = lam_h * (1.0 + hia) / (1.0 + hia_ref)
    lam_a_adj = lam_a * (1.0 - hia) / (1.0 - hia_ref)
    return max(lam_h_adj, 0.1), max(lam_a_adj, 0.1)


def _parse_date_match(date_str):
    """Parse une date de match (YYYY-MM-DD ou YYYY-MM-DD HH:MM) → date ou None."""
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _poids_recency_mle(match):
    """
    Poids exponentiel selon l'ancienneté du match : w = 0.5^(age_jours / demi_vie).
    Match d'hier ≈ 1.0 ; à 28 j (demi-vie par défaut) ≈ 0.5 ; à 56 j ≈ 0.25.
    """
    if not NHL_MLE_RECENCY_ACTIF or NHL_MLE_RECENCY_HALFLIFE_JOURS <= 0:
        return 1.0
    match_date = _parse_date_match(match.get("date"))
    if match_date is None:
        return 1.0
    age_jours = max((datetime.now().date() - match_date).days, 0)
    return 0.5 ** (age_jours / NHL_MLE_RECENCY_HALFLIFE_JOURS)


def log_likelihood_rho_hia(params, historique_matchs):
    rho, hia = params[0], params[1]
    ll = 0.0
    for match in historique_matchs:
        h_goals, a_goals = match["vrai_score_domicile"], match["vrai_score_exterieur"]
        lam_h = match["lambda_domicile_calcule"]
        lam_a = match["lambda_exterieur_calcule"]
        hia_ref = match.get("hia_ref", HIA_REF_CALIBRATION)
        lam_h, lam_a = _ajuster_lambdas_pour_hia(lam_h, lam_a, hia, hia_ref)
        p_h = (math.exp(-lam_h) * (lam_h ** h_goals)) / math.factorial(h_goals)
        p_a = (math.exp(-lam_a) * (lam_a ** a_goals)) / math.factorial(a_goals)
        tau = tau_dixon_coles(lam_h, lam_a, h_goals, a_goals, rho)
        ll += _poids_recency_mle(match) * math.log(max(p_h * p_a * tau, 1e-10))
    return -ll


def optimiser_rho_et_hia_saison(historique_matchs):
    """Calibre rho et HIA conjointement par MLE sur l'historique du journal."""
    meta = lire_rho_meta()
    x0 = [float(meta.get("rho", -0.12)), float(meta.get("hia", NHL_HIA_DEFAULT))]
    recency_label = (
        f", recency demi-vie {NHL_MLE_RECENCY_HALFLIFE_JOURS:.0f}j"
        if NHL_MLE_RECENCY_ACTIF and NHL_MLE_RECENCY_HALFLIFE_JOURS > 0
        else ""
    )
    log_nhl(f"🔬 Calibrage MLE rho+HIA sur {len(historique_matchs)} matchs (init {x0}{recency_label})...")
    resultat = minimize(
        log_likelihood_rho_hia,
        x0,
        args=(historique_matchs,),
        bounds=[(-0.30, 0.05), (0.0, 0.10)],
        method="L-BFGS-B",
    )
    rho_opt = round(max(-0.30, min(0.05, resultat.x[0])), 4)
    hia_opt = round(max(0.0, min(0.10, resultat.x[1])), 4)
    log_nhl(f"✅ Calibrage terminé — rho={rho_opt}, HIA={hia_opt:.1%} (attaque dom +{hia_opt:.1%} / défense dom -{hia_opt:.1%})")
    return rho_opt, hia_opt


def optimiser_rho_saison(historique_matchs):
    """Rétrocompatibilité : retourne rho seul."""
    rho, _ = optimiser_rho_et_hia_saison(historique_matchs)
    return rho


def _prob_score_avec_correction(lam_h, lam_a, h_obs, a_obs, rho, prob_tie, prob_en):
    """
    Probabilité fermée d'observer le score (h_obs, a_obs) sous Dixon-Coles +
    correction empty-net/OT-tie (équivalent cellule-par-cellule de la matrice
    construite dans calculate_master_odds_v4, sans reconstruire toute la matrice).
    """
    def p_brute(h, a):
        if h < 0 or a < 0:
            return 0.0
        return poisson(lam_h, h) * poisson(lam_a, a) * tau_dixon_coles(lam_h, lam_a, h, a, rho)

    margin = h_obs - a_obs
    p = p_brute(h_obs, a_obs)

    if margin == 1 or margin == -1:
        return p * (1.0 - prob_tie - prob_en)
    if margin == 0:
        return p + p_brute(h_obs, a_obs - 1) * prob_tie + p_brute(h_obs - 1, a_obs) * prob_tie
    if margin == 2:
        return p + p_brute(h_obs - 1, a_obs) * prob_en
    if margin == -2:
        return p + p_brute(h_obs, a_obs - 1) * prob_en
    return p


def log_likelihood_empty_net(params, historique_matchs, rho, hia):
    prob_tie, prob_en = params[0], params[1]
    if prob_tie < 0 or prob_en < 0 or (prob_tie + prob_en) >= 1.0:
        return 1e10
    ll = 0.0
    for match in historique_matchs:
        h_obs, a_obs = match["vrai_score_domicile"], match["vrai_score_exterieur"]
        lam_h = match["lambda_domicile_calcule"]
        lam_a = match["lambda_exterieur_calcule"]
        hia_ref = match.get("hia_ref", HIA_REF_CALIBRATION)
        lam_h, lam_a = _ajuster_lambdas_pour_hia(lam_h, lam_a, hia, hia_ref)
        p = _prob_score_avec_correction(lam_h, lam_a, h_obs, a_obs, rho, prob_tie, prob_en)
        ll += _poids_recency_mle(match) * math.log(max(p, 1e-10))
    return -ll


def optimiser_empty_net_ot(historique_matchs, rho, hia):
    """Calibre prob_tie (retour à égalité) et prob_en (but cage vide) par MLE, rho/HIA fixés."""
    meta = lire_rho_meta()
    x0 = [float(meta.get("prob_tie", 0.12)), float(meta.get("prob_en", 0.22))]
    recency_label = (
        f", recency demi-vie {NHL_MLE_RECENCY_HALFLIFE_JOURS:.0f}j"
        if NHL_MLE_RECENCY_ACTIF and NHL_MLE_RECENCY_HALFLIFE_JOURS > 0
        else ""
    )
    log_nhl(f"🔬 Calibrage MLE empty-net/OT-tie sur {len(historique_matchs)} matchs (init {x0}{recency_label})...")
    resultat = minimize(
        log_likelihood_empty_net,
        x0,
        args=(historique_matchs, rho, hia),
        bounds=[(0.02, 0.30), (0.02, 0.40)],
        method="L-BFGS-B",
    )
    prob_tie_opt = round(max(0.02, min(0.30, resultat.x[0])), 4)
    prob_en_opt = round(max(0.02, min(0.40, resultat.x[1])), 4)
    log_nhl(f"✅ Calibrage empty-net terminé — prob_tie={prob_tie_opt:.1%}, prob_en={prob_en_opt:.1%}")
    return prob_tie_opt, prob_en_opt

def calculate_master_odds_v4(
    teams_data, home_team, away_team, home_gsax, away_gsax,
    home_is_b2b=False, away_is_b2b=False,
    home_travel_miles=0.0, away_travel_miles=0.0,
    referee_pp_mult=1.0,
    rho=-0.12, hia=None, prob_tie=None, prob_en=None,
):
    if hia is None:
        hia = lire_hia_equipe(home_team)
    hia = min(max(float(hia), 0.0), 0.12)
    if prob_tie is None:
        prob_tie = lire_prob_tie_dynamique()
    if prob_en is None:
        prob_en = lire_prob_en_dynamique()
    prob_tie = min(max(float(prob_tie), 0.0), 0.30)
    prob_en = min(max(float(prob_en), 0.0), 0.40)
    league_avg_5v5 = sum(t['xGF_per_game'] for t in teams_data) / len(teams_data)
    league_avg_pp = sum(t['xGF_PP'] for t in teams_data) / len(teams_data)
    safe_league_pp = max(league_avg_pp, 0.01)
    fo_vals = [t.get("fo_pct", 0.5) for t in teams_data if t.get("fo_pct", 0) > 0]
    league_fo_pct = sum(fo_vals) / len(fo_vals) if fo_vals else 0.5

    home = next((t for t in teams_data if t['team'] == home_team), None)
    away = next((t for t in teams_data if t['team'] == away_team), None)
    if not home or not away: return None

    # HIA calibré (le xG utilisé intègre déjà la forme récente via get_team_stats)
    home_xgf = home['xGF_per_game'] * (1.0 + hia)
    home_xga = home['xGA_per_game'] * (1.0 - hia)
    away_xgf = away['xGF_per_game']
    away_xga = away['xGA_per_game']

    home_xgf, home_xga = _apply_faceoff_possession_adj(
        home_xgf, home_xga, home.get("fo_pct", league_fo_pct), league_fo_pct,
    )
    away_xgf, away_xga = _apply_faceoff_possession_adj(
        away_xgf, away_xga, away.get("fo_pct", league_fo_pct), league_fo_pct,
    )

    home_pp_att = home['xGF_PP']
    home_pk_def = home['xGA_PK']
    away_pp_att = away['xGF_PP']
    away_pk_def = away['xGA_PK']

    # Fatigue calendrier : B2B + fuseau + miles parcourus
    home_fatigue_atk, home_fatigue_def = calculate_schedule_fatigue(
        home_team, away_team, home_is_b2b, is_home=True, travel_miles=home_travel_miles,
    )
    away_fatigue_atk, away_fatigue_def = calculate_schedule_fatigue(
        away_team, home_team, away_is_b2b, is_home=False, travel_miles=away_travel_miles,
    )

    home_xgf *= home_fatigue_atk
    home_xga *= home_fatigue_def
    home_pp_att *= home_fatigue_atk
    home_pk_def *= home_fatigue_def

    away_xgf *= away_fatigue_atk
    away_xga *= away_fatigue_def
    away_pp_att *= away_fatigue_atk
    away_pk_def *= away_fatigue_def

    # Crew arbitral : plus de sifflets → plus d'occasions PP des deux côtés
    if referee_pp_mult != 1.0:
        home_pp_att = round(home_pp_att * referee_pp_mult, 3)
        away_pp_att = round(away_pp_att * referee_pp_mult, 3)

    lam_home_5v5 = (home_xgf / league_avg_5v5) * (away_xga / league_avg_5v5) * league_avg_5v5
    lam_away_5v5 = (away_xgf / league_avg_5v5) * (home_xga / league_avg_5v5) * league_avg_5v5
    lam_home_pp = (home_pp_att / safe_league_pp) * (away_pk_def / safe_league_pp) * safe_league_pp
    lam_away_pp = (away_pp_att / safe_league_pp) * (home_pk_def / safe_league_pp) * safe_league_pp

    adj_gsax_away = gsax_per_60_vers_lambda(away_gsax)
    adj_gsax_home = gsax_per_60_vers_lambda(home_gsax)
    final_lam_home = max(lam_home_5v5 + lam_home_pp - adj_gsax_away, 0.1)
    final_lam_away = max(lam_away_5v5 + lam_away_pp - adj_gsax_home, 0.1)

    # Matrice initiale Poisson + Dixon-Coles
    matrice_brute = [[0.0]*12 for _ in range(12)]
    for h in range(12):
        for a in range(12):
            matrice_brute[h][a] = poisson(final_lam_home, h) * poisson(final_lam_away, a) * tau_dixon_coles(final_lam_home, final_lam_away, h, a, rho)

    # Correction Surdispersion (Empty Net / retour à égalité) — prob_tie/prob_en calibrés par MLE
    matrice_finale = [[0.0]*12 for _ in range(12)]

    for h in range(12):
        for a in range(12):
            p = matrice_brute[h][a]
            if p == 0: continue
            if h - a == 1 and h < 11 and a < 11:
                matrice_finale[h][a] += p - (p*prob_tie + p*prob_en)
                matrice_finale[h][a+1] += p*prob_tie
                matrice_finale[h+1][a] += p*prob_en
            elif a - h == 1 and h < 11 and a < 11:
                matrice_finale[h][a] += p - (p*prob_tie + p*prob_en)
                matrice_finale[h+1][a] += p*prob_tie
                matrice_finale[h][a+1] += p*prob_en
            else:
                matrice_finale[h][a] += p

    # Renormalisation : garantit des probas cohérentes (PL, ML, O/U)
    masse_totale = sum(matrice_finale[h][a] for h in range(12) for a in range(12))
    if masse_totale > 0:
        for h in range(12):
            for a in range(12):
                matrice_finale[h][a] /= masse_totale

    # Lecture des probabilités
    prob_1, prob_X, prob_2 = 0, 0, 0
    prob_pl_home, prob_pl_away = 0, 0
    cuts_cibles = [4.5, 5.5, 6.5, 7.5]
    prob_over_cuts = {cut: 0.0 for cut in cuts_cibles}
    prob_under_cuts = {cut: 0.0 for cut in cuts_cibles}

    for h in range(12):
        for a in range(12):
            p_final = matrice_finale[h][a]
            if h > a: prob_1 += p_final
            elif a > h: prob_2 += p_final
            else: prob_X += p_final

            if (h - a) >= 2: prob_pl_home += p_final
            if (a - h) >= 2: prob_pl_away += p_final

            total_buts = h + a
            for cut in cuts_cibles:
                if total_buts > cut: prob_over_cuts[cut] += p_final
                else: prob_under_cuts[cut] += p_final

    s_1x2 = prob_1 + prob_X + prob_2
    if s_1x2 <= 0:
        return None

    # ML Pinnacle = 2-way incluant prolongation/tir au but
    ot_home = min(max(NHL_OT_HOME_ADVANTAGE, 0.45), 0.55)
    prob_ml_home = prob_1 + prob_X * ot_home
    prob_ml_away = prob_2 + prob_X * (1 - ot_home)

    prob_ml_home = min(max(prob_ml_home, 0.001), 0.999)
    prob_ml_away = min(max(prob_ml_away, 0.001), 0.999)
    prob_pl_home = min(max(prob_pl_home, 0.001), 0.999)
    prob_pl_away = min(max(prob_pl_away, 0.001), 0.999)

    for cut in cuts_cibles:
        s_ou = prob_over_cuts[cut] + prob_under_cuts[cut]
        if s_ou > 0:
            prob_over_cuts[cut] = min(max(prob_over_cuts[cut] / s_ou, 0.001), 0.999)
            prob_under_cuts[cut] = min(max(prob_under_cuts[cut] / s_ou, 0.001), 0.999)

    return {
        'prob_1': prob_ml_home, 'prob_2': prob_ml_away,
        'prob_pl_home': prob_pl_home, 'prob_pl_away': prob_pl_away,
        'prob_over_cuts': prob_over_cuts, 'prob_under_cuts': prob_under_cuts,
        'cote_1': round(1 / prob_ml_home, 2), 'cote_2': round(1 / prob_ml_away, 2),
        'cote_pl_home': round(1 / prob_pl_home, 2), 'cote_pl_away': round(1 / prob_pl_away, 2),
        'lam_home': round(final_lam_home, 3), 'lam_away': round(final_lam_away, 3),
    }


def _proba_no_vig_2way(cote_a, cote_b):
    """Probabilités 'fair' sans marge bookmaker (dévigorage multiplicatif simple)."""
    try:
        cote_a, cote_b = float(cote_a), float(cote_b)
    except (TypeError, ValueError):
        return None, None
    if cote_a <= 1.0 or cote_b <= 1.0:
        return None, None
    inv_a, inv_b = 1.0 / cote_a, 1.0 / cote_b
    total = inv_a + inv_b
    if total <= 0:
        return None, None
    return inv_a / total, inv_b / total


def _poids_confiance_modele(gp_moyen):
    """
    Poids accordé au modèle (vs marché no-vig) : croît avec la maturité de
    l'échantillon courant (GP moyen des deux équipes), plafonné à GP_PLEIN.
    """
    if NHL_MODEL_TRUST_GP_PLEIN <= 0:
        return NHL_MODEL_TRUST_MAX
    part_gp = min(1.0, max(gp_moyen, 0.0) / NHL_MODEL_TRUST_GP_PLEIN)
    return NHL_MODEL_TRUST_MIN + (NHL_MODEL_TRUST_MAX - NHL_MODEL_TRUST_MIN) * part_gp


def _blend_proba_marche(prob_modele_a, prob_modele_b, cote_marche_a, cote_marche_b, w_modele):
    """Blend (prob_a, prob_b) modèle avec le no-vig marché, renormalisé à 1."""
    no_vig_a, no_vig_b = _proba_no_vig_2way(cote_marche_a, cote_marche_b)
    if no_vig_a is None:
        return prob_modele_a, prob_modele_b
    blend_a = w_modele * prob_modele_a + (1.0 - w_modele) * no_vig_a
    blend_b = w_modele * prob_modele_b + (1.0 - w_modele) * no_vig_b
    total = blend_a + blend_b
    if total <= 0:
        return prob_modele_a, prob_modele_b
    return blend_a / total, blend_b / total


def _appliquer_coherence_marches(cotes):
    """
    Garde-fous après shrinkage marché indépendant par marché :
    - Puck line ≤ moneyline (gagner par 2+ ne peut pas être plus probable que gagner)
    - Over décroissant / Under croissant entre les cuts (4.5 → 7.5)
    - Renormalisation Over+Under = 1 par cut
    """
    global _marche_coherence_logue
    if not NHL_MARCHE_COHERENCE_ACTIF or not cotes:
        return cotes, []

    resultat = dict(cotes)
    corrections = []

    if "prob_1" in resultat and "prob_pl_home" in resultat:
        if resultat["prob_pl_home"] > resultat["prob_1"] + FLOAT_TOL:
            resultat["prob_pl_home"] = resultat["prob_1"]
            resultat["cote_pl_home"] = round(1 / max(resultat["prob_pl_home"], 0.001), 2)
            corrections.append("PL dom ≤ ML")
    if "prob_2" in resultat and "prob_pl_away" in resultat:
        if resultat["prob_pl_away"] > resultat["prob_2"] + FLOAT_TOL:
            resultat["prob_pl_away"] = resultat["prob_2"]
            resultat["cote_pl_away"] = round(1 / max(resultat["prob_pl_away"], 0.001), 2)
            corrections.append("PL vis ≤ ML")

    prob_over = dict(resultat.get("prob_over_cuts", {}))
    prob_under = dict(resultat.get("prob_under_cuts", {}))
    if prob_over and prob_under:
        cuts = sorted(prob_over.keys(), key=lambda k: float(k))
        ou_corrige = False
        for i in range(1, len(cuts)):
            c_prev, c_cur = cuts[i - 1], cuts[i]
            if prob_over[c_cur] > prob_over[c_prev] + FLOAT_TOL:
                prob_over[c_cur] = prob_over[c_prev]
                ou_corrige = True
            if prob_under[c_cur] < prob_under[c_prev] - FLOAT_TOL:
                prob_under[c_cur] = prob_under[c_prev]
                ou_corrige = True
        for cut in cuts:
            if cut not in prob_under:
                continue
            total_ou = prob_over[cut] + prob_under[cut]
            if total_ou > 0:
                prob_over[cut] = prob_over[cut] / total_ou
                prob_under[cut] = prob_under[cut] / total_ou
        if ou_corrige:
            corrections.append("O/U monotones")
        resultat["prob_over_cuts"] = prob_over
        resultat["prob_under_cuts"] = prob_under

    if corrections and not _marche_coherence_logue:
        _marche_coherence_logue = True
        log_nhl("🔗 Cohérence inter-marchés active — PL ≤ ML, totaux O/U monotones")

    return resultat, corrections


def shrink_cotes_vers_marche(cotes_vraies, cotes_bookmaker, cotes_puckline, gp_moyen_match):
    """
    Réduit la sur-confiance du modèle en le mélangeant avec la probabilité
    no-vig du marché (Pinnacle), pondéré par la maturité de l'échantillon (GP).
    Les lambdas (lam_home/lam_away) restent purs modèle pour la calibration MLE.
    """
    if not NHL_MARCHE_SHRINK_ACTIF or not cotes_vraies:
        return _appliquer_coherence_marches(cotes_vraies)
    w = _poids_confiance_modele(gp_moyen_match)
    resultat = dict(cotes_vraies)

    if cotes_bookmaker and "cote_1" in cotes_bookmaker and "cote_2" in cotes_bookmaker:
        p1, p2 = _blend_proba_marche(
            cotes_vraies["prob_1"], cotes_vraies["prob_2"],
            cotes_bookmaker["cote_1"], cotes_bookmaker["cote_2"], w,
        )
        resultat["prob_1"], resultat["prob_2"] = p1, p2
        resultat["cote_1"] = round(1 / max(p1, 0.001), 2)
        resultat["cote_2"] = round(1 / max(p2, 0.001), 2)

    if cotes_puckline and "cote_pl_home" in cotes_puckline and "cote_pl_away" in cotes_puckline:
        p_home, p_away = _blend_proba_marche(
            cotes_vraies["prob_pl_home"], cotes_vraies["prob_pl_away"],
            cotes_puckline["cote_pl_home"], cotes_puckline["cote_pl_away"], w,
        )
        resultat["prob_pl_home"], resultat["prob_pl_away"] = p_home, p_away
        resultat["cote_pl_home"] = round(1 / max(p_home, 0.001), 2)
        resultat["cote_pl_away"] = round(1 / max(p_away, 0.001), 2)

    if cotes_bookmaker and "totals" in cotes_bookmaker:
        prob_over = dict(cotes_vraies.get("prob_over_cuts", {}))
        prob_under = dict(cotes_vraies.get("prob_under_cuts", {}))
        for cut, prix in cotes_bookmaker["totals"].items():
            if "Over" not in prix or "Under" not in prix:
                continue
            cut_arrondi = _arrondir_cut(cut)
            cle_over, p_over_modele = _trouver_cle_float(prob_over, cut_arrondi)
            cle_under, p_under_modele = _trouver_cle_float(prob_under, cut_arrondi)
            if p_over_modele is None or p_under_modele is None:
                continue
            p_over, p_under = _blend_proba_marche(
                p_over_modele, p_under_modele, prix["Over"], prix["Under"], w,
            )
            prob_over[cle_over] = p_over
            prob_under[cle_under] = p_under
        resultat["prob_over_cuts"] = prob_over
        resultat["prob_under_cuts"] = prob_under

    return _appliquer_coherence_marches(resultat)

# ==========================================
# 5. INTEGRATION THE-ODDS-API & FINANCIAL
# ==========================================
def _noms_odds_pour_equipe(abbrev):
    """Liste des noms possibles pour une équipe (primaire + alias Pinnacle)."""
    primaire = NHL_TEAMS_MAPPING.get(abbrev)
    if not primaire:
        return []
    return [primaire] + ODDS_API_ALIASES.get(primaire, [])


def _nom_odds_vers_primaire(nom_api):
    """Mappe un nom Odds API vers notre nom primaire interne."""
    if nom_api in NHL_TEAMS_MAPPING.values():
        return nom_api
    return _ODDS_NOM_PRIMAIRE.get(nom_api, nom_api)


def _noms_equipe_equivalents(nom):
    """Ensemble de noms API interchangeables pour une même franchise."""
    if not nom:
        return set()
    primaire = _nom_odds_vers_primaire(nom)
    return {nom, primaire} | set(ODDS_API_ALIASES.get(primaire, []))


def _outcome_est_equipe(outcome_name, team_name):
    """True si le libellé Pinnacle correspond à l'équipe (alias inclus)."""
    if outcome_name == team_name:
        return True
    equiv_outcome = _noms_equipe_equivalents(outcome_name)
    equiv_team = _noms_equipe_equivalents(team_name)
    return bool(equiv_outcome & equiv_team)


def _est_puck_line_moins_15(point):
    return _float_proche(point, -1.5)


def _parse_pinnacle_game(game):
    """Extrait h2h, totals et puck line (-1.5) d'un match Pinnacle."""
    if not game.get("bookmakers"):
        return None
    home_full = game["home_team"]
    away_full = game["away_team"]
    bookmaker = game["bookmakers"][0]
    cotes = {"home_full": home_full, "away_full": away_full}
    for market in bookmaker["markets"]:
        if market["key"] == "h2h":
            for o in market["outcomes"]:
                if _outcome_est_equipe(o["name"], home_full):
                    cotes["cote_1"] = o["price"]
                elif _outcome_est_equipe(o["name"], away_full):
                    cotes["cote_2"] = o["price"]
        elif market["key"] == "totals":
            if "totals" not in cotes:
                cotes["totals"] = {}
            for o in market["outcomes"]:
                point = _arrondir_cut(o["point"])
                side = o["name"].capitalize()
                if side not in ("Over", "Under"):
                    continue
                if point not in cotes["totals"]:
                    cotes["totals"][point] = {}
                cotes["totals"][point][side] = o["price"]
        elif market["key"] == "spreads":
            for o in market["outcomes"]:
                if _est_puck_line_moins_15(o["point"]):
                    if _outcome_est_equipe(o["name"], home_full):
                        cotes["cote_pl_home"] = o["price"]
                    elif _outcome_est_equipe(o["name"], away_full):
                        cotes["cote_pl_away"] = o["price"]
    if "cote_1" not in cotes or "cote_2" not in cotes:
        return None
    return cotes


def _indexer_cotes_cache(cache, parsed):
    """Indexe un match sous toutes les combinaisons de noms connus."""
    home_api = parsed["home_full"]
    away_api = parsed["away_full"]
    home_primaire = _nom_odds_vers_primaire(home_api)
    away_primaire = _nom_odds_vers_primaire(away_api)
    cles_home = {home_api, home_primaire} | set(ODDS_API_ALIASES.get(home_primaire, []))
    cles_away = {away_api, away_primaire} | set(ODDS_API_ALIASES.get(away_primaire, []))
    for h in cles_home:
        for a in cles_away:
            cache[(h, a)] = parsed


def fetch_all_pinnacle_odds():
    """Une seule requête Odds API pour tous les matchs NHL (quota économisé)."""
    if not ODDS_API_KEY:
        return {}
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals,spreads",
        "bookmakers": "pinnacle",
        "oddsFormat": "decimal",
    }
    cache = {}
    try:
        response = requests.get(url, params=params, timeout=15)
        _traiter_quota_odds_api(response)
        if response.status_code != 200:
            log_nhl(f"⚠️ Odds API : HTTP {response.status_code}", level="warning")
            return {}
        for game in response.json():
            parsed = _parse_pinnacle_game(game)
            if parsed:
                _indexer_cotes_cache(cache, parsed)
        log_nhl(f"📡 Odds API : {len(cache)} clé(s) de match indexées")
        return cache
    except Exception as e:
        log_nhl(f"⚠️ Erreur Odds API globale : {e}", level="warning")
        return {}


def _traiter_quota_odds_api(response):
    """Log le quota mensuel et alerte Telegram si seuil bas."""
    restant_raw = response.headers.get("x-requests-remaining")
    utilise_raw = response.headers.get("x-requests-used")
    if restant_raw is None:
        return
    try:
        restant = int(restant_raw)
        utilise = int(utilise_raw) if utilise_raw else "?"
    except ValueError:
        return
    log_nhl(f"📊 Odds API quota : {restant} restantes ({utilise} utilisées ce mois)")
    if restant > NHL_ODDS_QUOTA_ALERT:
        _odds_quota_state["derniere_alerte"] = None
        return
    now = datetime.now()
    last = _odds_quota_state.get("derniere_alerte")
    if last and (now - last).total_seconds() < 6 * 3600:
        return
    _odds_quota_state["derniere_alerte"] = now
    envoyer_alerte_systeme(
        f"⚠️ **QUOTA ODDS API BAS**\n\n"
        f"Il reste **{restant}** requêtes ce mois (seuil {NHL_ODDS_QUOTA_ALERT}).\n"
        f"Envisager de réduire la fréquence CLV ou d'upgrader le plan Odds API."
    )


def _lire_odds_history():
    default = {"games": {}}
    if not os.path.exists(ODDS_HISTORY_FILE):
        return default
    try:
        with open(ODDS_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {**default, **data}
    except Exception:
        return default


def _sauver_odds_history(data):
    with open(ODDS_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _snapshot_from_cotes(cotes_book, cotes_pl):
    """Extrait les cotes Pinnacle clés pour historique line movement."""
    snap = {"ts": datetime.now(timezone.utc).isoformat()}
    if cotes_book:
        for key_src, key_dst in (("cote_1", "ml_home"), ("cote_2", "ml_away")):
            if key_src in cotes_book:
                snap[key_dst] = round(float(cotes_book[key_src]), 3)
        if "totals" in cotes_book:
            snap["totals"] = {}
            for cut, sides in cotes_book["totals"].items():
                cut_key = str(_arrondir_cut(cut))
                snap["totals"][cut_key] = {
                    side: round(float(price), 3) for side, price in sides.items()
                }
    if cotes_pl:
        for key_src, key_dst in (("cote_pl_home", "pl_home"), ("cote_pl_away", "pl_away")):
            if key_src in cotes_pl:
                snap[key_dst] = round(float(cotes_pl[key_src]), 3)
    return snap


def _snapshots_equivalents(a, b):
    """True si les cotes clés du snapshot n'ont pas bougé (évite doublons)."""
    for key in ("ml_home", "ml_away", "pl_home", "pl_away"):
        if key in a or key in b:
            if not _float_proche(a.get(key, 0), b.get(key, 0), tol=0.005):
                return False
    return True


def enregistrer_snapshots_cotes(matchs, odds_cache):
    """Historise les cotes Pinnacle par match (1 entrée/cycle si mouvement)."""
    if not NHL_LINE_MOVE_ACTIF or not matchs or not odds_cache:
        return

    data = _lire_odds_history()
    games = data.setdefault("games", {})
    now = datetime.now(timezone.utc)
    actifs = set()

    for m in matchs:
        gid = str(m["game_id"])
        actifs.add(gid)
        cotes = get_odds_for_match(m["home_team"], m["away_team"], odds_cache)
        if not cotes:
            continue
        cotes_pl = get_real_live_odds_puckline(m["home_team"], m["away_team"], odds_cache)
        snap = _snapshot_from_cotes(cotes, cotes_pl)

        entry = games.setdefault(gid, {
            "home": m["home_team"],
            "away": m["away_team"],
            "snapshots": [],
        })
        entry["home"] = m["home_team"]
        entry["away"] = m["away_team"]
        snaps = entry.setdefault("snapshots", [])
        if snaps and _snapshots_equivalents(snaps[-1], snap):
            continue
        snaps.append(snap)
        if NHL_LINE_MAX_SNAPSHOTS > 0 and len(snaps) > NHL_LINE_MAX_SNAPSHOTS:
            entry["snapshots"] = snaps[-NHL_LINE_MAX_SNAPSHOTS:]

    # Purge matchs terminés / hors fenêtre (> 36 h sans mise à jour)
    for gid in list(games.keys()):
        if gid in actifs:
            continue
        snaps = games[gid].get("snapshots", [])
        if not snaps:
            del games[gid]
            continue
        try:
            last_ts = datetime.fromisoformat(snaps[-1]["ts"].replace("Z", "+00:00"))
            if (now - last_ts).total_seconds() > 36 * 3600:
                del games[gid]
        except Exception:
            del games[gid]

    _sauver_odds_history(data)


def _cote_reference_pour_candidat(snapshot, candidat, m):
    """Retourne la cote historique correspondant au candidat."""
    marche = candidat.get("marche", "")
    type_pari = candidat.get("type", "")
    if marche == "ML":
        if m["home_team"] in type_pari:
            return snapshot.get("ml_home")
        if m["away_team"] in type_pari:
            return snapshot.get("ml_away")
    elif marche == "PL":
        if m["home_team"] in type_pari:
            return snapshot.get("pl_home")
        if m["away_team"] in type_pari:
            return snapshot.get("pl_away")
    elif marche == "OU" and "totals" in snapshot:
        parts = type_pari.split(" ")
        if len(parts) < 2:
            return None
        side = parts[0].capitalize()
        cut_key = str(_arrondir_cut(parts[1]))
        ligne = snapshot["totals"].get(cut_key)
        if ligne and side in ligne:
            return ligne[side]
    return None


def evaluer_steam_movement(game_id, candidat, m):
    """
    Mesure le drift Pinnacle sur notre sélection depuis le snapshot le plus ancien
    (≥ NHL_LINE_MIN_AGE_MIN). move_pct > 0 = cote allongée = steam contre nous.
    """
    if not NHL_LINE_MOVE_ACTIF:
        return {"contre_nous": False, "move_pct": 0.0, "reference": None, "age_min": 0}

    entry = _lire_odds_history().get("games", {}).get(str(game_id), {})
    snaps = entry.get("snapshots", [])
    if len(snaps) < 2:
        return {"contre_nous": False, "move_pct": 0.0, "reference": None, "age_min": 0}

    now = datetime.now(timezone.utc)
    ref_snap = None
    age_min = 0
    for snap in snaps:
        try:
            ts = datetime.fromisoformat(snap["ts"].replace("Z", "+00:00"))
        except Exception:
            continue
        delta_min = (now - ts).total_seconds() / 60.0
        if delta_min >= NHL_LINE_MIN_AGE_MIN:
            ref_snap = snap
            age_min = int(delta_min)
            break

    if ref_snap is None:
        return {"contre_nous": False, "move_pct": 0.0, "reference": None, "age_min": 0}

    ref_cote = _cote_reference_pour_candidat(ref_snap, candidat, m)
    try:
        cote_actuelle = float(candidat.get("cote_book"))
        ref_cote = float(ref_cote)
    except (TypeError, ValueError):
        return {"contre_nous": False, "move_pct": 0.0, "reference": None, "age_min": age_min}

    if ref_cote <= 1.0 or cote_actuelle <= 1.0:
        return {"contre_nous": False, "move_pct": 0.0, "reference": ref_cote, "age_min": age_min}

    move_pct = (cote_actuelle - ref_cote) / ref_cote
    return {
        "contre_nous": move_pct >= NHL_LINE_STEAM_WARN_PCT,
        "avec_nous": move_pct <= -NHL_LINE_STEAM_WARN_PCT,
        "move_pct": round(move_pct, 4),
        "reference": round(ref_cote, 3),
        "actuelle": round(cote_actuelle, 3),
        "age_min": age_min,
    }


def filtrer_candidats_line_movement(game_id, candidats, m, gardiens_confirmes=True):
    """
    Retire les paris pris contre un steam move Pinnacle (sharp money adverse).
    Zone intermédiaire : edge minimum majoré de NHL_LINE_STEAM_EDGE_EXTRA.
    """
    global _line_move_logue
    if not NHL_LINE_MOVE_ACTIF or not candidats:
        return candidats

    if not _line_move_logue:
        _line_move_logue = True
        log_nhl(
            f"📉 Line movement actif — ref ≥{NHL_LINE_MIN_AGE_MIN} min, "
            f"block ≥{NHL_LINE_STEAM_BLOCK_PCT:.1%} contre, "
            f"+{NHL_LINE_STEAM_EDGE_EXTRA:.1%} edge si ≥{NHL_LINE_STEAM_WARN_PCT:.1%} contre"
        )

    edge_min_pct = (EDGE_MINIMUM if gardiens_confirmes else NHL_EDGE_MIN_PROBABLE) * 100
    edge_min_steam_pct = edge_min_pct + NHL_LINE_STEAM_EDGE_EXTRA * 100
    filtres = []

    for cand in candidats:
        steam = evaluer_steam_movement(game_id, cand, m)
        cand["steam"] = steam
        move = steam["move_pct"]

        if steam.get("avec_nous") and abs(move) >= NHL_LINE_STEAM_WARN_PCT:
            log_nhl(
                f"📈 Steam avec nous — {cand['type']} {move:+.1%} "
                f"({steam['reference']}→{steam['actuelle']}, {steam['age_min']} min)"
            )

        if move >= NHL_LINE_STEAM_BLOCK_PCT:
            log_nhl(
                f"🚫 Steam block — {cand['type']} {move:+.1%} contre nous "
                f"({steam['reference']}→{steam['actuelle']}, seuil {NHL_LINE_STEAM_BLOCK_PCT:.1%})"
            )
            continue

        if steam["contre_nous"] and cand["inv"]["edge"] < edge_min_steam_pct:
            log_nhl(
                f"🚫 Steam edge insuffisant — {cand['type']} {move:+.1%} contre "
                f"(edge {cand['inv']['edge']:.1f}% < {edge_min_steam_pct:.1f}% requis)"
            )
            continue

        if steam["contre_nous"]:
            log_nhl(
                f"⚠️ Steam contre mais edge OK — {cand['type']} {move:+.1%} "
                f"(edge {cand['inv']['edge']:.1f}% ≥ {edge_min_steam_pct:.1f}%)"
            )

        filtres.append(cand)

    return filtres


def get_odds_for_match(home_team_abbrev, away_team_abbrev, odds_cache=None, log_si_absent=False):
    """Retourne les cotes Pinnacle pour un match (depuis le cache ou une requête dédiée)."""
    noms_home = _noms_odds_pour_equipe(home_team_abbrev)
    noms_away = _noms_odds_pour_equipe(away_team_abbrev)
    if not noms_home or not noms_away:
        if log_si_absent:
            log_nhl(f"⚠️ Abréviation inconnue : {away_team_abbrev} @ {home_team_abbrev}", level="warning")
        return None

    if odds_cache is not None:
        for home in noms_home:
            for away in noms_away:
                hit = odds_cache.get((home, away))
                if hit:
                    return hit
        if log_si_absent:
            log_nhl(
                f"⚠️ Pas de cotes Pinnacle : {away_team_abbrev} @ {home_team_abbrev} "
                f"(testé : {noms_away[0]} @ {noms_home[0]})",
                level="warning",
            )
        return None

    single = fetch_all_pinnacle_odds()
    return get_odds_for_match(home_team_abbrev, away_team_abbrev, single, log_si_absent)


def get_real_live_odds(home_team_abbrev, away_team_abbrev, odds_cache=None, log_si_absent=False):
    """Aspire Moneyline et tous les Cuts Over/Under."""
    cotes = get_odds_for_match(
        home_team_abbrev, away_team_abbrev, odds_cache, log_si_absent=log_si_absent,
    )
    if not cotes:
        return None
    return {k: v for k, v in cotes.items() if k not in ("home_full", "away_full")}


def get_real_live_odds_puckline(home_team_abbrev, away_team_abbrev, odds_cache=None):
    """Aspire les spreads pour le Puck Line."""
    cotes = get_odds_for_match(home_team_abbrev, away_team_abbrev, odds_cache)
    if not cotes:
        return None
    pl = {}
    if "cote_pl_home" in cotes:
        pl["cote_pl_home"] = cotes["cote_pl_home"]
    if "cote_pl_away" in cotes:
        pl["cote_pl_away"] = cotes["cote_pl_away"]
    return pl if pl else None


def _choisir_meilleur_pari(candidats):
    """Retourne le candidat avec le plus haut edge parmi les marchés autorisés."""
    best, max_edge = None, 0.0
    for cand in candidats:
        inv = cand.get("inv")
        if inv and inv["edge"] > max_edge:
            max_edge = inv["edge"]
            best = cand
    return best


def _construire_candidats_pari(
    m, cotes_vraies, cotes_bookmaker, cotes_puckline, bankroll, gardiens_verrouilles, kelly_mult=None,
):
    """Évalue ML / PL / O-U selon NHL_MARCHES_ACTIFS."""
    candidats = []
    if kelly_mult is None:
        kelly_mult = _multiplicateur_kelly_dynamique()

    if "ML" in NHL_MARCHES_ACTIFS:
        inv = calculate_kelly(
            cotes_vraies["prob_1"], cotes_bookmaker["cote_1"],
            bankroll, gardiens_confirmes=gardiens_verrouilles, kelly_mult=kelly_mult,
        )
        if inv:
            candidats.append({
                "type": f"Victoire {m['home_team']}", "inv": inv,
                "cote_book": cotes_bookmaker["cote_1"], "cote_vraie": cotes_vraies["cote_1"],
                "marche": "ML",
            })
        inv = calculate_kelly(
            cotes_vraies["prob_2"], cotes_bookmaker["cote_2"],
            bankroll, gardiens_confirmes=gardiens_verrouilles, kelly_mult=kelly_mult,
        )
        if inv:
            candidats.append({
                "type": f"Victoire {m['away_team']}", "inv": inv,
                "cote_book": cotes_bookmaker["cote_2"], "cote_vraie": cotes_vraies["cote_2"],
                "marche": "ML",
            })

    if "PL" in NHL_MARCHES_ACTIFS and cotes_puckline:
        if "cote_pl_home" in cotes_puckline:
            inv = calculate_kelly(
                cotes_vraies["prob_pl_home"], cotes_puckline["cote_pl_home"],
                bankroll, gardiens_confirmes=gardiens_verrouilles, kelly_mult=kelly_mult,
            )
            if inv:
                candidats.append({
                    "type": f"Puck Line {m['home_team']} -1.5", "inv": inv,
                    "cote_book": cotes_puckline["cote_pl_home"], "cote_vraie": cotes_vraies["cote_pl_home"],
                    "marche": "PL",
                })
        if "cote_pl_away" in cotes_puckline:
            inv = calculate_kelly(
                cotes_vraies["prob_pl_away"], cotes_puckline["cote_pl_away"],
                bankroll, gardiens_confirmes=gardiens_verrouilles, kelly_mult=kelly_mult,
            )
            if inv:
                candidats.append({
                    "type": f"Puck Line {m['away_team']} -1.5", "inv": inv,
                    "cote_book": cotes_puckline["cote_pl_away"], "cote_vraie": cotes_vraies["cote_pl_away"],
                    "marche": "PL",
                })

    if "OU" in NHL_MARCHES_ACTIFS and "totals" in cotes_bookmaker:
        for cut, prices in cotes_bookmaker["totals"].items():
            cut_arrondi = _arrondir_cut(cut)
            _, prob_over = _trouver_cle_float(cotes_vraies["prob_over_cuts"], cut_arrondi)
            _, prob_under = _trouver_cle_float(cotes_vraies["prob_under_cuts"], cut_arrondi)
            if prob_over is None:
                continue
            if "Over" in prices:
                inv = calculate_kelly(
                    prob_over, prices["Over"], bankroll,
                    gardiens_confirmes=gardiens_verrouilles, kelly_mult=kelly_mult,
                )
                if inv:
                    candidats.append({
                        "type": f"OVER {cut_arrondi}", "inv": inv,
                        "cote_book": prices["Over"],
                        "cote_vraie": round(1 / max(prob_over, 0.001), 2),
                        "marche": "OU",
                    })
            if "Under" in prices and prob_under is not None:
                inv = calculate_kelly(
                    prob_under, prices["Under"], bankroll,
                    gardiens_confirmes=gardiens_verrouilles, kelly_mult=kelly_mult,
                )
                if inv:
                    candidats.append({
                        "type": f"UNDER {cut_arrondi}", "inv": inv,
                        "cote_book": prices["Under"],
                        "cote_vraie": round(1 / max(prob_under, 0.001), 2),
                        "marche": "OU",
                    })

    return candidats


def _lire_brier_recent(fenetre=None):
    """
    Brier score sur les N derniers paris clôturés du journal + baseline théorique
    p*(1-p) (variance Bernoulli irréductible aux niveaux de probabilité pariés).
    """
    fenetre = fenetre or NHL_KELLY_BRIER_FENETRE
    if not os.path.exists(FICHIER_JOURNAL):
        return None, None, 0
    try:
        from utils import preparer_calibration_journal, calculer_brier_score
        import pandas as pd
        with open(FICHIER_JOURNAL, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None, None, 0
        df_cal = preparer_calibration_journal(pd.DataFrame(rows))
        if df_cal.empty:
            return None, None, 0
        df_recent = df_cal.tail(fenetre)
        brier = calculer_brier_score(df_recent)
        brier_baseline = float((df_recent["p_model"] * (1 - df_recent["p_model"])).mean())
        return brier, brier_baseline, len(df_recent)
    except Exception as e:
        log_nhl(f"⚠️ Erreur lecture Brier récent (Kelly dynamique) : {e}", level="warning")
        return None, None, 0


def _calculer_clv_ligne(cote_prise, cote_cloture):
    """CLV = (cote prise / cote clôture) - 1 — aligné dashboard_nhl / foot."""
    try:
        prise = float(cote_prise)
        cloture = float(cote_cloture)
    except (TypeError, ValueError):
        return None
    if prise <= 1.0 or cloture <= 1.0:
        return None
    return (prise / cloture) - 1.0


def _lire_clv_recent(fenetre=None):
    """
    CLV moyen sur les N derniers paris clôturés avec tracking effectif
    (Cote_CLV ≠ Cote_Prise). Signal complémentaire au Brier : bat-on la ligne Pinnacle ?
    """
    fenetre = fenetre or NHL_KELLY_CLV_FENETRE
    if not os.path.exists(FICHIER_JOURNAL):
        return None, 0
    try:
        with open(FICHIER_JOURNAL, "r", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("Statut") in ("GAGNÉ", "PERDU")]
        if not rows:
            return None, 0
        clv_vals = []
        for row in rows[-fenetre:]:
            prise = row.get("Cote_Prise")
            cloture = row.get("Cote_CLV")
            if not prise or not cloture:
                continue
            if _float_proche(prise, cloture):
                continue
            clv = _calculer_clv_ligne(prise, cloture)
            if clv is not None:
                clv_vals.append(clv)
        if not clv_vals:
            return None, 0
        return sum(clv_vals) / len(clv_vals), len(clv_vals)
    except Exception as e:
        log_nhl(f"⚠️ Erreur lecture CLV récent (Kelly dynamique) : {e}", level="warning")
        return None, 0


def _multiplicateur_kelly_brier():
    """Réduction Kelly si BSS récent négatif (calibration probabilités)."""
    if not NHL_KELLY_DYNAMIQUE_ACTIF:
        return 1.0
    brier, brier_baseline, n = _lire_brier_recent()
    if brier is None or n < NHL_KELLY_BRIER_MIN_PARIS or not brier_baseline or NHL_KELLY_BSS_SENSIBILITE <= 0:
        return 1.0
    bss = 1.0 - (brier / brier_baseline)
    if bss >= 0:
        return 1.0
    return round(min(max(1.0 + bss / NHL_KELLY_BSS_SENSIBILITE, NHL_KELLY_MULT_MIN), 1.0), 3)


def _multiplicateur_kelly_clv():
    """
    Réduction Kelly si CLV moyen récent négatif : on prend systématiquement
    une cote pire que la clôture Pinnacle → edge réel probablement surestimé.
    CLV ≥ 0 → mult=1.0 (pas de bonus, même logique que BSS).
    """
    if not NHL_KELLY_CLV_ACTIF:
        return 1.0
    clv_moy, n = _lire_clv_recent()
    if clv_moy is None or n < NHL_KELLY_CLV_MIN_PARIS or NHL_KELLY_CLV_SENSIBILITE <= 0:
        return 1.0
    if clv_moy >= 0:
        return 1.0
    return round(min(max(1.0 + clv_moy / NHL_KELLY_CLV_SENSIBILITE, NHL_KELLY_CLV_MULT_MIN), 1.0), 3)


def _multiplicateur_kelly_dynamique():
    """
    Kelly dynamique combiné : BSS (calibration) × CLV (qualité vs marché).
    Chaque signal ne réduit jamais au-dessus de 1.0 — pas de sur-mise sur hot streak.
    """
    mult_brier = _multiplicateur_kelly_brier()
    mult_clv = _multiplicateur_kelly_clv()
    mult = round(mult_brier * mult_clv, 3)
    if mult >= 1.0:
        return 1.0

    parts = []
    if mult_brier < 1.0:
        brier, brier_baseline, n_b = _lire_brier_recent()
        bss = 1.0 - (brier / brier_baseline) if brier and brier_baseline else 0.0
        parts.append(f"BSS {bss:+.2f} (n={n_b})→x{mult_brier:.2f}")
    if mult_clv < 1.0:
        clv_moy, n_c = _lire_clv_recent()
        parts.append(f"CLV {clv_moy:+.1%} (n={n_c})→x{mult_clv:.2f}")
    log_nhl(
        f"⚠️ Kelly dynamique : {' | '.join(parts)} → mises réduites x{mult:.2f}"
    )
    return mult


def calculate_kelly(true_prob, book_odds, bankroll, gardiens_confirmes=True, kelly_mult=None):
    if book_odds <= 1.0 or true_prob <= 0.01 or true_prob >= 0.99:
        return None
    edge = true_prob - (1 / book_odds)
    edge_min = EDGE_MINIMUM if gardiens_confirmes else NHL_EDGE_MIN_PROBABLE
    if edge <= edge_min:
        return None
    b = book_odds - 1.0
    fraction_kelly = KELLY_FRACTION if gardiens_confirmes else KELLY_FRACTION_GARDIEN_INCERTAIN
    if kelly_mult is None:
        kelly_mult = _multiplicateur_kelly_dynamique()
    fraction_kelly *= kelly_mult
    if not gardiens_confirmes:
        log_nhl(
            f"🛡️ Gardiens probables — Kelly ÷2, edge min {NHL_EDGE_MIN_PROBABLE:.0%} "
            f"(seuil confirmés : {EDGE_MINIMUM:.0%})"
        )
    safe_kelly = ((b * true_prob - (1 - true_prob)) / b) * fraction_kelly
    mise_brute = bankroll * safe_kelly
    if NHL_MISE_MAX_PCT > 0:
        cap_pct = bankroll * (NHL_MISE_MAX_PCT / 100.0)
        mise = round(min(mise_brute, cap_pct), 2)
    else:
        mise = round(mise_brute, 2)
    if mise <= 0:
        return None
    pct_effectif = round((mise / bankroll) * 100, 2) if bankroll > 0 else 0.0
    if NHL_MISE_MAX_PCT > 0 and mise < mise_brute:
        log_nhl(
            f"📉 Cap mise appliqué : {round(mise_brute, 2)} € → {mise} € "
            f"(max {NHL_MISE_MAX_PCT}% bankroll)"
        )
    return {
        'edge': round(edge * 100, 2),
        'pct_bankroll': pct_effectif,
        'mise': mise,
        'statut_gardiens': "CONFIRMÉ" if gardiens_confirmes else "PROBABLE",
    }

# ==========================================
# 6. JOURNAL DE TRADING & NOTIFICATIONS
# ==========================================
def compter_paris_du_jour():
    if not os.path.exists(FICHIER_JOURNAL):
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(FICHIER_JOURNAL, "r", encoding="utf-8") as f:
            return sum(1 for row in csv.DictReader(f) if row.get("Date", "").startswith(today))
    except Exception:
        return 0


def limite_paris_jour_atteinte():
    return NHL_PARIS_JOUR_MAX > 0 and compter_paris_du_jour() >= NHL_PARIS_JOUR_MAX


def migrer_journal_si_besoin():
    """Ajoute les colonnes P4 aux anciens journaux sans les perdre."""
    if not os.path.exists(FICHIER_JOURNAL):
        return
    with open(FICHIER_JOURNAL, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        if all(col in reader.fieldnames for col in JOURNAL_COLONNES):
            return
        lignes = list(reader)
    with open(FICHIER_JOURNAL, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_COLONNES, extrasaction="ignore")
        writer.writeheader()
        for row in lignes:
            writer.writerow({col: row.get(col, "-") for col in JOURNAL_COLONNES})
    log_nhl("📋 Journal migré vers le schéma enrichi (colonnes gardiens, B2B, rho)")


def _ecrire_journal(rows):
    with open(FICHIER_JOURNAL, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_COLONNES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def publier_journal_dashboard():
    """Journal déjà sur PA, ou upload FTP optionnel depuis une machine locale."""
    if os.path.isdir(PA_DATA_DIR) and FICHIER_JOURNAL.startswith(PA_DATA_DIR):
        return
    if not os.path.exists(FICHIER_JOURNAL):
        return
    ftp_user = os.environ.get("PA_FTP_USER", "")
    ftp_pass = os.environ.get("PA_FTP_PASSWORD", "")
    if not ftp_user or not ftp_pass:
        return
    ftp_host = os.environ.get("PA_FTP_HOST", "ftp.pythonanywhere.com")
    remote_dir = os.environ.get("PA_FTP_REMOTE_DIR", "/home/chienblanc/data")
    remote_name = os.path.basename(FICHIER_JOURNAL)
    try:
        with ftplib.FTP(ftp_host, timeout=30) as ftp:
            ftp.login(ftp_user, ftp_pass)
            ftp.cwd(remote_dir)
            with open(FICHIER_JOURNAL, "rb") as f:
                ftp.storbinary(f"STOR {remote_name}", f)
        log_nhl(f"📤 Journal uploadé → {ftp_host}{remote_dir}/{remote_name}")
    except Exception as e:
        log_nhl(f"⚠️ Upload FTP journal échoué : {e}", level="warning")


def match_deja_notifie(id_match):
    if not os.path.exists(FICHIER_MEMOIRE): return False
    with open(FICHIER_MEMOIRE, "r") as f: return id_match in f.read()

def enregistrer_notification(id_match):
    with open(FICHIER_MEMOIRE, "a") as f: f.write(id_match + "\n")


def envoyer_alerte_systeme(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log_nhl(f"⚠️ Alerte système (Telegram absent) : {message}", level="warning")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        log_nhl(f"⚠️ Erreur Telegram système : {e}", level="warning")


def enregistrer_transaction(
    id_match, ext, dom, type_pari, vraie_cote_pari, cotes_vraies_dict, investissement, cote_bookmaker,
    gardien_ext="-", gardien_dom="-", gardiens_confirmes=False, b2b_home=False, b2b_away=False, rho=-0.12,
    hia=NHL_HIA_DEFAULT,
):
    migrer_journal_si_besoin()
    fichier_existe = os.path.isfile(FICHIER_JOURNAL)
    row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ID_Match": id_match,
        "Visiteur": ext,
        "Local": dom,
        "Pari": type_pari,
        "Vraie_Cote_Bot": vraie_cote_pari,
        "Cote_Prise": cote_bookmaker,
        "Cote_CLV": cote_bookmaker,
        "Lam_Ext": cotes_vraies_dict["lam_away"],
        "Lam_Dom": cotes_vraies_dict["lam_home"],
        "Score_Ext": "-",
        "Score_Dom": "-",
        "Edge(%)": investissement["edge"],
        "Risque(%)": investissement["pct_bankroll"],
        "Mise_€": investissement["mise"],
        "Statut": "EN ATTENTE",
        "P&L": "0.00",
        "Gardien_Ext": gardien_ext,
        "Gardien_Dom": gardien_dom,
        "Gardiens_Confirmes": "OUI" if gardiens_confirmes else "NON",
        "B2B_Home": "OUI" if b2b_home else "NON",
        "B2B_Away": "OUI" if b2b_away else "NON",
        "Rho": rho,
        "Hia": hia,
    }
    with open(FICHIER_JOURNAL, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_COLONNES, extrasaction="ignore")
        if not fichier_existe:
            writer.writeheader()
        writer.writerow(row)
    publier_journal_dashboard()

def envoyer_alerte(ext, g_ext, dom, g_dom, vraie_cote_pari, investissement, type_pari, dry_run=False):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log_nhl("⚠️ Telegram non configuré — alerte non envoyée.", level="warning")
        return
    statut = investissement.get('statut_gardiens', 'CONFIRMÉ')
    alerte_gardien = "✅ Gardiens Confirmés" if statut == "CONFIRMÉ" else "🛡️ GARDIENS PROBABLES (Mise / 2)"
    prefix = "🧪 **[DRY RUN — SIMULATION]**\n\n" if dry_run else ""
    msg = prefix + f"🚨 **SNIPER NHL DÉCLENCHÉ** 🚨\n\nLoc: 🏟️ **{dom}** ({g_dom})\nVis: ✈️ **{ext}** ({g_ext})\n"
    msg += f"ℹ️ Statut : {alerte_gardien}\n──────────────\n"
    msg += f"🎯 **ORDRE : PARIER {type_pari}**\n🔥 Edge : **+{investissement['edge']}%**\n⚖️ Kelly : **{investissement['pct_bankroll']}%**\n"
    msg += f"💵 **MISE : {investissement['mise']} €**\n──────────────\n📊 True Odds: {vraie_cote_pari}"
    if dry_run:
        msg += "\n\n_(Aucune écriture journal — mode simulation)_"
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
        timeout=10,
    )


def _extraire_cote_clv(home_abbr, away_abbr, type_pari, cote_actuelle, odds_cache=None):
    """Détermine la cote Pinnacle actuelle pour un pari en attente."""
    cotes = get_odds_for_match(home_abbr, away_abbr, odds_cache)
    if not cotes:
        return cote_actuelle
    if "OVER" in type_pari or "UNDER" in type_pari:
        parts = type_pari.split(" ")
        side = parts[0].capitalize()
        cut = _arrondir_cut(parts[1])
        totals = cotes.get("totals", {})
        _, ligne = _trouver_cle_float(totals, cut)
        if ligne and side in ligne:
            return str(ligne[side])
    elif "Puck Line" in type_pari:
        if home_abbr in type_pari and "cote_pl_home" in cotes:
            return str(cotes["cote_pl_home"])
        if away_abbr in type_pari and "cote_pl_away" in cotes:
            return str(cotes["cote_pl_away"])
    else:
        if home_abbr in type_pari and "cote_1" in cotes:
            return str(cotes["cote_1"])
        if away_abbr in type_pari and "cote_2" in cotes:
            return str(cotes["cote_2"])
    return cote_actuelle


def compter_paris_en_attente():
    if not os.path.exists(FICHIER_JOURNAL):
        return 0
    try:
        with open(FICHIER_JOURNAL, "r", encoding="utf-8") as f:
            return sum(1 for row in csv.DictReader(f) if row.get("Statut") == "EN ATTENTE")
    except Exception:
        return 0


def traquer_et_actualiser_clv():
    if not os.path.exists(FICHIER_JOURNAL):
        return
    migrer_journal_si_besoin()
    odds_cache = fetch_all_pinnacle_odds()
    rows, mise_a_jour_effectuee = [], False
    with open(FICHIER_JOURNAL, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Statut") == "EN ATTENTE":
                ext, dom, type_pari = row["Visiteur"], row["Local"], row["Pari"]
                nv_cote = _extraire_cote_clv(dom, ext, type_pari, row["Cote_CLV"], odds_cache)
                if nv_cote != row["Cote_CLV"]:
                    row["Cote_CLV"] = nv_cote
                    mise_a_jour_effectuee = True
            rows.append(row)
    if mise_a_jour_effectuee:
        _ecrire_journal(rows)
        publier_journal_dashboard()

def calculer_bankroll_dynamique(capital_de_base=1000.0):
    """
    Lit le fichier CSV, additionne les profits et pertes (P&L),
    et retourne le capital actuel.
    """
    if not os.path.exists(FICHIER_JOURNAL):
        return capital_de_base

    profit_total = 0.0
    try:
        with open(FICHIER_JOURNAL, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # On ne compte que les paris qui sont terminés et balayés
                if row['Statut'] in ["GAGNÉ", "PERDU"]:
                    profit_total += float(row['P&L'])

        # On calcule le solde actuel
        bankroll_actuelle = capital_de_base + profit_total

        # Sécurité Anti-Ruine : Si la bankroll tombe sous 10€,
        # on la bloque à 10€ pour éviter que le bot ne plante avec des mises négatives ou à 0.
        return round(max(bankroll_actuelle, 10.0), 2)

    except Exception as e:
        print(f"⚠️ Erreur de calcul de la Bankroll : {e}")
        return capital_de_base

# ==========================================
# 7. BOUCLE DE SCANNAGE PRINCIPALE
# ==========================================
def run_sniper():
    mode = "DRY RUN (simulation)" if NHL_DRY_RUN else "LIVE"
    log_nhl(f"🤖 Lancement Sniper NHL — mode {mode}")
    if NHL_DRY_RUN:
        log_nhl("🧪 NHL_DRY_RUN actif : signaux Telegram sans écriture journal.")
    if NHL_PARIS_JOUR_MAX > 0:
        log_nhl(f"📊 Limite paris/jour : {NHL_PARIS_JOUR_MAX}")
    cap_label = f"{NHL_MISE_MAX_PCT}% bankroll" if NHL_MISE_MAX_PCT > 0 else "Kelly pur (pas de cap %)"
    log_nhl(f"💶 Cap mise : {cap_label} | journal → {FICHIER_JOURNAL}")
    log_nhl(f"📊 Alerte quota Odds API si ≤ {NHL_ODDS_QUOTA_ALERT} requêtes restantes")
    log_nhl(
        f"🎯 Marchés actifs : {', '.join(sorted(NHL_MARCHES_ACTIFS)) or 'aucun'} | "
        f"OT home adv ML : {NHL_OT_HOME_ADVANTAGE:.0%} | "
        f"edge min {EDGE_MINIMUM:.0%} (confirmés) / {NHL_EDGE_MIN_PROBABLE:.0%} (probables) | "
        f"rho+HIA recalibrés tous les {NHL_RHO_INTERVAL_MATCHS} matchs (min {NHL_RHO_MIN_MATCHS}) | "
        f"empty-net/tie recalibrés dès {NHL_EMPTY_NET_MIN_MATCHS} matchs"
    )
    if NHL_BLEND_GP_PLEIN > 0:
        log_nhl(f"🔀 Blend MoneyPuck N/N-1 jusqu'à {NHL_BLEND_GP_PLEIN:.0f} GP moyen/ligue")
    if NHL_PP_PK_SHRINK_GP > 0:
        log_nhl(f"📉 Shrinkage PP/PK vers moyenne ligue jusqu'à {NHL_PP_PK_SHRINK_GP:.0f} GP/équipe")
    if NHL_GSAX_RECENT_WINDOW > 0:
        log_nhl(
            f"🥅 GSAx gardiens : blend forme récente ({NHL_GSAX_RECENT_WINDOW:.0f} derniers matchs) "
            f"+ saison (plein à {NHL_GSAX_RECENT_GP_PLEIN:.0f} GP)"
        )
    if NHL_TEAM_RECENT_WINDOW > 0:
        log_nhl(
            f"📈 xG équipes : blend forme récente ({NHL_TEAM_RECENT_WINDOW:.0f} derniers matchs) "
            f"+ saison (plein à {NHL_TEAM_RECENT_GP_PLEIN:.0f} GP) — remplace le momentum L10"
        )
    if NHL_MARCHE_SHRINK_ACTIF:
        log_nhl(
            f"⚖️ Shrinkage marché actif — confiance modèle {NHL_MODEL_TRUST_MIN:.0%} à 0 GP → "
            f"{NHL_MODEL_TRUST_MAX:.0%} à {NHL_MODEL_TRUST_GP_PLEIN:.0f}+ GP (reste : no-vig Pinnacle)"
        )
    if NHL_MARCHE_COHERENCE_ACTIF:
        log_nhl("🔗 Cohérence inter-marchés active — PL ≤ ML, totaux O/U monotones après shrinkage")
    if NHL_KELLY_DYNAMIQUE_ACTIF:
        log_nhl(
            f"🎚️ Kelly dynamique actif — BSS fenêtre {NHL_KELLY_BRIER_FENETRE} paris (min {NHL_KELLY_BRIER_MIN_PARIS}), "
            f"mult. plancher x{NHL_KELLY_MULT_MIN:.2f}"
        )
    if NHL_KELLY_CLV_ACTIF:
        log_nhl(
            f"📊 Kelly CLV actif — fenêtre {NHL_KELLY_CLV_FENETRE} paris (min {NHL_KELLY_CLV_MIN_PARIS}), "
            f"sensibilité {NHL_KELLY_CLV_SENSIBILITE:.0%}, mult. plancher x{NHL_KELLY_CLV_MULT_MIN:.2f}"
        )
    if NHL_LINE_MOVE_ACTIF:
        log_nhl(
            f"📉 Line movement / steam — snapshots max {NHL_LINE_MAX_SNAPSHOTS}, "
            f"ref ≥{NHL_LINE_MIN_AGE_MIN} min, block ≥{NHL_LINE_STEAM_BLOCK_PCT:.1%}"
        )
    if NHL_TRAVEL_FATIGUE_ACTIF:
        log_nhl(
            f"✈️ Fatigue voyage : lookback {NHL_TRAVEL_LOOKBACK_JOURS}j, ref {NHL_TRAVEL_MILES_REF:.0f} mi "
            f"+ fuseau/B2B (long solo ≥ {NHL_TRAVEL_LONG_MILES:.0f} mi)"
        )
    if NHL_FACEOFF_ADJ_ACTIF:
        log_nhl(f"🏒 Ajustement faceoffs actif — sensibilité {NHL_FACEOFF_SENSIBILITE:.2f}")
    if NHL_REF_ADJ_ACTIF:
        log_nhl(
            f"👨‍⚖️ Ajustement arbitral actif — sensibilité {NHL_REF_SENSIBILITE:.2f}, "
            f"scan {NHL_REF_SCAN_JOURS}j/cycle, min {NHL_REF_MIN_MATCHS} matchs/arbitre"
        )
    if NHL_LIGUE_CALIB_ACTIF:
        log_nhl(
            f"📚 Calibration MLE élargie à tous les matchs ligue — bootstrap "
            f"{NHL_LIGUE_CALIB_LOOKBACK_JOURS}j/cycle puis {NHL_LIGUE_CALIB_SCAN_JOURS}j/cycle "
            f"après {NHL_LIGUE_CALIB_MIN_MATCHS} matchs indexés (max {NHL_LIGUE_CALIB_MAX_GAMES} conservés)"
        )
    if NHL_MLE_RECENCY_ACTIF and NHL_MLE_RECENCY_HALFLIFE_JOURS > 0:
        log_nhl(
            f"⏳ MLE recency actif — demi-vie {NHL_MLE_RECENCY_HALFLIFE_JOURS:.0f}j "
            f"(match récent poids ≈1, à {NHL_MLE_RECENCY_HALFLIFE_JOURS:.0f}j poids ≈0.5)"
        )
    if NHL_HIA_PAR_EQUIPE_ACTIF:
        log_nhl(
            f"🏠 HIA par équipe actif — shrinkage vers global MLE, confiance pleine à "
            f"{NHL_HIA_TEAM_GP_PLEIN:.0f} matchs domicile (min {NHL_HIA_TEAM_MIN_GAMES})"
        )
    migrer_journal_si_besoin()

    while True:
        try:
            lancer_la_balayeuse()

            log_nhl("📡 Synchronisation bases de données...")
            teams, goalies, stars_vip = get_team_stats(), get_goalie_stats(), get_stars_impact()

            if not teams or not goalies:
                log_nhl(
                    f"⚠️ Données MoneyPuck indisponibles (saison {NHL_SEASON}). Nouvelle tentative dans 5 min...",
                    level="warning",
                )
                time.sleep(300)
                continue

            if NHL_LIGUE_CALIB_ACTIF:
                actualiser_historique_ligue(teams)
            entrainer_ia_dixon_coles()
            actualiser_hia_equipes()
            actualiser_stats_arbitres()

            nb_attente = compter_paris_en_attente()
            log_nhl(f"🕵️ Tracking CLV ({nb_attente} pari(s) en attente)...")
            traquer_et_actualiser_clv()

            bankroll_actuelle = calculer_bankroll_dynamique(BANKROLL_INITIALE)
            log_nhl(f"💰 Capital Dynamique Disponible : {bankroll_actuelle} €")
            # ------------------------------------

            rho_actuel = lire_rho_dynamique()
            hia_actuel = lire_hia_dynamique()
            prob_tie_actuel = lire_prob_tie_dynamique()
            prob_en_actuel = lire_prob_en_dynamique()
            log_nhl(
                f"🧠 Configuration Mathématique : Rho = {rho_actuel} | HIA = {hia_actuel:.1%} | "
                f"Tie = {prob_tie_actuel:.1%} | EmptyNet = {prob_en_actuel:.1%}"
            )
            kelly_mult_actuel = _multiplicateur_kelly_dynamique()

            equipes_en_b2b_hier = get_teams_played_yesterday()
            derniers_lieux = get_team_last_game_venues()
            odds_cache = fetch_all_pinnacle_odds()

            matchs = get_nhl_games_today()
            if not matchs:
                log_nhl("🏒 Aucun match NHL éligible dans la fenêtre de scan — veille active.")
            else:
                log_nhl(f"🏒 {len(matchs)} match(s) dans la fenêtre de scan.")
                if NHL_LINE_MOVE_ACTIF:
                    enregistrer_snapshots_cotes(matchs, odds_cache)
            for m in matchs:
                id_match = f"{m['game_id']}_notified"
                if match_deja_notifie(id_match):
                    continue

                g_ext, g_dom, skaters_ext, skaters_dom, source_roster = get_rosters_avec_fallback(m["game_id"])
                if not g_ext or not g_dom:
                    log_nhl(
                        f"⏳ Skip alignement — {m['away_team']} @ {m['home_team']} "
                        f"(id {m['game_id']}, état {m.get('game_state', '?')})"
                    )
                    continue

                if source_roster == "landing_probable":
                    log_nhl(
                        f"ℹ️ Gardiens probables (landing) {m['away_team']} @ {m['home_team']} : {g_ext} / {g_dom}"
                    )
                elif not skaters_ext and not skaters_dom:
                    log_nhl(
                        f"ℹ️ Rosters patineurs non publiés {m['away_team']} @ {m['home_team']} "
                        f"— analyse sans détection d'absences stars."
                    )

                statut_confirmation = get_goalie_confirmation_status(m['game_id'])
                gardiens_verrouilles = statut_confirmation["away_confirmed"] and statut_confirmation["home_team_confirmed"]

                home_b2b = m['home_team'] in equipes_en_b2b_hier
                away_b2b = m['away_team'] in equipes_en_b2b_hier

                if home_b2b:
                    log_nhl(f"🔄 {m['home_team']} détecté en Back-to-Back !")
                if away_b2b:
                    log_nhl(f"🔄 {m['away_team']} détecté en Back-to-Back !")

                home_travel = get_travel_miles_for_team(m['home_team'], m['home_team'], derniers_lieux)
                away_travel = get_travel_miles_for_team(m['away_team'], m['home_team'], derniers_lieux)
                if NHL_TRAVEL_FATIGUE_ACTIF and (home_travel >= NHL_TRAVEL_LONG_MILES or away_travel >= NHL_TRAVEL_LONG_MILES):
                    log_nhl(
                        f"✈️ Voyage {m['away_team']} @ {m['home_team']} — "
                        f"dom {home_travel:.0f} mi, vis {away_travel:.0f} mi depuis dernier match"
                    )

                gsax_ext, gsax_dom = trouver_gsax(g_ext, goalies), trouver_gsax(g_dom, goalies)
                absents_ext = detecter_stars_absentes(m['away_team'], skaters_ext, stars_vip)
                absents_dom = detecter_stars_absentes(m['home_team'], skaters_dom, stars_vip)

                teams_match = copy.deepcopy(teams)
                home_base = next((t for t in teams_match if t['team'] == m['home_team']), None)
                away_base = next((t for t in teams_match if t['team'] == m['away_team']), None)
                if not home_base or not away_base:
                    log_nhl(f"⚠️ Skip stats MoneyPuck — {m['away_team']} @ {m['home_team']}", level="warning")
                    continue

                adj_xgf_ext, adj_xga_ext = apply_star_absence_penalty(
                    m['away_team'], away_base['xGF_per_game'], away_base['xGA_per_game'],
                    absents_ext, stars_vip,
                )
                adj_xgf_dom, adj_xga_dom = apply_star_absence_penalty(
                    m['home_team'], home_base['xGF_per_game'], home_base['xGA_per_game'],
                    absents_dom, stars_vip,
                )
                away_base['xGF_per_game'], away_base['xGA_per_game'] = adj_xgf_ext, adj_xga_ext
                home_base['xGF_per_game'], home_base['xGA_per_game'] = adj_xgf_dom, adj_xga_dom

                referee_names = get_game_referees(m["game_id"])
                ref_pp_mult, ref_info = compute_referee_pp_multiplier(referee_names)
                if ref_info and ref_pp_mult != 1.0:
                    log_nhl(
                        f"👨‍⚖️ Crew {', '.join(ref_info['refs'])} — "
                        f"{ref_info['crew_ppg']:.1f} vs ligue {ref_info['league_ppg']:.1f} pén./match "
                        f"→ PP x{ref_pp_mult:.2f} ({m['away_team']} @ {m['home_team']})"
                    )
                elif referee_names and not ref_info:
                    log_nhl(
                        f"ℹ️ Arbitres {', '.join(referee_names)} — historique insuffisant "
                        f"(<{NHL_REF_MIN_MATCHS} matchs), pas d'ajustement"
                    )

                hia_match = lire_hia_equipe(m['home_team'])
                if NHL_HIA_PAR_EQUIPE_ACTIF and abs(hia_match - hia_actuel) > 0.005:
                    log_nhl(
                        f"🏠 HIA {m['home_team']} : {hia_match:.1%} "
                        f"(global {hia_actuel:.1%}) — {m['away_team']} @ {m['home_team']}"
                    )

                cotes_vraies = calculate_master_odds_v4(
                    teams_match, m['home_team'], m['away_team'], gsax_dom, gsax_ext,
                    home_is_b2b=home_b2b, away_is_b2b=away_b2b,
                    home_travel_miles=home_travel, away_travel_miles=away_travel,
                    referee_pp_mult=ref_pp_mult,
                    rho=rho_actuel, hia=hia_match,
                    prob_tie=prob_tie_actuel, prob_en=prob_en_actuel,
                )
                cotes_bookmaker = get_real_live_odds(
                    m['home_team'], m['away_team'], odds_cache, log_si_absent=True,
                )
                if not cotes_bookmaker:
                    log_nhl(f"⚠️ Skip cotes Pinnacle — {m['away_team']} @ {m['home_team']}")
                    continue
                cotes_puckline = get_real_live_odds_puckline(m['home_team'], m['away_team'], odds_cache)

                gp_moyen_match = (
                    (home_base.get('games_played', 0) + away_base.get('games_played', 0)) / 2.0
                )
                cotes_vraies, corr_marches = shrink_cotes_vers_marche(
                    cotes_vraies, cotes_bookmaker, cotes_puckline, gp_moyen_match,
                )
                if corr_marches:
                    log_nhl(
                        f"🔗 Cohérence marchés corrigée ({', '.join(corr_marches)}) — "
                        f"{m['away_team']} @ {m['home_team']}"
                    )

                if cotes_vraies and cotes_bookmaker:
                    candidats = _construire_candidats_pari(
                        m, cotes_vraies, cotes_bookmaker, cotes_puckline,
                        bankroll_actuelle, gardiens_verrouilles, kelly_mult=kelly_mult_actuel,
                    )
                    candidats = filtrer_candidats_line_movement(
                        m["game_id"], candidats, m, gardiens_verrouilles,
                    )
                    best_pari = _choisir_meilleur_pari(candidats)

                    # --- Envoi de la transaction finale ---
                    if best_pari:
                        log_nhl(
                            f"🎯 Edge {best_pari['inv']['edge']}% [{best_pari.get('marche', '?')}] — "
                            f"{best_pari['type']} ({m['away_team']} @ {m['home_team']}) "
                            f"mise {best_pari['inv']['mise']} €"
                        )
                        if limite_paris_jour_atteinte():
                            log_nhl(
                                f"🛑 Limite paris/jour atteinte ({NHL_PARIS_JOUR_MAX}) — signal ignoré.",
                                level="warning",
                            )
                        elif NHL_DRY_RUN:
                            envoyer_alerte(
                                m['away_team'], g_ext, m['home_team'], g_dom,
                                best_pari['cote_vraie'], best_pari['inv'], best_pari['type'],
                                dry_run=True,
                            )
                            enregistrer_notification(id_match)
                        else:
                            envoyer_alerte(
                                m['away_team'], g_ext, m['home_team'], g_dom,
                                best_pari['cote_vraie'], best_pari['inv'], best_pari['type'],
                            )
                            enregistrer_transaction(
                                id_match, m['away_team'], m['home_team'], best_pari['type'],
                                best_pari['cote_vraie'], cotes_vraies, best_pari['inv'], best_pari['cote_book'],
                                gardien_ext=g_ext, gardien_dom=g_dom,
                                gardiens_confirmes=gardiens_verrouilles,
                                b2b_home=home_b2b, b2b_away=away_b2b, rho=rho_actuel, hia=hia_match,
                            )
                            enregistrer_notification(id_match)
                    else:
                        if not gardiens_verrouilles:
                            log_nhl(
                                f"— Pas d'edge ≥ {NHL_EDGE_MIN_PROBABLE:.0%} (gardiens probables) "
                                f"— attente confirmation {m['away_team']} @ {m['home_team']}"
                            )
                        else:
                            log_nhl(f"— Pas d'edge suffisant — {m['away_team']} @ {m['home_team']}")
            time.sleep(900)
        except Exception as e:
            log_nhl(f"⚠️ Erreur système : {e}", level="error")
            traceback.print_exc()
            time.sleep(60)

# ==========================================
# 8. LA BALAYEUSE & INTELLIGENCE
# ==========================================
def get_match_result(game_id):
    try:
        response = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore", timeout=10)
        data = response.json()
        if response.status_code == 200 and data["gameState"] == "FINAL": return data["awayScore"], data["homeScore"]
        return None
    except: return None

def lancer_la_balayeuse():
    if not os.path.exists(FICHIER_JOURNAL):
        return
    migrer_journal_si_besoin()
    rows = []
    modifie = False
    with open(FICHIER_JOURNAL, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Statut") == "EN ATTENTE":
                game_id = row["ID_Match"].split("_")[0]
                res = get_match_result(game_id)
                if res:
                    modifie = True
                    score_v, score_d = res
                    row["Score_Ext"] = str(score_v)
                    row["Score_Dom"] = str(score_d)
                    ext, dom, pari = row["Visiteur"], row["Local"], row["Pari"]
                    mise, cote_book = float(row["Mise_€"]), float(row["Cote_Prise"])

                    gagne = False
                    if "Victoire" in pari:
                        if (score_d > score_v and dom in pari) or (score_v > score_d and ext in pari):
                            gagne = True
                    elif "Puck Line" in pari:
                        if (score_d - score_v >= 2 and dom in pari) or (score_v - score_d >= 2 and ext in pari):
                            gagne = True
                    elif "OVER" in pari or "UNDER" in pari:
                        parts = pari.split(" ")
                        cut = float(parts[1])
                        total_buts = score_d + score_v
                        if "OVER" in pari and total_buts > cut:
                            gagne = True
                        elif "UNDER" in pari and total_buts < cut:
                            gagne = True

                    row["Statut"] = "GAGNÉ" if gagne else "PERDU"
                    row["P&L"] = f"{round(mise * (cote_book - 1), 2) if gagne else -mise}"
            rows.append(row)
    if modifie:
        _ecrire_journal(rows)
        publier_journal_dashboard()

def lire_rho_meta():
    default = {
        "rho": -0.12, "hia": NHL_HIA_DEFAULT,
        "prob_tie": 0.12, "prob_en": 0.22,
        "nb_matchs": 0,
    }
    if os.path.exists(RHO_META_FILE):
        try:
            with open(RHO_META_FILE, "r", encoding="utf-8") as f:
                return {**default, **json.load(f)}
        except Exception:
            pass
    if os.path.exists("rho_optimal.txt"):
        try:
            with open("rho_optimal.txt", "r", encoding="utf-8") as f:
                return {**default, "rho": float(f.read().strip())}
        except Exception:
            pass
    return default


def lire_rho_dynamique():
    return float(lire_rho_meta()["rho"])


def lire_hia_dynamique():
    return float(lire_rho_meta().get("hia", NHL_HIA_DEFAULT))


def lire_prob_tie_dynamique():
    return float(lire_rho_meta().get("prob_tie", 0.12))


def lire_prob_en_dynamique():
    return float(lire_rho_meta().get("prob_en", 0.22))


def lire_hia_equipes_meta():
    default = {"hia_global": NHL_HIA_DEFAULT, "teams": {}}
    if os.path.exists(HIA_TEAM_META_FILE):
        try:
            with open(HIA_TEAM_META_FILE, "r", encoding="utf-8") as f:
                return {**default, **json.load(f)}
        except Exception:
            pass
    return default


def _estimer_hia_brut_match(lam_h, lam_a, home_goals, away_goals):
    """
    HIA implicite d'un match à domicile : écart de score réel vs attendu (lambdas),
    normalisé pour rester dans l'échelle du paramètre HIA (+/- quelques %).
    """
    denom = max(float(lam_h) + float(lam_a), 2.0)
    expected_diff = float(lam_h) - float(lam_a)
    actual_diff = int(home_goals) - int(away_goals)
    residual = actual_diff - expected_diff
    return max(min(residual / denom, 0.15), -0.05)


def actualiser_hia_equipes():
    """
    Estime un HIA spécifique par équipe (matchs à domicile) depuis l'historique
    ligue, avec shrinkage empirique vers le HIA global MLE (James-Stein simplifié).
    """
    global _hia_equipe_logue
    if not NHL_HIA_PAR_EQUIPE_ACTIF:
        return

    games = lire_league_calib_meta().get("games", {})
    if not games:
        return

    hia_global = lire_hia_dynamique()
    accum = {}

    for data in games.values():
        home = data.get("home")
        if not home:
            continue
        try:
            lam_h = float(data["lambda_domicile_calcule"])
            lam_a = float(data["lambda_exterieur_calcule"])
            h_goals = int(data["vrai_score_domicile"])
            a_goals = int(data["vrai_score_exterieur"])
        except (KeyError, TypeError, ValueError):
            continue

        brut = _estimer_hia_brut_match(lam_h, lam_a, h_goals, a_goals)
        w = _poids_recency_mle({"date": data.get("date")})

        bucket = accum.setdefault(home, {"sum_w": 0.0, "sum_whia": 0.0, "n": 0})
        bucket["sum_w"] += w
        bucket["sum_whia"] += w * brut
        bucket["n"] += 1

    teams_hia = {}
    for team, stats in accum.items():
        n = stats["n"]
        if n < NHL_HIA_TEAM_MIN_GAMES:
            continue
        hia_raw = stats["sum_whia"] / stats["sum_w"] if stats["sum_w"] > 0 else hia_global
        w_shrink = min(1.0, n / NHL_HIA_TEAM_GP_PLEIN) if NHL_HIA_TEAM_GP_PLEIN > 0 else 1.0
        hia_shrunk = (1.0 - w_shrink) * hia_global + w_shrink * hia_raw
        hia_shrunk = round(min(max(hia_shrunk, 0.0), 0.12), 4)
        teams_hia[team] = {
            "hia": hia_shrunk,
            "hia_raw": round(hia_raw, 4),
            "home_games": n,
            "shrink_weight": round(w_shrink, 3),
        }

    meta = {
        "hia_global": round(hia_global, 4),
        "teams": teams_hia,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(HIA_TEAM_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not _hia_equipe_logue:
        _hia_equipe_logue = True
        log_nhl(
            f"🏠 HIA par équipe actif — shrinkage vers global {hia_global:.1%}, "
            f"confiance pleine à {NHL_HIA_TEAM_GP_PLEIN:.0f} matchs domicile "
            f"(min {NHL_HIA_TEAM_MIN_GAMES})"
        )
    if teams_hia:
        extremes = sorted(teams_hia.items(), key=lambda kv: kv[1]["hia"])
        bas, haut = extremes[0], extremes[-1]
        log_nhl(
            f"🏠 HIA équipes mis à jour — {len(teams_hia)} indexées "
            f"(min {bas[0]} {bas[1]['hia']:.1%}, max {haut[0]} {haut[1]['hia']:.1%})"
        )


def lire_hia_equipe(home_team):
    """HIA effectif pour une équipe à domicile (shrinkage vers moyenne ligue MLE)."""
    if not NHL_HIA_PAR_EQUIPE_ACTIF:
        return lire_hia_dynamique()
    team_data = lire_hia_equipes_meta().get("teams", {}).get(home_team)
    if not team_data:
        return lire_hia_dynamique()
    return float(team_data["hia"])


def lire_league_calib_meta():
    if os.path.exists(LEAGUE_CALIB_FILE):
        try:
            with open(LEAGUE_CALIB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"games": {}, "scanned_game_ids": []}


def _lambda_referentiel_match(teams_data, home_team, away_team):
    """
    Lambdas 'attendues' pour un match à partir des stats d'équipe courantes,
    sans ajustement spécifique (gardien/fatigue/arbitre) — sert de proxy
    rétrospectif pour élargir l'échantillon de calibration MLE (rho/HIA/
    empty-net) à tous les matchs ligue, pas seulement les paris placés.
    """
    hia = lire_hia_dynamique()
    resultat = calculate_master_odds_v4(
        teams_data, home_team, away_team,
        home_gsax=0.0, away_gsax=0.0,
        hia=hia,
    )
    if not resultat:
        return None, None, None
    return resultat["lam_home"], resultat["lam_away"], hia


def actualiser_historique_ligue(teams_data):
    """
    Scan incrémental de tous les matchs NHL terminés (pas seulement ceux
    pariés par le bot) pour alimenter la calibration MLE rho/HIA/empty-net/
    tie avec un échantillon bien plus large et rapide à constituer.
    Lambdas approximées via les stats d'équipe courantes (proxy rétrospectif) ;
    les paris réellement placés (lambdas pré-match exactes) restent
    prioritaires en cas de doublon lors de la fusion (voir entrainer_ia_dixon_coles).
    """
    if not NHL_LIGUE_CALIB_ACTIF or not teams_data:
        return

    meta = lire_league_calib_meta()
    games_db = meta.setdefault("games", {})
    scanned = set(meta.get("scanned_game_ids", []))
    etats_finaux = {"FINAL", "OFF", "OFFICIAL"}
    headers = {"User-Agent": "Mozilla/5.0"}

    bootstrap = len(games_db) < NHL_LIGUE_CALIB_MIN_MATCHS
    scan_jours = NHL_LIGUE_CALIB_LOOKBACK_JOURS if bootstrap else NHL_LIGUE_CALIB_SCAN_JOURS
    nouveaux = 0

    for offset in range(1, scan_jours + 1):
        date_str = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
        try:
            response = requests.get(
                f"https://api-web.nhle.com/v1/score/{date_str}",
                headers=headers,
                timeout=10,
            )
            if response.status_code != 200:
                continue
            for game in response.json().get("games", []):
                if game.get("gameState") not in etats_finaux:
                    continue
                gid = str(game["id"])
                if gid in scanned:
                    continue
                home_info = game.get("homeTeam", {}) or {}
                away_info = game.get("awayTeam", {}) or {}
                home_abbrev, away_abbrev = home_info.get("abbrev"), away_info.get("abbrev")
                home_score, away_score = home_info.get("score"), away_info.get("score")
                if not home_abbrev or not away_abbrev or home_score is None or away_score is None:
                    continue
                lam_h, lam_a, hia_ref = _lambda_referentiel_match(teams_data, home_abbrev, away_abbrev)
                if lam_h is None:
                    scanned.add(gid)
                    continue
                games_db[gid] = {
                    "date": date_str,
                    "home": home_abbrev, "away": away_abbrev,
                    "vrai_score_domicile": int(home_score),
                    "vrai_score_exterieur": int(away_score),
                    "lambda_domicile_calcule": lam_h,
                    "lambda_exterieur_calcule": lam_a,
                    "hia_ref": hia_ref,
                }
                scanned.add(gid)
                nouveaux += 1
        except Exception:
            continue

    if nouveaux <= 0:
        return

    if len(games_db) > NHL_LIGUE_CALIB_MAX_GAMES:
        plus_recents = sorted(games_db.items(), key=lambda kv: kv[1].get("date", ""))[-NHL_LIGUE_CALIB_MAX_GAMES:]
        games_db = dict(plus_recents)

    meta = {
        "games": games_db,
        "scanned_game_ids": list(scanned)[-4000:],
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(LEAGUE_CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log_nhl(
        f"📚 Historique ligue mis à jour — +{nouveaux} match(s), "
        f"{len(games_db)} matchs indexés pour calibration MLE"
    )


def entrainer_ia_dixon_coles():
    historique_unique = {}

    if NHL_LIGUE_CALIB_ACTIF:
        for gid, data in lire_league_calib_meta().get("games", {}).items():
            historique_unique[gid] = {
                "date": data.get("date"),
                "vrai_score_domicile": data["vrai_score_domicile"],
                "vrai_score_exterieur": data["vrai_score_exterieur"],
                "lambda_domicile_calcule": data["lambda_domicile_calcule"],
                "lambda_exterieur_calcule": data["lambda_exterieur_calcule"],
                "hia_ref": data.get("hia_ref", HIA_REF_CALIBRATION),
            }
    nb_ligue = len(historique_unique)

    if os.path.exists(FICHIER_JOURNAL):
        with open(FICHIER_JOURNAL, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["Statut"] in ["GAGNÉ", "PERDU"]:
                    game_id_brut = row["ID_Match"].split("_")[0]
                    try:
                        hia_ref = float(row.get("Hia", HIA_REF_CALIBRATION))
                    except (ValueError, TypeError):
                        hia_ref = HIA_REF_CALIBRATION
                    # Lambdas pré-match réelles (calculées au moment du pari) :
                    # priment sur le proxy rétrospectif du scan ligue.
                    historique_unique[game_id_brut] = {
                        "date": row.get("Date", "").split()[0] or None,
                        "vrai_score_domicile": int(row["Score_Dom"]),
                        "vrai_score_exterieur": int(row["Score_Ext"]),
                        "lambda_domicile_calcule": float(row["Lam_Dom"]),
                        "lambda_exterieur_calcule": float(row["Lam_Ext"]),
                        "hia_ref": hia_ref,
                    }

    dataset = list(historique_unique.values())
    nb = len(dataset)
    if nb < NHL_RHO_MIN_MATCHS:
        return

    meta = lire_rho_meta()
    nb_precedent = int(meta.get("nb_matchs", 0))
    nouveaux = nb - nb_precedent
    if nb_precedent > 0 and nouveaux < NHL_RHO_INTERVAL_MATCHS:
        return

    nouveau_rho, nouveau_hia = optimiser_rho_et_hia_saison(dataset)
    meta_precedent = lire_rho_meta()
    meta = {
        "rho": nouveau_rho,
        "hia": nouveau_hia,
        "prob_tie": meta_precedent.get("prob_tie", 0.12),
        "prob_en": meta_precedent.get("prob_en", 0.22),
        "nb_matchs": nb,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Empty-net/OT-tie : 2 paramètres de plus, besoin d'un échantillon plus large
    if nb >= NHL_EMPTY_NET_MIN_MATCHS:
        nouveau_prob_tie, nouveau_prob_en = optimiser_empty_net_ot(dataset, nouveau_rho, nouveau_hia)
        meta["prob_tie"], meta["prob_en"] = nouveau_prob_tie, nouveau_prob_en

    with open(RHO_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open("rho_optimal.txt", "w", encoding="utf-8") as f:
        f.write(str(nouveau_rho))
    nb_paris = nb - nb_ligue
    log_nhl(
        f"💾 Rho+HIA+EmptyNet sauvegardés ({nb} matchs = {nb_ligue} ligue + {nb_paris} paris, "
        f"+{nouveaux} depuis dernier run)"
    )


def rapport_calibration_journal(chemin=None, min_paris=5):
    """Diagnostic Brier + calibration sur le journal (sans impact live)."""
    from utils import formater_rapport_calibration_texte, preparer_calibration_journal

    chemin = chemin or FICHIER_JOURNAL
    if not os.path.exists(chemin):
        print(f"Journal introuvable : {chemin}")
        return
    migrer_journal_si_besoin()
    with open(chemin, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Journal vide.")
        return
    import pandas as pd

    df = pd.DataFrame(rows)
    df_cal = preparer_calibration_journal(df)
    print(formater_rapport_calibration_texte(df_cal, min_paris=min_paris))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("--calibration", "--calib"):
        rapport_calibration_journal()
        sys.exit(0)

    manquants = []
    if not ODDS_API_KEY:
        manquants.append("API_ODDS_KEY")
    if not TELEGRAM_TOKEN:
        manquants.append("TELEGRAM_TOKEN")
    if manquants:
        log_nhl(
            f"⚠️ Variables manquantes dans {env_files_hint('nhl')} : {', '.join(manquants)}",
            level="warning",
        )
        log_nhl("   Le bot peut tourner en veille, mais ne pourra pas parier sans clé Odds API.")
    mode_label = "DRY RUN" if NHL_DRY_RUN else "LIVE"
    cap_label = f"cap {NHL_MISE_MAX_PCT}%" if NHL_MISE_MAX_PCT > 0 else "Kelly pur"
    log_nhl(
        f"🏒 Sniper NHL Oméga — {mode_label} | saison {NHL_SEASON} | "
        f"bankroll {BANKROLL_INITIALE} € | {cap_label}"
    )
    run_sniper()