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
from dotenv import load_dotenv

load_dotenv("identifiants_différent_api.env")
load_dotenv()

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
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
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
KELLY_FRACTION = float(os.environ.get("NHL_KELLY_FRACTION", "0.25"))
KELLY_FRACTION_GARDIEN_INCERTAIN = float(os.environ.get("NHL_KELLY_GARDIEN", "0.125"))
NHL_MISE_MAX_PCT = float(os.environ.get("NHL_MISE_MAX_PCT", "2"))
NHL_DRY_RUN = _env_bool("NHL_DRY_RUN", False)
NHL_PARIS_JOUR_MAX = int(os.environ.get("NHL_PARIS_JOUR_MAX", "0"))
NHL_ODDS_QUOTA_ALERT = int(os.environ.get("NHL_ODDS_QUOTA_ALERT", "100"))
NHL_BLEND_GP_PLEIN = float(os.environ.get("NHL_BLEND_GP_PLEIN", "20"))
NHL_PP_PK_SHRINK_GP = float(os.environ.get("NHL_PP_PK_SHRINK_GP", "20"))
NHL_GSAX_EWMA_SPAN = float(os.environ.get("NHL_GSAX_EWMA_SPAN", "8"))
NHL_GSAX_EWMA_GP_PLEIN = float(os.environ.get("NHL_GSAX_EWMA_GP_PLEIN", "8"))
NHL_RHO_MIN_MATCHS = int(os.environ.get("NHL_RHO_MIN_MATCHS", "30"))
NHL_RHO_INTERVAL_MATCHS = int(os.environ.get("NHL_RHO_INTERVAL_MATCHS", "20"))
NHL_OT_HOME_ADVANTAGE = float(os.environ.get("NHL_OT_HOME_ADVANTAGE", "0.52"))
NHL_HIA_DEFAULT = float(os.environ.get("NHL_HIA_DEFAULT", "0.05"))
HIA_REF_CALIBRATION = 0.05  # HIA utilisé lors du calcul des lambdas historiques du journal
NHL_MARCHES_ACTIFS = {m.strip().upper() for m in os.environ.get("NHL_MARCHES_ACTIFS", "ML,PL,OU").split(",") if m.strip()}
RHO_META_FILE = "rho_calibrage_meta.json"
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
_gsax_ewma_logue = False
FLOAT_TOL = 0.01
# Part du temps de jeu gardien (~54 min) pour convertir GSAx/60 → impact/match
GSAX_MINUTES_PAR_MATCH = float(os.environ.get("NHL_GSAX_MINUTES", "54.0"))
NHL_SCAN_HEURES_AVANCE = float(os.environ.get("NHL_SCAN_HEURES_AVANCE", "18"))
# Minutes avant le puck drop où l'on arrête de chercher de nouveaux paris
NHL_MINUTES_AVANT_PUCK = float(os.environ.get("NHL_MINUTES_AVANT_PUCK", "5"))

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

JOURNAL_COLONNES = [
    "Date", "ID_Match", "Visiteur", "Local", "Pari",
    "Vraie_Cote_Bot", "Cote_Prise", "Cote_CLV",
    "Lam_Ext", "Lam_Dom", "Score_Ext", "Score_Dom",
    "Edge(%)", "Risque(%)", "Mise_€", "Statut", "P&L",
    "Gardien_Ext", "Gardien_Dom", "Gardiens_Confirmes", "B2B_Home", "B2B_Away", "Rho",
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


def _fetch_moneypuck_game_by_game(kind, season):
    """Télécharge un CSV MoneyPuck gameByGame (saison N, puis N-1 si échec)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    for s in (season, season - 1):
        url = f"https://moneypuck.com/moneypuck/playerData/gameByGame/{s}/regular/{kind}.csv"
        try:
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code == 200 and response.text.strip():
                return response.text, s
            print(f"⚠️ MoneyPuck gameByGame {kind} saison {s} : HTTP {response.status_code}")
        except Exception as e:
            print(f"⚠️ MoneyPuck gameByGame {kind} saison {s} : {e}")
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
            }
        gp = max(float(row.get("games_played", 1)), 1)
        if sit == "5on5":
            teams_dict[team]["xGF_per_game"] = round(float(row.get(col_for, 0) or 0) / gp, 3)
            teams_dict[team]["xGA_per_game"] = round(float(row.get(col_against, 0) or 0) / gp, 3)
            teams_dict[team]["games_played"] = int(gp)
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
        for key in ("xGF_per_game", "xGA_per_game", "xGF_PP", "xGA_PK"):
            v_n = team.get(key, 0.0)
            v_n1 = prev.get(key, v_n)
            merged[key] = round(poids_courant * v_n + (1 - poids_courant) * v_n1, 3)
        blended.append(merged)
    return blended


def get_team_stats(season=None, blend=True):
    if season is None:
        season = NHL_SEASON
    """Aspire les xG score/venue-adjusted (5v5 + PP/PK), blend N/N-1 en début de saison."""
    try:
        texte, saison_utilisee = _fetch_moneypuck_csv("teams", season)
        if not texte:
            return []
        if saison_utilisee != season:
            log_nhl(f"ℹ️ MoneyPuck équipes : repli sur la saison {saison_utilisee}")
        teams = _shrink_special_teams(_parser_team_stats_csv(texte))
        if not teams:
            return []
        if not blend or NHL_BLEND_GP_PLEIN <= 0:
            return teams

        gp_values = [t["games_played"] for t in teams if t.get("games_played", 0) > 0]
        gp_moyen = sum(gp_values) / len(gp_values) if gp_values else NHL_BLEND_GP_PLEIN
        poids_n = min(1.0, gp_moyen / NHL_BLEND_GP_PLEIN)
        if poids_n >= 1.0:
            return teams

        texte_n1, _ = _fetch_moneypuck_csv("teams", season - 1)
        if not texte_n1:
            return teams
        teams_n1 = _shrink_special_teams(_parser_team_stats_csv(texte_n1))
        if not teams_n1:
            return teams

        log_nhl(
            f"🔀 Blend MoneyPuck : {round(poids_n * 100)}% saison {season} / "
            f"{round((1 - poids_n) * 100)}% {season - 1} (GP moyen {gp_moyen:.1f})"
        )
        return _blend_team_stats(teams, teams_n1, poids_n)
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


def _ewma_series(values, span):
    """EWMA pandas-compatible (span = fenêtre effective, ordre chronologique)."""
    if not values:
        return 0.0
    alpha = 2.0 / (float(span) + 1.0)
    result = values[0]
    for val in values[1:]:
        result = alpha * val + (1.0 - alpha) * result
    return result


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


def _parser_goalie_games_by_game(texte):
    """Indexe les matchs gardiens (situation=all) par nom, triés chronologiquement."""
    by_name = {}
    csv_reader = csv.DictReader(StringIO(texte))
    for row in csv_reader:
        if row.get("situation") != "all":
            continue
        name = row.get("name") or row.get("playerName") or ""
        if not name:
            continue
        if _gsax_per_60_de_ligne(row) is None:
            continue
        by_name.setdefault(name, []).append(row)

    def _cle_tri(match_row):
        for col in ("gameDate", "game_date", "date"):
            if match_row.get(col):
                return match_row[col]
        return match_row.get("gameId", "")

    for name in by_name:
        by_name[name].sort(key=_cle_tri)
    return by_name


def _calculer_ewma_gsax_per_60(game_rows, span):
    """GSAx/60 EWMA sur les derniers matchs (plus récent = poids max)."""
    vals = []
    for row in game_rows:
        gsax = _gsax_per_60_de_ligne(row)
        if gsax is not None:
            vals.append(round(gsax, 4))
    if not vals:
        return None, 0
    fenetre = int(span) if span > 0 else len(vals)
    recent = vals[-fenetre:]
    return round(_ewma_series(recent, span), 3), len(recent)


def _index_goalie_games(games_by_name):
    idx = {}
    for name, games in games_by_name.items():
        idx[name] = games
        idx[normaliser_nom_joueur(name)] = games
    return idx


def _appliquer_ewma_gsax_goalies(goalies, games_by_name):
    """Blend GSAx saison + EWMA récent : w = min(1, n_matchs / GP_plein)."""
    global _gsax_ewma_logue
    if NHL_GSAX_EWMA_SPAN <= 0 or not goalies or not games_by_name:
        return goalies

    idx = _index_goalie_games(games_by_name)
    nb_blend = 0
    for gardien in goalies:
        name = gardien["name"]
        games = idx.get(name) or idx.get(normaliser_nom_joueur(name))
        if not games:
            continue
        gsax_ewma, n_jeux = _calculer_ewma_gsax_per_60(games, NHL_GSAX_EWMA_SPAN)
        if gsax_ewma is None or n_jeux == 0:
            continue
        gsax_saison = gardien.get("gsax_saison", gardien["gsax_per_60"])
        w = min(1.0, n_jeux / NHL_GSAX_EWMA_GP_PLEIN) if NHL_GSAX_EWMA_GP_PLEIN > 0 else 1.0
        gardien["gsax_ewma"] = gsax_ewma
        gardien["gsax_per_60"] = round(w * gsax_ewma + (1.0 - w) * gsax_saison, 3)
        if w < 1.0:
            nb_blend += 1

    if not _gsax_ewma_logue:
        _gsax_ewma_logue = True
        log_nhl(
            f"🥅 GSAx EWMA actif — span {NHL_GSAX_EWMA_SPAN:.0f} matchs, "
            f"confiance pleine à {NHL_GSAX_EWMA_GP_PLEIN:.0f} GP "
            f"({nb_blend}/{len(goalies)} gardiens partiellement blendés)"
        )
    return goalies


def get_goalie_stats(season=None):
    if season is None:
        season = NHL_SEASON
    """Aspire GSAx gardiens (saison + blend EWMA forme récente si activé)."""
    try:
        texte, saison_utilisee = _fetch_moneypuck_csv("goalies", season)
        if not texte:
            return []
        if saison_utilisee != season:
            print(f"ℹ️ MoneyPuck gardiens : repli sur la saison {saison_utilisee}")
        goalies = _parser_goalie_season_csv(texte)
        if not goalies:
            return []

        if NHL_GSAX_EWMA_SPAN > 0:
            texte_gbg, saison_gbg = _fetch_moneypuck_game_by_game("goalies", season)
            if texte_gbg:
                if saison_gbg != season:
                    log_nhl(f"ℹ️ MoneyPuck gardiens game-by-game : repli saison {saison_gbg}")
                games_map = _parser_goalie_games_by_game(texte_gbg)
                goalies = _appliquer_ewma_gsax_goalies(goalies, games_map)
            else:
                log_nhl("ℹ️ GSAx EWMA : game-by-game indisponible — saison seule", level="warning")

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

def get_nhl_momentum():
    """Calcule le différentiel de forme L10 (Momentum)."""
    url = "https://api-web.nhle.com/v1/standings/now"
    headers = {'User-Agent': 'Mozilla/5.0'}
    momentum_dict = {}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return {}
        data = response.json()
        for team in data.get("standings", []):
            abbrev = team.get("teamAbbrev", {}).get("default")
            gp = team.get("gamesPlayed", 1)
            if gp < 10:
                momentum_dict[abbrev] = 0.0
                continue
            season_pts_pct = team.get("points", 0) / (gp * 2.0)
            l10_wins = team.get("l10Wins", 0)
            l10_ot_losses = team.get("l10OtLosses", 0)
            l10_pts_pct = ((l10_wins * 2) + l10_ot_losses) / 20.0
            delta_form = l10_pts_pct - season_pts_pct
            momentum_dict[abbrev] = round(delta_form * 0.15, 3) # Modérateur conservateur
        return momentum_dict
    except: return {}

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
    Calcule le modificateur de fatigue basé sur le décalage horaire et le B2B.
    Retourne (modificateur_attaque, modificateur_defense).
    1.0 = Forme normale. < 1.0 = Baisse d'attaque. > 1.0 = Hausse des erreurs défensives.
    """
    # Pénalité de base pour un Back-to-Back classique (ce qu'on avait avant)
    mod_atk, mod_def = (0.95, 1.10) if is_b2b else (1.0, 1.0)

    # Si l'équipe joue à domicile, on considère qu'elle est "acclimatée" à son fuseau
    if is_home:
        return mod_atk, mod_def

    tz_team = NHL_TIMEZONES.get(team_abbrev, 0)
    tz_opp = NHL_TIMEZONES.get(opponent_abbrev, 0)

    # Calcul du décalage horaire absolu traversé
    decalage = abs(tz_team - tz_opp)

    if decalage >= 2:
        # Jet Lag modéré à sévère (ex: Est vers Montagne ou Pacifique)
        if is_b2b:
            # Le pire scénario possible : B2B + Jet Lag (Schedule Loss)
            mod_atk *= 0.93  # L'attaque s'effondre
            mod_def *= 1.15  # La défense craque complètement (+15% de xGA)
        else:
            # Juste le Jet Lag sans B2B
            mod_atk *= 0.98
            mod_def *= 1.04

    elif decalage == 1 and is_b2b:
        # Petit décalage mais en B2B (ex: Chicago joue à New York le lendemain)
        mod_atk *= 0.97
        mod_def *= 1.05

    return round(mod_atk, 3), round(mod_def, 3)

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
        ll += math.log(max(p_h * p_a * tau, 1e-10))
    return -ll


def optimiser_rho_et_hia_saison(historique_matchs):
    """Calibre rho et HIA conjointement par MLE sur l'historique du journal."""
    meta = lire_rho_meta()
    x0 = [float(meta.get("rho", -0.12)), float(meta.get("hia", NHL_HIA_DEFAULT))]
    log_nhl(f"🔬 Calibrage MLE rho+HIA sur {len(historique_matchs)} matchs (init {x0})...")
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

def calculate_master_odds_v4(
    teams_data, home_team, away_team, home_gsax, away_gsax,
    mom_home=0.0, mom_away=0.0, home_is_b2b=False, away_is_b2b=False,
    rho=-0.12, hia=None,
):
    if hia is None:
        hia = lire_hia_dynamique()
    hia = min(max(float(hia), 0.0), 0.10)
    league_avg_5v5 = sum(t['xGF_per_game'] for t in teams_data) / len(teams_data)
    league_avg_pp = sum(t['xGF_PP'] for t in teams_data) / len(teams_data)
    safe_league_pp = max(league_avg_pp, 0.01)

    home = next((t for t in teams_data if t['team'] == home_team), None)
    away = next((t for t in teams_data if t['team'] == away_team), None)
    if not home or not away: return None

    # HIA calibré + Momentum
    home_xgf = home['xGF_per_game'] * (1.0 + hia) * (1.0 + mom_home)
    home_xga = home['xGA_per_game'] * (1.0 - hia) * (1.0 - mom_home)
    away_xgf = away['xGF_per_game'] * (1.0 + mom_away)
    away_xga = away['xGA_per_game'] * (1.0 - mom_away)

    home_pp_att = home['xGF_PP'] * (1.0 + mom_home)
    home_pk_def = home['xGA_PK'] * (1.0 - mom_home)
    away_pp_att = away['xGF_PP'] * (1.0 + mom_away)
    away_pk_def = away['xGA_PK'] * (1.0 - mom_away)

    # Matrice de Fatigue Circadienne et Kilométrique
    home_fatigue_atk, home_fatigue_def = calculate_circadian_fatigue(home_team, away_team, home_is_b2b, is_home=True)
    away_fatigue_atk, away_fatigue_def = calculate_circadian_fatigue(away_team, home_team, away_is_b2b, is_home=False)

    home_xgf *= home_fatigue_atk
    home_xga *= home_fatigue_def
    home_pp_att *= home_fatigue_atk
    home_pk_def *= home_fatigue_def

    away_xgf *= away_fatigue_atk
    away_xga *= away_fatigue_def
    away_pp_att *= away_fatigue_atk
    away_pk_def *= away_fatigue_def

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

    # Correction Surdispersion (Empty Net)
    matrice_finale = [[0.0]*12 for _ in range(12)]
    prob_tie, prob_en = 0.12, 0.22

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


def get_real_live_odds(home_team_abbrev, away_team_abbrev, odds_cache=None):
    """Aspire Moneyline et tous les Cuts Over/Under."""
    cotes = get_odds_for_match(home_team_abbrev, away_team_abbrev, odds_cache)
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


def _construire_candidats_pari(m, cotes_vraies, cotes_bookmaker, cotes_puckline, bankroll, gardiens_verrouilles):
    """Évalue ML / PL / O-U selon NHL_MARCHES_ACTIFS."""
    candidats = []

    if "ML" in NHL_MARCHES_ACTIFS:
        inv = calculate_kelly(
            cotes_vraies["prob_1"], cotes_bookmaker["cote_1"],
            bankroll, gardiens_confirmes=gardiens_verrouilles,
        )
        if inv:
            candidats.append({
                "type": f"Victoire {m['home_team']}", "inv": inv,
                "cote_book": cotes_bookmaker["cote_1"], "cote_vraie": cotes_vraies["cote_1"],
                "marche": "ML",
            })
        inv = calculate_kelly(
            cotes_vraies["prob_2"], cotes_bookmaker["cote_2"],
            bankroll, gardiens_confirmes=gardiens_verrouilles,
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
                bankroll, gardiens_confirmes=gardiens_verrouilles,
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
                bankroll, gardiens_confirmes=gardiens_verrouilles,
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
                inv = calculate_kelly(prob_over, prices["Over"], bankroll, gardiens_confirmes=gardiens_verrouilles)
                if inv:
                    candidats.append({
                        "type": f"OVER {cut_arrondi}", "inv": inv,
                        "cote_book": prices["Over"],
                        "cote_vraie": round(1 / max(prob_over, 0.001), 2),
                        "marche": "OU",
                    })
            if "Under" in prices and prob_under is not None:
                inv = calculate_kelly(prob_under, prices["Under"], bankroll, gardiens_confirmes=gardiens_verrouilles)
                if inv:
                    candidats.append({
                        "type": f"UNDER {cut_arrondi}", "inv": inv,
                        "cote_book": prices["Under"],
                        "cote_vraie": round(1 / max(prob_under, 0.001), 2),
                        "marche": "OU",
                    })

    return candidats


def calculate_kelly(true_prob, book_odds, bankroll, gardiens_confirmes=True):
    if book_odds <= 1.0 or true_prob <= 0.01 or true_prob >= 0.99:
        return None
    edge = true_prob - (1 / book_odds)
    if edge <= EDGE_MINIMUM:
        return None
    b = book_odds - 1.0
    fraction_kelly = KELLY_FRACTION if gardiens_confirmes else KELLY_FRACTION_GARDIEN_INCERTAIN
    if not gardiens_confirmes:
        log_nhl("🛡️ SÉCURITÉ GARDIEN : Alignement non confirmé à 100%. Mise divisée par 2.")
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
        f"rho+HIA recalibrés tous les {NHL_RHO_INTERVAL_MATCHS} matchs (min {NHL_RHO_MIN_MATCHS})"
    )
    if NHL_BLEND_GP_PLEIN > 0:
        log_nhl(f"🔀 Blend MoneyPuck N/N-1 jusqu'à {NHL_BLEND_GP_PLEIN:.0f} GP moyen/ligue")
    if NHL_PP_PK_SHRINK_GP > 0:
        log_nhl(f"📉 Shrinkage PP/PK vers moyenne ligue jusqu'à {NHL_PP_PK_SHRINK_GP:.0f} GP/équipe")
    if NHL_GSAX_EWMA_SPAN > 0:
        log_nhl(
            f"🥅 GSAx gardiens : blend EWMA ({NHL_GSAX_EWMA_SPAN:.0f} derniers matchs) "
            f"+ saison (plein à {NHL_GSAX_EWMA_GP_PLEIN:.0f} GP)"
        )
    migrer_journal_si_besoin()

    while True:
        try:
            lancer_la_balayeuse()
            entrainer_ia_dixon_coles()

            nb_attente = compter_paris_en_attente()
            log_nhl(f"🕵️ Tracking CLV ({nb_attente} pari(s) en attente)...")
            traquer_et_actualiser_clv()

            bankroll_actuelle = calculer_bankroll_dynamique(BANKROLL_INITIALE)
            log_nhl(f"💰 Capital Dynamique Disponible : {bankroll_actuelle} €")
            # ------------------------------------

            rho_actuel = lire_rho_dynamique()
            hia_actuel = lire_hia_dynamique()
            log_nhl(f"🧠 Configuration Mathématique : Rho = {rho_actuel} | HIA = {hia_actuel:.1%}")

            log_nhl("📡 Synchronisation bases de données...")
            teams, goalies, stars_vip, momentum_data = (
                get_team_stats(), get_goalie_stats(), get_stars_impact(), get_nhl_momentum()
            )
            equipes_en_b2b_hier = get_teams_played_yesterday()
            odds_cache = fetch_all_pinnacle_odds()

            if not teams or not goalies:
                log_nhl(
                    f"⚠️ Données MoneyPuck indisponibles (saison {NHL_SEASON}). Nouvelle tentative dans 5 min...",
                    level="warning",
                )
                time.sleep(300)
                continue

            matchs = get_nhl_games_today()
            if not matchs:
                log_nhl("🏒 Aucun match NHL éligible dans la fenêtre de scan — veille active.")
            else:
                log_nhl(f"🏒 {len(matchs)} match(s) dans la fenêtre de scan.")
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

                mom_dom, mom_ext = momentum_data.get(m['home_team'], 0.0), momentum_data.get(m['away_team'], 0.0)

                cotes_vraies = calculate_master_odds_v4(
                    teams_match, m['home_team'], m['away_team'], gsax_dom, gsax_ext,
                    mom_home=mom_dom, mom_away=mom_ext,
                    home_is_b2b=home_b2b, away_is_b2b=away_b2b,
                    rho=rho_actuel, hia=hia_actuel,
                )
                cotes_bookmaker = get_real_live_odds(
                    m['home_team'], m['away_team'], odds_cache, log_si_absent=True,
                )
                if not cotes_bookmaker:
                    log_nhl(f"⚠️ Skip cotes Pinnacle — {m['away_team']} @ {m['home_team']}")
                    continue
                cotes_puckline = get_real_live_odds_puckline(m['home_team'], m['away_team'], odds_cache)

                if cotes_vraies and cotes_bookmaker:
                    candidats = _construire_candidats_pari(
                        m, cotes_vraies, cotes_bookmaker, cotes_puckline,
                        bankroll_actuelle, gardiens_verrouilles,
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
                                b2b_home=home_b2b, b2b_away=away_b2b, rho=rho_actuel,
                            )
                            enregistrer_notification(id_match)
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
    default = {"rho": -0.12, "hia": NHL_HIA_DEFAULT, "nb_matchs": 0}
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


def entrainer_ia_dixon_coles():
    if not os.path.exists(FICHIER_JOURNAL):
        return
    historique_unique = {}
    with open(FICHIER_JOURNAL, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Statut"] in ["GAGNÉ", "PERDU"]:
                if row["ID_Match"] not in historique_unique:
                    historique_unique[row["ID_Match"]] = {
                        "vrai_score_domicile": int(row["Score_Dom"]),
                        "vrai_score_exterieur": int(row["Score_Ext"]),
                        "lambda_domicile_calcule": float(row["Lam_Dom"]),
                        "lambda_exterieur_calcule": float(row["Lam_Ext"]),
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
    meta = {
        "rho": nouveau_rho,
        "hia": nouveau_hia,
        "nb_matchs": nb,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(RHO_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open("rho_optimal.txt", "w", encoding="utf-8") as f:
        f.write(str(nouveau_rho))
    log_nhl(f"💾 Rho+HIA sauvegardés ({nb} matchs, +{nouveaux} depuis dernier run)")


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
        manquants.append("ODDS_API_KEY")
    if not TELEGRAM_TOKEN:
        manquants.append("TELEGRAM_TOKEN")
    if manquants:
        log_nhl(
            f"⚠️ Variables manquantes dans identifiants_différent_api.env : {', '.join(manquants)}",
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