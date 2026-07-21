"""Sniper NBA — bot value bets moneyline/spread/total (paper trading).

Ce module contient :

- La **couche de collecte + cache PIT** (point-in-time, sans look-ahead) :
  récupération bulk (1 appel/saison/measure_type) des box scores et stats
  avancées par match via `nba_api` (stats.nba.com) : ``Base`` (score, W/L),
  ``Advanced`` (OFF_RATING/DEF_RATING/PACE/POSS) et ``Four Factors``
  (EFG%/FTA_RATE/TOV%/OREB% + équivalents adverses) ; fusion en un
  enregistrement par (match, équipe) et index PIT walk-forward (cumul
  pondéré possessions, strictement avant chaque match, pattern identique au
  PIT MoneyPuck de `nhl_sniper_omega.py`) ; calendrier saison
  (``ScheduleLeagueV2``) et scoreboard du jour (module live cdn.nba.com) ;
  cache disque JSON (``data/nba/``), permanent (saisons terminées) ou TTL
  (``NBA_PIT_CACHE_JOURS``, saison en cours).
- Le **moteur mathématique** (ratings ajustés Off/Def + Pace avec
  décroissance temporelle + shrinkage bayésien vers la moyenne ligue + blend
  saison N/N-1, ajustements home court/repos/B2B/altitude, distribution
  bivariée Normale sur la marge et le total, probabilités de marché
  moneyline/spread/total avec gestion du push sur lignes entières) — voir
  `calculer_probabilites_match_nba()`. Paramètres calibrables dans
  `nba_params.py`.

Intégration Odds API (basketball_nba, Pinnacle) + dévigorage Shin 2-way
(`odds_devig.py`) + shrink modèle↔marché, edge/Kelly fractionnel + cap %
bankroll, journal CSV + alertes Telegram (paper trading `NBA_DRY_RUN`), et
boucle live `run_sniper_nba()` — voir sections 7 à 9 ci-dessous (mirroring
`nhl_sniper_omega.py`, simplifié : pas de line-movement/steam, pas de Kelly
dynamique BSS/CLV, pas d'absences stars en v1, documenté comme limite
connue — voir le plan `sniper_nba_value_bets`).
"""
import copy
import csv
import json
import logging
import math
import os
import re
import time
import traceback
from datetime import datetime, timezone

import requests
from nba_api.stats.endpoints import scheduleleaguev2, teamgamelogs
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard

import nba_params
from config_env import load_project_env
from odds_devig import proba_no_vig_shin_2way

load_project_env("nba")

# ==========================================
# ⚙️ CONFIGURATION GLOBALE
# ==========================================
logging.basicConfig(
    filename="nba_sniper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)


def _env_bool(key, default=False):
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes", "on")


def log_nba(msg, level="info"):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))
    getattr(logging, level, logging.info)(msg)


# Saison en cours, convention "année de fin" (ex. 2027 = saison 2026-27),
# identique à NHL_SEASON pour rester cohérent entre bots du repo.
NBA_SEASON = int(os.environ.get("NBA_SEASON", "2027"))
# TTL cache disque pour la saison en cours (saisons terminées = cache permanent)
NBA_PIT_CACHE_JOURS = int(os.environ.get("NBA_PIT_CACHE_JOURS", "1"))
# Retry/backoff stats.nba.com : timeouts transitoires observés sur
# ScoreboardV3/ScheduleLeagueV2 en exploration (jamais de rate-limit dur sur
# les appels bulk teamgamelogs/leaguegamefinder, 1 appel/saison).
NBA_API_MAX_RETRIES = int(os.environ.get("NBA_API_MAX_RETRIES", "3"))
NBA_API_TIMEOUT = int(os.environ.get("NBA_API_TIMEOUT", "30"))
NBA_API_RETRY_BACKOFF = float(os.environ.get("NBA_API_RETRY_BACKOFF", "2.0"))

# Odds API (secret partagé common.env) + canal Telegram dédié optionnel
# (pattern MLB_TELEGRAM_* — vide = repli sur le canal commun TELEGRAM_*).
ODDS_API_KEY = os.environ.get("API_ODDS_KEY", "")
NBA_TELEGRAM_TOKEN = os.environ.get("NBA_TELEGRAM_TOKEN") or os.environ.get("TELEGRAM_TOKEN", "")
NBA_TELEGRAM_CHAT_ID = os.environ.get("NBA_TELEGRAM_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID", "")

# Fenêtre de scan live (heures avant tip-off) — matchs du programme du jour.
NBA_SCAN_HEURES_AVANCE = float(os.environ.get("NBA_SCAN_HEURES_AVANCE", "12"))

# Edge minimum (fraction, ex. 0.03 = 3%) ; dynamique = majoré en début de
# saison (échantillon faible = incertitude modèle plus grande), pattern
# NHL_EDGE_DYNAMIQUE_*.
NBA_EDGE_MIN = float(os.environ.get("NBA_EDGE_MIN", "0.03"))
NBA_EDGE_DYNAMIQUE_ACTIF = _env_bool("NBA_EDGE_DYNAMIQUE_ACTIF", True)
NBA_EDGE_DYNAMIQUE_EXTRA = float(os.environ.get("NBA_EDGE_DYNAMIQUE_EXTRA", "0.02"))
NBA_EDGE_DYNAMIQUE_GP_PLEIN = float(os.environ.get("NBA_EDGE_DYNAMIQUE_GP_PLEIN", "20"))

# Marchés actifs (ML=moneyline, SPREAD=handicap, TOTAL=total points).
NBA_MARCHES_ACTIFS = {
    m.strip().upper() for m in os.environ.get("NBA_MARCHES_ACTIFS", "ML,SPREAD,TOTAL").split(",") if m.strip()
}

# Kelly fractionnel + cap % bankroll (pattern NHL_KELLY_FRACTION / NHL_MISE_MAX_PCT).
NBA_KELLY_FRACTION = float(os.environ.get("NBA_KELLY_FRACTION", "0.25"))
NBA_MISE_MAX_PCT = float(os.environ.get("NBA_MISE_MAX_PCT", "2"))

# Shrinkage modèle↔marché (confiance croissante avec le volume de données
# saison, pattern NHL_MODEL_TRUST_MIN/MAX/GP_PLEIN) — le reste du poids va
# à la probabilité no-vig Pinnacle (Shin 2-way).
NBA_MARCHE_SHRINK_ACTIF = _env_bool("NBA_MARCHE_SHRINK_ACTIF", True)
NBA_MODEL_TRUST_MIN = float(os.environ.get("NBA_MODEL_TRUST_MIN", "0.55"))
NBA_MODEL_TRUST_MAX = float(os.environ.get("NBA_MODEL_TRUST_MAX", "0.80"))
NBA_MODEL_TRUST_GP_PLEIN = float(os.environ.get("NBA_MODEL_TRUST_GP_PLEIN", "20"))

# Paper trading — cette saison, aucune mise réelle (voir règle nba-paper-trading).
NBA_DRY_RUN = _env_bool("NBA_DRY_RUN", True)
# Paper : tous les candidats edge valides du match (pas seulement le max-edge).
NBA_TOUS_CANDIDATS_ACTIF = _env_bool("NBA_TOUS_CANDIDATS_ACTIF", True)
NBA_BANKROLL = float(os.environ.get("NBA_BANKROLL", "1000.0"))
NBA_ODDS_QUOTA_ALERT = int(os.environ.get("NBA_ODDS_QUOTA_ALERT", "100"))

PA_DATA_DIR = "/home/chienblanc/data"
JOURNAL_NOM_NBA = "journal_trading_nba.csv"
FICHIER_JOURNAL_NBA = (
    os.path.join(PA_DATA_DIR, JOURNAL_NOM_NBA) if os.path.isdir(PA_DATA_DIR) else JOURNAL_NOM_NBA
)
FICHIER_MEMOIRE_NBA = "alertes_nba_envoyees.txt"
JOURNAL_COLONNES_NBA = [
    "Date", "ID_Match", "Exterieur", "Domicile", "Pari",
    "Vraie_Cote_Bot", "Cote_Prise", "Cote_CLV",
    "Mu_Ext", "Mu_Dom", "Edge(%)", "Risque(%)", "Mise_€",
    "Statut", "P&L", "B2B_Ext", "B2B_Dom", "Confiance_Kelly",
]

# Mapping tricode nba.com -> nom complet Odds API (basketball_nba, régions eu/us).
NBA_TEAMS_MAPPING = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}
# Variantes de noms observées côté Odds API (regions eu/us confondues).
_NBA_ODDS_ALIASES = {
    "LAC": ["Los Angeles Clippers"],
}
_odds_quota_state_nba = {"derniere_alerte": None}

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NBA_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "nba")

# GAME_ID nba.com : 001 preseason, 002 saison régulière, 003 all-star,
# 004 playoffs, 005 play-in. On ne veut que la saison régulière (backtest et
# ratings walk-forward).
GAME_ID_PREFIX_REGULAR = "002"

# Saisons COVID atypiques à marquer/filtrer (option) — bulle 2019-20 (finie le
# 14/08/2020) et saison 2020-21 raccourcie à 72 matchs (début 22/12/2020).
NBA_SAISONS_COVID = {2020, 2021}

_pit_index_memo = {}


def is_saison_covid(season_year):
    """True si `season_year` (convention année de fin) désigne 2019-20 ou 2020-21."""
    return season_year in NBA_SAISONS_COVID


def _nba_season_str(season_year=None):
    """Convertit l'année de fin de saison (ex. 2027) en format nba_api 'YYYY-YY'
    (ex. '2026-27')."""
    if season_year is None:
        season_year = NBA_SEASON
    debut = season_year - 1
    return f"{debut}-{str(season_year)[-2:]}"


# ==========================================
# 1. APPELS API AVEC RETRY/BACKOFF
# ==========================================
def _nba_api_call(endpoint_cls, label, timeout=None, max_retries=None, **kwargs):
    """Instancie un endpoint nba_api (stats.nba.com) avec retry/backoff.

    Aucun rate-limit dur observé sur les appels bulk répétés (cache CDN côté
    serveur), mais des timeouts transitoires ponctuels sur certains endpoints
    lourds (ScoreboardV3, ScheduleLeagueV2) qui se résolvent au retry suivant.
    """
    max_retries = max_retries or NBA_API_MAX_RETRIES
    timeout = timeout or NBA_API_TIMEOUT
    derniere_erreur = None
    for tentative in range(1, max_retries + 1):
        try:
            return endpoint_cls(timeout=timeout, **kwargs)
        except Exception as e:
            derniere_erreur = e
            if tentative < max_retries:
                attente = NBA_API_RETRY_BACKOFF * tentative
                log_nba(
                    f"⚠️ {label} — tentative {tentative}/{max_retries} échouée "
                    f"({type(e).__name__}: {e}), retry dans {attente:.0f}s",
                    level="warning",
                )
                time.sleep(attente)
    log_nba(f"❌ {label} — échec après {max_retries} tentatives : {derniere_erreur}", level="error")
    return None


def _resultset_vers_dicts(payload, index=0):
    """Convertit le dict brut nba_api (resultSets[i].headers/rowSet) en liste
    de dicts — évite la dépendance pandas dans la logique du bot (nba_api
    l'installe déjà en transitif pour get_data_frames(), mais on n'en a pas
    besoin ici)."""
    result_sets = payload.get("resultSets") or payload.get("resultSet") or []
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    if not result_sets or index >= len(result_sets):
        return []
    rs = result_sets[index]
    headers = rs.get("headers", [])
    rows = rs.get("rowSet", [])
    return [dict(zip(headers, row)) for row in rows]


# ==========================================
# 2. CACHE DISQUE (data/nba/) — permanent si saison terminée, TTL sinon
# ==========================================
def _cache_est_frais(chemin, season_year):
    if not os.path.exists(chemin):
        return False
    if season_year < NBA_SEASON:
        return True
    age_jours = (time.time() - os.path.getmtime(chemin)) / 86400
    return age_jours < NBA_PIT_CACHE_JOURS


def _lire_cache_disque(chemin):
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _ecrire_cache_disque(chemin, data):
    try:
        os.makedirs(NBA_DATA_DIR, exist_ok=True)
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError as e:
        log_nba(f"⚠️ écriture cache {chemin} échouée : {e}", level="warning")


def _mesure_slug(measure_type):
    return measure_type.lower().replace(" ", "")


# ==========================================
# 3. COLLECTE BULK PAR SAISON (team game logs)
# ==========================================
def _fetch_team_game_logs(season_year, measure_type, force=False):
    """1 appel bulk nba_api par (saison, measure_type) → liste de dicts
    (1 ligne/match/équipe). Cache disque JSON permanent (saison terminée) ou
    TTL `NBA_PIT_CACHE_JOURS` (saison en cours)."""
    season_str = _nba_season_str(season_year)
    chemin = os.path.join(
        NBA_DATA_DIR, f"team_game_logs_{season_str}_{_mesure_slug(measure_type)}.json"
    )
    if not force and _cache_est_frais(chemin, season_year):
        cache = _lire_cache_disque(chemin)
        if cache is not None:
            return cache

    resultat = _nba_api_call(
        teamgamelogs.TeamGameLogs,
        label=f"teamgamelogs {measure_type} {season_str}",
        season_nullable=season_str,
        season_type_nullable="Regular Season",
        measure_type_player_game_logs_nullable=measure_type,
    )
    if resultat is None:
        cache = _lire_cache_disque(chemin)
        if cache is not None:
            log_nba(
                f"ℹ️ teamgamelogs {measure_type} {season_str} — repli sur cache disque (expiré)",
                level="warning",
            )
            return cache
        return []

    try:
        lignes = _resultset_vers_dicts(resultat.get_dict())
    except Exception as e:
        log_nba(f"⚠️ teamgamelogs {measure_type} {season_str} — parsing échoué : {e}", level="warning")
        return []

    lignes = [
        l for l in lignes
        if l.get("AVAILABLE_FLAG", 1) in (1, 1.0)
        and str(l.get("GAME_ID", "")).startswith(GAME_ID_PREFIX_REGULAR)
    ]
    _ecrire_cache_disque(chemin, lignes)
    return lignes


def _extraire_domicile(matchup):
    """'OKC vs. DAL' → True (domicile) ; 'OKC @ DAL' → False."""
    return " vs. " in (matchup or "")


def _extraire_adversaire(matchup):
    if not matchup:
        return ""
    sep = " vs. " if " vs. " in matchup else " @ "
    parts = matchup.split(sep)
    return parts[1].strip() if len(parts) == 2 else ""


def _fusionner_logs_saison(season_year, force=False):
    """Fusionne Base + Advanced + Four Factors → 1 enregistrement complet par
    (match, équipe), avec le score adverse (2e ligne du même GAME_ID)."""
    base = _fetch_team_game_logs(season_year, "Base", force=force)
    if not base:
        return []
    advanced = _fetch_team_game_logs(season_year, "Advanced", force=force)
    four_factors = _fetch_team_game_logs(season_year, "Four Factors", force=force)

    adv_par_cle = {(l.get("GAME_ID"), l.get("TEAM_ID")): l for l in advanced}
    ff_par_cle = {(l.get("GAME_ID"), l.get("TEAM_ID")): l for l in four_factors}

    par_match = {}
    for ligne in base:
        par_match.setdefault(ligne.get("GAME_ID"), []).append(ligne)

    records = []
    for ligne in base:
        game_id = ligne.get("GAME_ID")
        team_id = ligne.get("TEAM_ID")
        adversaires = [l for l in par_match.get(game_id, []) if l.get("TEAM_ID") != team_id]
        if len(adversaires) != 1:
            continue  # match incomplet (ligne équipe adverse manquante) → ignoré
        opp = adversaires[0]
        adv = adv_par_cle.get((game_id, team_id), {})
        ff = ff_par_cle.get((game_id, team_id), {})
        matchup = ligne.get("MATCHUP") or ""
        records.append({
            "season": season_year,
            "game_id": game_id,
            "team_id": team_id,
            "team": ligne.get("TEAM_ABBREVIATION"),
            "date": (ligne.get("GAME_DATE") or "")[:10],
            "domicile": _extraire_domicile(matchup),
            "opponent": _extraire_adversaire(matchup) or opp.get("TEAM_ABBREVIATION"),
            "pts": ligne.get("PTS"),
            "pts_adv": opp.get("PTS"),
            "poss": adv.get("POSS"),
            "off_rating": adv.get("OFF_RATING"),
            "def_rating": adv.get("DEF_RATING"),
            "pace": adv.get("PACE"),
            "efg_pct": ff.get("EFG_PCT"),
            "fta_rate": ff.get("FTA_RATE"),
            "tov_pct": ff.get("TM_TOV_PCT"),
            "oreb_pct": ff.get("OREB_PCT"),
            "opp_efg_pct": ff.get("OPP_EFG_PCT"),
            "opp_fta_rate": ff.get("OPP_FTA_RATE"),
            "opp_tov_pct": ff.get("OPP_TOV_PCT"),
            "opp_oreb_pct": ff.get("OPP_OREB_PCT"),
        })
    return records


# ==========================================
# 4. INDEX PIT WALK-FORWARD (cumul pondéré possessions, sans look-ahead)
# ==========================================
def _cumul_vers_stats_equipe_nba(cumul):
    """Convertit les cumuls pré-match en stats moyennes pondérées par
    possessions. `games_played == 0` (aucun match encore joué) → None partout,
    laissé au moteur de rating (phase ultérieure) de gérer le défaut/prior."""
    gp = cumul["gp"]
    poss = cumul["poss"]
    if gp <= 0 or poss <= 0:
        return {
            "games_played": 0,
            "off_rating": None,
            "def_rating": None,
            "pace": None,
            "efg_pct": None,
            "fta_rate": None,
            "tov_pct": None,
            "oreb_pct": None,
            "opp_efg_pct": None,
            "opp_fta_rate": None,
            "opp_tov_pct": None,
            "opp_oreb_pct": None,
        }
    return {
        "games_played": gp,
        "off_rating": round(cumul["off_num"] / poss, 3),
        "def_rating": round(cumul["def_num"] / poss, 3),
        "pace": round(cumul["pace_num"] / gp, 3),
        "efg_pct": round(cumul["efg_num"] / poss, 4),
        "fta_rate": round(cumul["fta_num"] / poss, 4),
        "tov_pct": round(cumul["tov_num"] / poss, 4),
        "oreb_pct": round(cumul["oreb_num"] / poss, 4),
        "opp_efg_pct": round(cumul["opp_efg_num"] / poss, 4),
        "opp_fta_rate": round(cumul["opp_fta_num"] / poss, 4),
        "opp_tov_pct": round(cumul["opp_tov_num"] / poss, 4),
        "opp_oreb_pct": round(cumul["opp_oreb_num"] / poss, 4),
    }


def _construire_index_saison(records):
    """Regroupe les enregistrements d'UNE saison par équipe, trie
    chronologiquement, puis calcule pour chaque match le snapshot cumulé
    strictement AVANT ce match (walk-forward, pattern identique au PIT
    MoneyPuck NHL — le cumul repart de 0 GP au 1er match de la saison, il n'y
    a pas de mélange avec une saison antérieure ici).

    Retourne `(snapshots_par_equipe, finaux_par_equipe)` : `finaux_par_equipe`
    est le cumul complet en fin de saison (tous les matchs inclus), utile
    comme prior « saison N-1 » pour le blend/shrinkage début de saison
    suivante (pattern `teams_n1` du NHL) — calculé par la phase moteur
    (hors scope collecte), voir `stats_fin_saison_nba`.
    """
    par_equipe = {}
    for r in records:
        if r.get("poss") is None or r.get("off_rating") is None:
            continue  # ligne Advanced/Four Factors manquante → ignorée
        par_equipe.setdefault(r["team"], []).append(r)

    snapshots_par_equipe = {}
    finaux_par_equipe = {}
    for team, games in par_equipe.items():
        games.sort(key=lambda g: (g["date"], g["game_id"]))
        cumul = {
            "gp": 0, "poss": 0.0,
            "off_num": 0.0, "def_num": 0.0, "pace_num": 0.0,
            "efg_num": 0.0, "fta_num": 0.0, "tov_num": 0.0, "oreb_num": 0.0,
            "opp_efg_num": 0.0, "opp_fta_num": 0.0, "opp_tov_num": 0.0, "opp_oreb_num": 0.0,
        }
        snapshots = []
        for g in games:
            snapshots.append({
                "date": g["date"],
                "game_id": g["game_id"],
                "season": g["season"],
                "opponent": g["opponent"],
                "domicile": g["domicile"],
                "pts": g["pts"],
                "pts_adv": g["pts_adv"],
                "stats": _cumul_vers_stats_equipe_nba(cumul),
            })
            poss = float(g["poss"] or 0)
            cumul["gp"] += 1
            cumul["poss"] += poss
            cumul["off_num"] += float(g["off_rating"] or 0) * poss
            cumul["def_num"] += float(g["def_rating"] or 0) * poss
            cumul["pace_num"] += float(g["pace"] or 0)
            cumul["efg_num"] += float(g["efg_pct"] or 0) * poss
            cumul["fta_num"] += float(g["fta_rate"] or 0) * poss
            cumul["tov_num"] += float(g["tov_pct"] or 0) * poss
            cumul["oreb_num"] += float(g["oreb_pct"] or 0) * poss
            cumul["opp_efg_num"] += float(g["opp_efg_pct"] or 0) * poss
            cumul["opp_fta_num"] += float(g["opp_fta_rate"] or 0) * poss
            cumul["opp_tov_num"] += float(g["opp_tov_pct"] or 0) * poss
            cumul["opp_oreb_num"] += float(g["opp_oreb_pct"] or 0) * poss
        snapshots_par_equipe[team] = snapshots
        finaux_par_equipe[team] = _cumul_vers_stats_equipe_nba(cumul)
    return snapshots_par_equipe, finaux_par_equipe


def construire_index_pit_nba(season_year=None, force=False):
    """
    Construit l'index PIT (point-in-time, sans look-ahead) d'UNE saison
    régulière : pour chaque équipe, la liste chronologique des snapshots
    pré-match (OFF/DEF rating, pace, Four Factors cumulés pondérés par
    possessions, strictement avant chaque match, cumul repartant de 0 GP au
    1er match de la saison).

    Le blend avec la saison précédente (prior N-1, décroissance temporelle,
    `n_prior`) est délégué à la phase moteur mathématique (hors scope
    collecte) ; utiliser `stats_fin_saison_nba(season_year - 1)` comme
    référence de fin de saison précédente.

    Cache mémoire (process) + cache disque par saison (`_fetch_team_game_logs`,
    permanent si saison terminée, TTL `NBA_PIT_CACHE_JOURS` sinon).
    """
    global _pit_index_memo
    if season_year is None:
        season_year = NBA_SEASON

    if not force and _pit_index_memo.get("season") == season_year:
        return _pit_index_memo["index"]

    records = _fusionner_logs_saison(season_year, force=force)
    if not records:
        log_nba(f"⚠️ nba_api — aucune donnée saison {_nba_season_str(season_year)}", level="warning")
        return _pit_index_memo.get("index") or {}

    index, finaux = _construire_index_saison(records)
    _pit_index_memo = {"season": season_year, "index": index, "finaux": finaux}

    nb_snapshots = sum(len(v) for v in index.values())
    log_nba(
        f"📅 Index PIT NBA construit — {len(index)} équipes, {nb_snapshots} snapshots "
        f"(saison {_nba_season_str(season_year)})"
    )
    return index


def stats_fin_saison_nba(season_year, force=False):
    """Cumul complet (stats moyennes pondérées par possessions sur
    l'intégralité de la saison régulière `season_year`) par équipe — sert de
    prior « saison N-1 » pour le blend début de saison suivante (pattern
    `teams_n1` du NHL), à utiliser par la phase moteur mathématique."""
    global _pit_index_memo
    if _pit_index_memo.get("season") == season_year and "finaux" in _pit_index_memo:
        return _pit_index_memo["finaux"]

    records = _fusionner_logs_saison(season_year, force=force)
    if not records:
        return {}
    _snapshots, finaux = _construire_index_saison(records)
    return finaux


def _invalider_pit_index_memo():
    global _pit_index_memo
    _pit_index_memo = {}


# ==========================================
# 5. CALENDRIER / LIVE (schedule + scoreboard du jour)
# ==========================================
def fetch_saison_schedule(season_year=None, force=False):
    """Calendrier complet saison régulière (ScheduleLeagueV2) : dates/heures
    UTC de tip-off + résultats si le match est terminé. 1 appel bulk, cache
    disque (permanent si saison terminée, TTL sinon)."""
    season_year = season_year or NBA_SEASON
    season_str = _nba_season_str(season_year)
    chemin = os.path.join(NBA_DATA_DIR, f"schedule_{season_str}.json")
    if not force and _cache_est_frais(chemin, season_year):
        cache = _lire_cache_disque(chemin)
        if cache is not None:
            return cache

    resultat = _nba_api_call(
        scheduleleaguev2.ScheduleLeagueV2,
        label=f"scheduleleaguev2 {season_str}",
        timeout=max(NBA_API_TIMEOUT, 45),
        season=season_str,
    )
    if resultat is None:
        cache = _lire_cache_disque(chemin)
        if cache is not None:
            log_nba(f"ℹ️ schedule {season_str} — repli sur cache disque (expiré)", level="warning")
            return cache
        return []

    try:
        payload = resultat.get_dict()
        dates = payload.get("leagueSchedule", {}).get("gameDates", [])
    except Exception as e:
        log_nba(f"⚠️ schedule {season_str} — parsing échoué : {e}", level="warning")
        return []

    matchs = []
    for jour in dates:
        for g in jour.get("games", []):
            game_id = g.get("gameId", "")
            if not game_id.startswith(GAME_ID_PREFIX_REGULAR):
                continue
            home = g.get("homeTeam", {})
            away = g.get("awayTeam", {})
            matchs.append({
                "game_id": game_id,
                "date_utc": g.get("gameDateTimeUTC"),
                "statut": g.get("gameStatusText"),
                "domicile": home.get("teamTricode"),
                "exterieur": away.get("teamTricode"),
                "score_domicile": home.get("score"),
                "score_exterieur": away.get("score"),
            })

    _ecrire_cache_disque(chemin, matchs)
    return matchs


def fetch_live_scoreboard():
    """Programme + scores du jour (module live nba_api, cdn.nba.com).
    Retourne une liste vide (sans lever d'exception) hors saison ou en cas de
    réponse vide/JSON invalide — comportement confirmé en exploration (aucun
    match programmé ne renvoie pas un JSON valide côté cdn.nba.com)."""
    try:
        r = live_scoreboard.ScoreBoard()
        d = r.get_dict()
    except Exception as e:
        log_nba(
            f"ℹ️ live scoreboard indisponible (normal hors saison) : {type(e).__name__}: {e}",
            level="info",
        )
        return []

    jeux = d.get("scoreboard", {}).get("games", [])
    matchs = []
    for g in jeux:
        home = g.get("homeTeam", {})
        away = g.get("awayTeam", {})
        matchs.append({
            "game_id": g.get("gameId"),
            "date_utc": g.get("gameTimeUTC"),
            "statut": g.get("gameStatusText"),
            "domicile": home.get("teamTricode"),
            "exterieur": away.get("teamTricode"),
            "score_domicile": home.get("score"),
            "score_exterieur": away.get("score"),
        })
    return matchs


# ==========================================
# 6. MOTEUR MATHÉMATIQUE — ratings ajustés, distribution, probabilités marché
# ==========================================
#
# Pipeline (walk-forward strict, aucune donnée >= date_ref utilisée) :
#
#   records par équipe (get_dict per-game)
#       -> stats brutes decay-pondérées (demi-vie NBA_RATING_HALF_LIFE_JOURS)
#       -> moyenne ligue du jour (cross-section des équipes déjà actives)
#       -> shrinkage bayésien vers la moyenne ligue (n_prior)
#       -> blend saison N / saison N-1 (rampe sur GP comptés, NBA_BLEND_GP_PLEIN)
#       -> mu_home / mu_away (cross Off x Def x Pace/lg_avg)
#       -> ajustements HCA / repos-B2B / altitude
#       -> distribution bivariée Normale (sigma_home, sigma_away, rho)
#       -> probabilités moneyline / spread (avec push) / total (avec push)
#
# Analogue au moteur foot (Dixon-Coles + shrink + blend N-1) et NHL (xG +
# shrink PP/PK + blend N-1 + fatigue calendrier), mais distribution Normale
# bivariée sur la marge/total plutôt qu'une matrice de score Poisson —
# justifié par le volume de points NBA (~110-120/équipe, quasi-gaussien par
# TCL) contre le faible effectif de buts foot/NHL. Voir le plan
# `sniper_nba_value_bets` (section "Pourquoi pas le pattern Poisson/DC").


def _parse_date_nba(date_str):
    """Parse une date 'YYYY-MM-DD' (ou 'YYYY-MM-DD HH:MM...') → date, ou None."""
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _decay_poids(age_jours, half_life_jours):
    """Poids exponentiel selon l'ancienneté (jours) : 0.5^(age/demi_vie).
    `half_life_jours <= 0` désactive la décroissance (poids uniforme = 1.0)."""
    if half_life_jours is None or half_life_jours <= 0:
        return 1.0
    return 0.5 ** (max(age_jours, 0) / half_life_jours)


def _grouper_records_par_equipe(records):
    """Regroupe les enregistrements fusionnés (`_fusionner_logs_saison`) par
    équipe, triés chronologiquement (date, game_id)."""
    par_equipe = {}
    for r in records:
        if r.get("poss") is None or r.get("off_rating") is None:
            continue
        par_equipe.setdefault(r["team"], []).append(r)
    for team in par_equipe:
        par_equipe[team].sort(key=lambda g: (g["date"], g["game_id"]))
    return par_equipe


def _stats_brutes_decay_equipe(games_avant, date_ref, half_life_jours):
    """Stats brutes (OFF/DEF rating pondérés possessions, Pace) d'une équipe
    à partir de ses matchs de la saison en cours strictement AVANT
    `date_ref`, pondérés par décroissance temporelle exponentielle.

    Retourne ``None`` si l'équipe n'a encore aucun match cette saison.
    `effective_gp` (somme des poids de décroissance, unité "matchs") sert de
    taille d'échantillon au shrinkage bayésien ; `gp` (compte brut, non
    pondéré) sert de rampe au blend saison N-1.
    """
    if not games_avant:
        return None
    ref = _parse_date_nba(date_ref)
    poids_poss_total = 0.0
    effective_gp = 0.0
    off_num = 0.0
    def_num = 0.0
    pace_num = 0.0
    for g in games_avant:
        d = _parse_date_nba(g.get("date"))
        age_jours = (ref - d).days if (ref and d) else 0
        w = _decay_poids(age_jours, half_life_jours)
        poss = float(g.get("poss") or 0)
        off_num += w * poss * float(g.get("off_rating") or 0)
        def_num += w * poss * float(g.get("def_rating") or 0)
        poids_poss_total += w * poss
        pace_num += w * float(g.get("pace") or 0)
        effective_gp += w
    if poids_poss_total <= 0 or effective_gp <= 0:
        return None
    return {
        "off_rating": off_num / poids_poss_total,
        "def_rating": def_num / poids_poss_total,
        "pace": pace_num / effective_gp,
        "effective_gp": effective_gp,
        "gp": len(games_avant),
    }


def _snapshot_ligue_decay(par_equipe, date_ref, half_life_jours):
    """Stats brutes decay-pondérées de chaque équipe à `date_ref` (walk-forward
    strict : ne regarde que les matchs de `par_equipe` avec `date < date_ref`).
    Retourne un dict team -> stats (ou ``None`` si l'équipe n'a aucun match)."""
    snap = {}
    for team, games in par_equipe.items():
        games_avant = [g for g in games if g["date"] < date_ref]
        snap[team] = _stats_brutes_decay_equipe(games_avant, date_ref, half_life_jours)
    return snap


def _moyennes_ligue_depuis_snapshot(snap):
    """Moyenne ligue (ORtg/DRtg/Pace) = moyenne cross-section des stats brutes
    des équipes déjà actives à `date_ref` — reste point-in-time car chaque
    stat d'équipe n'utilise que son propre passé. Retourne ``None`` si aucune
    équipe n'a encore joué (tout début de saison)."""
    offs = [s["off_rating"] for s in snap.values() if s]
    defs_ = [s["def_rating"] for s in snap.values() if s]
    paces = [s["pace"] for s in snap.values() if s]
    if not offs:
        return None
    return {
        "off_rating": sum(offs) / len(offs),
        "def_rating": sum(defs_) / len(defs_),
        "pace": sum(paces) / len(paces),
    }


def _rating_ajuste_equipe(team, snap, league_avg, prior_n1, n_prior, blend_gp_plein):
    """Rating ajusté d'une équipe : shrinkage bayésien vers `league_avg`
    (poids `n_prior`, taille d'échantillon = `effective_gp` decay-pondéré)
    puis blend avec le rating de fin de saison précédente `prior_n1` (rampe
    sur GP comptés jusqu'à `blend_gp_plein`, pattern `NHL_BLEND_GP_PLEIN`).

    Sans aucun match cette saison : pur prior N-1 si disponible, sinon
    moyenne ligue (équipe d'expansion / prior manquant)."""
    brut = snap.get(team)
    prior = (prior_n1 or {}).get(team)
    prior_valide = bool(prior) and prior.get("games_played", 0) > 0 and prior.get("off_rating") is not None

    if not brut:
        if prior_valide:
            return {
                "off_rating": prior["off_rating"],
                "def_rating": prior["def_rating"],
                "pace": prior["pace"],
                "gp": 0,
            }
        return {
            "off_rating": league_avg["off_rating"],
            "def_rating": league_avg["def_rating"],
            "pace": league_avg["pace"],
            "gp": 0,
        }

    eff_gp = brut["effective_gp"]
    shrink_off = (eff_gp * brut["off_rating"] + n_prior * league_avg["off_rating"]) / (eff_gp + n_prior)
    shrink_def = (eff_gp * brut["def_rating"] + n_prior * league_avg["def_rating"]) / (eff_gp + n_prior)
    shrink_pace = (eff_gp * brut["pace"] + n_prior * league_avg["pace"]) / (eff_gp + n_prior)

    gp = brut["gp"]
    if prior_valide and blend_gp_plein > 0:
        w = min(1.0, gp / blend_gp_plein)
        return {
            "off_rating": w * shrink_off + (1.0 - w) * prior["off_rating"],
            "def_rating": w * shrink_def + (1.0 - w) * prior["def_rating"],
            "pace": w * shrink_pace + (1.0 - w) * prior["pace"],
            "gp": gp,
        }
    return {"off_rating": shrink_off, "def_rating": shrink_def, "pace": shrink_pace, "gp": gp}


def _contexte_repos_equipe(games_avant, date_ref):
    """(is_b2b, rest_jours) d'une équipe à `date_ref`, à partir de son dernier
    match connu strictement avant. `is_b2b` = match la veille (0 ou 1 jour de
    repos). Retourne ``(False, None)`` si aucun match antérieur (1er match
    de la saison pour cette équipe)."""
    if not games_avant:
        return False, None
    dernier = games_avant[-1]
    d_ref = _parse_date_nba(date_ref)
    d_last = _parse_date_nba(dernier.get("date"))
    if not d_ref or not d_last:
        return False, None
    diff = (d_ref - d_last).days
    return diff <= 1, diff


def calculer_mu_points(
    rating_home,
    rating_away,
    league_avg,
    hca=None,
    home_b2b=False,
    away_b2b=False,
    home_rest_jours=None,
    away_rest_jours=None,
    altitude_domicile=False,
):
    """Points attendus (mu_home, mu_away) — cross-multiplicatif Off x Def x
    Pace/lg_avg (pattern identique foot/NHL), avec ajustements home
    court/repos-B2B/altitude appliqués aux ratings avant le cross.

    `rating_*` : dict avec ``off_rating``/``def_rating``/``pace`` (déjà
    shrink + blend, voir `_rating_ajuste_equipe`). `league_avg` : moyenne
    ligue ORtg/DRtg/Pace du jour (voir `_moyennes_ligue_depuis_snapshot`).
    """
    hca = nba_params.get_hca() if hca is None else hca
    b2b_atk_pct = nba_params.get_b2b_atk_pct()
    b2b_def_pct = nba_params.get_b2b_def_pct()
    rest_bonus_pct = nba_params.get_rest_bonus_pct()
    altitude_bonus = nba_params.get_altitude_bonus()

    home_off = rating_home["off_rating"] * (1.0 + hca)
    home_def = rating_home["def_rating"] * (1.0 - hca)
    away_off = rating_away["off_rating"]
    away_def = rating_away["def_rating"]

    if altitude_domicile:
        home_off *= 1.0 + altitude_bonus
        away_off *= 1.0 - altitude_bonus * 0.5

    if home_b2b:
        home_off *= 1.0 - b2b_atk_pct
        home_def *= 1.0 + b2b_def_pct
    elif home_rest_jours is not None and home_rest_jours >= 2:
        home_off *= 1.0 + rest_bonus_pct

    if away_b2b:
        away_off *= 1.0 - b2b_atk_pct
        away_def *= 1.0 + b2b_def_pct
    elif away_rest_jours is not None and away_rest_jours >= 2:
        away_off *= 1.0 + rest_bonus_pct

    lg_rtg = max(league_avg["off_rating"], 1.0)
    pace_match = (rating_home["pace"] + rating_away["pace"]) / 2.0 / 100.0

    mu_home = (home_off / lg_rtg) * (away_def / lg_rtg) * lg_rtg * pace_match
    mu_away = (away_off / lg_rtg) * (home_def / lg_rtg) * lg_rtg * pace_match
    return max(mu_home, 1.0), max(mu_away, 1.0)


# ------------------------------------------
# 6b. Distribution bivariée Normale + probabilités de marché
# ------------------------------------------
def _phi(x):
    """CDF de la loi Normale standard N(0,1)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sigma_diff(sigma_home, sigma_away, rho):
    """Écart-type de la marge (home - away) : Var(X-Y) = VarX + VarY - 2*Cov(X,Y)."""
    variance = sigma_home ** 2 + sigma_away ** 2 - 2.0 * rho * sigma_home * sigma_away
    return math.sqrt(max(variance, 1e-6))


def _sigma_total(sigma_home, sigma_away, rho):
    """Écart-type du total (home + away) : Var(X+Y) = VarX + VarY + 2*Cov(X,Y)."""
    variance = sigma_home ** 2 + sigma_away ** 2 + 2.0 * rho * sigma_home * sigma_away
    return math.sqrt(max(variance, 1e-6))


def prob_moneyline_nba(mu_home, mu_away, sigma_home=None, sigma_away=None, rho=None):
    """P(victoire domicile), P(victoire extérieur) — pas de push : une
    prolongation tranche toujours un score nul en fin de temps réglementaire."""
    sigma_home = nba_params.get_sigma_team() if sigma_home is None else sigma_home
    sigma_away = nba_params.get_sigma_team() if sigma_away is None else sigma_away
    rho = nba_params.get_rho_scores() if rho is None else rho

    sd = _sigma_diff(sigma_home, sigma_away, rho)
    p_domicile = _phi((mu_home - mu_away) / sd)
    return p_domicile, 1.0 - p_domicile


def _prob_ligne_normale(mu, sigma, ligne):
    """P(gagne, push, perdu) pour une variable à valeurs entières (marge ou
    total de points) approximée par N(mu, sigma), avec correction de
    continuité. Lignes entières (ex. spread -6, total 220) : push possible.
    Lignes .5 (ex. -6.5, 220.5) : pas de push, la correction de continuité
    s'annule exactement avec la ligne (cf. commentaire moteur NBA)."""
    if sigma <= 0:
        if mu > ligne:
            return 1.0, 0.0, 0.0
        if mu < ligne:
            return 0.0, 0.0, 1.0
        return 0.0, 1.0, 0.0

    est_entiere = float(ligne).is_integer()
    if est_entiere:
        p_perdu = _phi((ligne - 0.5 - mu) / sigma)
        p_gagne = 1.0 - _phi((ligne + 0.5 - mu) / sigma)
        p_push = max(0.0, 1.0 - p_gagne - p_perdu)
    else:
        p_gagne = 1.0 - _phi((ligne - mu) / sigma)
        p_perdu = 1.0 - p_gagne
        p_push = 0.0
    return p_gagne, p_push, p_perdu


def prob_spread_nba(mu_home, mu_away, ligne_domicile, sigma_home=None, sigma_away=None, rho=None):
    """P(domicile couvre, push, extérieur couvre) pour une ligne de spread
    exprimée du point de vue domicile (ex. ``-5.5`` si domicile favori de
    5.5 pts, ``+3`` si domicile outsider de 3 pts) — convention American
    spread standard (pas de handicap asiatique multi-lignes comme le foot)."""
    sigma_home = nba_params.get_sigma_team() if sigma_home is None else sigma_home
    sigma_away = nba_params.get_sigma_team() if sigma_away is None else sigma_away
    rho = nba_params.get_rho_scores() if rho is None else rho

    mu_diff = mu_home - mu_away
    sd = _sigma_diff(sigma_home, sigma_away, rho)
    # Domicile couvre si marge > -ligne_domicile (ex. ligne -5.5 -> marge > 5.5).
    return _prob_ligne_normale(mu_diff, sd, -ligne_domicile)


def prob_total_nba(mu_home, mu_away, ligne_total, sigma_home=None, sigma_away=None, rho=None):
    """P(over, push, under) pour une ligne de total de points."""
    sigma_home = nba_params.get_sigma_team() if sigma_home is None else sigma_home
    sigma_away = nba_params.get_sigma_team() if sigma_away is None else sigma_away
    rho = nba_params.get_rho_scores() if rho is None else rho

    mu_total = mu_home + mu_away
    st = _sigma_total(sigma_home, sigma_away, rho)
    return _prob_ligne_normale(mu_total, st, ligne_total)


# ------------------------------------------
# 6c. Orchestrateur — moteur complet pour un match
# ------------------------------------------
def calculer_probabilites_match_nba(
    home,
    away,
    date_ref,
    season_year=None,
    lignes_spread=None,
    lignes_total=None,
    force=False,
):
    """Moteur mathématique complet pour un match NBA (`home` vs `away`) à
    `date_ref` (``'YYYY-MM-DD'``) — walk-forward strict, aucune donnée à
    partir de `date_ref` n'est utilisée.

    `lignes_spread` : itérable de lignes côté domicile (ex. ``[-5.5]``).
    `lignes_total` : itérable de lignes de total (ex. ``[224.5]``).

    Retourne un dict avec `mu_home`/`mu_away`, `sigma_home`/`sigma_away`,
    `rho`, les ratings ajustés, le contexte repos/B2B, et les probabilités
    de marché (moneyline, spread par ligne, total par ligne — push géré sur
    lignes entières).
    """
    season_year = season_year or NBA_SEASON
    records = _fusionner_logs_saison(season_year, force=force)
    par_equipe = _grouper_records_par_equipe(records)

    half_life = nba_params.get_rating_half_life_jours()
    n_prior = nba_params.get_n_prior()
    blend_gp_plein = nba_params.get_blend_gp_plein()

    snap = _snapshot_ligue_decay(par_equipe, date_ref, half_life)
    league_avg = _moyennes_ligue_depuis_snapshot(snap) or nba_params.get_league_avg_fallback()
    prior_n1 = stats_fin_saison_nba(season_year - 1, force=force)

    rating_home = _rating_ajuste_equipe(home, snap, league_avg, prior_n1, n_prior, blend_gp_plein)
    rating_away = _rating_ajuste_equipe(away, snap, league_avg, prior_n1, n_prior, blend_gp_plein)

    games_home_avant = [g for g in par_equipe.get(home, []) if g["date"] < date_ref]
    games_away_avant = [g for g in par_equipe.get(away, []) if g["date"] < date_ref]
    home_b2b, home_rest = _contexte_repos_equipe(games_home_avant, date_ref)
    away_b2b, away_rest = _contexte_repos_equipe(games_away_avant, date_ref)

    mu_home, mu_away = calculer_mu_points(
        rating_home,
        rating_away,
        league_avg,
        hca=nba_params.get_hca(home),
        home_b2b=home_b2b,
        away_b2b=away_b2b,
        home_rest_jours=home_rest,
        away_rest_jours=away_rest,
        altitude_domicile=nba_params.is_altitude_team(home),
    )

    sigma_home = nba_params.get_sigma_team(home)
    sigma_away = nba_params.get_sigma_team(away)
    rho = nba_params.get_rho_scores()

    p_dom_ml, p_ext_ml = prob_moneyline_nba(mu_home, mu_away, sigma_home, sigma_away, rho)

    spreads = {}
    for ligne in (lignes_spread or []):
        p_dom, p_push, p_ext = prob_spread_nba(mu_home, mu_away, ligne, sigma_home, sigma_away, rho)
        spreads[ligne] = {"domicile": round(p_dom, 4), "push": round(p_push, 4), "exterieur": round(p_ext, 4)}

    totals = {}
    for ligne in (lignes_total or []):
        p_over, p_push, p_under = prob_total_nba(mu_home, mu_away, ligne, sigma_home, sigma_away, rho)
        totals[ligne] = {"over": round(p_over, 4), "push": round(p_push, 4), "under": round(p_under, 4)}

    return {
        "home": home,
        "away": away,
        "date": date_ref,
        "mu_home": round(mu_home, 2),
        "mu_away": round(mu_away, 2),
        "sigma_home": sigma_home,
        "sigma_away": sigma_away,
        "rho": rho,
        "rating_home": rating_home,
        "rating_away": rating_away,
        "home_b2b": home_b2b,
        "away_b2b": away_b2b,
        "home_rest_jours": home_rest,
        "away_rest_jours": away_rest,
        "moneyline": {"domicile": round(p_dom_ml, 4), "exterieur": round(p_ext_ml, 4)},
        "spread": spreads,
        "total": totals,
    }


# ==========================================
# 7. INTEGRATION THE ODDS API (basketball_nba, Pinnacle) + DEVIG + SHRINK
# ==========================================
def _cle_nom_odds_nba(nom):
    """Normalise un nom d'équipe Odds API pour comparaison fuzzy (accents/casse)."""
    if not nom:
        return ""
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(nom)).split())


def _resoudre_tricode_depuis_nom_odds(nom_api):
    """Résout un nom Odds API ('Los Angeles Lakers') vers notre tricode interne
    ('LAL'), ou ``None`` si inconnu (déclenche un log de mapping manquant)."""
    if not nom_api:
        return None
    for tricode, full in NBA_TEAMS_MAPPING.items():
        if full == nom_api:
            return tricode
    cle = _cle_nom_odds_nba(nom_api)
    for tricode, full in NBA_TEAMS_MAPPING.items():
        if _cle_nom_odds_nba(full) == cle:
            return tricode
        for alias in _NBA_ODDS_ALIASES.get(tricode, []):
            if _cle_nom_odds_nba(alias) == cle:
                return tricode
    return None


def _parse_pinnacle_game_nba(game):
    """Parse un événement Odds API (`basketball_nba`) → dict cotes Pinnacle
    indexé par tricode interne, ou ``None`` si équipes non résolues / pas de
    cotes Pinnacle h2h disponibles.

    ``spreads`` : ``{ligne_domicile: {"home": cote, "away": cote}}`` — la
    ligne est déjà exprimée du point de vue domicile (convention identique à
    `prob_spread_nba`). ``totals`` : ``{ligne: {"over": cote, "under": cote}}``.
    """
    home_tri = _resoudre_tricode_depuis_nom_odds(game.get("home_team"))
    away_tri = _resoudre_tricode_depuis_nom_odds(game.get("away_team"))
    if not home_tri or not away_tri:
        return None

    parsed = {"home": home_tri, "away": away_tri, "spreads": {}, "totals": {}}
    for bookmaker in game.get("bookmakers", []):
        if bookmaker.get("key") != "pinnacle":
            continue
        for market in bookmaker.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])
            if key == "h2h":
                for o in outcomes:
                    tri = _resoudre_tricode_depuis_nom_odds(o.get("name"))
                    if tri == home_tri:
                        parsed["cote_1"] = float(o["price"])
                    elif tri == away_tri:
                        parsed["cote_2"] = float(o["price"])
            elif key == "spreads":
                ligne_home, prix_home, prix_away = None, None, None
                for o in outcomes:
                    tri = _resoudre_tricode_depuis_nom_odds(o.get("name"))
                    if tri == home_tri:
                        ligne_home = o.get("point")
                        prix_home = o.get("price")
                    elif tri == away_tri:
                        prix_away = o.get("price")
                if ligne_home is not None and prix_home and prix_away:
                    parsed["spreads"][round(float(ligne_home), 1)] = {
                        "home": float(prix_home), "away": float(prix_away),
                    }
            elif key == "totals":
                for o in outcomes:
                    point = o.get("point")
                    if point is None:
                        continue
                    ligne = round(float(point), 1)
                    side = "over" if str(o.get("name", "")).strip().lower() == "over" else "under"
                    parsed["totals"].setdefault(ligne, {})[side] = float(o["price"])

    if "cote_1" not in parsed or "cote_2" not in parsed:
        return None
    return parsed


def _traiter_quota_odds_api_nba(response):
    """Log le quota mensuel Odds API et alerte Telegram si seuil bas (partagé
    avec les autres bots du repo, mais loggé/alerté côté canal NBA)."""
    restant_raw = response.headers.get("x-requests-remaining")
    if restant_raw is None:
        return
    try:
        restant = int(restant_raw)
        utilise = int(response.headers.get("x-requests-used", "0"))
    except ValueError:
        return
    log_nba(f"📊 Odds API quota : {restant} restantes ({utilise} utilisées ce mois)")
    if restant > NBA_ODDS_QUOTA_ALERT:
        _odds_quota_state_nba["derniere_alerte"] = None
        return
    now = datetime.now()
    last = _odds_quota_state_nba.get("derniere_alerte")
    if last and (now - last).total_seconds() < 6 * 3600:
        return
    _odds_quota_state_nba["derniere_alerte"] = now
    envoyer_alerte_systeme_nba(
        f"⚠️ **QUOTA ODDS API BAS**\n\nIl reste **{restant}** requêtes ce mois (seuil {NBA_ODDS_QUOTA_ALERT})."
    )


def fetch_all_pinnacle_odds_nba():
    """Une seule requête Odds API pour tous les matchs NBA du jour (quota
    économisé, pattern `fetch_all_pinnacle_odds` NHL)."""
    if not ODDS_API_KEY:
        return {}
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,spreads,totals",
        "bookmakers": "pinnacle",
        "oddsFormat": "decimal",
    }
    cache = {}
    try:
        response = requests.get(url, params=params, timeout=15)
        _traiter_quota_odds_api_nba(response)
        if response.status_code != 200:
            log_nba(f"⚠️ Odds API NBA : HTTP {response.status_code}", level="warning")
            return {}
        non_resolus = set()
        for game in response.json():
            parsed = _parse_pinnacle_game_nba(game)
            if parsed:
                cache[(parsed["home"], parsed["away"])] = parsed
            else:
                for nom in (game.get("home_team"), game.get("away_team")):
                    if nom and _resoudre_tricode_depuis_nom_odds(nom) is None:
                        non_resolus.add(nom)
        if non_resolus:
            log_nba(f"⚠️ Noms Odds API NBA non mappés : {', '.join(sorted(non_resolus))}", level="warning")
        log_nba(f"📡 Odds API NBA : {len(cache)} match(s) Pinnacle indexés")
        return cache
    except Exception as e:
        log_nba(f"⚠️ Erreur Odds API NBA : {e}", level="warning")
        return {}


def get_odds_for_match_nba(home_tricode, away_tricode, odds_cache=None, log_si_absent=False):
    """Retourne les cotes Pinnacle pour un match (depuis le cache ou une
    requête dédiée si `odds_cache` est ``None``)."""
    if odds_cache is None:
        odds_cache = fetch_all_pinnacle_odds_nba()
    hit = odds_cache.get((home_tricode, away_tricode))
    if not hit and log_si_absent:
        log_nba(f"⚠️ Pas de cotes Pinnacle : {away_tricode} @ {home_tricode}", level="warning")
    return hit


def _poids_confiance_modele_nba(gp_moyen_match):
    """Confiance modèle croissante avec le volume de données saison (rampe
    linéaire trust_min → trust_max jusqu'à `NBA_MODEL_TRUST_GP_PLEIN` GP
    moyen/match) — le reste du poids va à la probabilité no-vig Pinnacle."""
    if NBA_MODEL_TRUST_GP_PLEIN <= 0:
        return NBA_MODEL_TRUST_MAX
    gp = max(gp_moyen_match or 0.0, 0.0)
    ramp = min(1.0, gp / NBA_MODEL_TRUST_GP_PLEIN)
    return NBA_MODEL_TRUST_MIN + (NBA_MODEL_TRUST_MAX - NBA_MODEL_TRUST_MIN) * ramp


def _blend_proba_marche_nba(p_model_a, p_model_b, cote_a, cote_b, poids_modele):
    """Blend probabilité modèle / no-vig marché (Shin 2-way), renormalisé à 1."""
    p_mkt_a, p_mkt_b = proba_no_vig_shin_2way(cote_a, cote_b)
    if p_mkt_a is None or p_mkt_b is None:
        return p_model_a, p_model_b
    p_a = poids_modele * p_model_a + (1.0 - poids_modele) * p_mkt_a
    p_b = poids_modele * p_model_b + (1.0 - poids_modele) * p_mkt_b
    total = p_a + p_b
    if total <= 0:
        return p_model_a, p_model_b
    return p_a / total, p_b / total


def _blend_avec_push_nba(p_win_model, p_push_model, p_loss_model, cote_win, cote_loss, poids_modele):
    """Blend modèle/marché pour un marché à 3 issues (win/push/loss, spread ou
    total sur ligne entière) : le marché (2 prix, sans push explicite) blend
    la probabilité *conditionnelle* win/loss (hors push) ; `p_push` reste
    celui du modèle (Pinnacle ne cote pas le push séparément)."""
    denom = p_win_model + p_loss_model
    if denom <= 0:
        return p_win_model, p_push_model, p_loss_model
    p_win_cond, p_loss_cond = p_win_model / denom, p_loss_model / denom
    p_win_mkt, p_loss_mkt = proba_no_vig_shin_2way(cote_win, cote_loss)
    if p_win_mkt is None:
        return p_win_model, p_push_model, p_loss_model
    p_win_blend = poids_modele * p_win_cond + (1.0 - poids_modele) * p_win_mkt
    p_loss_blend = poids_modele * p_loss_cond + (1.0 - poids_modele) * p_loss_mkt
    s = p_win_blend + p_loss_blend
    if s <= 0:
        return p_win_model, p_push_model, p_loss_model
    scale = 1.0 - p_push_model
    return (p_win_blend / s) * scale, p_push_model, (p_loss_blend / s) * scale


def shrink_probabilites_vers_marche_nba(probas, cotes_match, gp_moyen_match):
    """Réduit la sur-confiance du modèle en le mélangeant avec la probabilité
    no-vig Pinnacle (Shin 2-way), pondéré par la maturité de l'échantillon
    saison (`gp_moyen_match`) — pattern `shrink_cotes_vers_marche` NHL.
    `mu_home`/`mu_away` restent purs modèle (calibration MLE inchangée)."""
    if not NBA_MARCHE_SHRINK_ACTIF or not cotes_match:
        return probas
    poids = _poids_confiance_modele_nba(gp_moyen_match)
    resultat = copy.deepcopy(probas)

    if "cote_1" in cotes_match and "cote_2" in cotes_match:
        p_dom, p_ext = _blend_proba_marche_nba(
            probas["moneyline"]["domicile"], probas["moneyline"]["exterieur"],
            cotes_match["cote_1"], cotes_match["cote_2"], poids,
        )
        resultat["moneyline"] = {"domicile": round(p_dom, 4), "exterieur": round(p_ext, 4)}

    for ligne, prix in cotes_match.get("spreads", {}).items():
        probs_ligne = probas.get("spread", {}).get(ligne)
        if not probs_ligne or "home" not in prix or "away" not in prix:
            continue
        p_dom, p_push, p_ext = _blend_avec_push_nba(
            probs_ligne["domicile"], probs_ligne["push"], probs_ligne["exterieur"],
            prix["home"], prix["away"], poids,
        )
        resultat.setdefault("spread", {})[ligne] = {
            "domicile": round(p_dom, 4), "push": round(p_push, 4), "exterieur": round(p_ext, 4),
        }

    for ligne, prix in cotes_match.get("totals", {}).items():
        probs_ligne = probas.get("total", {}).get(ligne)
        if not probs_ligne or "over" not in prix or "under" not in prix:
            continue
        p_over, p_push, p_under = _blend_avec_push_nba(
            probs_ligne["over"], probs_ligne["push"], probs_ligne["under"],
            prix["over"], prix["under"], poids,
        )
        resultat.setdefault("total", {})[ligne] = {
            "over": round(p_over, 4), "push": round(p_push, 4), "under": round(p_under, 4),
        }

    return resultat


# ==========================================
# 8. EDGE / KELLY / SELECTION DES CANDIDATS
# ==========================================
def _edge_minimum_dynamique_nba(gp_moyen):
    """Edge minimum majoré en début de saison (échantillon faible = incertitude
    modèle plus grande), rampe linéaire jusqu'à `NBA_EDGE_DYNAMIQUE_GP_PLEIN`
    GP moyen/match — pattern `_edge_minimum_dynamique` NHL."""
    if not NBA_EDGE_DYNAMIQUE_ACTIF or NBA_EDGE_DYNAMIQUE_GP_PLEIN <= 0:
        return NBA_EDGE_MIN
    gp = max(gp_moyen or 0.0, 0.0)
    ramp = max(0.0, 1.0 - min(gp, NBA_EDGE_DYNAMIQUE_GP_PLEIN) / NBA_EDGE_DYNAMIQUE_GP_PLEIN)
    return NBA_EDGE_MIN + NBA_EDGE_DYNAMIQUE_EXTRA * ramp


def calculate_kelly_nba(true_prob, book_odds, bankroll, kelly_mult=1.0, gp_moyen=None, edge_min_override=None):
    """Kelly fractionnel + cap % bankroll — pattern `calculate_kelly` NHL
    (sans les composantes gardiens, non applicables en NBA v1)."""
    if book_odds is None or book_odds <= 1.0 or true_prob is None or true_prob <= 0.01 or true_prob >= 0.99:
        return None
    edge = true_prob - (1.0 / book_odds)
    edge_min = _edge_minimum_dynamique_nba(gp_moyen)
    if edge_min_override is not None:
        edge_min = max(edge_min, float(edge_min_override))
    if edge <= edge_min:
        return None
    b = book_odds - 1.0
    fraction_kelly = NBA_KELLY_FRACTION * max(kelly_mult, 0.0)
    safe_kelly = ((b * true_prob - (1.0 - true_prob)) / b) * fraction_kelly
    mise_brute = bankroll * safe_kelly
    if NBA_MISE_MAX_PCT > 0:
        mise = round(min(mise_brute, bankroll * (NBA_MISE_MAX_PCT / 100.0)), 2)
    else:
        mise = round(mise_brute, 2)
    if mise <= 0:
        return None
    pct_effectif = round((mise / bankroll) * 100, 2) if bankroll > 0 else 0.0
    return {"edge": round(edge * 100, 2), "pct_bankroll": pct_effectif, "mise": mise}


def _construire_candidats_pari_nba(m, probas, cotes_match, bankroll, gp_moyen=None, kelly_mult=1.0):
    """Évalue ML / SPREAD / TOTAL selon `NBA_MARCHES_ACTIFS`, à partir des
    probabilités déjà shrink (`shrink_probabilites_vers_marche_nba`) et des
    cotes Pinnacle (`get_odds_for_match_nba`)."""
    candidats = []
    kelly_kw = dict(bankroll=bankroll, kelly_mult=kelly_mult, gp_moyen=gp_moyen)

    def _ajouter(marche, type_pari, inv, cote_book, cote_vraie):
        if not inv:
            return
        candidats.append({
            "type": type_pari, "inv": inv, "cote_book": cote_book, "cote_vraie": cote_vraie, "marche": marche,
        })

    if "ML" in NBA_MARCHES_ACTIFS and "cote_1" in cotes_match and "cote_2" in cotes_match:
        p_dom = probas["moneyline"]["domicile"]
        p_ext = probas["moneyline"]["exterieur"]
        inv = calculate_kelly_nba(p_dom, cotes_match["cote_1"], **kelly_kw)
        _ajouter("ML", f"Victoire {m['home']}", inv, cotes_match["cote_1"], round(1 / max(p_dom, 0.001), 2))
        inv = calculate_kelly_nba(p_ext, cotes_match["cote_2"], **kelly_kw)
        _ajouter("ML", f"Victoire {m['away']}", inv, cotes_match["cote_2"], round(1 / max(p_ext, 0.001), 2))

    if "SPREAD" in NBA_MARCHES_ACTIFS:
        for ligne, prix in cotes_match.get("spreads", {}).items():
            probs_ligne = probas.get("spread", {}).get(ligne)
            if not probs_ligne:
                continue
            if "home" in prix:
                inv = calculate_kelly_nba(probs_ligne["domicile"], prix["home"], **kelly_kw)
                _ajouter(
                    "SPREAD", f"{m['home']} {ligne:+g}", inv, prix["home"],
                    round(1 / max(probs_ligne["domicile"], 0.001), 2),
                )
            if "away" in prix:
                inv = calculate_kelly_nba(probs_ligne["exterieur"], prix["away"], **kelly_kw)
                _ajouter(
                    "SPREAD", f"{m['away']} {-ligne:+g}", inv, prix["away"],
                    round(1 / max(probs_ligne["exterieur"], 0.001), 2),
                )

    if "TOTAL" in NBA_MARCHES_ACTIFS:
        for ligne, prix in cotes_match.get("totals", {}).items():
            probs_ligne = probas.get("total", {}).get(ligne)
            if not probs_ligne:
                continue
            if "over" in prix:
                inv = calculate_kelly_nba(probs_ligne["over"], prix["over"], **kelly_kw)
                _ajouter(
                    "TOTAL", f"OVER {ligne:g}", inv, prix["over"], round(1 / max(probs_ligne["over"], 0.001), 2),
                )
            if "under" in prix:
                inv = calculate_kelly_nba(probs_ligne["under"], prix["under"], **kelly_kw)
                _ajouter(
                    "TOTAL", f"UNDER {ligne:g}", inv, prix["under"], round(1 / max(probs_ligne["under"], 0.001), 2),
                )

    return candidats


def _choisir_meilleur_pari_nba(candidats):
    best, max_edge = None, 0.0
    for cand in candidats:
        inv = cand.get("inv")
        if inv and inv["edge"] > max_edge:
            max_edge = inv["edge"]
            best = cand
    return best


def _selectionner_paris_nba(candidats):
    """Max-edge seul, ou tous les candidats (paper trading — `NBA_TOUS_CANDIDATS_ACTIF`)."""
    if not candidats:
        return []
    if NBA_TOUS_CANDIDATS_ACTIF:
        return list(candidats)
    best = _choisir_meilleur_pari_nba(candidats)
    return [best] if best else []


# ==========================================
# 9. JOURNAL DE TRADING, TELEGRAM & BOUCLE LIVE
# ==========================================
def match_deja_notifie_nba(id_match):
    if not os.path.exists(FICHIER_MEMOIRE_NBA):
        return False
    with open(FICHIER_MEMOIRE_NBA, "r", encoding="utf-8") as f:
        return id_match in f.read()


def enregistrer_notification_nba(id_match):
    with open(FICHIER_MEMOIRE_NBA, "a", encoding="utf-8") as f:
        f.write(id_match + "\n")


def publier_journal_dashboard_nba():
    """Upload FTP optionnel du journal vers PythonAnywhere (pattern NHL/foot) —
    no-op si le journal est déjà écrit directement dans `PA_DATA_DIR`."""
    if os.path.isdir(PA_DATA_DIR) and FICHIER_JOURNAL_NBA.startswith(PA_DATA_DIR):
        return
    if not os.path.exists(FICHIER_JOURNAL_NBA):
        return
    ftp_user = os.environ.get("PA_FTP_USER", "")
    ftp_pass = os.environ.get("PA_FTP_PASSWORD", "")
    if not ftp_user or not ftp_pass:
        return
    import ftplib
    ftp_host = os.environ.get("PA_FTP_HOST", "ftp.pythonanywhere.com")
    remote_dir = os.environ.get("PA_FTP_REMOTE_DIR", "/home/chienblanc/data")
    remote_name = os.path.basename(FICHIER_JOURNAL_NBA)
    try:
        with ftplib.FTP(ftp_host, timeout=30) as ftp:
            ftp.login(ftp_user, ftp_pass)
            ftp.cwd(remote_dir)
            with open(FICHIER_JOURNAL_NBA, "rb") as f:
                ftp.storbinary(f"STOR {remote_name}", f)
        log_nba(f"📤 Journal NBA uploadé → {ftp_host}{remote_dir}/{remote_name}")
    except Exception as e:
        log_nba(f"⚠️ Upload FTP journal NBA échoué : {e}", level="warning")


def enregistrer_transaction_nba(
    id_match, ext, dom, type_pari, vraie_cote_pari, investissement, cote_bookmaker,
    mu_ext=None, mu_dom=None, b2b_ext=False, b2b_dom=False,
):
    fichier_existe = os.path.isfile(FICHIER_JOURNAL_NBA)
    row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ID_Match": id_match,
        "Exterieur": ext,
        "Domicile": dom,
        "Pari": type_pari,
        "Vraie_Cote_Bot": vraie_cote_pari,
        "Cote_Prise": cote_bookmaker,
        "Cote_CLV": cote_bookmaker,
        "Mu_Ext": mu_ext,
        "Mu_Dom": mu_dom,
        "Edge(%)": investissement["edge"],
        "Risque(%)": investissement["pct_bankroll"],
        "Mise_€": investissement["mise"],
        "Statut": "EN ATTENTE",
        "P&L": "0.00",
        "B2B_Ext": "OUI" if b2b_ext else "NON",
        "B2B_Dom": "OUI" if b2b_dom else "NON",
        "Confiance_Kelly": 1.0,
    }
    with open(FICHIER_JOURNAL_NBA, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_COLONNES_NBA, extrasaction="ignore")
        if not fichier_existe:
            writer.writeheader()
        writer.writerow(row)
    publier_journal_dashboard_nba()


def envoyer_alerte_systeme_nba(message):
    if not NBA_TELEGRAM_TOKEN or not NBA_TELEGRAM_CHAT_ID:
        log_nba(f"⚠️ Alerte système NBA (Telegram absent) : {message}", level="warning")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{NBA_TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": NBA_TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        log_nba(f"⚠️ Erreur Telegram système NBA : {e}", level="warning")


def envoyer_alerte_nba(ext, dom, vraie_cote_pari, investissement, type_pari, dry_run=False):
    if not NBA_TELEGRAM_TOKEN or not NBA_TELEGRAM_CHAT_ID:
        log_nba("⚠️ Telegram NBA non configuré — alerte non envoyée.", level="warning")
        return
    prefix = "🧪 **[DRY RUN — SIMULATION]**\n\n" if dry_run else ""
    msg = (
        prefix + f"🚨 **SNIPER NBA DÉCLENCHÉ** 🚨\n\nExt: 🏀 **{ext}**\nDom: 🏠 **{dom}**\n"
        f"──────────────\n🎯 **ORDRE : PARIER {type_pari}**\n"
        f"🔥 Edge : **+{investissement['edge']}%**\n⚖️ Kelly : **{investissement['pct_bankroll']}%**\n"
        f"💵 **MISE : {investissement['mise']} €**\n──────────────\n📊 True Odds: {vraie_cote_pari}"
    )
    if dry_run:
        msg += "\n\n_(Paper / DRY RUN — journal écrit, pas de mise réelle)_"
    try:
        requests.post(
            f"https://api.telegram.org/bot{NBA_TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": NBA_TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        log_nba(f"⚠️ Erreur Telegram NBA : {e}", level="warning")


def compter_paris_en_attente_nba():
    if not os.path.exists(FICHIER_JOURNAL_NBA):
        return 0
    try:
        with open(FICHIER_JOURNAL_NBA, "r", encoding="utf-8") as f:
            return sum(1 for row in csv.DictReader(f) if row.get("Statut") == "EN ATTENTE")
    except Exception:
        return 0


def _extraire_cote_clv_nba(home, away, type_pari, cote_actuelle, odds_cache=None):
    """Cote Pinnacle actuelle pour un pari en attente — approximation v1 par
    parsing du libellé `Pari` (pas de ligne stockée séparément dans le journal)."""
    cotes = get_odds_for_match_nba(home, away, odds_cache)
    if not cotes:
        return cote_actuelle
    tp = str(type_pari).strip()
    tp_upper = tp.upper()

    if tp_upper.startswith("OVER") or tp_upper.startswith("UNDER"):
        m = re.search(r"[-+]?\d+\.?\d*", tp)
        if not m:
            return cote_actuelle
        ligne = round(float(m.group()), 1)
        side = "over" if tp_upper.startswith("OVER") else "under"
        prix = cotes.get("totals", {}).get(ligne)
        return str(prix[side]) if prix and side in prix else cote_actuelle

    if tp_upper.startswith("VICTOIRE"):
        if home in tp and "cote_1" in cotes:
            return str(cotes["cote_1"])
        if away in tp and "cote_2" in cotes:
            return str(cotes["cote_2"])
        return cote_actuelle

    m = re.search(r"[-+]\d+\.?\d*$", tp)
    if not m:
        return cote_actuelle
    ligne = float(m.group())
    if home in tp:
        prix = cotes.get("spreads", {}).get(round(ligne, 1))
        return str(prix["home"]) if prix and "home" in prix else cote_actuelle
    if away in tp:
        prix = cotes.get("spreads", {}).get(round(-ligne, 1))
        return str(prix["away"]) if prix and "away" in prix else cote_actuelle
    return cote_actuelle


def traquer_et_actualiser_clv_nba():
    if not os.path.exists(FICHIER_JOURNAL_NBA):
        return
    odds_cache = fetch_all_pinnacle_odds_nba()
    rows, maj = [], False
    with open(FICHIER_JOURNAL_NBA, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Statut") == "EN ATTENTE":
                dom, ext, type_pari = row.get("Domicile"), row.get("Exterieur"), row.get("Pari")
                nv_cote = _extraire_cote_clv_nba(dom, ext, type_pari, row.get("Cote_CLV"), odds_cache)
                if nv_cote != row.get("Cote_CLV"):
                    row["Cote_CLV"] = nv_cote
                    maj = True
            rows.append(row)
    if maj:
        with open(FICHIER_JOURNAL_NBA, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=JOURNAL_COLONNES_NBA, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        publier_journal_dashboard_nba()


def calculer_bankroll_dynamique_nba(capital_de_base=None):
    capital_de_base = NBA_BANKROLL if capital_de_base is None else capital_de_base
    if not os.path.exists(FICHIER_JOURNAL_NBA):
        return capital_de_base
    profit_total = 0.0
    try:
        with open(FICHIER_JOURNAL_NBA, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Statut") in ("GAGNÉ", "PERDU"):
                    profit_total += float(row["P&L"])
        return round(max(capital_de_base + profit_total, 10.0), 2)
    except Exception as e:
        log_nba(f"⚠️ Erreur calcul bankroll NBA : {e}", level="warning")
        return capital_de_base


def get_match_result_nba(game_id):
    """Score final `(score_away, score_home)` si le match est terminé
    (`gameStatus == 3`, module live cdn.nba.com), sinon ``None``."""
    try:
        from nba_api.live.nba.endpoints import boxscore as live_boxscore
        r = live_boxscore.BoxScore(game_id=str(game_id), timeout=15)
        d = r.get_dict()
        game = d.get("game", {})
        if game.get("gameStatus") != 3:
            return None
        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})
        return int(away.get("score", 0)), int(home.get("score", 0))
    except Exception:
        return None


def regler_pari_nba(pari, home, away, score_home, score_away):
    """Règle un pari (libellé `Pari` du journal) selon le score final — pure
    fonction réutilisée par le bot live (`lancer_la_balayeuse_nba`) ET le
    backtest (`backtest_nba.py`), garantissant la parité de règlement.

    Retourne ``(gagne: bool, push: bool)``.
    """
    pari_upper = str(pari).upper()
    if pari_upper.startswith("VICTOIRE"):
        gagne = (score_home > score_away and home in pari) or (score_away > score_home and away in pari)
        return gagne, False

    if pari_upper.startswith("OVER") or pari_upper.startswith("UNDER"):
        m = re.search(r"[-+]?\d+\.?\d*", pari)
        cut = float(m.group()) if m else None
        total_pts = score_home + score_away
        if cut is None:
            return False, False
        if total_pts == cut:
            return False, True
        if pari_upper.startswith("OVER"):
            return total_pts > cut, False
        return total_pts < cut, False

    m = re.search(r"[-+]\d+\.?\d*$", pari)
    ligne = float(m.group()) if m else None
    if ligne is None:
        return False, False
    marge = (score_home - score_away) if home in pari else (score_away - score_home)
    couverture = marge + ligne
    if couverture == 0:
        return False, True
    return couverture > 0, False


def lancer_la_balayeuse_nba():
    """Règle les paris `EN ATTENTE` dont le match est terminé (ML/SPREAD/TOTAL,
    avec gestion du push) — pattern `lancer_la_balayeuse` NHL."""
    if not os.path.exists(FICHIER_JOURNAL_NBA):
        return
    rows, modifie = [], False
    with open(FICHIER_JOURNAL_NBA, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Statut") != "EN ATTENTE":
                rows.append(row)
                continue
            game_id = str(row["ID_Match"]).split("|")[0]
            res = get_match_result_nba(game_id)
            if not res:
                rows.append(row)
                continue
            modifie = True
            score_ext, score_dom = res
            ext, dom, pari = row["Exterieur"], row["Domicile"], row["Pari"]
            mise, cote_book = float(row["Mise_€"]), float(row["Cote_Prise"])
            gagne, push = regler_pari_nba(pari, dom, ext, score_dom, score_ext)

            if push:
                row["Statut"] = "PUSH"
                row["P&L"] = "0.00"
            else:
                row["Statut"] = "GAGNÉ" if gagne else "PERDU"
                row["P&L"] = f"{round(mise * (cote_book - 1), 2) if gagne else -mise}"
            rows.append(row)
    if modifie:
        with open(FICHIER_JOURNAL_NBA, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=JOURNAL_COLONNES_NBA, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        publier_journal_dashboard_nba()


def _parse_utc_nba(date_str):
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
    except ValueError:
        return None


def get_nba_games_today():
    """Programme du jour (module live cdn.nba.com), filtré aux matchs pas
    encore terminés et dans la fenêtre `NBA_SCAN_HEURES_AVANCE`."""
    matchs = fetch_live_scoreboard()
    if not matchs:
        return []
    maintenant = datetime.now(timezone.utc)
    eligibles = []
    for m in matchs:
        if "final" in str(m.get("statut", "")).strip().lower():
            continue
        date_utc = _parse_utc_nba(m.get("date_utc"))
        if date_utc and (date_utc - maintenant).total_seconds() / 3600.0 > NBA_SCAN_HEURES_AVANCE:
            continue
        eligibles.append(m)
    return eligibles


def _executer_opportunite_nba(opp):
    """Envoie Telegram + journal pour une opportunité validée."""
    home, away = opp["home"], opp["away"]
    best = opp["best_pari"]
    log_nba(
        f"🎯 Edge {best['inv']['edge']}% [{best.get('marche', '?')}] — "
        f"{best['type']} ({away} @ {home}) mise {best['inv']['mise']} €"
        + (" [DRY RUN]" if NBA_DRY_RUN else "")
    )
    envoyer_alerte_nba(away, home, best["cote_vraie"], best["inv"], best["type"], dry_run=NBA_DRY_RUN)
    enregistrer_transaction_nba(
        opp["id_match"], away, home, best["type"], best["cote_vraie"], best["inv"], best["cote_book"],
        mu_ext=opp.get("mu_away"), mu_dom=opp.get("mu_home"),
        b2b_ext=opp.get("away_b2b", False), b2b_dom=opp.get("home_b2b", False),
    )
    enregistrer_notification_nba(opp["id_match"])


def run_sniper_nba():
    """Boucle live principale — scan du programme du jour, cotes Pinnacle,
    moteur mathématique + shrink marché, edge/Kelly, journal + Telegram
    (paper trading `NBA_DRY_RUN`, pattern `run_sniper` NHL)."""
    mode = "DRY RUN (paper trading)" if NBA_DRY_RUN else "LIVE"
    log_nba(f"🤖 Lancement Sniper NBA — mode {mode}")
    if NBA_DRY_RUN:
        log_nba("🧪 NBA_DRY_RUN actif : Telegram paper + écriture journal (pas de mise réelle).")
    if NBA_TOUS_CANDIDATS_ACTIF:
        log_nba("📚 Tous candidats actifs — chaque edge valide est pris (pas seulement le max du match).")
    cap_label = f"{NBA_MISE_MAX_PCT}% bankroll" if NBA_MISE_MAX_PCT > 0 else "Kelly pur (pas de cap %)"
    log_nba(f"💶 Cap mise : {cap_label} | journal → {FICHIER_JOURNAL_NBA}")
    log_nba(
        f"🎯 Marchés actifs : {', '.join(sorted(NBA_MARCHES_ACTIFS)) or 'aucun'} | "
        f"edge min {NBA_EDGE_MIN:.0%}"
        + (
            f" + dynamique jusqu'à +{NBA_EDGE_DYNAMIQUE_EXTRA:.0%} à 0 GP → plein à "
            f"{NBA_EDGE_DYNAMIQUE_GP_PLEIN:.0f} GP moy./match" if NBA_EDGE_DYNAMIQUE_ACTIF else ""
        )
    )
    if NBA_MARCHE_SHRINK_ACTIF:
        log_nba(
            f"⚖️ Shrinkage marché actif — confiance modèle {NBA_MODEL_TRUST_MIN:.0%} à 0 GP → "
            f"{NBA_MODEL_TRUST_MAX:.0%} à {NBA_MODEL_TRUST_GP_PLEIN:.0f}+ GP (reste : no-vig Pinnacle Shin)"
        )

    while True:
        try:
            lancer_la_balayeuse_nba()
            _invalider_pit_index_memo()

            nb_attente = compter_paris_en_attente_nba()
            log_nba(f"🕵️ Tracking CLV ({nb_attente} pari(s) en attente)...")
            traquer_et_actualiser_clv_nba()

            bankroll_actuelle = calculer_bankroll_dynamique_nba()
            log_nba(f"💰 Capital Dynamique Disponible : {bankroll_actuelle} €")

            odds_cache = fetch_all_pinnacle_odds_nba()
            matchs = get_nba_games_today()
            opportunites = []

            if not matchs:
                log_nba("🏀 Aucun match NBA éligible dans la fenêtre de scan — veille active.")
            else:
                log_nba(f"🏀 {len(matchs)} match(s) dans la fenêtre de scan.")

            for m in matchs:
                home, away = m.get("domicile"), m.get("exterieur")
                if not home or not away:
                    continue
                cotes_match = get_odds_for_match_nba(home, away, odds_cache, log_si_absent=True)
                if not cotes_match:
                    continue

                date_ref = (m.get("date_utc") or datetime.now(timezone.utc).isoformat())[:10]
                lignes_spread = list(cotes_match.get("spreads", {}).keys())
                lignes_total = list(cotes_match.get("totals", {}).keys())
                probas = calculer_probabilites_match_nba(
                    home, away, date_ref, lignes_spread=lignes_spread, lignes_total=lignes_total,
                )
                gp_moyen = (probas["rating_home"].get("gp", 0) + probas["rating_away"].get("gp", 0)) / 2.0
                probas_shrink = shrink_probabilites_vers_marche_nba(probas, cotes_match, gp_moyen)

                candidats = _construire_candidats_pari_nba(
                    {"home": home, "away": away}, probas_shrink, cotes_match, bankroll_actuelle, gp_moyen,
                )
                paris_retenus = _selectionner_paris_nba(candidats)
                if not paris_retenus:
                    log_nba(
                        f"— Pas d'edge ≥ {_edge_minimum_dynamique_nba(gp_moyen):.0%} "
                        f"(GP moy {gp_moyen:.0f}) — {away} @ {home}"
                    )
                    continue

                for pari in paris_retenus:
                    id_signal = f"{m['game_id']}|{pari['type']}"
                    if match_deja_notifie_nba(id_signal):
                        continue
                    opportunites.append({
                        "home": home, "away": away, "id_match": id_signal, "best_pari": pari,
                        "mu_home": probas.get("mu_home"), "mu_away": probas.get("mu_away"),
                        "home_b2b": probas.get("home_b2b", False), "away_b2b": probas.get("away_b2b", False),
                    })

            for opp in opportunites:
                _executer_opportunite_nba(opp)

            time.sleep(900)
        except Exception as e:
            log_nba(f"⚠️ Erreur système NBA : {e}", level="error")
            traceback.print_exc()
            time.sleep(60)


if __name__ == "__main__":
    manquants = []
    if not ODDS_API_KEY:
        manquants.append("API_ODDS_KEY")
    if not NBA_TELEGRAM_TOKEN:
        manquants.append("NBA_TELEGRAM_TOKEN/TELEGRAM_TOKEN")
    if not NBA_TELEGRAM_CHAT_ID:
        manquants.append("NBA_TELEGRAM_CHAT_ID/TELEGRAM_CHAT_ID")
    if manquants:
        log_nba(f"⚠️ Variables manquantes ({', '.join(manquants)}) — vérifier {load_project_env}", level="warning")
    run_sniper_nba()
