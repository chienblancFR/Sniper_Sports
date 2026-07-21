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

L'intégration Odds API/edge/Kelly et la boucle live `run_sniper()` sont des
phases ultérieures du plan — pas encore implémentées ici.
"""
import json
import logging
import math
import os
import time
from datetime import datetime

from nba_api.stats.endpoints import scheduleleaguev2, teamgamelogs
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard

import nba_params
from config_env import load_project_env

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
