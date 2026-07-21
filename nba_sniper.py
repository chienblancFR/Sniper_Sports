"""Sniper NBA — bot value bets moneyline/spread/total (paper trading).

Ce module contient pour l'instant la **couche de collecte + cache PIT**
(point-in-time, sans look-ahead) :

- Récupération bulk (1 appel/saison/measure_type) des box scores et stats
  avancées par match via `nba_api` (stats.nba.com) : ``Base`` (score, W/L),
  ``Advanced`` (OFF_RATING/DEF_RATING/PACE/POSS) et ``Four Factors``
  (EFG%/FTA_RATE/TOV%/OREB% + équivalents adverses).
- Fusion en un enregistrement par (match, équipe) et construction d'un index
  PIT walk-forward : pour chaque équipe, la liste chronologique des snapshots
  de stats cumulées **strictement avant** chaque match (pondérées par
  possessions), pattern identique au PIT MoneyPuck de `nhl_sniper_omega.py`.
- Calendrier saison (``ScheduleLeagueV2``, heures de tip-off UTC) et
  scoreboard du jour (module live cdn.nba.com) pour la fenêtre live à venir.
- Cache disque JSON (``data/nba/``) : permanent pour les saisons terminées,
  TTL (``NBA_PIT_CACHE_JOURS``) pour la saison en cours ; retry/backoff sur
  les appels stats.nba.com (timeouts transitoires observés en exploration).

Le moteur mathématique (ratings ajustés/shrinkage, distribution bivariée,
probabilités de marché), l'intégration Odds API/edge/Kelly et la boucle live
`run_sniper()` sont des phases ultérieures du plan — pas encore implémentées
ici.
"""
import json
import logging
import os
import time

from nba_api.stats.endpoints import scheduleleaguev2, teamgamelogs
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard

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
