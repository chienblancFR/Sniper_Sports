"""
backtest_nba.py — Back-test NBA aligné sur nba_sniper.py
=========================================================
Parité live : `calculer_probabilites_match_nba`, `shrink_probabilites_vers_marche_nba`,
`_construire_candidats_pari_nba`, `_selectionner_paris_nba`, `regler_pari_nba`,
`_extraire_cote_clv_nba` (CLV) — aucune logique de marché/règlement dupliquée,
tout est importé depuis `nba_sniper.py` (mirroring `backtest_nhl.py`).

Usage :
  python backtest_nba.py --collect              # Matchs NBA (nba_api) + cotes Pinnacle historiques
  python backtest_nba.py --simulate             # Simulation walk-forward (parité bot live)
  python backtest_nba.py --report               # Rapport console + CSV
  python backtest_nba.py                        # Collect + simulate + report

  python backtest_nba.py --reset                # Vide signaux, garde la DB
  python backtest_nba.py --reset-full           # Supprime backtest_nba.db + CSV
  python backtest_nba.py --collect --odds-only  # Recollecte cotes uniquement
  python backtest_nba.py --fill-close           # Complète closes manquantes (tip-5/15/30/60)
  python backtest_nba.py --collect --saisons 2024,2025,2026
  python backtest_nba.py --tune --saisons 2022,2023,2024,2025,2026
      # Calibre n_prior / demi-vie / HCA / sigma_team / rho / B2B (Brier ML) → nba_params_tuned.json

  python backtest_nba.py --tune --saisons 2022,2023,2024 --holdout-saisons 2025,2026
      # Validation train/test chronologique : tune UNIQUEMENT sur --saisons (train),
      # puis simulate+report sur --holdout-saisons (jamais vues en tune) → ROI/CLV/Brier
      # honnêtes hors-échantillon. Écrit nba_params_tuned.json (train only) + deux CSV
      # (NBA_BT_TRAIN_RESULTS_CSV / NBA_BT_HOLDOUT_RESULTS_CSV) pour comparaison directe.

Limites documentées (v1, voir le plan `sniper_nba_value_bets`) :
  - Pas de line movement / steam (pas d'historique snapshots intra-cycle)
  - Pas d'absences stars / injury report (proxy minutes/impact = phase 2 roadmap)
  - Cotes Pinnacle historiques : snapshot H-X avant tip-off par match (repli H-1 / H-0.5)
  - Résultats des matchs dérivés des box scores `nba_api` (teamgamelogs, déjà PIT-safe côté
    moteur) ; tip-off exact (`start_utc`) via `ScheduleLeagueV2` (bucketing des snapshots cotes)

Prérequis : `API_ODDS_KEY` avec accès à l'endpoint `/v4/historical/` (plan payant Odds API).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import aiohttp

from config_env import env_files_hint, load_project_env

load_project_env("nba")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import bot live — logique métier partagée (parité stricte)
import nba_sniper as nba
import nba_params

DB_PATH = os.environ.get("NBA_BT_DB", "backtest_nba.db")
RESULTS_CSV = os.environ.get("NBA_BT_RESULTS_CSV", "backtest_nba_results.csv")
# Validation train/test chronologique (--tune --holdout-saisons) : CSV séparés pour
# comparer directement le in-sample (train) et le out-of-sample (holdout).
TRAIN_RESULTS_CSV = os.environ.get("NBA_BT_TRAIN_RESULTS_CSV", "backtest_nba_train_results.csv")
HOLDOUT_RESULTS_CSV = os.environ.get("NBA_BT_HOLDOUT_RESULTS_CSV", "backtest_nba_holdout_results.csv")
SPORT_KEY = "basketball_nba"
DEFAULT_SAISONS = [
    int(s.strip()) for s in os.environ.get("NBA_BT_SAISONS", "2022,2023,2024,2025,2026").split(",") if s.strip()
]
API_ODDS_KEY = os.environ.get("API_ODDS_KEY", "")
BANKROLL_BT = float(os.environ.get("NBA_BT_BANKROLL", os.environ.get("NBA_BANKROLL", "1000")))
# Cotes de prise : par match, X h avant tip-off (repli si ligne pas encore publiée)
ODDS_PRISE_HEURES = float(os.environ.get("NBA_BT_ODDS_HEURES_AVANT", "3"))
ODDS_PRISE_FALLBACK_HEURES = [
    float(s.strip()) for s in os.environ.get("NBA_BT_ODDS_FALLBACK_HEURES", "1,0.5").split(",") if s.strip()
]
NBA_BT_MINUTES_AVANT_TIPOFF = float(os.environ.get("NBA_BT_MINUTES_AVANT_TIPOFF", "5"))
# Fallbacks clôture (minutes avant tip) si snapshot tip-X vide — ordre croissant recommandé.
ODDS_CLOSE_FALLBACK_MINUTES = [
    float(s.strip())
    for s in os.environ.get("NBA_BT_ODDS_CLOSE_FALLBACK_MINUTES", "5,15,30,60").split(",")
    if s.strip()
]

# ─────────────────────────────────────────────────────────────
# Cache mémoire : records/regroupements bruts nba_api (indépendants des
# hyperparamètres) — accélère --simulate et surtout --tune (des centaines
# d'évaluations du même dataset avec des params différents).
# ─────────────────────────────────────────────────────────────
_records_cache: dict[int, list] = {}
_orig_fusionner = nba._fusionner_logs_saison


def _fusionner_logs_saison_cached(season_year, force=False):
    if force or season_year not in _records_cache:
        _records_cache[season_year] = _orig_fusionner(season_year, force=force)
    return _records_cache[season_year]


nba._fusionner_logs_saison = _fusionner_logs_saison_cached

_grouped_cache: dict[int, dict] = {}
_orig_grouper = nba._grouper_records_par_equipe


def _grouper_records_par_equipe_cached(records):
    key = id(records)
    if key not in _grouped_cache:
        _grouped_cache[key] = _orig_grouper(records)
    return _grouped_cache[key]


nba._grouper_records_par_equipe = _grouper_records_par_equipe_cached

# `_snapshot_ligue_decay` (moyenne ligue decay-pondérée) est appelé 1x/équipe
# pour CHAQUE match — coûteux en --tune (des centaines d'évaluations du même
# dataset). Beaucoup de matchs partagent la même date (plusieurs matchs/soir)
# → cache keyed (par_equipe, date_ref, half_life) : gain ~10x sur --tune.
_snapshot_cache: dict[tuple, dict] = {}
_orig_snapshot_ligue = nba._snapshot_ligue_decay


def _snapshot_ligue_decay_cached(par_equipe, date_ref, half_life_jours):
    key = (id(par_equipe), date_ref, half_life_jours)
    if key not in _snapshot_cache:
        _snapshot_cache[key] = _orig_snapshot_ligue(par_equipe, date_ref, half_life_jours)
    return _snapshot_cache[key]


nba._snapshot_ligue_decay = _snapshot_ligue_decay_cached


# ─────────────────────────────────────────────────────────────
# DB
# ─────────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nba_games (
            game_id   TEXT PRIMARY KEY,
            season    INTEGER,
            date_utc  TEXT,
            start_utc TEXT,
            home      TEXT,
            away      TEXT,
            gh        INTEGER,
            ga        INTEGER
        );
        CREATE TABLE IF NOT EXISTS nba_odds_prise (
            game_id    TEXT PRIMARY KEY,
            cotes_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS nba_odds_cloture (
            game_id    TEXT PRIMARY KEY,
            cotes_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS nba_signaux (
            game_id      TEXT,
            season       INTEGER,
            date_utc     TEXT,
            marche       TEXT,
            type_pari    TEXT,
            home         TEXT,
            away         TEXT,
            cote_prise   REAL,
            cote_cloture REAL,
            cote_modele  REAL,
            prob_modele  REAL,
            edge_pct     REAL,
            mise         REAL,
            gh           INTEGER,
            ga           INTEGER,
            gagne        INTEGER,
            pnl          REAL,
            clv          REAL,
            mu_home      REAL,
            mu_away      REAL,
            clv_exact    INTEGER,
            clv_ligne    REAL,
            PRIMARY KEY (game_id, marche, type_pari)
        );
    """)
    # Migration douce si DB antérieure
    cols = {r[1] for r in conn.execute("PRAGMA table_info(nba_signaux)")}
    if "clv_exact" not in cols:
        conn.execute("ALTER TABLE nba_signaux ADD COLUMN clv_exact INTEGER")
    if "clv_ligne" not in cols:
        conn.execute("ALTER TABLE nba_signaux ADD COLUMN clv_ligne REAL")
    conn.commit()


def _normaliser_cles_lignes(cotes: dict) -> dict:
    """JSON sérialise les clés dict en str ('-5.5') — reconverties en float
    pour matcher les clés `lignes_spread`/`lignes_total` du moteur (`nba_sniper.py`)."""
    for champ in ("spreads", "totals"):
        if champ in cotes and isinstance(cotes[champ], dict):
            cotes[champ] = {float(k): v for k, v in cotes[champ].items()}
    return cotes


def _charger_odds_json(conn: sqlite3.Connection, table: str) -> dict[str, dict]:
    out = {}
    for game_id, cotes_json in conn.execute(f"SELECT game_id, cotes_json FROM {table}"):
        try:
            out[game_id] = _normaliser_cles_lignes(json.loads(cotes_json))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def _parse_utc(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────
# Phase 1a — Collecte matchs (résultats via ScheduleLeagueV2, déjà PIT-safe
# côté moteur puisque le walk-forward ne regarde que records < date_ref)
# ─────────────────────────────────────────────────────────────
def collecter_matchs_saison(conn: sqlite3.Connection, season: int, force: bool = False) -> int:
    matchs = nba.fetch_saison_schedule(season_year=season, force=force)
    inserted = 0
    for m in matchs:
        statut = str(m.get("statut") or "")
        gh, ga = m.get("score_domicile"), m.get("score_exterieur")
        if "final" not in statut.lower() or gh is None or ga is None:
            continue
        home, away = m.get("domicile"), m.get("exterieur")
        date_utc = str(m.get("date_utc") or "")[:10]
        if not home or not away or not date_utc:
            continue
        try:
            conn.execute(
                """INSERT OR REPLACE INTO nba_games
                   (game_id, season, date_utc, start_utc, home, away, gh, ga)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(m["game_id"]), season, date_utc, m.get("date_utc"), home, away, int(gh), int(ga)),
            )
            inserted += 1
        except (sqlite3.Error, ValueError, TypeError):
            continue
    conn.commit()
    return inserted


# ─────────────────────────────────────────────────────────────
# Phase 1b — Collecte cotes historiques Pinnacle (Odds API /v4/historical/)
# ─────────────────────────────────────────────────────────────
async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> object:
    try:
        async with session.get(url, params=params, timeout=30) as resp:
            if resp.status == 200:
                return await resp.json()
            print(f"  ⚠️ HTTP {resp.status} — {url[:90]}")
    except Exception as e:
        print(f"  ⚠️ fetch error : {e}")
    return None


def _trouver_game_id_nba(event: dict, games_bucket: list[dict]) -> str | None:
    h_tri = nba._resoudre_tricode_depuis_nom_odds(event.get("home_team", ""))
    a_tri = nba._resoudre_tricode_depuis_nom_odds(event.get("away_team", ""))
    if not h_tri or not a_tri:
        return None
    for g in games_bucket:
        if g["home"] == h_tri and g["away"] == a_tri:
            return g["game_id"]
    return None


async def collecter_odds_instant(
    session: aiohttp.ClientSession, conn: sqlite3.Connection,
    date_utc: datetime, table: str, games_bucket: list[dict],
) -> int:
    if not API_ODDS_KEY:
        return 0
    date_str = date_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/odds"
        f"?apiKey={API_ODDS_KEY}&regions=eu&markets=h2h,spreads,totals"
        f"&oddsFormat=decimal&bookmakers=pinnacle&date={date_str}"
    )
    raw = await fetch_json(session, url)
    if not raw:
        return 0
    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    if not isinstance(data, list):
        return 0
    inserted = 0
    for event in data:
        gid = _trouver_game_id_nba(event, games_bucket)
        if not gid:
            continue
        parsed = nba._parse_pinnacle_game_nba(event)
        if not parsed:
            continue
        conn.execute(f"INSERT OR REPLACE INTO {table} (game_id, cotes_json) VALUES (?, ?)", (gid, json.dumps(parsed)))
        inserted += 1
    if inserted:
        conn.commit()
    return inserted


def _limite_prise_avant_tipoff(start_utc: datetime) -> datetime:
    return start_utc - timedelta(minutes=NBA_BT_MINUTES_AVANT_TIPOFF)


def _instant_prise_pour_match(game: dict, heures_avant: float) -> datetime | None:
    start = _parse_utc(game["start_utc"])
    if not start:
        return None
    dt = start - timedelta(hours=heures_avant)
    limite = _limite_prise_avant_tipoff(start)
    return min(dt, limite)


def _instant_cloture_pour_match(game: dict, minutes_avant: float) -> datetime | None:
    """Snapshot de clôture : tip − N minutes, jamais à/après tip−1 min."""
    start = _parse_utc(game["start_utc"])
    if not start:
        return None
    dt = start - timedelta(minutes=max(minutes_avant, 1.0))
    limite = start - timedelta(minutes=1.0)
    return min(dt, limite)


def _bucketiser_par_instant(games: list[dict], heures_avant: float) -> dict[datetime, list[dict]]:
    buckets: dict[datetime, list[dict]] = defaultdict(list)
    for game in games:
        dt = _instant_prise_pour_match(game, heures_avant)
        if dt:
            buckets[dt].append(game)
    return buckets


def _bucketiser_cloture_par_instant(games: list[dict], minutes_avant: float) -> dict[datetime, list[dict]]:
    buckets: dict[datetime, list[dict]] = defaultdict(list)
    for game in games:
        dt = _instant_cloture_pour_match(game, minutes_avant)
        if dt:
            buckets[dt].append(game)
    return buckets


def _games_sans_cotes(conn: sqlite3.Connection, table: str, game_ids: list[str]) -> list[str]:
    if not game_ids:
        return []
    placeholders = ",".join("?" * len(game_ids))
    couverts = {r[0] for r in conn.execute(f"SELECT game_id FROM {table} WHERE game_id IN ({placeholders})", game_ids)}
    return [gid for gid in game_ids if gid not in couverts]


async def _collecter_odds_heures(
    conn: sqlite3.Connection, session: aiohttp.ClientSession, games: list[dict], heures_avant: float,
) -> int:
    total = 0
    for dt, bucket in sorted(_bucketiser_par_instant(games, heures_avant).items()):
        total += await collecter_odds_instant(session, conn, dt, "nba_odds_prise", bucket)
        await asyncio.sleep(0.35)
    return total


async def _collecter_odds_cloture(
    conn: sqlite3.Connection,
    session: aiohttp.ClientSession,
    games: list[dict],
    *,
    only_missing: bool = False,
) -> int:
    """Collecte cotes de clôture avec fallbacks tip−5/15/30/60 min.

    `only_missing=True` : ne touche pas les matchs déjà couverts (mode --fill-close).
    """
    if not games:
        return 0
    minutes_list = ODDS_CLOSE_FALLBACK_MINUTES or [NBA_BT_MINUTES_AVANT_TIPOFF]
    # Toujours commencer par la marge tip-off configurée si absente de la liste
    if NBA_BT_MINUTES_AVANT_TIPOFF not in minutes_list:
        minutes_list = [NBA_BT_MINUTES_AVANT_TIPOFF] + list(minutes_list)
    # Ordre croissant (du plus proche du tip au plus loin)
    minutes_list = sorted({float(m) for m in minutes_list if m > 0})
    if only_missing:
        # Les manquants ont déjà échoué au tip-N primaire lors de la collecte initiale
        # → démarrer aux fallbacks plus éloignés pour économiser le quota Odds API.
        plus_loin = [m for m in minutes_list if m > NBA_BT_MINUTES_AVANT_TIPOFF + 1e-9]
        minutes_list = plus_loin or minutes_list

    if only_missing:
        restants = _games_sans_cotes(conn, "nba_odds_cloture", [g["game_id"] for g in games])
    else:
        restants = [g["game_id"] for g in games]

    total = 0
    for minutes in minutes_list:
        if not restants:
            break
        subset = [g for g in games if g["game_id"] in restants]
        print(
            f"  📊 Cotes clôture : tip-{minutes:g} min — {len(subset)} match(s)"
            + (" (manquants seulement)" if only_missing else ""),
            flush=True,
        )
        n_snap = 0
        for i, (dt_close, bucket) in enumerate(sorted(_bucketiser_cloture_par_instant(subset, minutes).items()), 1):
            total += await collecter_odds_instant(session, conn, dt_close, "nba_odds_cloture", bucket)
            n_snap += 1
            if i % 200 == 0:
                print(f"     … clôture tip-{minutes:g} {i} snapshots", flush=True)
            await asyncio.sleep(0.35)
        restants = _games_sans_cotes(conn, "nba_odds_cloture", restants)
        if restants:
            print(f"     ↪ {len(restants)} match(s) encore sans close après tip-{minutes:g}", flush=True)
        else:
            print(f"     ✅ couverture clôture complète après tip-{minutes:g} ({n_snap} snapshots)", flush=True)
    if restants:
        print(f"  ⚠️ {len(restants)} match(s) toujours sans cote de clôture Pinnacle")
    return total


async def collecter_odds_saison(conn: sqlite3.Connection, session: aiohttp.ClientSession, season: int) -> int:
    games = [
        {"game_id": r[0], "start_utc": r[1], "home": r[2], "away": r[3]}
        for r in conn.execute("SELECT game_id, start_utc, home, away FROM nba_games WHERE season=?", (season,))
    ]
    if not games:
        print(f"  ⚠️ Aucun match en base pour saison {season}")
        return 0

    conn.execute("DELETE FROM nba_odds_prise WHERE game_id IN (SELECT game_id FROM nba_games WHERE season=?)", (season,))
    conn.execute("DELETE FROM nba_odds_cloture WHERE game_id IN (SELECT game_id FROM nba_games WHERE season=?)", (season,))
    conn.commit()

    print(f"  📊 Cotes prise : H-{ODDS_PRISE_HEURES:g} par match (repli H-{'/H-'.join(f'{h:g}' for h in ODDS_PRISE_FALLBACK_HEURES)})")
    total = await _collecter_odds_heures(conn, session, games, ODDS_PRISE_HEURES)

    restants = _games_sans_cotes(conn, "nba_odds_prise", [g["game_id"] for g in games])
    for fb_h in ODDS_PRISE_FALLBACK_HEURES:
        if not restants or fb_h >= ODDS_PRISE_HEURES:
            continue
        subset = [g for g in games if g["game_id"] in restants]
        print(f"     ↪ repli H-{fb_h:g} pour {len(subset)} match(s) sans ligne", flush=True)
        total += await _collecter_odds_heures(conn, session, subset, fb_h)
        restants = _games_sans_cotes(conn, "nba_odds_prise", restants)

    total += await _collecter_odds_cloture(conn, session, games, only_missing=False)

    n_cov = conn.execute(
        "SELECT COUNT(DISTINCT o.game_id) FROM nba_odds_prise o JOIN nba_games g ON g.game_id=o.game_id WHERE g.season=?",
        (season,),
    ).fetchone()[0]
    n_close = conn.execute(
        "SELECT COUNT(DISTINCT o.game_id) FROM nba_odds_cloture o JOIN nba_games g ON g.game_id=o.game_id WHERE g.season=?",
        (season,),
    ).fetchone()[0]
    print(
        f"  ✅ Saison {season} : {total} événement(s) cotes | "
        f"prise {n_cov}/{len(games)} | clôture {n_close}/{len(games)}"
    )
    if restants:
        print(f"     ⚠️ {len(restants)} match(s) toujours sans cote Pinnacle de prise")
    return total


async def phase_fill_close(conn: sqlite3.Connection, saisons: list[int]) -> None:
    """Complète uniquement les cotes de clôture manquantes (ne touche pas aux prises)."""
    print("\n" + "=" * 60)
    print("📥  FILL-CLOSE — cotes de clôture manquantes")
    print("=" * 60)
    if not API_ODDS_KEY:
        print(f"\n⚠️ API_ODDS_KEY absente ({env_files_hint('nba')}) — impossible.")
        return
    async with aiohttp.ClientSession() as session:
        for season in saisons:
            games = [
                {"game_id": r[0], "start_utc": r[1], "home": r[2], "away": r[3]}
                for r in conn.execute(
                    "SELECT game_id, start_utc, home, away FROM nba_games WHERE season=?", (season,),
                )
            ]
            manquants = _games_sans_cotes(conn, "nba_odds_cloture", [g["game_id"] for g in games])
            print(f"\n🔄 Saison {season} — {len(manquants)}/{len(games)} match(s) sans close")
            if not manquants:
                continue
            await _collecter_odds_cloture(conn, session, games, only_missing=True)
            n_close = conn.execute(
                "SELECT COUNT(DISTINCT o.game_id) FROM nba_odds_cloture o "
                "JOIN nba_games g ON g.game_id=o.game_id WHERE g.season=?",
                (season,),
            ).fetchone()[0]
            print(f"  ✅ Saison {season} clôture : {n_close}/{len(games)}")
    print("\n✅ Fill-close terminé.")


async def phase_collecte(conn: sqlite3.Connection, saisons: list[int], odds_only: bool = False) -> None:
    print("\n" + "=" * 60)
    print("📥  PHASE 1 — COLLECTE NBA")
    print("=" * 60)
    if not odds_only:
        for season in saisons:
            n = collecter_matchs_saison(conn, season)
            print(f"🔄 Saison {season} ({season - 1}-{str(season)[-2:]}) — {n} match(s) terminé(s) indexés")
            # Warm cache teamgamelogs (saison + prior N-1) pour le walk-forward
            for sy in {season, season - 1}:
                n_rec = len(nba._fusionner_logs_saison(sy))
                print(f"   📦 teamgamelogs {sy - 1}-{str(sy)[-2:]} : {n_rec} lignes équipe-match")
    if not API_ODDS_KEY:
        print(f"\n⚠️ API_ODDS_KEY absente ({env_files_hint('nba')}) — matchs indexés mais pas de cotes.")
        return
    async with aiohttp.ClientSession() as session:
        for season in saisons:
            print(f"\n🔄 Cotes Pinnacle saison {season}")
            await collecter_odds_saison(conn, session, season)
    print("\n✅ Phase 1 terminée.")


# ─────────────────────────────────────────────────────────────
# Phase 2 — Simulation (parité bot live)
# ─────────────────────────────────────────────────────────────
def _trouver_cote_cloture_nba(odds_close: dict, game_id: str, type_pari: str, home: str, away: str) -> tuple[float | None, bool]:
    """Retourne ``(cote_cloture, exact)`` — `exact=False` si ligne proche (tolérance)."""
    cotes = odds_close.get(game_id)
    if not cotes:
        return None, False
    cache = {(home, away): cotes}
    val, meta = nba._extraire_cote_clv_nba(home, away, type_pari, None, cache, retourner_meta=True)
    if val is None:
        return None, False
    try:
        return float(val), bool(meta.get("exact"))
    except (TypeError, ValueError):
        return None, False


def simuler_saison(conn: sqlite3.Connection, season: int, bankroll: float) -> tuple[float, list[dict]]:
    odds_prise = _charger_odds_json(conn, "nba_odds_prise")
    odds_close = _charger_odds_json(conn, "nba_odds_cloture")
    games = list(conn.execute(
        "SELECT game_id, date_utc, home, away, gh, ga FROM nba_games WHERE season=? ORDER BY date_utc, game_id",
        (season,),
    ))
    signaux: list[dict] = []
    skipped = {"pas_cotes": 0, "pas_edge": 0}
    # Mises dimensionnées sur bankroll de référence FIXE (pas de compounding
    # Kelly sur 5 saisons × tous-candidats → explosion numérique du P&L).
    bankroll_sizing = BANKROLL_BT

    for game_id, date_utc, home, away, gh, ga in games:
        cotes_match = odds_prise.get(game_id)
        if not cotes_match:
            skipped["pas_cotes"] += 1
            continue

        lignes_spread = list(cotes_match.get("spreads", {}).keys())
        lignes_total = list(cotes_match.get("totals", {}).keys())
        probas = nba.calculer_probabilites_match_nba(
            home, away, date_utc, season_year=season, lignes_spread=lignes_spread, lignes_total=lignes_total,
        )
        gp_moyen = (probas["rating_home"].get("gp", 0) + probas["rating_away"].get("gp", 0)) / 2.0
        probas_shrink = nba.shrink_probabilites_vers_marche_nba(probas, cotes_match, gp_moyen)

        candidats = nba._construire_candidats_pari_nba(
            {"home": home, "away": away}, probas_shrink, cotes_match, bankroll_sizing, gp_moyen,
        )
        paris_retenus = nba._selectionner_paris_nba(candidats)
        if not paris_retenus:
            skipped["pas_edge"] += 1
            continue

        for best in paris_retenus:
            marche = best.get("marche", "?")
            type_pari = best["type"]
            cote_prise = float(best["cote_book"])
            cote_modele = float(best["cote_vraie"])
            prob = round(1.0 / cote_modele, 4) if cote_modele > 1 else None
            mise = float(best["inv"]["mise"])

            gagne, push = nba.regler_pari_nba(type_pari, home, away, int(gh), int(ga))
            if push:
                pnl = 0.0
            else:
                pnl = round(mise * (cote_prise - 1), 2) if gagne else round(-mise, 2)
            bankroll = round(bankroll + pnl, 2)

            cote_cloture, clv_exact = _trouver_cote_cloture_nba(odds_close, game_id, type_pari, home, away)
            clv = round((cote_prise / cote_cloture) - 1, 4) if cote_cloture and cote_cloture > 1 else None
            cotes_close_match = odds_close.get(game_id)
            clv_ligne = nba.calculer_clv_ligne_nba(type_pari, home, away, cotes_close_match) if cotes_close_match else None

            signaux.append({
                "game_id": game_id, "season": season, "date_utc": date_utc,
                "marche": marche, "type_pari": type_pari, "home": home, "away": away,
                "cote_prise": cote_prise, "cote_cloture": cote_cloture, "cote_modele": cote_modele,
                "prob_modele": prob, "edge_pct": best["inv"]["edge"], "mise": mise,
                "gh": gh, "ga": ga, "gagne": (None if push else int(gagne)), "pnl": pnl, "clv": clv,
                "clv_exact": (None if clv is None else int(clv_exact)),
                "clv_ligne": clv_ligne,
                "mu_home": probas.get("mu_home"), "mu_away": probas.get("mu_away"),
            })

    print(f"  Saison {season} : {len(signaux)} signaux | skip cotes={skipped['pas_cotes']} edge={skipped['pas_edge']}")
    return bankroll, signaux


def persister_signaux(conn: sqlite3.Connection, signaux: list[dict]) -> None:
    conn.execute("DELETE FROM nba_signaux")
    if signaux:
        conn.executemany(
            """INSERT OR REPLACE INTO nba_signaux (
                game_id, season, date_utc, marche, type_pari, home, away,
                cote_prise, cote_cloture, cote_modele, prob_modele, edge_pct,
                mise, gh, ga, gagne, pnl, clv, mu_home, mu_away, clv_exact, clv_ligne
            ) VALUES (
                :game_id, :season, :date_utc, :marche, :type_pari, :home, :away,
                :cote_prise, :cote_cloture, :cote_modele, :prob_modele, :edge_pct,
                :mise, :gh, :ga, :gagne, :pnl, :clv, :mu_home, :mu_away, :clv_exact, :clv_ligne
            )""",
            signaux,
        )
    conn.commit()


def _verifier_pit_disponible(season: int) -> bool:
    records = nba._fusionner_logs_saison(season)
    if records:
        return True
    print(f"\n❌ Aucune donnée nba_api pour la saison {season} — simulation impossible.")
    print("   Vérifier la connectivité stats.nba.com ou relancer --collect.")
    return False


def phase_simulation(conn: sqlite3.Connection, saisons: list[int]) -> None:
    print("\n" + "=" * 60)
    print("🔬  PHASE 2 — SIMULATION (parité bot live)")
    print("=" * 60)
    print(f"  Bankroll sizing : {BANKROLL_BT:.0f} € fixe (pas de compounding Kelly) | pipeline : moteur → shrink → Kelly → journal")

    n_odds = conn.execute("SELECT COUNT(*) FROM nba_odds_prise").fetchone()[0]
    if n_odds == 0:
        print("\n  ⚠️ Aucune cote de prise — relancez --collect (API_ODDS_KEY requise).")
        return
    if not _verifier_pit_disponible(saisons[0]):
        return

    bankroll = BANKROLL_BT
    tous: list[dict] = []
    for season in saisons:
        bankroll, sigs = simuler_saison(conn, season, bankroll)
        tous.extend(sigs)

    persister_signaux(conn, tous)
    print(f"\n✅ Phase 2 terminée — {len(tous)} signaux | bankroll finale {bankroll:.2f} €")


# ─────────────────────────────────────────────────────────────
# Phase 3 — Rapport
# ─────────────────────────────────────────────────────────────
def _brier(signaux: list[dict]) -> float | None:
    clos = [s for s in signaux if s.get("gagne") is not None and s.get("prob_modele") is not None]
    if not clos:
        return None
    return sum((s["prob_modele"] - s["gagne"]) ** 2 for s in clos) / len(clos)


def _resume_segment(signaux: list[dict], label: str) -> None:
    if not signaux:
        print(f"  {label:<12} — aucun signal")
        return
    n = len(signaux)
    clos = [s for s in signaux if s.get("gagne") is not None]
    wins = sum(s["gagne"] for s in clos)
    pnl = sum(s["pnl"] for s in signaux)
    mise_tot = sum(s["mise"] for s in signaux)
    roi = pnl / mise_tot if mise_tot else 0.0
    clvs = [s["clv"] for s in signaux if s.get("clv") is not None]
    clv_exact = [s["clv"] for s in signaux if s.get("clv") is not None and s.get("clv_exact") == 1]
    clv_lignes = [s["clv_ligne"] for s in signaux if s.get("clv_ligne") is not None]
    clv_moy = sum(clvs) / len(clvs) if clvs else float("nan")
    clv_ex_moy = sum(clv_exact) / len(clv_exact) if clv_exact else float("nan")
    clv_lig_moy = sum(clv_lignes) / len(clv_lignes) if clv_lignes else float("nan")
    cov = len(clvs) / n if n else 0.0
    brier = _brier(signaux)
    brier_s = f"{brier:.4f}" if brier is not None else "—"
    wr = wins / len(clos) if clos else 0.0
    clv_ex_s = f"{clv_ex_moy:+.2%}" if clv_exact else "—"
    clv_lig_s = f"{clv_lig_moy:+.2f}pt" if clv_lignes else "—"
    print(
        f"  {label:<12} n={n:>4}  WR={wr:.1%}  ROI={roi:+.1%}  P&L={pnl:+.1f}u  "
        f"CLV={clv_moy:+.2%} (exact {clv_ex_s}, cov={cov:.0%})  "
        f"CLVligne={clv_lig_s}  Brier={brier_s}"
    )


def exporter_csv(signaux: list[dict], path: str = RESULTS_CSV) -> None:
    if not signaux:
        return
    cols = list(signaux[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(signaux)
    print(f"\n📁 Export → {path}")


def phase_rapport(conn: sqlite3.Connection, csv_path: str = RESULTS_CSV, label: str | None = None) -> None:
    print("\n" + "=" * 60)
    print("📊  PHASE 3 — RAPPORT" + (f"  [{label}]" if label else ""))
    print("=" * 60)
    rows = conn.execute("SELECT * FROM nba_signaux ORDER BY date_utc").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM nba_signaux LIMIT 0").description]
    signaux = [dict(zip(cols, row)) for row in rows]
    if not signaux:
        print("  Aucun signal — lancez --simulate après --collect.")
        return

    exporter_csv(signaux, csv_path)
    _resume_segment(signaux, "GLOBAL")
    for marche in sorted({s["marche"] for s in signaux}):
        _resume_segment([s for s in signaux if s["marche"] == marche], marche)
    for season in sorted({s["season"] for s in signaux}):
        _resume_segment([s for s in signaux if s["season"] == season], f"S{season}")

    n_clv = sum(1 for s in signaux if s.get("clv") is not None)
    n_exact = sum(1 for s in signaux if s.get("clv") is not None and s.get("clv_exact") == 1)
    n_ligne = sum(1 for s in signaux if s.get("clv_ligne") is not None)
    clv_lignes = [s["clv_ligne"] for s in signaux if s.get("clv_ligne") is not None]
    if n_clv:
        print(
            f"\n  CLV-prix coverage : {n_clv}/{len(signaux)} ({n_clv / len(signaux):.0%}) | "
            f"dont exacte (même ligne) : {n_exact}/{n_clv} ({n_exact / n_clv:.0%} des CLV mesurées)"
        )
    else:
        print("\n  CLV-prix coverage : 0")
    if n_ligne:
        pos = sum(1 for v in clv_lignes if v > 0) / n_ligne
        print(
            f"  CLV-ligne coverage : {n_ligne}/{len(signaux)} ({n_ligne / len(signaux):.0%}) | "
            f"moy={sum(clv_lignes) / n_ligne:+.2f} pt | %ligne>0={pos:.0%}"
        )
    print(
        f"  Filtres process : EDGE_MIN={nba.NBA_EDGE_MIN:.0%} | "
        f"EDGE_MIN_SPREAD={nba.NBA_EDGE_MIN_SPREAD:.0%} | "
        f"CLV_LIGNE_TOL={nba.NBA_CLV_LIGNE_TOLERANCE:g} pt | "
        f"marchés={','.join(sorted(nba.NBA_MARCHES_ACTIFS)) or 'aucun'}"
    )
    print(
        "\n  Note : P&L Kelly (sizing fixe) ; CLV-prix approx = ligne ±tol ; "
        "CLV-ligne = pts de numéro battus vs close (spread/total)."
    )


# ─────────────────────────────────────────────────────────────
# Phase 4 — Calibration (--tune) : Brier moneyline sur TOUS les matchs
# (pas seulement les paris retenus) → nba_params_tuned.json
#
# Stratégie en 2 étapes (rapide + robuste) :
#   1. Fixe les defaults de rating, estime sigma_team / rho_scores sur les
#      résidus (mu vs score réel) — estimation analytique 1 passe.
#   2. Nelder-Mead sur n_prior / demi-vie / blend / HCA / B2B (Brier ML),
#      avec params en mémoire (pas d'écriture disque à chaque éval).
# ─────────────────────────────────────────────────────────────
PARAM_NAMES = [
    "n_prior", "rating_half_life_jours", "blend_gp_plein",
    "hca", "b2b_atk_pct", "b2b_def_pct", "sigma_team", "rho_scores",
]
# Sous-échantillon optionnel pour accélérer Nelder-Mead (0 = dataset complet).
TUNE_SUBSAMPLE = int(os.environ.get("NBA_BT_TUNE_SUBSAMPLE", "0"))


def _construire_dataset_calibration(conn: sqlite3.Connection, saisons: list[int]) -> list[dict]:
    dataset = []
    for season in saisons:
        rows = conn.execute(
            "SELECT game_id, date_utc, home, away, gh, ga FROM nba_games WHERE season=? ORDER BY date_utc, game_id",
            (season,),
        ).fetchall()
        for game_id, date_utc, home, away, gh, ga in rows:
            dataset.append({
                "season": season, "date": date_utc, "home": home, "away": away,
                "gh": int(gh), "ga": int(ga),
            })
    return dataset


def _sous_echantillon_tune(dataset: list[dict], n_max: int) -> list[dict]:
    """Sous-échantillon stratifié par saison (pas de look-ahead : ordre chronologique)."""
    if n_max <= 0 or len(dataset) <= n_max:
        return dataset
    par_saison: dict[int, list] = defaultdict(list)
    for row in dataset:
        par_saison[row["season"]].append(row)
    saisons = sorted(par_saison)
    quota = max(1, n_max // max(len(saisons), 1))
    out: list[dict] = []
    for season in saisons:
        rows = par_saison[season]
        # Prend 1 match sur k pour couvrir toute la saison (début → playoffs-edge)
        step = max(1, len(rows) // quota)
        out.extend(rows[::step][:quota])
    return out[:n_max]


def _clip(v: float, lo: float, hi: float) -> float:
    return min(max(float(v), lo), hi)


# Bornes physiquement plausibles (garde-fou numérique, indépendant du fix de
# fuite de look-ahead sur les dates) — empêche Nelder-Mead de converger vers
# un optimum dégénéré (ex. half_life≈1j + n_prior≈0 = quasi aucune histoire,
# ou rho≈0.9 = sigma_diff qui s'effondre et rend le modèle absurdement
# confiant). Plages dérivées de la littérature NBA analytics / ordres de
# grandeur observés (pas calibrées à la main sur ce dataset).
_BOUNDS = {
    "n_prior": (3.0, 40.0),               # < 3 : shrinkage quasi nul (bruit 1 match domine)
    "rating_half_life_jours": (10.0, 120.0),  # < 10j : ne garde que les 1-2 derniers matchs
    "blend_gp_plein": (5.0, 41.0),         # rampe saison N vs N-1 (max = mi-saison, 82 matchs)
    "hca": (-0.02, 0.06),                  # home court ~1.5-3 pts/114 rtg ≈ 0.013-0.026 typique
    "b2b_atk_pct": (0.0, 0.06),            # effet B2B documenté ~1-3 pts (~1-3% de 110)
    "b2b_def_pct": (0.0, 0.06),
    "sigma_team": (8.0, 16.0),             # écart-type pts/équipe/match NBA ≈ 11-13 historique
    "rho_scores": (-0.3, 0.6),             # corrélation scores dom/ext (effet pace partagé, modéré)
}


def _global_depuis_vecteur(x: list[float]) -> dict:
    n_prior, half_life, blend_gp_plein, hca, b2b_atk, b2b_def, sigma_team, rho = x
    return {
        "n_prior": round(_clip(n_prior, *_BOUNDS["n_prior"]), 3),
        "rating_half_life_jours": round(_clip(half_life, *_BOUNDS["rating_half_life_jours"]), 2),
        "blend_gp_plein": round(_clip(blend_gp_plein, *_BOUNDS["blend_gp_plein"]), 2),
        "hca": round(_clip(hca, *_BOUNDS["hca"]), 5),
        "b2b_atk_pct": round(_clip(b2b_atk, *_BOUNDS["b2b_atk_pct"]), 5),
        "b2b_def_pct": round(_clip(b2b_def, *_BOUNDS["b2b_def_pct"]), 5),
        "sigma_team": round(_clip(sigma_team, *_BOUNDS["sigma_team"]), 3),
        "rho_scores": round(_clip(rho, *_BOUNDS["rho_scores"]), 4),
    }


def _appliquer_candidat_params(x: list[float], *, persist: bool = False, source: str = "backtest_tune_wip") -> dict:
    """Applique un vecteur de params. Par défaut : mémoire seule (rapide pour --tune)."""
    global_params = _global_depuis_vecteur(x)
    if persist:
        nba_params.save_tuned_params({"_global": global_params}, source=source, merge_existing=True)
    else:
        nba_params.set_tuned_global_in_memory(global_params)
    return global_params


def _brier_ml_pour_params(x: list[float], dataset: list[dict]) -> float:
    _appliquer_candidat_params(x, persist=False)
    total, n = 0.0, 0
    for row in dataset:
        probas = nba.calculer_probabilites_match_nba(
            row["home"], row["away"], row["date"], season_year=row["season"],
        )
        p_dom = float(probas["moneyline"]["domicile"])
        outcome = 1.0 if row["gh"] > row["ga"] else 0.0
        total += (p_dom - outcome) ** 2
        n += 1
    return total / n if n else 1.0


def _estimer_sigma_rho_residus(dataset: list[dict]) -> tuple[float, float, int]:
    """1 passe : sigma_team = écart-type des résidus points/équipe, rho = corr(rés_h, rés_a)."""
    res_h: list[float] = []
    res_a: list[float] = []
    for row in dataset:
        probas = nba.calculer_probabilites_match_nba(
            row["home"], row["away"], row["date"], season_year=row["season"],
        )
        res_h.append(float(row["gh"]) - float(probas["mu_home"]))
        res_a.append(float(row["ga"]) - float(probas["mu_away"]))
    n = len(res_h)
    if n < 50:
        return nba_params.SIGMA_TEAM_DEFAULT, nba_params.RHO_SCORES_DEFAULT, n

    mean_h = sum(res_h) / n
    mean_a = sum(res_a) / n
    var_h = sum((r - mean_h) ** 2 for r in res_h) / (n - 1)
    var_a = sum((r - mean_a) ** 2 for r in res_a) / (n - 1)
    cov = sum((res_h[i] - mean_h) * (res_a[i] - mean_a) for i in range(n)) / (n - 1)
    sigma = (var_h ** 0.5 + var_a ** 0.5) / 2.0
    denom = (var_h * var_a) ** 0.5
    rho = (cov / denom) if denom > 1e-9 else 0.0
    return float(sigma), float(rho), n


def phase_tune(conn: sqlite3.Connection, saisons: list[int], maxiter: int) -> None:
    from scipy.optimize import minimize

    print("\n" + "=" * 60)
    print("🎛️   PHASE 4 — CALIBRATION (--tune)")
    print("=" * 60)
    dataset_full = _construire_dataset_calibration(conn, saisons)
    if len(dataset_full) < 100:
        print(f"  ⚠️ Dataset trop petit ({len(dataset_full)} matchs) — lancez --collect sur plusieurs saisons.")
        return

    dataset = _sous_echantillon_tune(dataset_full, TUNE_SUBSAMPLE)
    print(
        f"  Dataset calibration : {len(dataset_full)} matchs (saisons {saisons})"
        + (f" → sous-échantillon {len(dataset)}" if len(dataset) < len(dataset_full) else "")
        + " | objectif : Brier moneyline"
    )

    # Étape 0 — defaults en mémoire (évite de partir d'un JSON WIP partiel)
    x_defaults = [
        float(nba_params.N_PRIOR_DEFAULT),
        float(nba_params.RATING_HALF_LIFE_JOURS_DEFAULT),
        float(nba_params.BLEND_GP_PLEIN_DEFAULT),
        float(nba_params.HCA_DEFAULT),
        float(nba_params.B2B_ATK_PCT_DEFAULT),
        float(nba_params.B2B_DEF_PCT_DEFAULT),
        float(nba_params.SIGMA_TEAM_DEFAULT),
        float(nba_params.RHO_SCORES_DEFAULT),
    ]
    _appliquer_candidat_params(x_defaults, persist=False)
    brier_defaut = _brier_ml_pour_params(x_defaults, dataset)
    print(f"  Brier ML (defaults) : {brier_defaut:.4f}")

    # Étape 1 — sigma / rho analytiques sur résidus (defaults rating)
    print("  … estimation sigma_team / rho_scores sur résidus (1 passe)", flush=True)
    sigma_hat, rho_hat, n_res = _estimer_sigma_rho_residus(dataset)
    print(f"  σ̂={sigma_hat:.2f}  ρ̂={rho_hat:.3f}  (n={n_res})")
    x0 = list(x_defaults)
    x0[6], x0[7] = sigma_hat, rho_hat
    brier_apres_sigma = _brier_ml_pour_params(x0, dataset)
    print(f"  Brier ML (après σ/ρ) : {brier_apres_sigma:.4f}")

    # Étape 2 — Nelder-Mead sur les 6 params de rating (+ σ/ρ affinés ensemble)
    n_evals = [0]

    def _objectif(x):
        n_evals[0] += 1
        b = _brier_ml_pour_params(list(x), dataset)
        if n_evals[0] % 10 == 0:
            print(f"     … évaluation {n_evals[0]} (maxiter {maxiter}) — Brier courant {b:.4f}", flush=True)
        return b

    resultat = minimize(
        _objectif, x0=x0, method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5, "adaptive": True},
    )
    brier_final = float(resultat.fun)
    print(f"\n  Brier ML (calibré)   : {brier_final:.4f} ({n_evals[0]} évaluations, convergé={resultat.success})")

    if brier_final >= brier_defaut - 1e-6:
        print("  ⚠️ Aucune amélioration vs defaults — conservation des valeurs par défaut (pas d'écriture).")
        # Restaure defaults en mémoire ; ne wipe PAS un JSON déjà utile.
        _appliquer_candidat_params(x_defaults, persist=False)
        nba_params.reload_tuned_params()
        return

    # Re-score sur le dataset COMPLET pour le méta (honnêteté hors sous-échantillon)
    global_final = _appliquer_candidat_params(list(resultat.x), persist=False)
    brier_full = _brier_ml_pour_params(list(resultat.x), dataset_full) if len(dataset) < len(dataset_full) else brier_final
    brier_defaut_full = (
        _brier_ml_pour_params(x_defaults, dataset_full) if len(dataset) < len(dataset_full) else brier_defaut
    )

    nba_params.save_tuned_params(
        {"_global": global_final},
        source="backtest_tune",
        merge_existing=True,
    )
    # Enrichit _meta avec les métriques de calib (lecture-écriture légère)
    try:
        with open(nba_params._PARAMS_FILE, encoding="utf-8") as f:
            payload = json.load(f)
        payload.setdefault("_meta", {})
        payload["_meta"].update({
            "source": "backtest_tune",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "saisons": saisons,
            "n_matchs_tune": len(dataset),
            "n_matchs_full": len(dataset_full),
            "brier_ml_defaults": round(brier_defaut_full, 6),
            "brier_ml_tuned": round(brier_full, 6),
            "tune_maxiter": maxiter,
            "nelder_mead_success": bool(resultat.success),
        })
        with open(nba_params._PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        nba_params.reload_tuned_params()
    except OSError as e:
        print(f"  ⚠️ Méta non enrichi : {e}")

    print("\n  Paramètres calibrés :")
    for k, v in global_final.items():
        print(f"    {k}={v}")
    print(f"\n✅ nba_params_tuned.json figé (Brier {brier_defaut_full:.4f} → {brier_full:.4f}).")


# ─────────────────────────────────────────────────────────────
# Phase 5 — Validation hold-out (train/test chronologique)
#
# `phase_tune` calibre et rapporte historiquement sur les MÊMES saisons
# (risque de surapprentissage in-sample). Ce mode calibre strictement sur
# `train_saisons`, persiste `nba_params_tuned.json`, puis simule/rapporte
# séparément sur `train_saisons` (in-sample, référence) et `holdout_saisons`
# (out-of-sample, jamais vues en tune) pour un ROI/CLV/Brier honnêtes.
# ─────────────────────────────────────────────────────────────
def phase_holdout_validation(
    conn: sqlite3.Connection, train_saisons: list[int], holdout_saisons: list[int], maxiter: int,
) -> None:
    print("\n" + "=" * 60)
    print("🧪  VALIDATION HOLD-OUT (train/test chronologique)")
    print("=" * 60)
    print(f"  Train (calibration)  : saisons {train_saisons}")
    print(f"  Holdout (évaluation) : saisons {holdout_saisons} — jamais vues par --tune")

    # 1) Calibration sur TRAIN uniquement → écrit nba_params_tuned.json
    phase_tune(conn, train_saisons, maxiter=maxiter)

    # 2) Rapport IN-SAMPLE (train) — référence de comparaison
    phase_simulation(conn, train_saisons)
    phase_rapport(conn, csv_path=TRAIN_RESULTS_CSV, label="TRAIN / IN-SAMPLE — params calibrés sur ces mêmes saisons")

    # 3) Rapport OUT-OF-SAMPLE (holdout) — mêmes params, saisons jamais vues en tune
    phase_simulation(conn, holdout_saisons)
    phase_rapport(
        conn, csv_path=HOLDOUT_RESULTS_CSV,
        label="HOLDOUT / OUT-OF-SAMPLE — aucune donnée holdout utilisée pour le tune",
    )

    print("\n" + "=" * 60)
    print("✅ Validation hold-out terminée.")
    print(f"   → comparer {TRAIN_RESULTS_CSV} (in-sample) vs {HOLDOUT_RESULTS_CSV} (out-of-sample)")
    print("   ⚠️ nba_params_tuned.json reflète désormais la calibration TRAIN SEULE (pas la saison complète).")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# Reset / main
# ─────────────────────────────────────────────────────────────
def reset_backtest(full: bool = False) -> None:
    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)
    if full and os.path.exists(DB_PATH):
        for suffix in ("", "-wal", "-shm"):
            p = DB_PATH + suffix
            if os.path.exists(p):
                os.remove(p)
        print(f"🗑️  Supprimé {DB_PATH}")
    elif os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM nba_signaux")
        conn.commit()
        conn.close()
        print("🗑️  Signaux vidés (DB conservée)")
    else:
        print("🗑️  Rien à reset")


async def main_async(args: argparse.Namespace) -> None:
    logging.getLogger().setLevel(logging.WARNING)
    saisons = [int(s) for s in args.saisons.split(",")] if args.saisons else DEFAULT_SAISONS

    if args.reset_full:
        reset_backtest(full=True)
    elif args.reset:
        reset_backtest(full=False)

    if args.holdout_saisons:
        if not args.tune:
            print("⚠️ --holdout-saisons nécessite --tune (calibration sur --saisons, puis évaluation holdout).")
            return
        holdout_saisons = [int(s) for s in args.holdout_saisons.split(",")]
        conn = sqlite3.connect(DB_PATH)
        init_db(conn)
        try:
            if args.collect:
                await phase_collecte(conn, sorted(set(saisons) | set(holdout_saisons)), odds_only=args.odds_only)
            phase_holdout_validation(conn, saisons, holdout_saisons, maxiter=args.tune_maxiter)
        finally:
            conn.close()
        return

    run_phases = args.collect or args.simulate or args.report or args.tune or args.fill_close
    all_phases = not run_phases and not args.reset and not args.reset_full
    if not run_phases and not all_phases:
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    try:
        if args.fill_close:
            await phase_fill_close(conn, saisons)
        if args.collect or all_phases:
            await phase_collecte(conn, saisons, odds_only=args.odds_only)
        if args.tune:
            phase_tune(conn, saisons, maxiter=args.tune_maxiter)
        if args.simulate or all_phases:
            phase_simulation(conn, saisons)
        if args.report or all_phases:
            phase_rapport(conn)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest NBA — parité nba_sniper.py")
    parser.add_argument("--collect", action="store_true", help="Phase 1 : matchs + cotes historiques")
    parser.add_argument("--simulate", action="store_true", help="Phase 2 : simulation")
    parser.add_argument("--report", action="store_true", help="Phase 3 : rapport + CSV")
    parser.add_argument("--tune", action="store_true", help="Phase 4 : calibration hyperparamètres (Brier ML)")
    parser.add_argument("--tune-maxiter", type=int, default=150, help="Nelder-Mead maxiter (défaut 150)")
    parser.add_argument(
        "--holdout-saisons", type=str, default=None,
        help="Phase 5 : validation train/test — avec --tune, calibre sur --saisons (train) "
             "puis simulate+report sur ces saisons holdout (jamais vues en tune). "
             "Ex: --tune --saisons 2022,2023,2024 --holdout-saisons 2025,2026",
    )
    parser.add_argument(
        "--fill-close", action="store_true",
        help="Complète uniquement les cotes de clôture manquantes (tip-5/15/30/60, sans toucher aux prises)",
    )
    parser.add_argument("--reset", action="store_true", help="Vide signaux, garde la DB")
    parser.add_argument("--reset-full", action="store_true", help="Supprime DB + CSV")
    parser.add_argument("--odds-only", action="store_true", help="Recollecte cotes uniquement")
    parser.add_argument("--saisons", type=str, default=None, help="Ex: 2024,2025,2026 (convention NBA_SEASON)")
    args = parser.parse_args()

    if args.odds_only and not args.collect:
        args.collect = True

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
