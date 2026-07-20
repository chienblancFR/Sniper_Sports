"""
backtest_nhl.py — Back-test NHL aligné sur nhl_sniper_omega.py
==============================================================
Parité live : calculate_master_odds_v4, shrink_cotes_vers_marche,
_construire_candidats_pari, calculate_kelly, PIT MoneyPuck (équipes + gardiens).

Usage :
  python backtest_nhl.py --collect              # Matchs NHL + cotes Pinnacle historiques
  python backtest_nhl.py --simulate             # Simulation (1 pari max / match)
  python backtest_nhl.py --report               # Rapport console + CSV
  python backtest_nhl.py                        # Collect + simulate + report

  python backtest_nhl.py --reset                # Vide signaux, garde la DB
  python backtest_nhl.py --reset-full           # Supprime backtest_nhl.db + CSV
  python backtest_nhl.py --collect --saisons 2025,2026
  python backtest_nhl.py --simulate --calib defaults   # Pas de rho_meta.json (honnête)
  python backtest_nhl.py --simulate --calib frozen     # Lit rho_calibrage_meta.json (lookahead)
  python backtest_nhl.py --calib-pl-scale --saisons 2023,2024,2025
  python backtest_nhl.py --calib-nb-ou --saisons 2023,2024,2025
  python backtest_nhl.py --calib-ou-mu-scale --saisons 2023,2024,2025
      # Calibre pl_scale / nb_ou_dispersion / ou_mu_scale (PIT) → rho_calibrage_meta.json

Limites documentées (v1) :
  - Pas de line movement / steam (pas d'historique snapshots)
  - Pas d'absences stars (rosters historiques non collectés)
  - Gardiens : all_goalies.csv (auto depuis zips shots MoneyPuck) → GSAx PIT ; sinon 0
  - Arbitres : crews boxscore NHL (post-match) + rates PIT chrono ; fallback mult=1.0
  - Kelly dynamique journal / CLV : kelly_mult=1.0
  - Cotes Pinnacle : snapshot H-X avant puck drop par match (pas H-24 jour calendaire)

  python backtest_nhl.py --collect-refs --saisons 2023,2024,2025,2026
  python backtest_nhl.py --simulate --report --saisons 2023,2024,2025,2026 --calib defaults

Prérequis : API_ODDS_KEY avec endpoint /v4/historical/ (plan Odds API).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from config_env import env_files_hint, load_project_env

load_project_env("nhl")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import bot live — logique métier partagée
import nhl_sniper_omega as nhl

DB_PATH = os.environ.get("NHL_BT_DB", "backtest_nhl.db")
RESULTS_CSV = os.environ.get("NHL_BT_RESULTS_CSV", "backtest_nhl_results.csv")
SPORT_KEY = "icehockey_nhl"
DEFAULT_SAISONS = [
    int(s.strip())
    for s in os.environ.get("NHL_BT_SAISONS", "2025,2026").split(",")
    if s.strip()
]
API_ODDS_KEY = os.environ.get("API_ODDS_KEY", "")
BANKROLL_BT = float(os.environ.get("NHL_BT_BANKROLL", os.environ.get("NHL_BANKROLL", "1000")))
# Cotes de prise : par match, X h avant puck drop (pas H-24 — lignes NHL souvent plus tard)
ODDS_PRISE_HEURES = float(os.environ.get("NHL_BT_ODDS_HEURES_AVANT", "3"))
ODDS_PRISE_FALLBACK_HEURES = [
    float(s.strip())
    for s in os.environ.get("NHL_BT_ODDS_FALLBACK_HEURES", "1,0.5").split(",")
    if s.strip()
]
ODDS_TABLE_PRISE = "nhl_odds_prise"

NHL_HEADERS = {"User-Agent": "Mozilla/5.0"}
ETATS_FINaux = {"FINAL", "OFF", "OFFICIAL"}
NHL_BT_REFS_ACTIF = os.environ.get("NHL_BT_REFS_ACTIF", "true").strip().lower() in (
    "1", "true", "yes", "oui",
)
NHL_BT_REFS_CONCURRENCY = int(os.environ.get("NHL_BT_REFS_CONCURRENCY", "12"))


# ─────────────────────────────────────────────────────────────
# Helpers mapping / cotes
# ─────────────────────────────────────────────────────────────
def _abbrev_depuis_nom_odds(nom_api: str) -> str | None:
    primaire = nhl._nom_odds_vers_primaire(nom_api)
    for abbr, full in nhl.NHL_TEAMS_MAPPING.items():
        if full == primaire:
            return abbr
    return None


def _saison_dates(season: int) -> tuple[datetime, datetime]:
    """Saison NHL convention MoneyPuck (season=2026 → 2025-10 → 2026-06)."""
    return datetime(season - 1, 10, 1), datetime(season, 6, 25)


def _parse_utc(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _appliquer_calib_mode(mode: str) -> dict:
    """
    defaults : constantes env — pas de lookahead sur rho_calibrage_meta.json
    frozen   : lit le meta actuel (utile sanity check, lookahead possible)
    """
    if mode == "frozen" and os.path.exists(nhl.RHO_META_FILE):
        meta = nhl.lire_rho_meta()
    else:
        meta = _meta_defaults_bt()

    def _lire_meta():
        return dict(meta)

    nhl.lire_rho_meta = _lire_meta
    nhl._rho_meta_cache = dict(meta)
    return meta


def _assembler_cotes_book(rows: list[tuple], home_abbr: str, away_abbr: str) -> dict | None:
    """Reconstruit le dict cotes Pinnacle depuis la DB (format bot live)."""
    home_full = nhl.NHL_TEAMS_MAPPING.get(home_abbr)
    away_full = nhl.NHL_TEAMS_MAPPING.get(away_abbr)
    if not home_full or not away_full:
        return None
    cotes: dict[str, Any] = {"home_full": home_full, "away_full": away_full, "totals": {}}
    for market, outcome, point, price in rows:
        price = float(price)
        if market == "h2h":
            if nhl._outcome_est_equipe(outcome, home_full):
                cotes["cote_1"] = price
            elif nhl._outcome_est_equipe(outcome, away_full):
                cotes["cote_2"] = price
        elif market == "spreads" and nhl._est_puck_line_moins_15(point):
            if nhl._outcome_est_equipe(outcome, home_full):
                cotes["cote_pl_home"] = price
            elif nhl._outcome_est_equipe(outcome, away_full):
                cotes["cote_pl_away"] = price
        elif market == "totals":
            cut = nhl._arrondir_cut(point)
            side = str(outcome).capitalize()
            if side in ("Over", "Under"):
                cotes["totals"].setdefault(cut, {})[side] = price
    if "cote_1" not in cotes or "cote_2" not in cotes:
        return None
    return cotes


def _extraire_puckline(cotes_book: dict) -> dict | None:
    pl = {}
    if "cote_pl_home" in cotes_book:
        pl["cote_pl_home"] = cotes_book["cote_pl_home"]
    if "cote_pl_away" in cotes_book:
        pl["cote_pl_away"] = cotes_book["cote_pl_away"]
    return pl if pl else None


def _contexte_calendrier_bt(games_avant: dict, date_str: str, home: str, away: str) -> tuple:
    """B2B + miles — même logique que _contexte_calendrier_match."""
    return nhl._contexte_calendrier_match(date_str, home, away, games_avant)


def _regler_pari(marche: str, type_pari: str, home: str, away: str, gh: int, ga: int) -> bool:
    if marche == "ML":
        if home in type_pari:
            return gh > ga
        return ga > gh
    if marche == "PL":
        if home in type_pari:
            return (gh - ga) >= 2
        return (ga - gh) >= 2
    if marche == "OU":
        total = gh + ga
        parts = type_pari.split()
        cut = float(parts[1])
        if parts[0].upper() == "OVER":
            return total > cut
        return total < cut
    return False


def _prob_modele_pari(marche: str, type_pari: str, home: str, cotes_vraies: dict) -> float | None:
    if marche == "ML":
        return cotes_vraies["prob_1"] if home in type_pari else cotes_vraies["prob_2"]
    if marche == "PL":
        return cotes_vraies["prob_pl_home"] if home in type_pari else cotes_vraies["prob_pl_away"]
    if marche == "OU":
        parts = type_pari.split()
        cut = nhl._arrondir_cut(float(parts[1]))
        if parts[0].upper() == "OVER":
            _, p = nhl._trouver_cle_float(cotes_vraies["prob_over_cuts"], cut)
            return p
        _, p = nhl._trouver_cle_float(cotes_vraies["prob_under_cuts"], cut)
        return p
    return None


# ─────────────────────────────────────────────────────────────
# SQLite
# ─────────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nhl_games (
            game_id     TEXT PRIMARY KEY,
            season      INTEGER,
            date_utc    TEXT,
            home        TEXT,
            away        TEXT,
            gh          INTEGER,
            ga          INTEGER,
            start_utc   TEXT
        );
        CREATE TABLE IF NOT EXISTS nhl_odds_prise (
            game_id  TEXT,
            market   TEXT,
            outcome  TEXT,
            point    REAL,
            cote     REAL,
            PRIMARY KEY (game_id, market, outcome, point)
        );
        CREATE TABLE IF NOT EXISTS nhl_odds_cloture (
            game_id  TEXT,
            market   TEXT,
            outcome  TEXT,
            point    REAL,
            cote     REAL,
            PRIMARY KEY (game_id, market, outcome, point)
        );
        CREATE TABLE IF NOT EXISTS nhl_signaux (
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
            lam_home     REAL,
            lam_away     REAL,
            PRIMARY KEY (game_id, marche, type_pari)
        );
        CREATE TABLE IF NOT EXISTS nhl_refs (
            game_id    TEXT PRIMARY KEY,
            refs_json  TEXT NOT NULL,
            penalties  INTEGER
        );
    """)
    _migrer_schema_odds(conn)
    conn.commit()


def _migrer_schema_odds(conn: sqlite3.Connection) -> None:
    """Repli nhl_odds_h24 → nhl_odds_prise (ancienne collecte H-24 par journée)."""
    legacy = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='nhl_odds_h24'",
    ).fetchone()
    if not legacy:
        return
    n_new = conn.execute(f"SELECT COUNT(*) FROM {ODDS_TABLE_PRISE}").fetchone()[0]
    if n_new == 0:
        conn.execute(
            f"INSERT OR IGNORE INTO {ODDS_TABLE_PRISE} "
            "SELECT game_id, market, outcome, point, cote FROM nhl_odds_h24",
        )
    cols = [r[1] for r in conn.execute("PRAGMA table_info(nhl_signaux)")]
    if "cote_h24" in cols and "cote_prise" not in cols:
        conn.execute("ALTER TABLE nhl_signaux RENAME COLUMN cote_h24 TO cote_prise")


def _charger_odds_table(conn: sqlite3.Connection, table: str) -> dict[str, list]:
    out: dict[str, list] = defaultdict(list)
    for row in conn.execute(f"SELECT game_id, market, outcome, point, cote FROM {table}"):
        out[row[0]].append((row[1], row[2], row[3], row[4]))
    return out


# ─────────────────────────────────────────────────────────────
# Phase 1 — Collecte
# ─────────────────────────────────────────────────────────────
async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> Any:
    try:
        async with session.get(url, params=params, headers=NHL_HEADERS, timeout=30) as resp:
            if resp.status == 200:
                return await resp.json()
            print(f"  ⚠️ HTTP {resp.status} — {url[:80]}")
    except Exception as e:
        print(f"  ⚠️ fetch error : {e}")
    return None


async def collecter_matchs_saison(conn: sqlite3.Connection, session: aiohttp.ClientSession, season: int) -> int:
    debut, fin = _saison_dates(season)
    nouveaux = 0
    day = debut.date()
    fin_day = fin.date()
    while day <= fin_day:
        date_str = day.isoformat()
        url = f"https://api-web.nhle.com/v1/score/{date_str}"
        data = await fetch_json(session, url)
        day += timedelta(days=1)
        if not data or "games" not in data:
            continue
        for game in data["games"]:
            if game.get("gameState") not in ETATS_FINaux:
                continue
            home_info = game.get("homeTeam", {}) or {}
            away_info = game.get("awayTeam", {}) or {}
            home, away = home_info.get("abbrev"), away_info.get("abbrev")
            gh, ga = home_info.get("score"), away_info.get("score")
            if not home or not away or gh is None or ga is None:
                continue
            gid = str(game["id"])
            start_utc = game.get("startTimeUTC", f"{date_str}T00:00:00Z")
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO nhl_games
                       (game_id, season, date_utc, home, away, gh, ga, start_utc)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (gid, season, date_str, home, away, int(gh), int(ga), start_utc),
                )
                nouveaux += 1
            except sqlite3.Error:
                pass
        await asyncio.sleep(0.15)
    conn.commit()
    return nouveaux


def _games_par_date(conn: sqlite3.Connection, season: int) -> dict[str, list[dict]]:
    par_date: dict[str, list[dict]] = defaultdict(list)
    for row in conn.execute(
        "SELECT game_id, date_utc, home, away, start_utc FROM nhl_games WHERE season=?",
        (season,),
    ):
        par_date[row[1]].append({
            "game_id": row[0], "date_utc": row[1], "home": row[2],
            "away": row[3], "start_utc": row[4],
        })
    return par_date


def _trouver_game_id(event: dict, games_du_jour: list[dict]) -> str | None:
    h_abbr = _abbrev_depuis_nom_odds(event.get("home_team", ""))
    a_abbr = _abbrev_depuis_nom_odds(event.get("away_team", ""))
    if not h_abbr or not a_abbr:
        return None
    for g in games_du_jour:
        if g["home"] == h_abbr and g["away"] == a_abbr:
            return g["game_id"]
    return None


async def collecter_odds_instant(
    session: aiohttp.ClientSession,
    conn: sqlite3.Connection,
    date_utc: datetime,
    table: str,
    games_du_jour: list[dict],
) -> int:
    if not API_ODDS_KEY:
        return 0
    date_str = date_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/odds"
        f"?apiKey={API_ODDS_KEY}&regions=eu"
        f"&markets=h2h,totals,spreads&oddsFormat=decimal&bookmakers=pinnacle"
        f"&date={date_str}"
    )
    raw = await fetch_json(session, url)
    if not raw:
        return 0
    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    if not isinstance(data, list):
        return 0
    inserted = 0
    for event in data:
        gid = _trouver_game_id(event, games_du_jour)
        if not gid:
            continue
        pinnacle = next((b for b in event.get("bookmakers", []) if b["key"] == "pinnacle"), None)
        if not pinnacle:
            continue
        rows = []
        for market in pinnacle["markets"]:
            for out in market["outcomes"]:
                rows.append((
                    gid, market["key"], out["name"],
                    float(out.get("point", 0)), float(out["price"]),
                ))
        if rows:
            conn.executemany(
                f"INSERT OR IGNORE INTO {table} VALUES (?,?,?,?,?)",
                rows,
            )
            inserted += len(rows)
    if inserted:
        conn.commit()
    return inserted


def _limite_prise_avant_puck(start_utc: datetime) -> datetime:
    """Dernière minute où le bot live peut encore parier (NHL_MINUTES_AVANT_PUCK)."""
    return start_utc - timedelta(minutes=nhl.NHL_MINUTES_AVANT_PUCK)


def _instant_prise_pour_match(game: dict, heures_avant: float) -> datetime | None:
    start = _parse_utc(game["start_utc"])
    if not start:
        return None
    dt = start - timedelta(hours=heures_avant)
    limite = _limite_prise_avant_puck(start)
    if dt > limite:
        dt = limite
    return dt


def _bucketiser_par_instant(games: list[dict], heures_avant: float) -> dict[datetime, list[dict]]:
    buckets: dict[datetime, list[dict]] = defaultdict(list)
    for game in games:
        dt = _instant_prise_pour_match(game, heures_avant)
        if dt:
            buckets[dt].append(game)
    return buckets


def _games_sans_cotes(conn: sqlite3.Connection, game_ids: list[str]) -> list[str]:
    if not game_ids:
        return []
    placeholders = ",".join("?" * len(game_ids))
    couverts = {
        row[0]
        for row in conn.execute(
            f"SELECT DISTINCT game_id FROM {ODDS_TABLE_PRISE} WHERE game_id IN ({placeholders})",
            game_ids,
        )
    }
    return [gid for gid in game_ids if gid not in couverts]


async def _collecter_odds_heures(
    conn: sqlite3.Connection,
    session: aiohttp.ClientSession,
    games: list[dict],
    heures_avant: float,
) -> int:
    total = 0
    for dt, bucket in sorted(_bucketiser_par_instant(games, heures_avant).items()):
        total += await collecter_odds_instant(session, conn, dt, ODDS_TABLE_PRISE, bucket)
        await asyncio.sleep(0.35)
    return total


async def collecter_odds_saison(conn: sqlite3.Connection, session: aiohttp.ClientSession, season: int) -> int:
    games = [
        {
            "game_id": row[0], "date_utc": row[1], "home": row[2],
            "away": row[3], "start_utc": row[4],
        }
        for row in conn.execute(
            "SELECT game_id, date_utc, home, away, start_utc FROM nhl_games WHERE season=?",
            (season,),
        )
    ]
    if not games:
        print(f"  ⚠️ Aucun match en base pour saison {season}")
        return 0

    conn.execute(
        f"DELETE FROM {ODDS_TABLE_PRISE} WHERE game_id IN "
        "(SELECT game_id FROM nhl_games WHERE season=?)",
        (season,),
    )
    conn.execute(
        "DELETE FROM nhl_odds_cloture WHERE game_id IN (SELECT game_id FROM nhl_games WHERE season=?)",
        (season,),
    )
    conn.commit()

    print(
        f"  📊 Cotes prise : H-{ODDS_PRISE_HEURES:g} par match "
        f"(fenêtre live ≤{nhl.NHL_SCAN_HEURES_AVANCE:g}h, stop {nhl.NHL_MINUTES_AVANT_PUCK:g} min avant puck)"
    )
    total = 0
    total += await _collecter_odds_heures(conn, session, games, ODDS_PRISE_HEURES)

    restants = _games_sans_cotes(conn, [g["game_id"] for g in games])
    for fb_h in ODDS_PRISE_FALLBACK_HEURES:
        if not restants:
            break
        if fb_h >= ODDS_PRISE_HEURES:
            continue
        subset = [g for g in games if g["game_id"] in restants]
        print(f"     ↪ repli H-{fb_h:g} pour {len(subset)} match(s) sans ligne", flush=True)
        total += await _collecter_odds_heures(conn, session, subset, fb_h)
        restants = _games_sans_cotes(conn, restants)

    close_buckets: dict[datetime, list[dict]] = defaultdict(list)
    for game in games:
        start = _parse_utc(game["start_utc"])
        if start:
            close_buckets[_limite_prise_avant_puck(start)].append(game)
    for i, (dt_close, bucket) in enumerate(sorted(close_buckets.items()), 1):
        total += await collecter_odds_instant(session, conn, dt_close, "nhl_odds_cloture", bucket)
        if i % 200 == 0:
            print(f"     … clôture {i}/{len(close_buckets)} snapshots", flush=True)
        await asyncio.sleep(0.35)

    n_cov = conn.execute(
        f"SELECT COUNT(DISTINCT o.game_id) FROM {ODDS_TABLE_PRISE} o "
        "JOIN nhl_games g ON g.game_id=o.game_id WHERE g.season=?",
        (season,),
    ).fetchone()[0]
    n_games = len(games)
    print(f"  ✅ Saison {season} : {total} lignes cotes | {n_cov}/{n_games} matchs avec ligne de prise")
    if restants:
        print(f"     ⚠️ {len(restants)} match(s) toujours sans cote Pinnacle (ligne jamais publiée ?)")
    return total


async def phase_collecte(conn: sqlite3.Connection, saisons: list[int], odds_only: bool = False) -> None:
    print("\n" + "=" * 60)
    print("📥  PHASE 1 — COLLECTE NHL")
    print("=" * 60)
    if not API_ODDS_KEY:
        print(f"\n❌ API_ODDS_KEY absente ({env_files_hint('nhl')})")
        return
    async with aiohttp.ClientSession() as session:
        for season in saisons:
            print(f"\n🔄 Saison MoneyPuck {season} ({season - 1}-{str(season)[-2:]})")
            if not odds_only:
                n = await collecter_matchs_saison(conn, session, season)
                print(f"  📋 {n} match(s) indexés")
            await collecter_odds_saison(conn, session, season)
    print("\n✅ Phase 1 terminée.")


# ─────────────────────────────────────────────────────────────
# Arbitres (B2) — crews NHL post-match + rates PIT chronologiques
# ─────────────────────────────────────────────────────────────
def _charger_refs_lookup(conn: sqlite3.Connection) -> dict[str, dict]:
    """game_id → {refs: [str], penalties: int|None}."""
    out = {}
    for gid, refs_json, pens in conn.execute(
        "SELECT game_id, refs_json, penalties FROM nhl_refs"
    ):
        try:
            refs = json.loads(refs_json) if refs_json else []
        except json.JSONDecodeError:
            refs = []
        out[str(gid)] = {"refs": refs, "penalties": pens}
    return out


def _meta_refs_vide() -> dict:
    return {
        "referees": {},
        "league_penalties_pg": 10.0,
        "total_games": 0,
        "league_penalties_total": 0,
        "scanned_game_ids": [],
    }


def _enrichir_meta_refs(meta: dict, ref_names: list[str], pen_count: int | None) -> None:
    """Ajoute un match terminé aux stats arbitres (in-place)."""
    if not ref_names or pen_count is None:
        return
    refs_db = meta.setdefault("referees", {})
    for name in ref_names:
        key = nhl._normaliser_nom_arbitre(name)
        if key not in refs_db:
            refs_db[key] = {"name": name, "games": 0, "penalties": 0}
        refs_db[key]["games"] += 1
        refs_db[key]["penalties"] += int(pen_count)
    meta["league_penalties_total"] = int(meta.get("league_penalties_total", 0)) + int(pen_count)
    meta["total_games"] = int(meta.get("total_games", 0)) + 1
    tg = meta["total_games"]
    meta["league_penalties_pg"] = round(meta["league_penalties_total"] / tg, 2) if tg else 10.0


def _warmstart_meta_refs(conn: sqlite3.Connection, avant_saison: int) -> dict:
    """Stats arbitres sur saisons < avant_saison (PIT inter-saisons)."""
    meta = _meta_refs_vide()
    rows = conn.execute(
        """
        SELECT r.refs_json, r.penalties
        FROM nhl_refs r
        JOIN nhl_games g ON g.game_id = r.game_id
        WHERE g.season < ? AND r.penalties IS NOT NULL
        ORDER BY g.date_utc, g.game_id
        """,
        (avant_saison,),
    )
    for refs_json, pens in rows:
        try:
            refs = json.loads(refs_json) if refs_json else []
        except json.JSONDecodeError:
            refs = []
        _enrichir_meta_refs(meta, refs, pens)
    return meta


def _installer_meta_refs_live(meta: dict):
    """Patch lire_referee_meta pour le multiplicateur live pendant la sim."""

    def _lire():
        return dict(meta)

    nhl.lire_referee_meta = _lire


async def _fetch_refs_et_penalites(session: aiohttp.ClientSession, game_id: str) -> tuple[list[str], int | None]:
    """right-rail (refs) + landing (pénalités)."""
    refs: list[str] = []
    pens: int | None = None
    url_rr = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail"
    data = await fetch_json(session, url_rr)
    if data:
        raw = (data.get("gameInfo") or {}).get("referees") or []
        for entry in raw:
            nom = nhl._extraire_nom_officiel(entry)
            if nom:
                refs.append(nom)
    url_land = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/landing"
    data2 = await fetch_json(session, url_land)
    if data2:
        total = 0
        for bloc in (data2.get("summary") or {}).get("penalties") or []:
            plist = bloc.get("penalties") if isinstance(bloc, dict) else None
            if isinstance(plist, list):
                total += len(plist)
        pens = total
    return refs, pens


async def collecter_refs_saison(
    conn: sqlite3.Connection,
    session: aiohttp.ClientSession,
    season: int,
) -> int:
    """Collecte crews + pénalités pour les matchs de la saison (skip déjà en DB)."""
    deja = {r[0] for r in conn.execute("SELECT game_id FROM nhl_refs")}
    games = [
        r[0]
        for r in conn.execute(
            "SELECT game_id FROM nhl_games WHERE season=? AND gh IS NOT NULL ORDER BY date_utc",
            (season,),
        )
        if r[0] not in deja
    ]
    if not games:
        n = conn.execute(
            "SELECT COUNT(*) FROM nhl_refs r JOIN nhl_games g ON g.game_id=r.game_id WHERE g.season=?",
            (season,),
        ).fetchone()[0]
        print(f"  👨‍⚖️ Saison {season} : {n} crews déjà en DB")
        return 0

    print(f"  👨‍⚖️ Saison {season} : collecte {len(games)} crews (concurrence {NHL_BT_REFS_CONCURRENCY})…")
    sem = asyncio.Semaphore(NHL_BT_REFS_CONCURRENCY)
    ok = 0
    fail = 0

    async def _one(gid: str):
        nonlocal ok, fail
        async with sem:
            refs, pens = await _fetch_refs_et_penalites(session, gid)
            if not refs:
                fail += 1
                return
            conn.execute(
                "INSERT OR REPLACE INTO nhl_refs (game_id, refs_json, penalties) VALUES (?,?,?)",
                (gid, json.dumps(refs, ensure_ascii=False), pens),
            )
            ok += 1
            if ok % 200 == 0:
                conn.commit()
                print(f"     … {ok}/{len(games)} (échecs {fail})")

    await asyncio.gather(*[_one(g) for g in games])
    conn.commit()
    print(f"  ✅ Saison {season} : +{ok} crews | échecs {fail}")
    return ok


async def phase_collecte_refs(conn: sqlite3.Connection, saisons: list[int]) -> None:
    print("\n" + "=" * 60)
    print("👨‍⚖️  COLLECTE ARBITRES (boxscore NHL)")
    print("=" * 60)
    print("  Caveat : crew connu post-match (léger look-ahead d'identité).")
    async with aiohttp.ClientSession() as session:
        for season in saisons:
            await collecter_refs_saison(conn, session, season)
    n = conn.execute("SELECT COUNT(*) FROM nhl_refs").fetchone()[0]
    print(f"\n✅ Collecte refs terminée — {n} matchs en nhl_refs")


# ─────────────────────────────────────────────────────────────
# Calibrage pl_scale (hors live, lambdas PIT sans look-ahead)
# ─────────────────────────────────────────────────────────────
DEFAULT_PL_CALIB_SAISONS = [2023, 2024, 2025]


def _meta_defaults_bt() -> dict:
    """Même dict que --calib defaults (pas de lookahead meta)."""
    return {
        "rho": -0.12,
        "hia": nhl.NHL_HIA_DEFAULT,
        "prob_tie": 0.12,
        "prob_en": 0.22,
        "nb_ou_dispersion": nhl.NHL_NB_OU_DISPERSION_DEFAULT,
        "ot_home_adv": nhl.NHL_OT_HOME_ADVANTAGE,
        "ref_sensibilite": nhl.NHL_REF_SENSIBILITE,
        "faceoff_sensibilite": nhl.NHL_FACEOFF_SENSIBILITE,
        "pp_lam_share": nhl.NHL_PP_LAM_SHARE,
        "travel_b2b_atk_pct": nhl.NHL_TRAVEL_B2B_ATK_PCT,
        "travel_b2b_def_pct": nhl.NHL_TRAVEL_B2B_DEF_PCT,
        "travel_solo_atk_pct": nhl.NHL_TRAVEL_SOLO_ATK_PCT,
        "travel_solo_def_pct": nhl.NHL_TRAVEL_SOLO_DEF_PCT,
        "gsax_lam_mult": nhl.NHL_GSAX_LAM_MULT_DEFAULT,
        "pl_scale": nhl.NHL_PL_SCALE_DEFAULT,
        "ou_mu_scale": nhl.NHL_OU_MU_SCALE_DEFAULT,
        "nb_matchs": 0,
    }


def _construire_dataset_pl_pit(conn: sqlite3.Connection, saisons: list[int]) -> list[dict]:
    """
    Matchs avec lambdas PIT pré-match (HIA_REF fixe, comme _lambda_referentiel_match).
    Scores réels pour Brier PL home (marge ≥ 2).
    """
    meta = _meta_defaults_bt()
    rho = float(meta["rho"])
    prob_tie = float(meta["prob_tie"])
    prob_en = float(meta["prob_en"])
    hia_ref = nhl.HIA_REF_CALIBRATION
    dataset: list[dict] = []

    for season in saisons:
        pit_index = (
            nhl.construire_index_pit_moneypuck(season=season) if nhl.NHL_PIT_CALIB_ACTIF else None
        )
        teams_n1 = (
            nhl.get_team_stats(season=season - 1, blend=False) if nhl.NHL_BLEND_GP_PLEIN > 0 else None
        )
        goalie_by_game = (
            nhl.lire_goalie_by_game_lookup(season=season) if nhl.NHL_GSAX_CALIB_ACTIF else {}
        )
        games = list(conn.execute(
            "SELECT game_id, date_utc, home, away, gh, ga FROM nhl_games "
            "WHERE season=? AND gh IS NOT NULL ORDER BY date_utc, game_id",
            (season,),
        ))
        games_avant: dict = {}
        n_ok = 0
        for game_id, date_utc, home, away, gh, ga in games:
            home_b2b, away_b2b, home_travel, away_travel = _contexte_calendrier_bt(
                games_avant, date_utc, home, away,
            )
            starters = goalie_by_game.get(str(game_id), {})
            home_gsax = float((starters.get(home) or {}).get("gsax_per_60", 0.0))
            away_gsax = float((starters.get(away) or {}).get("gsax_per_60", 0.0))
            teams_snap = (
                nhl._teams_data_snapshot_pit(pit_index, date_utc, teams_n1=teams_n1)
                if pit_index else None
            )
            games_avant[game_id] = {"date": date_utc, "home": home, "away": away}
            if not teams_snap:
                continue
            cotes = nhl.calculate_master_odds_v4(
                teams_snap, home, away,
                home_gsax=home_gsax, away_gsax=away_gsax,
                home_is_b2b=home_b2b, away_is_b2b=away_b2b,
                home_travel_miles=home_travel, away_travel_miles=away_travel,
                referee_pp_mult=1.0,
                rho=rho, hia=hia_ref,
                prob_tie=prob_tie, prob_en=prob_en,
            )
            if not cotes:
                continue
            date_str = str(date_utc)[:10] if date_utc else None
            dataset.append({
                "game_id": str(game_id),
                "date": date_str,
                "home": home,
                "away": away,
                "vrai_score_domicile": int(gh),
                "vrai_score_exterieur": int(ga),
                "lambda_domicile_calcule": float(cotes["lam_home"]),
                "lambda_exterieur_calcule": float(cotes["lam_away"]),
                "hia_ref": hia_ref,
                "lambdas_pit": True,
                "season": season,
            })
            n_ok += 1
        print(f"  Saison {season} : {n_ok} matchs PIT pour calib")
    return dataset


def _brier_ou_nb(dataset: list[dict], r_disp: float, cut: float = 5.5) -> float:
    """Brier Over @ cut avec NB (lambdas déjà HIA_REF)."""
    sse, n = 0.0, 0
    for match in dataset:
        try:
            lam_h = float(match["lambda_domicile_calcule"])
            lam_a = float(match["lambda_exterieur_calcule"])
            total = int(match["vrai_score_domicile"]) + int(match["vrai_score_exterieur"])
        except (KeyError, TypeError, ValueError):
            continue
        mu = max(lam_h + lam_a, 1.0)
        po, _ = nhl._prob_over_under_nb(mu, r_disp, [cut])
        pred = float(po.get(cut, 0.5))
        outcome = 1.0 if total > cut else 0.0
        sse += (pred - outcome) ** 2
        n += 1
    return sse / max(n, 1)


def phase_calib_pl_scale(conn: sqlite3.Connection, saisons: list[int]) -> float:
    """Calibre pl_scale (grille Brier) et persiste dans rho_calibrage_meta.json."""
    print("\n" + "=" * 60)
    print("🔬  CALIBRAGE pl_scale (PIT → rho_calibrage_meta.json)")
    print("=" * 60)
    print(f"  Saisons : {saisons}")

    dataset = _construire_dataset_pl_pit(conn, saisons)
    if len(dataset) < nhl.NHL_PL_CALIB_MIN_MATCHS:
        print(
            f"❌ Trop peu de matchs ({len(dataset)} < {nhl.NHL_PL_CALIB_MIN_MATCHS}) — abort"
        )
        sys.exit(1)

    meta = _meta_defaults_bt()
    rho = float(meta["rho"])
    hia = float(meta["hia"])
    # Recency live (demi-vie 28j) rend les saisons 2023-25 quasi-invisibles en 2026.
    # Calib offline pré-saison = poids uniformes sur la fenêtre demandée.
    recency_prev = nhl.NHL_MLE_RECENCY_ACTIF
    nhl.NHL_MLE_RECENCY_ACTIF = False
    try:
        mse_def = nhl._mse_pl_scale(nhl.NHL_PL_SCALE_DEFAULT, dataset, rho, hia)
        pl_scale = nhl.calibrer_pl_scale(dataset, rho, hia)
        mse_opt = nhl._mse_pl_scale(pl_scale, dataset, rho, hia)
    finally:
        nhl.NHL_MLE_RECENCY_ACTIF = recency_prev

    out = dict(meta)
    if os.path.exists(nhl.RHO_META_FILE):
        try:
            with open(nhl.RHO_META_FILE, "r", encoding="utf-8") as f:
                out = {**out, **json.load(f)}
        except Exception:
            pass
    out["pl_scale"] = pl_scale
    out["pl_scale_calib_n"] = len(dataset)
    out["pl_scale_calib_saisons"] = saisons
    out["pl_scale_mse_default"] = round(mse_def, 6)
    out["pl_scale_mse_opt"] = round(mse_opt, 6)
    out["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    if "nb_matchs" not in out or not out.get("nb_matchs"):
        out["nb_matchs"] = len(dataset)

    with open(nhl.RHO_META_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    nhl._invalider_rho_meta_cache()

    print(
        f"  pl_scale {nhl.NHL_PL_SCALE_DEFAULT:.3f} → {pl_scale:.3f} | "
        f"Brier {mse_def:.4f} → {mse_opt:.4f} | n={len(dataset)}"
    )
    print(f"  💾 Écrit → {nhl.RHO_META_FILE}")
    return pl_scale


def phase_calib_nb_ou(conn: sqlite3.Connection, saisons: list[int]) -> float:
    """Calibre nb_ou_dispersion (MLE NB) sur lambdas PIT → rho_calibrage_meta.json."""
    print("\n" + "=" * 60)
    print("🔬  CALIBRAGE nb_ou_dispersion (PIT → rho_calibrage_meta.json)")
    print("=" * 60)
    print(f"  Saisons : {saisons}")

    dataset = _construire_dataset_pl_pit(conn, saisons)
    if len(dataset) < nhl.NHL_NB_OU_MIN_MATCHS:
        print(
            f"❌ Trop peu de matchs ({len(dataset)} < {nhl.NHL_NB_OU_MIN_MATCHS}) — abort"
        )
        sys.exit(1)

    r_def = float(nhl.NHL_NB_OU_DISPERSION_DEFAULT)
    recency_prev = nhl.NHL_MLE_RECENCY_ACTIF
    nhl.NHL_MLE_RECENCY_ACTIF = False
    try:
        # Point de départ = défaut env (pas un meta déjà biaisé)
        nhl._rho_meta_cache = {**_meta_defaults_bt(), "nb_ou_dispersion": r_def}
        nll_def = nhl.log_likelihood_nb_ou_dispersion(r_def, dataset)
        brier_def = _brier_ou_nb(dataset, r_def)
        r_opt = nhl.optimiser_nb_ou_dispersion(dataset)
        nll_opt = nhl.log_likelihood_nb_ou_dispersion(r_opt, dataset)
        brier_opt = _brier_ou_nb(dataset, r_opt)
    finally:
        nhl.NHL_MLE_RECENCY_ACTIF = recency_prev
        nhl._invalider_rho_meta_cache()

    out = dict(_meta_defaults_bt())
    if os.path.exists(nhl.RHO_META_FILE):
        try:
            with open(nhl.RHO_META_FILE, "r", encoding="utf-8") as f:
                out = {**out, **json.load(f)}
        except Exception:
            pass

    out["nb_ou_dispersion"] = r_opt
    out["nb_ou_dispersion_opt"] = r_opt
    out["nb_ou_calib_n"] = len(dataset)
    out["nb_ou_calib_saisons"] = saisons
    out["nb_ou_nll_default"] = round(float(nll_def), 4)
    out["nb_ou_nll_opt"] = round(float(nll_opt), 4)
    out["nb_ou_brier55_default"] = round(brier_def, 6)
    out["nb_ou_brier55_opt"] = round(brier_opt, 6)
    out["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    if not out.get("nb_matchs"):
        out["nb_matchs"] = len(dataset)

    with open(nhl.RHO_META_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    nhl._invalider_rho_meta_cache()

    print(
        f"  r {r_def:.1f} → {r_opt:.1f} | NLL {nll_def:.1f} → {nll_opt:.1f} | "
        f"Brier Over5.5 {brier_def:.4f} → {brier_opt:.4f} | n={len(dataset)}"
    )
    print(f"  💾 Écrit → {nhl.RHO_META_FILE}")
    return r_opt


def phase_calib_ou_mu_scale(conn: sqlite3.Connection, saisons: list[int]) -> float:
    """Calibre ou_mu_scale (grille Brier Over 5.5/6.5) → rho_calibrage_meta.json."""
    print("\n" + "=" * 60)
    print("🔬  CALIBRAGE ou_mu_scale (PIT → rho_calibrage_meta.json)")
    print("=" * 60)
    print(f"  Saisons : {saisons}")

    dataset = _construire_dataset_pl_pit(conn, saisons)
    if len(dataset) < nhl.NHL_OU_MU_SCALE_CALIB_MIN_MATCHS:
        print(
            f"❌ Trop peu de matchs ({len(dataset)} < {nhl.NHL_OU_MU_SCALE_CALIB_MIN_MATCHS}) — abort"
        )
        sys.exit(1)

    r_disp = float(nhl.NHL_NB_OU_DISPERSION_DEFAULT)
    recency_prev = nhl.NHL_MLE_RECENCY_ACTIF
    nhl.NHL_MLE_RECENCY_ACTIF = False
    try:
        # Scale=1 pendant la grille ; r opérationnel (défaut env, pas meta biaisé)
        nhl._rho_meta_cache = {**_meta_defaults_bt(), "nb_ou_dispersion": r_disp, "ou_mu_scale": 1.0}
        brier_def = nhl._brier_ou_mu_scale(nhl.NHL_OU_MU_SCALE_DEFAULT, dataset, r_disp=r_disp)
        scale_opt = nhl.calibrer_ou_mu_scale(dataset, r_disp=r_disp)
        brier_opt = nhl._brier_ou_mu_scale(scale_opt, dataset, r_disp=r_disp)
    finally:
        nhl.NHL_MLE_RECENCY_ACTIF = recency_prev
        nhl._invalider_rho_meta_cache()

    out = dict(_meta_defaults_bt())
    if os.path.exists(nhl.RHO_META_FILE):
        try:
            with open(nhl.RHO_META_FILE, "r", encoding="utf-8") as f:
                out = {**out, **json.load(f)}
        except Exception:
            pass

    out["ou_mu_scale"] = scale_opt
    out["ou_mu_scale_opt"] = scale_opt
    out["ou_mu_scale_calib_n"] = len(dataset)
    out["ou_mu_scale_calib_saisons"] = saisons
    out["ou_mu_scale_brier_default"] = round(brier_def, 6)
    out["ou_mu_scale_brier_opt"] = round(brier_opt, 6)
    out["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    if not out.get("nb_matchs"):
        out["nb_matchs"] = len(dataset)

    with open(nhl.RHO_META_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    nhl._invalider_rho_meta_cache()

    print(
        f"  ou_mu_scale {nhl.NHL_OU_MU_SCALE_DEFAULT:.3f} → {scale_opt:.3f} | "
        f"Brier Over5.5/6.5 {brier_def:.4f} → {brier_opt:.4f} | n={len(dataset)}"
    )
    print(f"  💾 Écrit → {nhl.RHO_META_FILE}")
    return scale_opt


# ─────────────────────────────────────────────────────────────
# Phase 2 — Simulation
# ─────────────────────────────────────────────────────────────
def simuler_saison(
    conn: sqlite3.Connection,
    season: int,
    calib_mode: str,
    bankroll: float,
) -> tuple[float, list[dict]]:
    _appliquer_calib_mode(calib_mode)
    meta = nhl.lire_rho_meta()
    rho = float(meta["rho"])
    prob_tie = float(meta.get("prob_tie", 0.12))
    prob_en = float(meta.get("prob_en", 0.22))

    pit_index = nhl.construire_index_pit_moneypuck(season=season) if nhl.NHL_PIT_CALIB_ACTIF else None
    teams_n1 = nhl.get_team_stats(season=season - 1, blend=False) if nhl.NHL_BLEND_GP_PLEIN > 0 else None
    goalie_by_game = nhl.lire_goalie_by_game_lookup(season=season) if nhl.NHL_GSAX_CALIB_ACTIF else {}

    odds_prise = _charger_odds_table(conn, ODDS_TABLE_PRISE)
    odds_close = _charger_odds_table(conn, "nhl_odds_cloture")
    games = list(conn.execute(
        "SELECT game_id, date_utc, home, away, gh, ga FROM nhl_games "
        "WHERE season=? AND gh IS NOT NULL ORDER BY date_utc, game_id",
        (season,),
    ))
    refs_lookup = _charger_refs_lookup(conn) if NHL_BT_REFS_ACTIF else {}
    meta_refs = _warmstart_meta_refs(conn, season) if NHL_BT_REFS_ACTIF else _meta_refs_vide()
    if NHL_BT_REFS_ACTIF:
        _installer_meta_refs_live(meta_refs)
        n_warm = int(meta_refs.get("total_games", 0))
        n_crew = sum(1 for g in games if str(g[0]) in refs_lookup)
        print(
            f"  👨‍⚖️ Refs saison {season} — warmstart {n_warm} matchs antérieurs, "
            f"{n_crew}/{len(games)} crews dispo"
        )

    games_avant: dict = {}
    signaux: list[dict] = []
    skipped = {"pas_cotes": 0, "pas_pit": 0, "pas_edge": 0}

    for game_id, date_utc, home, away, gh, ga in games:
        rows_prise = odds_prise.get(game_id)
        if not rows_prise:
            skipped["pas_cotes"] += 1
            # Même sans cote : enrichir meta refs pour PIT des matchs suivants
            if NHL_BT_REFS_ACTIF:
                info = refs_lookup.get(str(game_id), {})
                _enrichir_meta_refs(meta_refs, info.get("refs") or [], info.get("penalties"))
            continue

        cotes_book = _assembler_cotes_book(rows_prise, home, away)
        if not cotes_book:
            skipped["pas_cotes"] += 1
            if NHL_BT_REFS_ACTIF:
                info = refs_lookup.get(str(game_id), {})
                _enrichir_meta_refs(meta_refs, info.get("refs") or [], info.get("penalties"))
            continue

        home_b2b, away_b2b, home_travel, away_travel = _contexte_calendrier_bt(
            games_avant, date_utc, home, away,
        )
        starters = goalie_by_game.get(str(game_id), {})
        home_gsax = float((starters.get(home) or {}).get("gsax_per_60", 0.0))
        away_gsax = float((starters.get(away) or {}).get("gsax_per_60", 0.0))
        gardiens_confirmes = home in starters and away in starters

        ref_info = refs_lookup.get(str(game_id), {})
        ref_names = ref_info.get("refs") or []
        ref_pp_mult = 1.0
        if NHL_BT_REFS_ACTIF and ref_names and nhl.NHL_REF_ADJ_ACTIF:
            _installer_meta_refs_live(meta_refs)
            ref_pp_mult, _ = nhl.compute_referee_pp_multiplier(ref_names)

        teams_snap = nhl._teams_data_snapshot_pit(pit_index, date_utc, teams_n1=teams_n1) if pit_index else None
        if not teams_snap:
            skipped["pas_pit"] += 1
            games_avant[game_id] = {"date": date_utc, "home": home, "away": away}
            if NHL_BT_REFS_ACTIF:
                _enrichir_meta_refs(meta_refs, ref_names, ref_info.get("penalties"))
            continue

        hia_match = float(meta.get("hia", nhl.NHL_HIA_DEFAULT))

        cotes_vraies = nhl.calculate_master_odds_v4(
            teams_snap, home, away,
            home_gsax=home_gsax, away_gsax=away_gsax,
            home_is_b2b=home_b2b, away_is_b2b=away_b2b,
            home_travel_miles=home_travel, away_travel_miles=away_travel,
            referee_pp_mult=ref_pp_mult,
            rho=rho, hia=hia_match,
            prob_tie=prob_tie, prob_en=prob_en,
        )
        if not cotes_vraies:
            skipped["pas_pit"] += 1
            games_avant[game_id] = {"date": date_utc, "home": home, "away": away}
            if NHL_BT_REFS_ACTIF:
                _enrichir_meta_refs(meta_refs, ref_names, ref_info.get("penalties"))
            continue

        cotes_puckline = _extraire_puckline(cotes_book)
        home_base = next((t for t in teams_snap if t["team"] == home), {})
        away_base = next((t for t in teams_snap if t["team"] == away), {})
        gp_moyen = (home_base.get("games_played", 0) + away_base.get("games_played", 0)) / 2.0

        cotes_vraies, _ = nhl.shrink_cotes_vers_marche(
            cotes_vraies, cotes_book, cotes_puckline, gp_moyen,
        )

        m = {"game_id": game_id, "home_team": home, "away_team": away}
        candidats = nhl._construire_candidats_pari(
            m, cotes_vraies, cotes_book, cotes_puckline,
            bankroll, gardiens_verrouilles=gardiens_confirmes,
            kelly_mult=1.0, gp_moyen_match=gp_moyen,
            kelly_param_mult=1.0,
        )
        paris_retenus = nhl._selectionner_paris(candidats)
        if not paris_retenus:
            skipped["pas_edge"] += 1
            games_avant[game_id] = {"date": date_utc, "home": home, "away": away}
            if NHL_BT_REFS_ACTIF:
                _enrichir_meta_refs(meta_refs, ref_names, ref_info.get("penalties"))
            continue

        for best in paris_retenus:
            marche = best.get("marche", "?")
            type_pari = best["type"]
            cote_prise = float(best["cote_book"])
            cote_modele = float(best["cote_vraie"])
            prob = _prob_modele_pari(marche, type_pari, home, cotes_vraies) or (1.0 / cote_modele)
            mise = float(best["inv"]["mise"])
            gagne = _regler_pari(marche, type_pari, home, away, int(gh), int(ga))
            pnl = round(mise * (cote_prise - 1), 2) if gagne else round(-mise, 2)
            bankroll = round(bankroll + pnl, 2)

            cote_cloture = _trouver_cote_cloture(odds_close, game_id, marche, type_pari, home, away, cotes_book)
            clv = round((cote_prise / cote_cloture) - 1, 4) if cote_cloture and cote_cloture > 1 else None

            sig = {
                "game_id": game_id, "season": season, "date_utc": date_utc,
                "marche": marche, "type_pari": type_pari,
                "home": home, "away": away,
                "cote_prise": cote_prise, "cote_cloture": cote_cloture,
                "cote_modele": cote_modele, "prob_modele": round(prob, 4),
                "edge_pct": best["inv"]["edge"], "mise": mise,
                "gh": gh, "ga": ga, "gagne": int(gagne), "pnl": pnl, "clv": clv,
                "lam_home": cotes_vraies.get("lam_home"), "lam_away": cotes_vraies.get("lam_away"),
            }
            signaux.append(sig)

        games_avant[game_id] = {"date": date_utc, "home": home, "away": away}
        if NHL_BT_REFS_ACTIF:
            _enrichir_meta_refs(meta_refs, ref_names, ref_info.get("penalties"))

    print(
        f"  Saison {season} : {len(signaux)} signaux | "
        f"skip cotes={skipped['pas_cotes']} pit={skipped['pas_pit']} edge={skipped['pas_edge']}"
    )
    return bankroll, signaux


def _trouver_cote_cloture(
    odds_close: dict, game_id: str, marche: str, type_pari: str,
    home: str, away: str, cotes_book_prise: dict,
) -> float | None:
    rows = odds_close.get(game_id)
    if not rows:
        return None
    cotes = _assembler_cotes_book(rows, home, away)
    if not cotes:
        return None
    if marche == "ML":
        return cotes.get("cote_1") if home in type_pari else cotes.get("cote_2")
    if marche == "PL":
        return cotes.get("cote_pl_home") if home in type_pari else cotes.get("cote_pl_away")
    if marche == "OU":
        parts = type_pari.split()
        cut = nhl._arrondir_cut(float(parts[1]))
        side = "Over" if parts[0].upper() == "OVER" else "Under"
        return cotes.get("totals", {}).get(cut, {}).get(side)
    return None


def _verifier_pit_disponible(season: int) -> bool:
    """Bloque tôt si MoneyPuck PIT inaccessible (HTTP 403 fréquent en script)."""
    if not nhl.NHL_PIT_CALIB_ACTIF:
        return True
    pit = nhl.construire_index_pit_moneypuck(season=season)
    if pit:
        if nhl.NHL_GSAX_CALIB_ACTIF:
            nhl.construire_index_gsax_gardiens(season=season)
        return True
    print("\n❌ Index PIT MoneyPuck indisponible — simulation impossible.")
    print("   MoneyPuck bloque souvent les requêtes script (HTTP 403).")
    print("   Placez all_teams.csv dans l'un de ces dossiers (auto-détecté + sync env/nhl.env) :")
    for chemin in nhl._candidats_pit_teams_csv()[:4]:
        print(f"   • {chemin}")
    print(f"   ou téléchargez : {nhl.MONEYPUCK_GBG_ALL_TEAMS_URL}")
    print("   Relancez : python backtest_nhl.py --simulate --report")
    return False


def persister_signaux(conn: sqlite3.Connection, signaux: list[dict]) -> None:
    conn.execute("DELETE FROM nhl_signaux")
    if not signaux:
        conn.commit()
        return
    conn.executemany(
        """INSERT OR REPLACE INTO nhl_signaux VALUES (
            :game_id, :season, :date_utc, :marche, :type_pari, :home, :away,
            :cote_prise, :cote_cloture, :cote_modele, :prob_modele, :edge_pct,
            :mise, :gh, :ga, :gagne, :pnl, :clv, :lam_home, :lam_away
        )""",
        signaux,
    )
    conn.commit()


def phase_simulation(conn: sqlite3.Connection, saisons: list[int], calib_mode: str) -> None:
    print("\n" + "=" * 60)
    print("🔬  PHASE 2 — SIMULATION (parité bot live)")
    print("=" * 60)
    print(f"  Calib : {calib_mode} | Bankroll init : {BANKROLL_BT:.0f} €")
    print(
        f"  Pipeline : PIT → master_odds_v4 → shrink → Kelly → 1 pari/match "
        f"(cote prise H-{ODDS_PRISE_HEURES:g} / match)"
    )

    n_odds = conn.execute(f"SELECT COUNT(*) FROM {ODDS_TABLE_PRISE}").fetchone()[0]
    if n_odds == 0:
        print("\n  ⚠️ Aucune cote de prise — relancez --collect (H-24 abandonné, recollecte requise).")
        return

    if not _verifier_pit_disponible(saisons[0]):
        return

    bankroll = BANKROLL_BT
    tous: list[dict] = []
    for season in saisons:
        bankroll, sigs = simuler_saison(conn, season, calib_mode, bankroll)
        tous.extend(sigs)

    persister_signaux(conn, tous)
    print(f"\n✅ Phase 2 terminée — {len(tous)} signaux | bankroll finale {bankroll:.2f} €")


# ─────────────────────────────────────────────────────────────
# Phase 3 — Rapport
# ─────────────────────────────────────────────────────────────
def _brier(signaux: list[dict]) -> float | None:
    if not signaux:
        return None
    return sum((s["prob_modele"] - s["gagne"]) ** 2 for s in signaux) / len(signaux)


def _resume_segment(signaux: list[dict], label: str) -> None:
    if not signaux:
        print(f"  {label:<12} — aucun signal")
        return
    n = len(signaux)
    wins = sum(s["gagne"] for s in signaux)
    pnl = sum(s["pnl"] for s in signaux)
    mise_tot = sum(s["mise"] for s in signaux)
    roi = pnl / mise_tot if mise_tot else 0.0
    clvs = [s["clv"] for s in signaux if s.get("clv") is not None]
    clv_moy = sum(clvs) / len(clvs) if clvs else float("nan")
    brier = _brier(signaux)
    brier_s = f"{brier:.4f}" if brier is not None else "—"
    print(
        f"  {label:<12} n={n:>4}  WR={wins/n:.1%}  ROI={roi:+.1%}  "
        f"P&L={pnl:+.1f}u  CLV={clv_moy:+.2%}  Brier={brier_s}"
    )


def exporter_csv(signaux: list[dict]) -> None:
    if not signaux:
        return
    cols = list(signaux[0].keys())
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(signaux)
    print(f"\n📁 Export → {RESULTS_CSV}")


def phase_rapport(conn: sqlite3.Connection) -> None:
    print("\n" + "=" * 60)
    print("📊  PHASE 3 — RAPPORT")
    print("=" * 60)
    rows = conn.execute("SELECT * FROM nhl_signaux ORDER BY date_utc").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM nhl_signaux LIMIT 0").description]
    signaux = [dict(zip(cols, row)) for row in rows]
    if not signaux:
        print("  Aucun signal — lancez --simulate après --collect.")
        return

    exporter_csv(signaux)
    _resume_segment(signaux, "GLOBAL")
    for marche in sorted({s["marche"] for s in signaux}):
        _resume_segment([s for s in signaux if s["marche"] == marche], marche)
    for season in sorted({s["season"] for s in signaux}):
        _resume_segment([s for s in signaux if s["season"] == season], f"S{season}")

    print("\n  Note : P&L simulé avec mises Kelly du bot ; variance élevée sur <100 paris.")


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
        conn.execute("DELETE FROM nhl_signaux")
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

    if args.build_goalies:
        path = nhl.construire_all_goalies_csv_depuis_shots(force=True)
        nhl._invalider_goalie_pit_index_memo()
        if not path:
            print("❌ Échec construction all_goalies.csv")
            sys.exit(1)
        print(f"✅ all_goalies.csv → {path}")

    run_phases = (
        args.collect or args.simulate or args.report
        or args.calib_pl_scale or args.calib_nb_ou or args.calib_ou_mu_scale
        or args.collect_refs
    )
    all_phases = (
        not run_phases and not args.reset and not args.reset_full
        and not args.build_goalies
    )
    if not run_phases and not all_phases and not args.build_goalies:
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    try:
        if args.calib_pl_scale:
            pl_saisons = (
                [int(s) for s in args.saisons.split(",")]
                if args.saisons
                else DEFAULT_PL_CALIB_SAISONS
            )
            phase_calib_pl_scale(conn, pl_saisons)
        if args.calib_nb_ou:
            nb_saisons = (
                [int(s) for s in args.saisons.split(",")]
                if args.saisons
                else DEFAULT_PL_CALIB_SAISONS
            )
            phase_calib_nb_ou(conn, nb_saisons)
        if args.calib_ou_mu_scale:
            ou_saisons = (
                [int(s) for s in args.saisons.split(",")]
                if args.saisons
                else DEFAULT_PL_CALIB_SAISONS
            )
            phase_calib_ou_mu_scale(conn, ou_saisons)
        if args.collect or all_phases:
            await phase_collecte(conn, saisons, odds_only=args.odds_only)
        if args.collect_refs:
            await phase_collecte_refs(conn, saisons)
        if args.simulate or all_phases:
            phase_simulation(conn, saisons, args.calib)
        if args.report or all_phases:
            phase_rapport(conn)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Back-test NHL (parité nhl_sniper_omega)")
    parser.add_argument("--collect", action="store_true", help="Phase 1 : matchs + cotes historiques")
    parser.add_argument("--simulate", action="store_true", help="Phase 2 : simulation")
    parser.add_argument("--report", action="store_true", help="Phase 3 : rapport + CSV")
    parser.add_argument(
        "--collect-refs", action="store_true",
        help="Collecte crews arbitres + pénalités (API NHL) → nhl_refs",
    )
    parser.add_argument(
        "--calib-pl-scale", action="store_true",
        help="Calibre pl_scale sur lambdas PIT (défaut saisons 2023-2025) → rho_calibrage_meta.json",
    )
    parser.add_argument(
        "--calib-nb-ou", action="store_true",
        help="Calibre nb_ou_dispersion (MLE NB) sur lambdas PIT → rho_calibrage_meta.json",
    )
    parser.add_argument(
        "--calib-ou-mu-scale", action="store_true",
        help="Calibre ou_mu_scale (Brier Over 5.5/6.5) sur lambdas PIT → rho_calibrage_meta.json",
    )
    parser.add_argument(
        "--build-goalies", action="store_true",
        help="Construit data/moneypuck/all_goalies.csv depuis les zips shots MoneyPuck",
    )
    parser.add_argument("--reset", action="store_true", help="Vide signaux, garde la DB")
    parser.add_argument("--reset-full", action="store_true", help="Supprime DB + CSV")
    parser.add_argument("--odds-only", action="store_true", help="Recollecte cotes uniquement")
    parser.add_argument("--saisons", type=str, default=None, help="Ex: 2025,2026 (MoneyPuck)")
    parser.add_argument(
        "--calib", choices=("defaults", "frozen"), default="defaults",
        help="defaults=constantes env (honnête) | frozen=rho_meta.json actuel",
    )
    args = parser.parse_args()

    if args.odds_only and not args.collect:
        print("❌ --odds-only requiert --collect")
        sys.exit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
