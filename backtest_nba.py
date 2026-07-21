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
  python backtest_nba.py --collect --saisons 2024,2025,2026
  python backtest_nba.py --tune --saisons 2022,2023,2024,2025,2026
      # Calibre n_prior / demi-vie / HCA / sigma_team / rho / B2B (Brier ML) → nba_params_tuned.json

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
from datetime import datetime, timedelta

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
            PRIMARY KEY (game_id, marche, type_pari)
        );
    """)
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


def _bucketiser_par_instant(games: list[dict], heures_avant: float) -> dict[datetime, list[dict]]:
    buckets: dict[datetime, list[dict]] = defaultdict(list)
    for game in games:
        dt = _instant_prise_pour_match(game, heures_avant)
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

    close_buckets: dict[datetime, list[dict]] = defaultdict(list)
    for game in games:
        start = _parse_utc(game["start_utc"])
        if start:
            close_buckets[_limite_prise_avant_tipoff(start)].append(game)
    for i, (dt_close, bucket) in enumerate(sorted(close_buckets.items()), 1):
        total += await collecter_odds_instant(session, conn, dt_close, "nba_odds_cloture", bucket)
        if i % 200 == 0:
            print(f"     … clôture {i}/{len(close_buckets)} snapshots", flush=True)
        await asyncio.sleep(0.35)

    n_cov = conn.execute(
        "SELECT COUNT(DISTINCT o.game_id) FROM nba_odds_prise o JOIN nba_games g ON g.game_id=o.game_id WHERE g.season=?",
        (season,),
    ).fetchone()[0]
    print(f"  ✅ Saison {season} : {total} événement(s) cotes | {n_cov}/{len(games)} matchs avec ligne de prise")
    if restants:
        print(f"     ⚠️ {len(restants)} match(s) toujours sans cote Pinnacle (ligne jamais publiée ?)")
    return total


async def phase_collecte(conn: sqlite3.Connection, saisons: list[int], odds_only: bool = False) -> None:
    print("\n" + "=" * 60)
    print("📥  PHASE 1 — COLLECTE NBA")
    print("=" * 60)
    if not odds_only:
        for season in saisons:
            n = collecter_matchs_saison(conn, season)
            print(f"🔄 Saison {season} ({season - 1}-{str(season)[-2:]}) — {n} match(s) terminé(s) indexés")
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
def _trouver_cote_cloture_nba(odds_close: dict, game_id: str, type_pari: str, home: str, away: str) -> float | None:
    cotes = odds_close.get(game_id)
    if not cotes:
        return None
    cache = {(home, away): cotes}
    val = nba._extraire_cote_clv_nba(home, away, type_pari, None, cache)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def simuler_saison(conn: sqlite3.Connection, season: int, bankroll: float) -> tuple[float, list[dict]]:
    odds_prise = _charger_odds_json(conn, "nba_odds_prise")
    odds_close = _charger_odds_json(conn, "nba_odds_cloture")
    games = list(conn.execute(
        "SELECT game_id, date_utc, home, away, gh, ga FROM nba_games WHERE season=? ORDER BY date_utc, game_id",
        (season,),
    ))
    signaux: list[dict] = []
    skipped = {"pas_cotes": 0, "pas_edge": 0}

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
            {"home": home, "away": away}, probas_shrink, cotes_match, bankroll, gp_moyen,
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

            cote_cloture = _trouver_cote_cloture_nba(odds_close, game_id, type_pari, home, away)
            clv = round((cote_prise / cote_cloture) - 1, 4) if cote_cloture and cote_cloture > 1 else None

            signaux.append({
                "game_id": game_id, "season": season, "date_utc": date_utc,
                "marche": marche, "type_pari": type_pari, "home": home, "away": away,
                "cote_prise": cote_prise, "cote_cloture": cote_cloture, "cote_modele": cote_modele,
                "prob_modele": prob, "edge_pct": best["inv"]["edge"], "mise": mise,
                "gh": gh, "ga": ga, "gagne": (None if push else int(gagne)), "pnl": pnl, "clv": clv,
                "mu_home": probas.get("mu_home"), "mu_away": probas.get("mu_away"),
            })

    print(f"  Saison {season} : {len(signaux)} signaux | skip cotes={skipped['pas_cotes']} edge={skipped['pas_edge']}")
    return bankroll, signaux


def persister_signaux(conn: sqlite3.Connection, signaux: list[dict]) -> None:
    conn.execute("DELETE FROM nba_signaux")
    if signaux:
        conn.executemany(
            """INSERT OR REPLACE INTO nba_signaux VALUES (
                :game_id, :season, :date_utc, :marche, :type_pari, :home, :away,
                :cote_prise, :cote_cloture, :cote_modele, :prob_modele, :edge_pct,
                :mise, :gh, :ga, :gagne, :pnl, :clv, :mu_home, :mu_away
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
    print(f"  Bankroll init : {BANKROLL_BT:.0f} € | pipeline : moteur → shrink marché → Kelly → journal")

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
    clv_moy = sum(clvs) / len(clvs) if clvs else float("nan")
    brier = _brier(signaux)
    brier_s = f"{brier:.4f}" if brier is not None else "—"
    wr = wins / len(clos) if clos else 0.0
    print(
        f"  {label:<12} n={n:>4}  WR={wr:.1%}  ROI={roi:+.1%}  P&L={pnl:+.1f}u  CLV={clv_moy:+.2%}  Brier={brier_s}"
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
    rows = conn.execute("SELECT * FROM nba_signaux ORDER BY date_utc").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM nba_signaux LIMIT 0").description]
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
# Phase 4 — Calibration (--tune) : Brier moneyline sur TOUS les matchs
# (pas seulement les paris retenus) → nba_params_tuned.json
# ─────────────────────────────────────────────────────────────
PARAM_NAMES = [
    "n_prior", "rating_half_life_jours", "blend_gp_plein",
    "hca", "b2b_atk_pct", "b2b_def_pct", "sigma_team", "rho_scores",
]


def _construire_dataset_calibration(conn: sqlite3.Connection, saisons: list[int]) -> list[dict]:
    dataset = []
    for season in saisons:
        rows = conn.execute(
            "SELECT game_id, date_utc, home, away, gh, ga FROM nba_games WHERE season=? ORDER BY date_utc, game_id",
            (season,),
        ).fetchall()
        for game_id, date_utc, home, away, gh, ga in rows:
            dataset.append({"season": season, "date": date_utc, "home": home, "away": away, "gh": gh, "ga": ga})
    return dataset


def _appliquer_candidat_params(x: list[float]) -> None:
    n_prior, half_life, blend_gp_plein, hca, b2b_atk, b2b_def, sigma_team, rho = x
    nba_params.save_tuned_params(
        {"_global": {
            "n_prior": round(max(n_prior, 0.5), 3),
            "rating_half_life_jours": round(max(half_life, 1.0), 2),
            "blend_gp_plein": round(max(blend_gp_plein, 1.0), 2),
            "hca": round(hca, 5),
            "b2b_atk_pct": round(b2b_atk, 5),
            "b2b_def_pct": round(b2b_def, 5),
            "sigma_team": round(max(sigma_team, 3.0), 3),
            "rho_scores": round(min(max(rho, -0.9), 0.9), 4),
        }},
        source="backtest_tune_wip",
        merge_existing=True,
    )
    nba_params.reload_tuned_params()


def _brier_ml_pour_params(x: list[float], dataset: list[dict]) -> float:
    _appliquer_candidat_params(x)
    total, n = 0.0, 0
    for row in dataset:
        probas = nba.calculer_probabilites_match_nba(row["home"], row["away"], row["date"], season_year=row["season"])
        p_dom = probas["moneyline"]["domicile"]
        outcome = 1.0 if row["gh"] > row["ga"] else 0.0
        total += (p_dom - outcome) ** 2
        n += 1
    return total / n if n else 1.0


def phase_tune(conn: sqlite3.Connection, saisons: list[int], maxiter: int) -> None:
    from scipy.optimize import minimize

    print("\n" + "=" * 60)
    print("🎛️   PHASE 4 — CALIBRATION (--tune)")
    print("=" * 60)
    dataset = _construire_dataset_calibration(conn, saisons)
    if len(dataset) < 100:
        print(f"  ⚠️ Dataset trop petit ({len(dataset)} matchs) — lancez --collect sur plusieurs saisons.")
        return
    print(f"  Dataset calibration : {len(dataset)} matchs (saisons {saisons}) | objectif : Brier moneyline")

    x0 = [
        nba_params.N_PRIOR_DEFAULT, nba_params.RATING_HALF_LIFE_JOURS_DEFAULT, nba_params.BLEND_GP_PLEIN_DEFAULT,
        nba_params.HCA_DEFAULT, nba_params.B2B_ATK_PCT_DEFAULT, nba_params.B2B_DEF_PCT_DEFAULT,
        nba_params.SIGMA_TEAM_DEFAULT, nba_params.RHO_SCORES_DEFAULT,
    ]
    brier_defaut = _brier_ml_pour_params(x0, dataset)
    print(f"  Brier ML (defaults) : {brier_defaut:.4f}")

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
    brier_final = resultat.fun
    print(f"\n  Brier ML (calibré)   : {brier_final:.4f} ({n_evals[0]} évaluations, convergé={resultat.success})")

    if brier_final >= brier_defaut:
        print("  ⚠️ Aucune amélioration vs defaults — conservation des valeurs par défaut (pas d'écriture).")
        nba_params.save_tuned_params({"_global": {}}, source="backtest_tune_no_improve", merge_existing=True)
        nba_params.reload_tuned_params()
        return

    _appliquer_candidat_params(list(resultat.x))
    n_prior, half_life, blend_gp_plein, hca, b2b_atk, b2b_def, sigma_team, rho = resultat.x
    print("\n  Paramètres calibrés :")
    print(f"    n_prior={n_prior:.2f}  rating_half_life_jours={half_life:.1f}  blend_gp_plein={blend_gp_plein:.1f}")
    print(f"    hca={hca:.4f}  b2b_atk_pct={b2b_atk:.4f}  b2b_def_pct={b2b_def:.4f}")
    print(f"    sigma_team={sigma_team:.2f}  rho_scores={rho:.3f}")
    print(f"\n✅ nba_params_tuned.json mis à jour (Brier {brier_defaut:.4f} → {brier_final:.4f}).")


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

    run_phases = args.collect or args.simulate or args.report or args.tune
    all_phases = not run_phases and not args.reset and not args.reset_full
    if not run_phases and not all_phases:
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    try:
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
