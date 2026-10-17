"""
Sniper tennis ATP — paper trading (moneyline + total jeux).

Usage :
  python tennis_sniper.py --once
  python tennis_sniper.py

Liste les sports `tennis_atp_*` actifs sur The Odds API, blend le modèle avec
Pinnacle no-vig (Shin) : p_final = (1-λ)*p_model + λ*p_market, puis edge / Kelly
sur p_final. Alerte Telegram + journal même en DRY_RUN.
"""
from __future__ import annotations

import argparse
import csv
import ftplib
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from config_env import env_files_hint, load_project_env

load_project_env("tennis")

from odds_devig import proba_no_vig_shin_2way
from tennis_calib import blend_model_market
from tennis_params import (
    PA_DATA_DIR,
    TENNIS_BANKROLL,
    TENNIS_BEST_OF_DEFAULT,
    TENNIS_DRY_RUN,
    TENNIS_EDGE_MIN,
    TENNIS_JOURNAL_NOM,
    TENNIS_JOURNAL_PATH,
    TENNIS_MARKET_BLEND,
    TENNIS_MARKET_BLEND_ML,
    TENNIS_MARKET_BLEND_OU,
    TENNIS_MINUTES_AVANT_TIP,
    TENNIS_PRISE_HEURES_MAX,
    TENNIS_PRISE_HEURES_MIN,
    TENNIS_SACKMANN_AUTO_REFRESH,
    TENNIS_SACKMANN_REFRESH_HOURS,
    TENNIS_SCAN_HEURES_AVANCE,
    TENNIS_STATE_CACHE_HOURS,
    TENNIS_STATE_RELOAD_HOURS,
)
from tennis_state import TennisLiveState, get_live_state, infer_surface

_ROOT = Path(__file__).resolve().parent

API_ODDS_KEY = os.environ.get("API_ODDS_KEY", "")
TENNIS_TELEGRAM_TOKEN = os.environ.get("TENNIS_TELEGRAM_TOKEN") or os.environ.get(
    "TELEGRAM_TOKEN", ""
)
TENNIS_TELEGRAM_CHAT_ID = os.environ.get("TENNIS_TELEGRAM_CHAT_ID") or os.environ.get(
    "TELEGRAM_CHAT_ID", ""
)
TENNIS_BOOKMAKER = os.environ.get("TENNIS_BOOKMAKER", "pinnacle")
TENNIS_REGIONS = os.environ.get("TENNIS_REGIONS", "eu")
TENNIS_TOUR_PREFIX = os.environ.get("TENNIS_TOUR_PREFIX", "tennis_atp_")
TENNIS_LOOP_SECONDS = float(os.environ.get("TENNIS_LOOP_SECONDS", "300"))
TENNIS_KELLY_FRACTION = float(os.environ.get("TENNIS_KELLY_FRACTION", "0.25"))
TENNIS_MISE_MAX_PCT = float(os.environ.get("TENNIS_MISE_MAX_PCT", "0.02"))
JOURNAL_PATH = Path(TENNIS_JOURNAL_PATH)

JOURNAL_COLS = [
    "Date",
    "EventId",
    "SportKey",
    "Tourney",
    "Surface",
    "PlayerA",
    "PlayerB",
    "Marche",
    "Selection",
    "Cote",
    "FairCote",
    "ProbaModele",
    "ProbaMarche",
    "ProbaFinal",
    "BlendLam",
    "Edge",
    "Mise",
    "DryRun",
    "Statut",
    "CommenceTime",
    "CoteClose",
    "CLV",
    "Resultat",
    "PnL",
]


def log_tennis(msg: str, level: str = "info") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"warning": "⚠️", "error": "❌"}.get(level, "🎾")
    print(f"[{ts}] {prefix} {msg}", flush=True)


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class EdgeCandidate:
    event_id: str
    sport_key: str
    tourney: str
    surface: str
    player_a: str
    player_b: str
    marche: str  # ML_A / ML_B / OU_over_23.0 / OU_under_23.0
    selection: str
    cote: float
    fair_cote: float
    proba_modele: float
    proba_marche: float
    proba_final: float
    blend_lam: float
    edge: float
    mise: float
    commence_time: str = ""


def kelly_mise(p: float, cote: float, bankroll: float) -> float:
    if cote <= 1.0 or p <= 0:
        return 0.0
    b = cote - 1.0
    k = (b * p - (1.0 - p)) / b
    if k <= 0:
        return 0.0
    frac = max(0.0, min(1.0, TENNIS_KELLY_FRACTION)) * k
    frac = min(frac, TENNIS_MISE_MAX_PCT)
    return round(bankroll * frac, 2)


def list_active_atp_sports() -> list[dict]:
    if not API_ODDS_KEY:
        return []
    url = "https://api.the-odds-api.com/v4/sports/"
    r = requests.get(url, params={"apiKey": API_ODDS_KEY}, timeout=20)
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining")
    if remaining is not None:
        log_tennis(f"Odds API quota restant={remaining}")
    out = []
    for s in r.json() or []:
        key = s.get("key") or ""
        if not key.startswith(TENNIS_TOUR_PREFIX):
            continue
        if not s.get("active"):
            continue
        out.append(s)
    return out


def fetch_odds_for_sport(sport_key: str) -> list[dict]:
    params = {
        "apiKey": API_ODDS_KEY,
        "regions": TENNIS_REGIONS,
        "markets": "h2h,totals",
        "bookmakers": TENNIS_BOOKMAKER,
        "oddsFormat": "decimal",
    }
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        log_tennis(f"Odds {sport_key}: HTTP {r.status_code}", level="warning")
        return []
    remaining = r.headers.get("x-requests-remaining")
    if remaining is not None:
        log_tennis(f"Odds API quota restant={remaining}")
    return r.json() or []


def _parse_book_markets(event: dict) -> dict | None:
    """Extrait h2h + totals du bookmaker configuré."""
    home = event.get("home_team") or ""
    away = event.get("away_team") or ""
    out = {
        "event_id": event.get("id") or "",
        "home": home,
        "away": away,
        "commence_time": event.get("commence_time"),
        "h2h": {},  # name -> cote
        "totals": {},  # line -> {over, under}
    }
    for book in event.get("bookmakers") or []:
        if book.get("key") != TENNIS_BOOKMAKER:
            continue
        for market in book.get("markets") or []:
            key = market.get("key")
            if key == "h2h":
                for o in market.get("outcomes") or []:
                    name = (o.get("name") or "").strip()
                    if name:
                        out["h2h"][name] = float(o["price"])
            elif key == "totals":
                for o in market.get("outcomes") or []:
                    point = o.get("point")
                    if point is None:
                        continue
                    line = round(float(point), 1)
                    side = str(o.get("name", "")).strip().lower()
                    bucket = out["totals"].setdefault(line, {})
                    if side == "over":
                        bucket["over"] = float(o["price"])
                    elif side == "under":
                        bucket["under"] = float(o["price"])
    if len(out["h2h"]) < 2:
        return None
    return out


def _hours_to_tip(commence_time: str | None) -> float | None:
    if not commence_time:
        return None
    try:
        tip = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    now = datetime.now(timezone.utc)
    return (tip - now).total_seconds() / 3600.0


def _within_scan_window(commence_time: str | None, heures: float) -> bool:
    """Match encore à venir et tip dans ≤ heures (scan Odds)."""
    delta_h = _hours_to_tip(commence_time)
    if delta_h is None:
        return True
    return 0.0 <= delta_h <= heures


def _within_prise_window(commence_time: str | None) -> tuple[bool, str]:
    """
    Fenêtre de prise paper : tip entre PRISE_HEURES_MIN et PRISE_HEURES_MAX,
    plus filet MINUTES_AVANT_TIP (jamais de quasi-live).
    """
    delta_h = _hours_to_tip(commence_time)
    if delta_h is None:
        return True, ""
    if delta_h > TENNIS_PRISE_HEURES_MAX:
        return False, f"trop tot ({delta_h:.1f}h > {TENNIS_PRISE_HEURES_MAX:.0f}h)"
    if delta_h < TENNIS_PRISE_HEURES_MIN:
        return False, f"trop tard ({delta_h:.1f}h < {TENNIS_PRISE_HEURES_MIN:.0f}h)"
    minutes = delta_h * 60.0
    if minutes < TENNIS_MINUTES_AVANT_TIP:
        return False, f"trop proche tip ({minutes:.0f}min < {TENNIS_MINUTES_AVANT_TIP:.0f}min)"
    return True, ""


def evaluate_event(
    state: TennisLiveState,
    sport: dict,
    event: dict,
    *,
    edge_min: float,
) -> list[EdgeCandidate]:
    parsed = _parse_book_markets(event)
    if parsed is None:
        return []
    if not _within_scan_window(parsed.get("commence_time"), TENNIS_SCAN_HEURES_AVANCE):
        return []

    home = parsed["home"]
    away = parsed["away"]
    # Convention modèle : A = away (visiteur Odds), B = home — peu importe, on mappe par nom
    id_home = state.resolve_player(home)
    id_away = state.resolve_player(away)
    if not id_home or not id_away:
        missing = []
        if not id_home:
            missing.append(home)
        if not id_away:
            missing.append(away)
        log_tennis(f"Joueur(s) non résolu(s): {', '.join(missing)}", level="warning")
        return []

    surface = infer_surface(sport.get("key", ""), sport.get("title", ""))
    commence = parsed.get("commence_time") or ""
    match_date = None
    if commence:
        try:
            tip = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            match_date = int(tip.strftime("%Y%m%d"))
        except ValueError:
            match_date = None
    proj, p_fav = state.project_match(
        id_home,
        id_away,
        surface,
        best_of=TENNIS_BEST_OF_DEFAULT,
        match_date=match_date,
    )

    cotes_h2h = parsed["h2h"]
    cote_home = cotes_h2h.get(home)
    cote_away = cotes_h2h.get(away)
    # parfois le nom ne match pas exactement la clé h2h
    if cote_home is None or cote_away is None:
        vals = list(cotes_h2h.items())
        if len(vals) == 2:
            # associer par normalize
            from tennis_state import normalize_player_name

            nh, na = normalize_player_name(home), normalize_player_name(away)
            for nom, cote in vals:
                nn = normalize_player_name(nom)
                if nn == nh:
                    cote_home = cote
                elif nn == na:
                    cote_away = cote
    if not cote_home or not cote_away:
        return []

    candidates: list[EdgeCandidate] = []
    sport_key = sport.get("key") or ""
    tourney = sport.get("title") or sport_key
    eid = parsed["event_id"]
    use_blend = bool(TENNIS_MARKET_BLEND)
    lam_ml = float(TENNIS_MARKET_BLEND_ML) if use_blend else 0.0
    lam_ou = float(TENNIS_MARKET_BLEND_OU) if use_blend else 0.0

    # --- Moneyline ---
    p_fair_h, p_fair_a = proba_no_vig_shin_2way(cote_home, cote_away)
    if p_fair_h is not None and p_fair_a is not None:
        for side, cote, p_model, p_mkt, name in (
            ("ML_home", cote_home, proj.p_a_win, p_fair_h, home),
            ("ML_away", cote_away, proj.p_b_win, p_fair_a, away),
        ):
            p_final, lam_eff = blend_model_market(p_model, p_mkt, lam_ml)
            edge = p_final * cote - 1.0
            if edge < edge_min:
                continue
            fair = 1.0 / p_final if p_final > 1e-12 else None
            if fair is None:
                continue
            candidates.append(
                EdgeCandidate(
                    event_id=eid,
                    sport_key=sport_key,
                    tourney=tourney,
                    surface=surface,
                    player_a=home,
                    player_b=away,
                    marche=side,
                    selection=name,
                    cote=float(cote),
                    fair_cote=round(fair, 3),
                    proba_modele=round(p_model, 4),
                    proba_marche=round(p_mkt, 4),
                    proba_final=round(p_final, 4),
                    blend_lam=round(lam_eff, 3),
                    edge=round(edge, 4),
                    mise=kelly_mise(p_final, float(cote), TENNIS_BANKROLL),
                    commence_time=commence,
                )
            )

    # --- Totaux jeux (proba O/U calibrée si disponible) ---
    for line, sides in parsed["totals"].items():
        co = sides.get("over")
        cu = sides.get("under")
        if not co or not cu:
            continue
        p_over_m = proj.p_over(line)
        p_under_m = proj.p_under(line)
        push = proj.p_push(line)
        if state.totals_calib is not None and state.totals_calib.leaves:
            p_over_m, p_under_m = state.totals_calib.calibrate_under(
                line, p_over_m, p_under_m, p_fav, p_push=push
            )
        elif push > 0:
            live = 1.0 - push
            if live > 1e-9:
                p_over_m = p_over_m / live
                p_under_m = p_under_m / live
        p_over_mkt, p_under_mkt = proba_no_vig_shin_2way(co, cu)
        for marche, cote, p_model, p_mkt, sel in (
            (f"OU_over_{line}", co, p_over_m, p_over_mkt, f"Over {line}"),
            (f"OU_under_{line}", cu, p_under_m, p_under_mkt, f"Under {line}"),
        ):
            p_final, lam_eff = blend_model_market(p_model, p_mkt, lam_ou)
            edge = p_final * cote - 1.0
            if edge < edge_min:
                continue
            fair = 1.0 / p_final if p_final > 1e-12 else None
            if fair is None:
                continue
            candidates.append(
                EdgeCandidate(
                    event_id=eid,
                    sport_key=sport_key,
                    tourney=tourney,
                    surface=surface,
                    player_a=home,
                    player_b=away,
                    marche=marche,
                    selection=sel,
                    cote=float(cote),
                    fair_cote=round(fair, 3),
                    proba_modele=round(p_model, 4),
                    proba_marche=round(float(p_mkt), 4) if p_mkt is not None else 0.0,
                    proba_final=round(p_final, 4),
                    blend_lam=round(lam_eff, 3),
                    edge=round(edge, 4),
                    mise=kelly_mise(p_final, float(cote), TENNIS_BANKROLL),
                    commence_time=commence,
                )
            )
    return candidates


def _journal_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not JOURNAL_PATH.is_file():
        return keys
    try:
        with JOURNAL_PATH.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                keys.add((row.get("EventId") or "", row.get("Marche") or ""))
    except Exception:
        pass
    return keys


def publier_journal_dashboard() -> None:
    """
    Rend le journal visible pour Streamlit via l'URL PA /data/...

    Sur PythonAnywhere, le static `/data/` pointe souvent sur `~/` et non `~/data/`.
    Canonique : `~/data/` ; miroir : `~/JOURNAL_NOM`. En local : FTP si PA_FTP_*.
    """
    if not JOURNAL_PATH.is_file():
        return

    journal_str = str(JOURNAL_PATH)
    if os.path.isdir(PA_DATA_DIR) and journal_str.startswith(PA_DATA_DIR):
        legacy = os.path.join(os.path.expanduser("~"), TENNIS_JOURNAL_NOM)
        try:
            if os.path.abspath(journal_str) != os.path.abspath(legacy):
                shutil.copy2(journal_str, legacy)
        except OSError as e:
            log_tennis(f"Miroir journal static (~/) echoue : {e}", level="warning")
        return

    ftp_user = os.environ.get("PA_FTP_USER", "")
    ftp_pass = os.environ.get("PA_FTP_PASSWORD", "")
    if not ftp_user or not ftp_pass:
        return
    ftp_host = os.environ.get("PA_FTP_HOST", "ftp.pythonanywhere.com")
    remote_dir = os.environ.get("PA_FTP_REMOTE_DIR", "/home/chienblanc/data")
    remote_name = JOURNAL_PATH.name
    try:
        with ftplib.FTP(ftp_host, timeout=30) as ftp:
            ftp.login(ftp_user, ftp_pass)
            ftp.cwd(remote_dir)
            with JOURNAL_PATH.open("rb") as f:
                ftp.storbinary(f"STOR {remote_name}", f)
            try:
                ftp.cwd("/home/chienblanc")
                with JOURNAL_PATH.open("rb") as f:
                    ftp.storbinary(f"STOR {remote_name}", f)
            except Exception:
                pass
        log_tennis(f"Journal uploade → {ftp_host} ({remote_name})")
    except Exception as e:
        log_tennis(f"Upload FTP journal echoue : {e}", level="warning")


def append_journal(c: EdgeCandidate, *, dry_run: bool) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = JOURNAL_PATH.is_file()
    with JOURNAL_PATH.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_COLS)
        if not exists:
            w.writeheader()
        w.writerow(
            {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "EventId": c.event_id,
                "SportKey": c.sport_key,
                "Tourney": c.tourney,
                "Surface": c.surface,
                "PlayerA": c.player_a,
                "PlayerB": c.player_b,
                "Marche": c.marche,
                "Selection": c.selection,
                "Cote": c.cote,
                "FairCote": c.fair_cote,
                "ProbaModele": c.proba_modele,
                "ProbaMarche": c.proba_marche,
                "ProbaFinal": c.proba_final,
                "BlendLam": c.blend_lam,
                "Edge": round(c.edge * 100, 2),
                "Mise": c.mise,
                "DryRun": "OUI" if dry_run else "NON",
                "Statut": "EN ATTENTE",
                # Seed CoteClose = Cote (comme NHL/NBA) ; tracking pré-tip ensuite
                "CommenceTime": c.commence_time,
                "CoteClose": f"{c.cote:.3f}",
                "CLV": "0.00",
                "Resultat": "",
                "PnL": "",
            }
        )
    publier_journal_dashboard()


def send_telegram(c: EdgeCandidate, *, dry_run: bool) -> None:
    if not TENNIS_TELEGRAM_TOKEN or not TENNIS_TELEGRAM_CHAT_ID:
        log_tennis("Telegram non configure — alerte console seule", level="warning")
        return
    prefix = "🧪 *[DRY RUN — PAPER]*\n\n" if dry_run else ""
    msg = (
        prefix
        + f"🎾 *SNIPER TENNIS*\n"
        f"{c.tourney} ({c.surface})\n"
        f"*{c.player_a}* vs *{c.player_b}*\n"
        f"──────────────\n"
        f"Cible : `{c.selection}` ({c.marche})\n"
        f"Cote : *{c.cote}*  |  Fair : *{c.fair_cote}*\n"
        f"Edge : *+{c.edge * 100:.1f}%*  |  Mise paper : *{c.mise} €*\n"
        f"p_model {c.proba_modele:.1%} → p_mkt {c.proba_marche:.1%} "
        f"→ p_final {c.proba_final:.1%} (λ={c.blend_lam:.2f})"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{TENNIS_TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TENNIS_TELEGRAM_CHAT_ID,
                "text": msg,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
    except Exception as e:
        log_tennis(f"Telegram erreur: {e}", level="warning")


def scan_once(state: TennisLiveState, *, edge_min: float | None = None) -> list[EdgeCandidate]:
    emin = TENNIS_EDGE_MIN if edge_min is None else edge_min
    dry = TENNIS_DRY_RUN if isinstance(TENNIS_DRY_RUN, bool) else _env_bool("TENNIS_DRY_RUN", True)
    sports = list_active_atp_sports()
    log_tennis(f"{len(sports)} tournoi(s) ATP actif(s)")
    seen = _journal_keys()
    found: list[EdgeCandidate] = []
    for sport in sports:
        key = sport.get("key") or ""
        events = fetch_odds_for_sport(key)
        log_tennis(f"{sport.get('title', key)}: {len(events)} event(s)")
        for event in events:
            for cand in evaluate_event(state, sport, event, edge_min=emin):
                dedup = (cand.event_id, cand.marche)
                if dedup in seen:
                    continue
                ok_prise, why = _within_prise_window(cand.commence_time)
                if not ok_prise:
                    log_tennis(
                        f"EDGE ignore ({why}): {cand.selection} @ {cand.cote} "
                        f"+{cand.edge * 100:.1f}% | {cand.player_a} vs {cand.player_b}"
                    )
                    continue
                found.append(cand)
                seen.add(dedup)
                log_tennis(
                    f"EDGE {cand.selection} @ {cand.cote} "
                    f"(fair {cand.fair_cote}) +{cand.edge * 100:.1f}% "
                    f"| p {cand.proba_modele:.1%}->{cand.proba_final:.1%} "
                    f"(mkt {cand.proba_marche:.1%}, λ={cand.blend_lam:.2f}) "
                    f"| {cand.player_a} vs {cand.player_b}"
                )
                append_journal(cand, dry_run=dry)
                send_telegram(cand, dry_run=dry)
    if not found:
        log_tennis("Aucun nouvel edge au-dessus du seuil")
    from tennis_params import TENNIS_AUTO_SETTLE, TENNIS_CLV_TRACK

    if TENNIS_CLV_TRACK:
        try:
            from tennis_clv import reparer_clv_aberrants, traquer_et_actualiser_clv

            reparer_clv_aberrants()
            n_clv = traquer_et_actualiser_clv()
            if n_clv:
                log_tennis(f"CLV pre-tip: {n_clv} cote(s) close maj (gele au tip)")
        except Exception as e:
            log_tennis(f"CLV track skip: {e}", level="warning")
    if TENNIS_AUTO_SETTLE:
        try:
            from tennis_clv import settle_results

            n_set = settle_results()
            if n_set:
                log_tennis(f"Reglement auto: {n_set} pari(s) ML → REGLE")
        except Exception as e:
            log_tennis(f"Settle skip: {e}", level="warning")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Sniper tennis ATP (paper)")
    parser.add_argument("--once", action="store_true", help="Un seul scan puis exit")
    parser.add_argument("--rebuild-state", action="store_true", help="Force rebuild Elo/serve")
    parser.add_argument(
        "--settle-clv",
        action="store_true",
        help="MAJ cote close + reglement ML + rapport CLV puis exit",
    )
    parser.add_argument(
        "--edge-min",
        type=float,
        default=None,
        help="Seuil edge (défaut TENNIS_EDGE_MIN)",
    )
    args = parser.parse_args()

    if args.settle_clv:
        from tennis_clv import (
            print_report,
            reparer_clv_aberrants,
            settle_results,
            traquer_et_actualiser_clv,
        )
        from tennis_params import TENNIS_CLV_TRACK

        if not TENNIS_CLV_TRACK:
            log_tennis("TENNIS_CLV_TRACK=false — skip", level="warning")
            return 0
        if not API_ODDS_KEY:
            log_tennis("API_ODDS_KEY manquante pour CLV", level="error")
            return 1
        reparer_clv_aberrants()
        traquer_et_actualiser_clv()  # pre-tip only — jamais de live
        settle_results()
        print_report()
        return 0

    if not API_ODDS_KEY:
        log_tennis(
            f"API_ODDS_KEY manquante — verifier {env_files_hint('tennis')}",
            level="error",
        )
        return 1

    dry = TENNIS_DRY_RUN if isinstance(TENNIS_DRY_RUN, bool) else _env_bool("TENNIS_DRY_RUN", True)
    mode = "DRY RUN (paper)" if dry else "LIVE"
    log_tennis(f"Demarrage sniper tennis — mode {mode}")
    if dry:
        log_tennis("TENNIS_DRY_RUN: Telegram paper + journal (pas de mise reelle)")
    log_tennis(f"Journal: {JOURNAL_PATH}")
    log_tennis(
        f"Fenetre: scan ≤{TENNIS_SCAN_HEURES_AVANCE:.0f}h | "
        f"prise {TENNIS_PRISE_HEURES_MIN:.0f}–{TENNIS_PRISE_HEURES_MAX:.0f}h avant tip "
        f"(stop {TENNIS_MINUTES_AVANT_TIP:.0f}min)"
    )
    if TENNIS_MARKET_BLEND:
        log_tennis(
            f"Market blend ON — λ_ML={TENNIS_MARKET_BLEND_ML:.2f} "
            f"λ_OU={TENNIS_MARKET_BLEND_OU:.2f} (p_final vs cote book)"
        )
    else:
        log_tennis("Market blend OFF — edge sur p_model pur")

    if TENNIS_SACKMANN_AUTO_REFRESH:
        log_tennis(
            f"Sackmann auto-refresh ON — CSV in-season si age ≥ "
            f"{TENNIS_SACKMANN_REFRESH_HOURS:.0f}h ; reload etat chaque "
            f"{TENNIS_STATE_RELOAD_HOURS:.0f}h"
        )
    log_tennis("Chargement etat PIT (Elo + service)…")
    t0 = time.perf_counter()
    state = get_live_state(
        force_rebuild=args.rebuild_state,
        max_age_hours=TENNIS_STATE_CACHE_HOURS,
        persist=True,
        refresh_sackmann=True,
    )
    log_tennis(
        f"Etat pret: {state.n_matches} matchs, {len(state.name_to_id)} noms "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    last_state_load = time.time()

    if args.once:
        scan_once(state, edge_min=args.edge_min)
        return 0

    reload_s = max(300.0, float(TENNIS_STATE_RELOAD_HOURS) * 3600.0)
    while True:
        try:
            if time.time() - last_state_load >= reload_s:
                log_tennis("Reload etat PIT (Sackmann + Elo/service)…")
                t0 = time.perf_counter()
                state = get_live_state(
                    force_rebuild=False,
                    max_age_hours=TENNIS_STATE_CACHE_HOURS,
                    persist=True,
                    refresh_sackmann=True,
                )
                last_state_load = time.time()
                log_tennis(
                    f"Etat recharge: {state.n_matches} matchs "
                    f"({time.perf_counter() - t0:.1f}s)"
                )
            scan_once(state, edge_min=args.edge_min)
        except Exception as e:
            log_tennis(f"Erreur scan: {e}", level="error")
        log_tennis(f"Pause {TENNIS_LOOP_SECONDS:.0f}s…")
        time.sleep(TENNIS_LOOP_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
