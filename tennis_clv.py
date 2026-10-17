"""
Suivi CLV paper tennis — même logique NHL/NBA :

1. À la prise : CoteClose = Cote (seed)
2. Pré-tip uniquement : rafraîchir CoteClose avec Pinnacle courant
3. Dès tip passé : GELER (jamais de cote live)
4. Garde plausibilité : rejeter écart absurde vs prise (blowout live)

Usage :
  python tennis_clv.py --track
  python tennis_clv.py --settle
  python tennis_clv.py --report
  python tennis_clv.py --repair
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

from config_env import load_project_env

load_project_env("tennis")

from tennis_params import TENNIS_JOURNAL_PATH

_ROOT = Path(__file__).resolve().parent
JOURNAL_PATH = Path(TENNIS_JOURNAL_PATH)
API_ODDS_KEY = os.environ.get("API_ODDS_KEY", "")
TENNIS_BOOKMAKER = os.environ.get("TENNIS_BOOKMAKER", "pinnacle")
TENNIS_REGIONS = os.environ.get("TENNIS_REGIONS", "eu")
# Écart max relatif prise→close accepté (au-delà = live blowout ignoré / réparé)
TENNIS_CLV_MAX_DEV_REL = float(os.environ.get("TENNIS_CLV_MAX_DEV_REL", "0.35"))

CLV_COLS = [
    "CommenceTime",
    "CoteClose",
    "CLV",
    "Resultat",
    "PnL",
]


def _log(msg: str, level: str = "info") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"warning": "!", "error": "X"}.get(level, "CLV")
    print(f"[{ts}] {prefix} {msg}", flush=True)


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    if not path.is_file():
        return [], []
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fields = list(r.fieldnames or [])
        rows = list(r)
    for c in CLV_COLS:
        if c not in fields:
            fields.append(c)
    return fields, rows


def _write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    try:
        from tennis_sniper import publier_journal_dashboard

        publier_journal_dashboard()
    except Exception as e:
        _log(f"publish dashboard skip: {e}", level="warning")


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _cote_clv_plausible(
    cote_prise,
    cote_clv,
    max_dev_rel: float | None = None,
) -> bool:
    """Rejette les blowouts live (ex. 1.90 → 4.50) — même garde NBA/NHL."""
    max_dev = TENNIS_CLV_MAX_DEV_REL if max_dev_rel is None else float(max_dev_rel)
    try:
        prise = float(cote_prise)
        clv = float(cote_clv)
    except (TypeError, ValueError):
        return False
    if prise <= 1.0 or clv <= 1.0:
        return False
    return abs(clv - prise) / prise <= max_dev


def _match_deja_tippe(
    commence_time: str | None,
    maintenant: datetime | None = None,
    *,
    date_prise: str | None = None,
) -> bool:
    """
    True si tip déjà passé → figer la CLV (plus de cotes live).
    Sans CommenceTime : geler si la prise a >3 h (évite live tardif, comme NBA).
    """
    now = maintenant or datetime.now(timezone.utc)
    tip = _parse_iso(commence_time)
    if tip is not None:
        return now >= tip
    if date_prise:
        try:
            # Journal : "YYYY-MM-DD HH:MM:SS" local — traite comme naive→UTC approx
            prise_dt = datetime.strptime(date_prise.strip()[:19], "%Y-%m-%d %H:%M:%S")
            prise_dt = prise_dt.replace(tzinfo=timezone.utc)
            if (now - prise_dt).total_seconds() > 3 * 3600:
                return True
        except ValueError:
            pass
    return False


def _clv_pct(cote_prise, cote_close) -> str:
    try:
        p = float(cote_prise)
        c = float(cote_close)
    except (TypeError, ValueError):
        return ""
    if p <= 1.0 or c <= 1.0:
        return ""
    return f"{(p / c - 1.0) * 100:.2f}"


def _fetch_sport_odds(sport_key: str) -> list[dict]:
    if not API_ODDS_KEY or not sport_key:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    try:
        r = requests.get(
            url,
            params={
                "apiKey": API_ODDS_KEY,
                "regions": TENNIS_REGIONS,
                "markets": "h2h,totals",
                "bookmakers": TENNIS_BOOKMAKER,
                "oddsFormat": "decimal",
            },
            timeout=20,
        )
        if r.status_code != 200:
            return []
        return r.json() or []
    except Exception:
        return []


def _extract_cote(event: dict, marche: str, selection: str) -> float | None:
    home = event.get("home_team") or ""
    away = event.get("away_team") or ""
    for book in event.get("bookmakers") or []:
        if book.get("key") != TENNIS_BOOKMAKER:
            continue
        for market in book.get("markets") or []:
            key = market.get("key")
            if key == "h2h" and marche in ("ML_home", "ML_away"):
                target = home if marche == "ML_home" else away
                for o in market.get("outcomes") or []:
                    if (o.get("name") or "").strip() == target:
                        return float(o["price"])
                for o in market.get("outcomes") or []:
                    if (o.get("name") or "").strip() == selection:
                        return float(o["price"])
            elif key == "totals" and marche.startswith("OU_"):
                parts = marche.split("_")
                if len(parts) < 3:
                    continue
                side = parts[1].lower()
                try:
                    line = float(parts[2])
                except ValueError:
                    continue
                for o in market.get("outcomes") or []:
                    point = o.get("point")
                    if point is None:
                        continue
                    if abs(float(point) - line) > 0.05:
                        continue
                    name = str(o.get("name", "")).strip().lower()
                    if name == side:
                        return float(o["price"])
    return None


def reparer_clv_aberrants(path: Path | None = None) -> int:
    """Remet CoteClose = Cote si écart absurde (CLV live post-tip)."""
    dest = path or JOURNAL_PATH
    fields, rows = _read_rows(dest)
    if not rows:
        return 0
    n = 0
    for row in rows:
        prise = row.get("Cote")
        close = row.get("CoteClose")
        if prise and close and not _cote_clv_plausible(prise, close):
            row["CoteClose"] = prise
            row["CLV"] = "0.00"
            n += 1
    if n:
        _write_rows(dest, fields, rows)
        _log(
            f"repare {n} ligne(s) CoteClose aberrante "
            f"(ecart >{TENNIS_CLV_MAX_DEV_REL:.0%} vs prise) → reset a Cote"
        )
    return n


def traquer_et_actualiser_clv(path: Path | None = None) -> int:
    """
    Met à jour CoteClose uniquement en pré-tip (comme NHL/NBA).
    Une fois le tip passé, la closing est figée — pas de cotes live.
    """
    dest = path or JOURNAL_PATH
    fields, rows = _read_rows(dest)
    if not rows:
        _log("journal vide")
        return 0

    # Seed manquant : CoteClose ← Cote (jamais laissé vide en attente)
    seeded = 0
    for row in rows:
        if (row.get("Statut") or "") != "EN ATTENTE":
            continue
        if not (row.get("CoteClose") or "").strip() and row.get("Cote"):
            row["CoteClose"] = row["Cote"]
            row["CLV"] = "0.00"
            seeded += 1

    maintenant = datetime.now(timezone.utc)
    # Grouper les events encore pré-tip
    need: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if (row.get("Statut") or "") != "EN ATTENTE":
            continue
        if _match_deja_tippe(
            row.get("CommenceTime"),
            maintenant,
            date_prise=row.get("Date"),
        ):
            # Tip passé (ou prise trop vieille sans tip) → geler
            continue
        sk = row.get("SportKey") or ""
        if sk:
            need[sk].append(i)

    updated = 0
    odds_by_event: dict[str, dict] = {}
    for sk, idxs in need.items():
        for ev in _fetch_sport_odds(sk):
            eid = ev.get("id")
            if eid:
                odds_by_event[eid] = ev

    for sk, idxs in need.items():
        for i in idxs:
            row = rows[i]
            # Double-check tip (race)
            if _match_deja_tippe(
                row.get("CommenceTime"),
                maintenant,
                date_prise=row.get("Date"),
            ):
                continue
            eid = row.get("EventId") or ""
            event = odds_by_event.get(eid)
            if not event:
                continue
            nv = _extract_cote(
                event, row.get("Marche") or "", row.get("Selection") or ""
            )
            if nv is None:
                continue
            prise = row.get("Cote")
            old = row.get("CoteClose") or prise
            if not _cote_clv_plausible(prise, nv):
                _log(
                    f"CLV ignore (ecart trop fort) {row.get('Selection')} : "
                    f"prise {prise} → {nv}",
                    level="warning",
                )
                continue
            try:
                old_f = float(old) if old else 0.0
            except ValueError:
                old_f = 0.0
            if abs(nv - old_f) < 1e-6:
                continue
            row["CoteClose"] = f"{nv:.3f}"
            row["CLV"] = _clv_pct(prise, nv)
            updated += 1

    if updated or seeded:
        _write_rows(dest, fields, rows)
    _log(
        f"track pre-tip: {updated} maj, {seeded} seed ; "
        f"post-tip geles (pas de live)"
    )
    return updated


# Alias rétrocompat (ancien nom trompeur — ne plus capturer post-tip)
def update_closing_odds(path: Path | None = None) -> int:
    return traquer_et_actualiser_clv(path)


def _fetch_scores(sport_key: str, days_from: int = 3) -> list[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/"
    try:
        r = requests.get(
            url,
            params={"apiKey": API_ODDS_KEY, "daysFrom": days_from},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        return r.json() or []
    except Exception:
        return []


def settle_results(path: Path | None = None) -> int:
    """Règle ML depuis les scores Odds API (completed). O/U jeux non dispo."""
    dest = path or JOURNAL_PATH
    fields, rows = _read_rows(dest)
    if not rows:
        return 0
    pending = [
        r
        for r in rows
        if (r.get("Statut") or "EN ATTENTE") in ("EN ATTENTE", "OUVERT", "")
        and not (r.get("Resultat") or "").strip()
    ]
    sports = {r.get("SportKey") or "" for r in pending if r.get("SportKey")}
    scores_by_event: dict[str, dict] = {}
    for sk in sports:
        for ev in _fetch_scores(sk, days_from=3):
            if ev.get("completed") and ev.get("id"):
                scores_by_event[ev["id"]] = ev

    settled = 0
    for row in rows:
        if (row.get("Resultat") or "").strip():
            continue
        eid = row.get("EventId") or ""
        ev = scores_by_event.get(eid)
        if not ev:
            continue
        scores = ev.get("scores") or []
        if len(scores) < 2:
            continue
        try:
            by_name = {s["name"]: int(float(s["score"])) for s in scores}
        except (KeyError, TypeError, ValueError):
            continue
        home = ev.get("home_team") or ""
        away = ev.get("away_team") or ""
        if home not in by_name or away not in by_name:
            continue
        marche = row.get("Marche") or ""
        cote = float(row.get("Cote") or 0)
        mise = float(row.get("Mise") or 0)
        won: bool | None = None
        if marche == "ML_home":
            won = by_name[home] > by_name[away]
        elif marche == "ML_away":
            won = by_name[away] > by_name[home]
        elif marche.startswith("OU_"):
            continue
        else:
            continue
        if won is None:
            continue
        # Figer CLV % à la clôture connue
        if row.get("CoteClose"):
            row["CLV"] = _clv_pct(row.get("Cote"), row.get("CoteClose"))
        row["Resultat"] = "W" if won else "L"
        row["Statut"] = "REGLE"
        row["PnL"] = f"{mise * (cote - 1.0):.2f}" if won else f"{-mise:.2f}"
        settled += 1

    if settled:
        _write_rows(dest, fields, rows)
    _log(f"resultats regles: {settled} (ML only)")
    return settled


def print_report(path: Path | None = None) -> None:
    dest = path or JOURNAL_PATH
    _, rows = _read_rows(dest)
    if not rows:
        _log("journal vide")
        return
    # Recalcule CLV depuis Cote / CoteClose (source de vérité = prix, pas %)
    with_clv = []
    for r in rows:
        close = r.get("CoteClose") or ""
        prise = r.get("Cote") or ""
        if not close or not prise:
            continue
        try:
            p, c = float(prise), float(close)
        except ValueError:
            continue
        if p <= 1.0 or c <= 1.0:
            continue
        # Exclure seed identique non informatif seulement si voulu — garder tous
        with_clv.append((r, p / c - 1.0))

    print("=== CLV paper tennis ===")
    print(f"lignes journal={len(rows)}  avec close={len(with_clv)}")
    if with_clv:
        vals = [v for _, v in with_clv]
        print(f"CLV moyen={sum(vals) / len(vals) * 100:+.2f}%  n={len(vals)}")
        by_m: dict[str, list[float]] = defaultdict(list)
        for r, v in with_clv:
            m = r.get("Marche") or "?"
            key = "ML" if m.startswith("ML") else ("OU" if m.startswith("OU") else m)
            by_m[key].append(v)
        for k, vs in sorted(by_m.items()):
            print(f"  {k}: n={len(vs)}  CLV moy={sum(vs) / len(vs) * 100:+.2f}%")
    settled = [r for r in rows if (r.get("Resultat") or "").strip() in ("W", "L")]
    if settled:
        pnl = sum(float(r.get("PnL") or 0) for r in settled)
        nw = sum(1 for r in settled if r["Resultat"] == "W")
        print(f"P&L paper regle: {pnl:+.2f} €  ({nw}/{len(settled)} W)")


def main() -> int:
    p = argparse.ArgumentParser(description="CLV paper tennis (pre-tip only)")
    p.add_argument("--track", action="store_true", help="MAJ CoteClose pre-tip seulement")
    p.add_argument("--update-close", action="store_true", help="alias de --track")
    p.add_argument("--settle", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--repair", action="store_true", help="Reset CoteClose aberrantes")
    p.add_argument("--journal", type=Path, default=None)
    args = p.parse_args()
    path = args.journal
    ran = False
    if args.repair:
        reparer_clv_aberrants(path)
        ran = True
    if args.track or args.update_close:
        if not API_ODDS_KEY:
            _log("API_ODDS_KEY manquante", level="error")
            return 1
        traquer_et_actualiser_clv(path)
        ran = True
    if args.settle:
        if not API_ODDS_KEY:
            _log("API_ODDS_KEY manquante", level="error")
            return 1
        settle_results(path)
        ran = True
    if args.report or not ran:
        print_report(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
