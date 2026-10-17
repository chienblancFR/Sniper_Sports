"""
Chargement / téléchargement des matchs ATP (format Sackmann).

Le dépôt upstream JeffSackmann/tennis_atp est 404 (2026). Par défaut on tire
depuis le miroir Aneeshers/tennis-sackmann-archive (CC BY-NC-SA 4.0 — Jeff Sackmann).
"""
from __future__ import annotations

import csv
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tennis_params import (
    TENNIS_SACKMANN_AUTO_REFRESH,
    TENNIS_SACKMANN_REFRESH_HOURS,
    normalize_surface,
)

_ROOT = Path(__file__).resolve().parent
_DEFAULT_DIR = _ROOT / "data" / "tennis" / "sackmann"
_DEFAULT_SOURCE = (
    "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp"
)

_SET_RE = re.compile(r"(\d+)-(\d+)(?:\(\d+\))?")
_UA = "paris-sportif-dashboard-tennis/0.1 (research; +local)"


@dataclass(frozen=True)
class ParsedScore:
    raw: str
    sets: tuple[tuple[int, int], ...]
    total_games: int | None
    complete: bool
    retired: bool
    walkover: bool

    @property
    def usable_for_totals(self) -> bool:
        return self.complete and self.total_games is not None and not self.retired and not self.walkover

    @property
    def tiebreak_winner_won(self) -> tuple[bool, ...]:
        """Pour chaque set 7-6 / 6-7 : True si le vainqueur du match a gagné le TB."""
        out: list[bool] = []
        for a, b in self.sets:
            if (a, b) == (7, 6):
                out.append(True)
            elif (a, b) == (6, 7):
                out.append(False)
        return tuple(out)


@dataclass(frozen=True)
class MatchServeStats:
    """Points service Sackmann (winner / loser), avec split 1er / 2e si dispo."""

    w_svpt: int
    w_serve_won: int  # w_1stWon + w_2ndWon
    l_svpt: int
    l_serve_won: int
    w_1st_in: int | None = None
    w_1st_won: int | None = None
    w_2nd_won: int | None = None
    l_1st_in: int | None = None
    l_1st_won: int | None = None
    l_2nd_won: int | None = None

    @property
    def valid(self) -> bool:
        return self.w_svpt > 0 and self.l_svpt > 0 and self.w_serve_won >= 0 and self.l_serve_won >= 0

    @property
    def has_split(self) -> bool:
        """True si 1stIn / 1stWon / 2ndWon cohérents des deux côtés."""
        vals = (
            self.w_1st_in,
            self.w_1st_won,
            self.w_2nd_won,
            self.l_1st_in,
            self.l_1st_won,
            self.l_2nd_won,
        )
        if any(v is None for v in vals):
            return False
        w_in, w1, w2, l_in, l1, l2 = vals  # type: ignore[misc]
        if w_in < 0 or l_in < 0 or w_in > self.w_svpt or l_in > self.l_svpt:
            return False
        if w1 < 0 or w2 < 0 or l1 < 0 or l2 < 0:
            return False
        if w1 > w_in or l1 > l_in:
            return False
        if w2 > (self.w_svpt - w_in) or l2 > (self.l_svpt - l_in):
            return False
        return True

    @property
    def w_2nd_att(self) -> int:
        if self.w_1st_in is None:
            return 0
        return max(0, self.w_svpt - self.w_1st_in)

    @property
    def l_2nd_att(self) -> int:
        if self.l_1st_in is None:
            return 0
        return max(0, self.l_svpt - self.l_1st_in)


@dataclass(frozen=True)
class AtpMatch:
    tourney_id: str
    tourney_name: str
    tourney_date: int  # YYYYMMDD
    match_num: int
    surface: str
    best_of: int
    round: str
    winner_id: str
    loser_id: str
    winner_name: str
    loser_name: str
    score: ParsedScore
    year: int
    serve: MatchServeStats | None = None
    minutes: int | None = None

    @property
    def sort_key(self) -> tuple[int, str, int]:
        return (self.tourney_date, self.tourney_id, self.match_num)


def _int_field(row: dict, *keys: str) -> int | None:
    for key in keys:
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return None
    return None


def parse_serve_stats(row: dict) -> MatchServeStats | None:
    """Extrait SPW winner/loser (+ split 1er/2e si colonnes présentes)."""
    w_svpt = _int_field(row, "w_svpt")
    l_svpt = _int_field(row, "l_svpt")
    w1 = _int_field(row, "w_1stWon")
    w2 = _int_field(row, "w_2ndWon")
    l1 = _int_field(row, "l_1stWon")
    l2 = _int_field(row, "l_2ndWon")
    if None in (w_svpt, l_svpt, w1, w2, l1, l2):
        return None
    w_in = _int_field(row, "w_1stIn")
    l_in = _int_field(row, "l_1stIn")
    stats = MatchServeStats(
        w_svpt=w_svpt,
        w_serve_won=w1 + w2,
        l_svpt=l_svpt,
        l_serve_won=l1 + l2,
        w_1st_in=w_in,
        w_1st_won=w1,
        w_2nd_won=w2,
        l_1st_in=l_in,
        l_1st_won=l1,
        l_2nd_won=l2,
    )
    return stats if stats.valid else None


def parse_score(raw: str | None) -> ParsedScore:
    text = (raw or "").strip()
    if not text:
        return ParsedScore("", (), None, False, False, False)
    upper = text.upper().replace(".", "")
    walkover = upper in ("W/O", "WO", "DEF", "WALKOVER") or upper.startswith("W/O")
    retired = "RET" in upper
    if walkover:
        return ParsedScore(text, (), None, False, False, True)

    sets: list[tuple[int, int]] = []
    total = 0
    for m in _SET_RE.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        sets.append((a, b))
        total += a + b
    if not sets:
        return ParsedScore(text, (), None, False, retired, False)

    complete = (not retired) and len(sets) >= 2
    return ParsedScore(
        raw=text,
        sets=tuple(sets),
        total_games=total if complete else None,
        complete=complete,
        retired=retired,
        walkover=False,
    )


def sackmann_dir() -> Path:
    raw = os.environ.get("TENNIS_SACKMANN_DIR", str(_DEFAULT_DIR))
    p = Path(raw)
    if not p.is_absolute():
        p = _ROOT / p
    return p


def sackmann_source() -> str:
    return os.environ.get("TENNIS_SACKMANN_SOURCE", _DEFAULT_SOURCE).rstrip("/")


def download_atp_years(years: list[int], *, force: bool = False) -> list[Path]:
    """Télécharge atp_matches_YYYY.csv dans TENNIS_SACKMANN_DIR."""
    dest_dir = sackmann_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = sackmann_source()
    out: list[Path] = []
    for y in years:
        name = f"atp_matches_{y}.csv"
        path = dest_dir / name
        out.append(path)
        if path.is_file() and path.stat().st_size > 1000 and not force:
            continue
        url = f"{base}/{name}"
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    return out


def parse_year_range(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def years_from_env(default: str = "2018-2026") -> list[int]:
    """Années env + année civile courante (toujours incluse)."""
    years = parse_year_range(os.environ.get("TENNIS_SACKMANN_YEARS", default))
    cy = datetime.now().year
    if cy not in years:
        years = sorted(set(years) | {cy})
    return years


def in_season_years(*, now: datetime | None = None) -> list[int]:
    """Année courante (+ précédente en janvier — tournois à cheval)."""
    now = now or datetime.now()
    years = [now.year]
    if now.month <= 1:
        years.append(now.year - 1)
    return sorted(set(years))


def csv_age_hours(year: int) -> float | None:
    """Âge du CSV local en heures, ou None si absent / trop petit."""
    path = sackmann_dir() / f"atp_matches_{year}.csv"
    if not path.is_file() or path.stat().st_size <= 1000:
        return None
    return (time.time() - path.stat().st_mtime) / 3600.0


def maybe_refresh_sackmann(*, force: bool = False) -> list[int]:
    """
    Re-télécharge les CSV in-season périmés (qualité Elo / service).

    Returns:
        Années effectivement re-téléchargées.
    """
    if not force and not TENNIS_SACKMANN_AUTO_REFRESH:
        return []
    max_age = float(TENNIS_SACKMANN_REFRESH_HOURS)
    # Uniquement l'in-season (années historiques = stables, téléchargées une fois)
    to_force: list[int] = []
    for y in in_season_years():
        age = csv_age_hours(y)
        if force or age is None or age >= max_age:
            to_force.append(y)
    if not to_force:
        return []
    # Garantit aussi les années historiques manquantes (sans re-download)
    download_atp_years(years_from_env(), force=False)
    download_atp_years(to_force, force=True)
    return to_force


def load_atp_matches(
    years: list[int] | None = None,
    *,
    download_missing: bool = True,
) -> list[AtpMatch]:
    """Charge et trie chronologiquement les matchs main-draw ATP."""
    ys = years if years is not None else years_from_env()
    if download_missing:
        download_atp_years(ys)

    matches: list[AtpMatch] = []
    dest = sackmann_dir()
    for y in ys:
        path = dest / f"atp_matches_{y}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"CSV manquant: {path}")
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    best_of = int(row.get("best_of") or 3)
                except ValueError:
                    best_of = 3
                try:
                    tourney_date = int(row.get("tourney_date") or 0)
                except ValueError:
                    continue
                try:
                    match_num = int(row.get("match_num") or 0)
                except ValueError:
                    match_num = 0
                wid = (row.get("winner_id") or "").strip()
                lid = (row.get("loser_id") or "").strip()
                if not wid or not lid:
                    continue
                matches.append(
                    AtpMatch(
                        tourney_id=(row.get("tourney_id") or "").strip(),
                        tourney_name=(row.get("tourney_name") or "").strip(),
                        tourney_date=tourney_date,
                        match_num=match_num,
                        surface=normalize_surface(row.get("surface")),
                        best_of=best_of,
                        round=(row.get("round") or "").strip(),
                        winner_id=wid,
                        loser_id=lid,
                        winner_name=(row.get("winner_name") or "").strip(),
                        loser_name=(row.get("loser_name") or "").strip(),
                        score=parse_score(row.get("score")),
                        year=y,
                        serve=parse_serve_stats(row),
                        minutes=_int_field(row, "minutes"),
                    )
                )
    matches.sort(key=lambda m: m.sort_key)
    return matches


def _self_check() -> None:
    s = parse_score("7-6(5) 6-4")
    assert s.complete and s.total_games == 23, s
    s2 = parse_score("6-4 3-6 6-2")
    assert s2.total_games == 27
    s3 = parse_score("6-4 2-1 RET")
    assert s3.retired and not s3.usable_for_totals
    s4 = parse_score("W/O")
    assert s4.walkover
    print("tennis_data OK", s.total_games, s2.total_games)


if __name__ == "__main__":
    _self_check()
