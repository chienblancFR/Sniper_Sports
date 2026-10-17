"""
Elo tennis par surface (+ Elo global) + fiabilité (n matchs) + fatigue book.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from tennis_params import (
    SURFACES,
    TENNIS_ELO_BASE,
    TENNIS_ELO_K,
    TENNIS_ELO_N_PRIOR,
    TENNIS_ELO_SURFACE_BLEND,
    TENNIS_FATIGUE,
    TENNIS_FATIGUE_MIN_THRESH,
    TENNIS_FATIGUE_REST_DAYS,
    TENNIS_FATIGUE_SCALE,
)


def expected_score(elo_a: float, elo_b: float, scale: float = 400.0) -> float:
    """P(A bat B) selon Elo classique."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / scale))


def _parse_yyyymmdd(d: int | str | None) -> date | None:
    if d is None:
        return None
    try:
        s = str(int(d))
        if len(s) != 8:
            return None
        return datetime.strptime(s, "%Y%m%d").date()
    except (TypeError, ValueError):
        return None


@dataclass
class SurfaceEloBook:
    """Notes Elo par joueur : global + une note par surface + compteurs."""

    base: float = TENNIS_ELO_BASE
    k: float = TENNIS_ELO_K
    surface_blend: float = TENNIS_ELO_SURFACE_BLEND
    n_prior: float = TENNIS_ELO_N_PRIOR
    global_elo: dict[str, float] = field(default_factory=dict)
    surface_elo: dict[str, dict[str, float]] = field(
        default_factory=lambda: {s: {} for s in SURFACES}
    )
    n_global: dict[str, int] = field(default_factory=dict)
    n_surface: dict[str, dict[str, int]] = field(
        default_factory=lambda: {s: {} for s in SURFACES}
    )

    def _ensure(self, player_id: str, surface: str) -> None:
        if player_id not in self.global_elo:
            self.global_elo[player_id] = self.base
            self.n_global.setdefault(player_id, 0)
        bucket = self.surface_elo.setdefault(surface, {})
        if player_id not in bucket:
            bucket[player_id] = self.base
        self.n_surface.setdefault(surface, {}).setdefault(player_id, 0)

    def get_raw_surface(self, player_id: str, surface: str) -> float:
        self._ensure(player_id, surface)
        return self.surface_elo[surface][player_id]

    def get_global(self, player_id: str) -> float:
        self._ensure(player_id, "Hard")
        return self.global_elo[player_id]

    def get_elo(self, player_id: str, surface: str) -> float:
        """
        Elo effectif pour un match sur `surface` :
        blend * Elo_surface + (1-blend) * Elo_global.
        """
        s = surface if surface in self.surface_elo else "Hard"
        surf = self.get_raw_surface(player_id, s)
        glob = self.get_global(player_id)
        b = max(0.0, min(1.0, self.surface_blend))
        return b * surf + (1.0 - b) * glob

    def n_eff(self, player_id: str, surface: str) -> float:
        s = surface if surface in self.n_surface else "Hard"
        self._ensure(player_id, s)
        b = max(0.0, min(1.0, self.surface_blend))
        ng = float(self.n_global.get(player_id, 0))
        ns = float(self.n_surface.get(s, {}).get(player_id, 0))
        return b * ns + (1.0 - b) * ng

    def reliability(self, player_id: str, surface: str) -> float:
        """rel = n_eff / (n_eff + prior) ∈ (0,1)."""
        n = self.n_eff(player_id, surface)
        prior = max(1e-6, float(self.n_prior))
        return n / (n + prior)

    def matchup_reliability(
        self, player_a: str, player_b: str, surface: str
    ) -> float:
        """Min des deux fiabilités (match limité par le moins connu)."""
        return min(
            self.reliability(player_a, surface),
            self.reliability(player_b, surface),
        )

    def update_after_match(
        self,
        winner_id: str,
        loser_id: str,
        surface: str,
        *,
        k: float | None = None,
    ) -> None:
        """Met à jour Elo global + Elo surface + compteurs après un match."""
        s = surface if surface in SURFACES else "Hard"
        kk = self.k if k is None else float(k)
        self._ensure(winner_id, s)
        self._ensure(loser_id, s)

        for store in (self.global_elo, self.surface_elo[s]):
            ea = store[winner_id]
            eb = store[loser_id]
            exp_w = expected_score(ea, eb)
            store[winner_id] = ea + kk * (1.0 - exp_w)
            store[loser_id] = eb + kk * (0.0 - (1.0 - exp_w))

        self.n_global[winner_id] = self.n_global.get(winner_id, 0) + 1
        self.n_global[loser_id] = self.n_global.get(loser_id, 0) + 1
        self.n_surface[s][winner_id] = self.n_surface[s].get(winner_id, 0) + 1
        self.n_surface[s][loser_id] = self.n_surface[s].get(loser_id, 0) + 1

    def matchup_elos(self, player_a: str, player_b: str, surface: str) -> tuple[float, float]:
        return self.get_elo(player_a, surface), self.get_elo(player_b, surface)


@dataclass
class FatigueBook:
    """Dernier match (date + minutes) par joueur — PIT fatigue SPW."""

    enabled: bool = TENNIS_FATIGUE
    scale: float = TENNIS_FATIGUE_SCALE
    min_thresh: float = TENNIS_FATIGUE_MIN_THRESH
    rest_days: float = TENNIS_FATIGUE_REST_DAYS
    last_date: dict[str, date] = field(default_factory=dict)
    last_minutes: dict[str, float] = field(default_factory=dict)

    def penalty(self, player_id: str, match_date: int | str | date | None) -> float:
        """
        Pénalité à soustraire du SPW (≈ 0..0.05).
        pen = scale * max(0, last_min - thresh) * max(0, rest_need - days) / 60
        """
        if not self.enabled:
            return 0.0
        prev = self.last_date.get(player_id)
        mins = self.last_minutes.get(player_id)
        if prev is None or mins is None:
            return 0.0
        if isinstance(match_date, date):
            cur = match_date
        else:
            cur = _parse_yyyymmdd(match_date)
        if cur is None:
            return 0.0
        days = (cur - prev).days
        if days < 0:
            return 0.0
        excess_min = max(0.0, float(mins) - float(self.min_thresh))
        short_rest = max(0.0, float(self.rest_days) - float(days))
        if excess_min <= 0 or short_rest <= 0:
            return 0.0
        pen = float(self.scale) * excess_min * short_rest / 60.0
        return max(0.0, min(0.08, pen))

    def matchup_penalties(
        self,
        player_a: str,
        player_b: str,
        match_date: int | str | date | None,
    ) -> tuple[float, float]:
        return self.penalty(player_a, match_date), self.penalty(player_b, match_date)

    def update_after_match(
        self,
        winner_id: str,
        loser_id: str,
        match_date: int | str | date | None,
        minutes: int | float | None,
    ) -> None:
        cur = match_date if isinstance(match_date, date) else _parse_yyyymmdd(match_date)
        if cur is None:
            return
        mins: float | None
        try:
            mins = float(minutes) if minutes is not None else None
        except (TypeError, ValueError):
            mins = None
        if mins is not None and mins <= 0:
            mins = None
        for pid in (winner_id, loser_id):
            self.last_date[pid] = cur
            if mins is not None:
                self.last_minutes[pid] = mins


def _self_check() -> None:
    book = SurfaceEloBook(n_prior=15.0)
    assert book.reliability("x", "Hard") < 0.05
    book.update_after_match("a", "b", "Hard")
    for _ in range(20):
        book.update_after_match("a", "b", "Hard")
    assert book.reliability("a", "Hard") > 0.5
    fat = FatigueBook(enabled=True, scale=0.015, min_thresh=120, rest_days=2)
    fat.update_after_match("a", "b", 20240101, 150)
    pen = fat.penalty("a", 20240102)
    assert pen > 0, pen
    assert fat.penalty("a", 20240110) == 0.0
    print("tennis_ratings OK", f"rel_a={book.reliability('a','Hard'):.3f}", f"pen={pen:.4f}")


if __name__ == "__main__":
    _self_check()
