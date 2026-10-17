"""
État live tennis : Elo + stats service PIT + index noms + calib sets/O/U.

Construit depuis les CSV Sackmann, cache disque optionnel pour démarrage rapide.
"""
from __future__ import annotations

import pickle
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from tennis_calib import (
    CalibSample,
    SetsCalibrator,
    SetsSample,
    TbCalibrator,
    TbSample,
    TotalsCalibrator,
    apply_sets_calibrator,
    tb_raw_p_fav_wins,
)
from tennis_data import load_atp_matches, maybe_refresh_sackmann, years_from_env
from tennis_model import project_matchup
from tennis_params import (
    TENNIS_SERVE_SOURCE,
    TENNIS_SETS_CALIB,
    TENNIS_SETS_CALIB_MIN_SAMPLES,
    TENNIS_SETS_COND_SHIFT,
    TENNIS_TB_CALIB,
    TENNIS_TB_CALIB_MIN_SAMPLES,
    TENNIS_TOTALS_CALIB,
    TENNIS_TOTALS_CALIB_BANDS,
    TENNIS_TOTALS_CALIB_MIN_SAMPLES,
    TENNIS_TOTAL_GAMES_LINES,
)
from tennis_ratings import FatigueBook, SurfaceEloBook
from tennis_serve import ServeStatsBook

_ROOT = Path(__file__).resolve().parent
_DEFAULT_CACHE = _ROOT / "data" / "tennis" / "state_cache.pkl"
# v7 : Vague A — break-back + Elo rel + fatigue + CLV hooks
_STATE_VERSION = 7


def normalize_player_name(name: str) -> str:
    """Normalise un nom Odds API / Sackmann pour matching."""
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace(".", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clamp_serve(p: float) -> float:
    return max(0.50, min(0.85, float(p)))


@dataclass
class TennisLiveState:
    elo: SurfaceEloBook = field(default_factory=SurfaceEloBook)
    serve: ServeStatsBook = field(default_factory=ServeStatsBook)
    fatigue: FatigueBook = field(default_factory=FatigueBook)
    name_to_id: dict[str, str] = field(default_factory=dict)
    id_to_name: dict[str, str] = field(default_factory=dict)
    totals_calib: TotalsCalibrator | None = None
    sets_calib: SetsCalibrator | None = None
    tb_calib: TbCalibrator | None = None
    n_matches: int = 0
    built_at: float = 0.0
    version: int = _STATE_VERSION

    def register_name(self, player_id: str, name: str) -> None:
        if not player_id or not name:
            return
        self.id_to_name[player_id] = name
        key = normalize_player_name(name)
        if key:
            self.name_to_id[key] = player_id

    def resolve_player(self, name: str) -> str | None:
        key = normalize_player_name(name)
        if not key:
            return None
        if key in self.name_to_id:
            return self.name_to_id[key]
        parts = key.split()
        if not parts:
            return None
        last = parts[-1]
        cands = {
            pid
            for nn, pid in self.name_to_id.items()
            if nn.split()[-1] == last
        }
        if len(cands) == 1:
            return next(iter(cands))
        cands2 = {
            pid
            for nn, pid in self.name_to_id.items()
            if all(tok in nn.split() for tok in parts)
        }
        if len(cands2) == 1:
            return next(iter(cands2))
        return None

    def project_match(
        self,
        id_a: str,
        id_b: str,
        surface: str,
        *,
        best_of: int = 3,
        match_date: int | str | None = None,
    ):
        """Projection complète : fatigue → moteur → sets-calib → (O/U côté sniper)."""
        from datetime import datetime, timezone

        ea, eb = self.elo.matchup_elos(id_a, id_b, surface)
        rel = self.elo.matchup_reliability(id_a, id_b, surface)
        p_sa = p_sb = None
        if TENNIS_SERVE_SOURCE == "stats":
            p_sa, p_sb = self.serve.matchup_serve_probs_engine(
                id_a, id_b, surface, ea, eb
            )
        md = match_date
        if md is None:
            md = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
        pen_a, pen_b = self.fatigue.matchup_penalties(id_a, id_b, md)
        if p_sa is not None and p_sb is not None:
            p_sa = _clamp_serve(p_sa - pen_a)
            p_sb = _clamp_serve(p_sb - pen_b)
        pred = project_matchup(
            ea,
            eb,
            surface,
            best_of=best_of,
            p_serve_a=p_sa,
            p_serve_b=p_sb,
            tb_calibrator=self.tb_calib,
            reliability=rel,
        )
        p_fav = max(pred.p_a_win, pred.p_b_win)
        return apply_sets_calibrator(pred, self.sets_calib, p_fav), p_fav


def build_live_state(years: list[int] | None = None) -> TennisLiveState:
    """Rejoue l'historique (PIT) : Elo + serve + fit calib TB/sets/O/U."""
    matches = load_atp_matches(years or years_from_env(), download_missing=True)
    state = TennisLiveState()
    method_ou = TENNIS_TOTALS_CALIB.strip().lower()
    method_sets = TENNIS_SETS_CALIB.strip().lower()
    method_tb = TENNIS_TB_CALIB.strip().lower()
    collect = method_ou not in ("none", "off", "0", "false") or method_sets not in (
        "none",
        "off",
        "0",
        "false",
    )
    need_tb = method_tb not in ("none", "off", "0", "false")
    snaps: list[tuple] = []
    tb_samples: list[TbSample] = []
    lines = TENNIS_TOTAL_GAMES_LINES

    for m in matches:
        if m.score.walkover:
            continue
        state.register_name(m.winner_id, m.winner_name)
        state.register_name(m.loser_id, m.loser_name)

        if need_tb and not m.score.retired and m.score.tiebreak_winner_won:
            ea, eb = state.elo.matchup_elos(m.winner_id, m.loser_id, m.surface)
            if TENNIS_SERVE_SOURCE == "stats":
                pa, pb = state.serve.matchup_serve_probs_engine(
                    m.winner_id, m.loser_id, m.surface, ea, eb
                )
            else:
                from tennis_model import elo_to_serve_probs
                from tennis_params import (
                    TENNIS_ELO_POINT_SCALE,
                    TENNIS_POINT_SENSITIVITY,
                    mu_serve_for_surface,
                )

                pa, pb = elo_to_serve_probs(
                    ea,
                    eb,
                    mu_serve=mu_serve_for_surface(m.surface),
                    elo_point_scale=TENNIS_ELO_POINT_SCALE,
                    point_sensitivity=TENNIS_POINT_SENSITIVITY,
                )
            fav_is_w = ea >= eb
            p_raw = tb_raw_p_fav_wins(
                pa, pb, surface=m.surface, fav_is_winner=fav_is_w
            )
            for w_won_tb in m.score.tiebreak_winner_won:
                y = 1.0 if (bool(w_won_tb) == fav_is_w) else 0.0
                tb_samples.append(TbSample(p_raw=float(p_raw), y=y))

        if (
            collect
            and m.best_of == 3
            and m.score.usable_for_totals
            and m.score.total_games is not None
        ):
            ea, eb = state.elo.matchup_elos(m.winner_id, m.loser_id, m.surface)
            rel = state.elo.matchup_reliability(
                m.winner_id, m.loser_id, m.surface
            )
            p_sa = p_sb = None
            if TENNIS_SERVE_SOURCE == "stats":
                p_sa, p_sb = state.serve.matchup_serve_probs_engine(
                    m.winner_id, m.loser_id, m.surface, ea, eb
                )
            pen_a, pen_b = state.fatigue.matchup_penalties(
                m.winner_id, m.loser_id, m.tourney_date
            )
            if p_sa is not None and p_sb is not None:
                p_sa = _clamp_serve(p_sa - pen_a)
                p_sb = _clamp_serve(p_sb - pen_b)
            pred = project_matchup(
                ea,
                eb,
                m.surface,
                best_of=m.best_of,
                p_serve_a=p_sa,
                p_serve_b=p_sb,
                reliability=rel,
            )
            p_fav = max(pred.p_a_win, pred.p_b_win)
            snaps.append((pred, int(m.score.total_games), len(m.score.sets), p_fav))

        state.elo.update_after_match(m.winner_id, m.loser_id, m.surface)
        if m.serve is not None and m.serve.valid:
            state.serve.update_after_match(m)
        if not m.score.retired and not m.score.walkover:
            state.fatigue.update_after_match(
                m.winner_id, m.loser_id, m.tourney_date, m.minutes
            )
        state.n_matches += 1

    if need_tb and tb_samples:
        state.tb_calib = TbCalibrator.fit(
            tb_samples,
            method="isotonic" if method_tb == "isotonic" else "platt",
            min_samples=TENNIS_TB_CALIB_MIN_SAMPLES,
        )

    if snaps and method_sets not in ("none", "off", "0", "false"):
        sets_rows = [
            SetsSample(
                p3_raw=float(pred.p_sets_3),
                y_three=1.0 if n_sets == 3 else 0.0,
                p_fav=p_fav,
                games=actual,
                n_sets=n_sets,
                exp_games_2=sum(g * p for g, p in (pred.games_pmf_2 or {}).items()),
                exp_games_3=sum(g * p for g, p in (pred.games_pmf_3 or {}).items()),
            )
            for pred, actual, n_sets, p_fav in snaps
        ]
        state.sets_calib = SetsCalibrator.fit(
            sets_rows,
            method=method_sets if method_sets in ("platt", "band_offset") else "isotonic",
            cond_shift=TENNIS_SETS_COND_SHIFT,
            min_samples=TENNIS_SETS_CALIB_MIN_SAMPLES,
        )

    if snaps and method_ou not in ("none", "off", "0", "false"):
        ou_rows: list[CalibSample] = []
        for pred, actual, _ns, p_fav in snaps:
            pred2 = apply_sets_calibrator(pred, state.sets_calib, p_fav)
            for line in lines:
                if abs(line - round(line)) < 1e-9 and actual == int(round(line)):
                    continue
                push = pred2.p_push(line)
                live = 1.0 - push
                p_over = pred2.p_over(line)
                p_cond = p_over / live if live > 1e-9 else p_over
                ou_rows.append(
                    CalibSample(
                        line=float(line),
                        p_raw=float(p_cond),
                        y_over=1.0 if actual > line else 0.0,
                        p_fav=float(p_fav),
                    )
                )
        state.totals_calib = TotalsCalibrator.fit(
            ou_rows,
            method="platt" if method_ou == "platt" else "isotonic",
            use_bands=TENNIS_TOTALS_CALIB_BANDS,
            min_samples=TENNIS_TOTALS_CALIB_MIN_SAMPLES,
        )

    state.built_at = time.time()
    return state


def save_state(state: TennisLiveState, path: Path | None = None) -> Path:
    dest = path or _DEFAULT_CACHE
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dest


def load_state_cache(
    path: Path | None = None,
    *,
    max_age_hours: float = 24.0,
) -> TennisLiveState | None:
    dest = path or _DEFAULT_CACHE
    if not dest.is_file():
        return None
    age_h = (time.time() - dest.stat().st_mtime) / 3600.0
    if age_h > max_age_hours:
        return None
    try:
        with dest.open("rb") as f:
            state = pickle.load(f)
        if isinstance(state, TennisLiveState) and getattr(state, "version", 0) >= _STATE_VERSION:
            return state
    except Exception:
        return None
    return None


def get_live_state(
    *,
    force_rebuild: bool = False,
    max_age_hours: float = 24.0,
    persist: bool = True,
    refresh_sackmann: bool = True,
) -> TennisLiveState:
    """
    Charge l'état PIT (cache disque si frais).

    Si refresh_sackmann : re-télécharge les CSV année courante périmés,
    puis force un rebuild Elo/service pour intégrer les nouveaux matchs.
    """
    refreshed: list[int] = []
    if refresh_sackmann:
        try:
            # force_rebuild CLI → re-télécharge l'in-season même s'il est frais
            refreshed = maybe_refresh_sackmann(force=force_rebuild)
            if refreshed:
                print(
                    f"[tennis_state] Sackmann refresh: {refreshed}",
                    flush=True,
                )
        except Exception as e:
            print(f"[tennis_state] Sackmann refresh skip: {e}", flush=True)
            refreshed = []
    if refreshed:
        force_rebuild = True
    if not force_rebuild:
        cached = load_state_cache(max_age_hours=max_age_hours)
        if cached is not None:
            return cached
    state = build_live_state()
    if persist:
        save_state(state)
    return state


def infer_surface(sport_key: str, title: str = "") -> str:
    """Heuristique surface depuis la clé / titre Odds API."""
    blob = f"{sport_key} {title}".lower()
    if any(x in blob for x in ("clay", "roland", "rome", "madrid", "monte.carlo", "bastad")):
        return "Clay"
    if any(
        x in blob
        for x in ("grass", "wimbledon", "halle", "queens", "s-hertogenbosch", "eastbourne")
    ):
        return "Grass"
    return "Hard"


def _self_check() -> None:
    from config_env import load_project_env

    load_project_env("tennis")
    assert normalize_player_name("Alexander Zverev") == "alexander zverev"
    assert infer_surface("tennis_atp_roland_garros", "Roland Garros") == "Clay"
    print("tennis_state OK")


if __name__ == "__main__":
    _self_check()
