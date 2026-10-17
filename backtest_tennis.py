"""
Calibration tennis (ATP) — Elo surface PIT + moneyline / total jeux.

Usage :
  python backtest_tennis.py
  python backtest_tennis.py --serve-source stats
  python backtest_tennis.py --totals-calib none
  python backtest_tennis.py --totals-calib isotonic

Données : miroir Sackmann (voir tennis_data.py / TENNIS_SACKMANN_*).
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field

from config_env import load_project_env

load_project_env("tennis")

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
from tennis_data import AtpMatch, download_atp_years, load_atp_matches
from tennis_model import MatchProjection, project_matchup
from tennis_params import (
    TENNIS_BREAK_BACK,
    TENNIS_ELO_N_PRIOR,
    TENNIS_FATIGUE,
    TENNIS_PMF_SIGMA,
    TENNIS_SERVE_SOURCE,
    TENNIS_SETS_CALIB,
    TENNIS_SETS_CALIB_MIN_SAMPLES,
    TENNIS_SETS_COND_SHIFT,
    TENNIS_TB_CALIB,
    TENNIS_TB_CALIB_MIN_SAMPLES,
    TENNIS_TB_DEDICATED,
    TENNIS_TB_SERVE_SCALE,
    TENNIS_TOTALS_CALIB,
    TENNIS_TOTALS_CALIB_BANDS,
    TENNIS_TOTALS_CALIB_MIN_SAMPLES,
    TENNIS_TOTAL_GAMES_LINES,
)
from tennis_ratings import FatigueBook, SurfaceEloBook
from tennis_serve import ServeStatsBook


def _clamp_serve(p: float) -> float:
    return max(0.50, min(0.85, float(p)))


@dataclass
class _MlBucket:
    n: int = 0
    brier: float = 0.0
    logloss: float = 0.0
    correct: int = 0

    def add(self, p_winner: float, model_picked_winner: bool) -> None:
        p = min(max(p_winner, 1e-6), 1.0 - 1e-6)
        self.n += 1
        self.brier += (p - 1.0) ** 2
        self.logloss += -math.log(p)
        if model_picked_winner:
            self.correct += 1


@dataclass
class _Running:
    ml: _MlBucket = field(default_factory=_MlBucket)
    ml_engine: _MlBucket = field(default_factory=_MlBucket)
    n_games: int = 0
    abs_err: float = 0.0
    bias: float = 0.0
    ou: dict[float, dict[str, float]] = field(default_factory=dict)
    ou_raw: dict[float, dict[str, float]] = field(default_factory=dict)
    # diag sets
    n_sets: int = 0
    p3_real: float = 0.0
    p3_mod: float = 0.0

    def add_games(
        self,
        pred: MatchProjection,
        actual_games: int,
        lines: tuple[float, ...],
        *,
        p_fav: float,
        calib: TotalsCalibrator | None = None,
    ) -> None:
        self.n_games += 1
        self.abs_err += abs(pred.expected_games - actual_games)
        self.bias += pred.expected_games - actual_games
        for line in lines:
            y = 1.0 if actual_games > line else 0.0
            if abs(line - round(line)) < 1e-9 and actual_games == int(round(line)):
                continue

            push = pred.p_push(line)
            live = 1.0 - push
            p_over_raw = pred.p_over(line)
            p_under_raw = pred.p_under(line)
            if live > 1e-9 and abs(line - round(line)) < 1e-9:
                p_cond = p_over_raw / live
            else:
                p_cond = p_over_raw

            # Raw
            raw_bucket = self.ou_raw.setdefault(
                line, {"n": 0.0, "brier": 0.0, "over_rate": 0.0, "p_over_sum": 0.0}
            )
            raw_bucket["n"] += 1
            raw_bucket["brier"] += (p_cond - y) ** 2
            raw_bucket["over_rate"] += y
            raw_bucket["p_over_sum"] += p_cond

            # Calibré
            if calib is not None and calib.leaves:
                p_o, _p_u = calib.calibrate_under(
                    line, p_over_raw, p_under_raw, p_fav, p_push=push
                )
                p_use = p_o / live if live > 1e-9 else p_o
            else:
                p_use = p_cond

            bucket = self.ou.setdefault(
                line, {"n": 0.0, "brier": 0.0, "over_rate": 0.0, "p_over_sum": 0.0}
            )
            bucket["n"] += 1
            bucket["brier"] += (p_use - y) ** 2
            bucket["over_rate"] += y
            bucket["p_over_sum"] += p_use


def _warmup_until_date(year: int) -> int:
    return int(f"{year}1231")


def _proj_cache_key(
    elo_a: float,
    elo_b: float,
    surface: str,
    best_of: int,
    p_serve_a: float | None,
    p_serve_b: float | None,
) -> tuple:
    if p_serve_a is None or p_serve_b is None:
        return ("elo", round(elo_a / 5.0) * 5, round(elo_b / 5.0) * 5, surface, best_of)
    return ("sv", round(p_serve_a, 3), round(p_serve_b, 3), surface, best_of)


def _project(
    book: SurfaceEloBook,
    serve_book: ServeStatsBook,
    m: AtpMatch,
    src_serve: str,
    cache: dict,
    meta: dict,
    *,
    tb_cal: TbCalibrator | None = None,
    fatigue: FatigueBook | None = None,
) -> tuple[MatchProjection, float, float]:
    ea, eb = book.matchup_elos(m.winner_id, m.loser_id, m.surface)
    rel = book.matchup_reliability(m.winner_id, m.loser_id, m.surface)
    p_sa = p_sb = None
    if src_serve == "stats":
        p_sa, p_sb = serve_book.matchup_serve_probs_engine(
            m.winner_id, m.loser_id, m.surface, ea, eb
        )
    if fatigue is not None and p_sa is not None and p_sb is not None:
        pen_a, pen_b = fatigue.matchup_penalties(
            m.winner_id, m.loser_id, m.tourney_date
        )
        p_sa = _clamp_serve(p_sa - pen_a)
        p_sb = _clamp_serve(p_sb - pen_b)
    key = _proj_cache_key(ea, eb, m.surface, m.best_of, p_sa, p_sb) + (
        round(rel, 3),
    )
    if key in cache:
        pred = cache[key]
        meta["cache_hits"] += 1
    else:
        pred = project_matchup(
            ea,
            eb,
            m.surface,
            best_of=m.best_of,
            p_serve_a=p_sa,
            p_serve_b=p_sb,
            tb_calibrator=tb_cal,
            reliability=rel,
        )
        cache[key] = pred
    return pred, ea, eb


def _collect_tb_samples(
    book: SurfaceEloBook,
    serve_book: ServeStatsBook,
    m: AtpMatch,
    src_serve: str,
    out: list[TbSample],
) -> None:
    """Échantillons PIT : P(favori gagne TB) vs réalité (sets 7-6 / 6-7)."""
    flags = m.score.tiebreak_winner_won
    if not flags:
        return
    ea, eb = book.matchup_elos(m.winner_id, m.loser_id, m.surface)
    if src_serve == "stats":
        pa, pb = serve_book.matchup_serve_probs_engine(
            m.winner_id, m.loser_id, m.surface, ea, eb
        )
    else:
        from tennis_model import elo_to_serve_probs
        from tennis_params import TENNIS_ELO_POINT_SCALE, TENNIS_POINT_SENSITIVITY, mu_serve_for_surface

        pa, pb = elo_to_serve_probs(
            ea,
            eb,
            mu_serve=mu_serve_for_surface(m.surface),
            elo_point_scale=TENNIS_ELO_POINT_SCALE,
            point_sensitivity=TENNIS_POINT_SENSITIVITY,
        )
    fav_is_w = ea >= eb
    p_raw = tb_raw_p_fav_wins(pa, pb, surface=m.surface, fav_is_winner=fav_is_w)
    for w_won_tb in flags:
        y = 1.0 if (bool(w_won_tb) == fav_is_w) else 0.0
        out.append(TbSample(p_raw=float(p_raw), y=y))


def _append_calib_samples(
    samples: list[CalibSample],
    pred: MatchProjection,
    actual_games: int,
    lines: tuple[float, ...],
    p_fav: float,
) -> None:
    for line in lines:
        if abs(line - round(line)) < 1e-9 and actual_games == int(round(line)):
            continue
        y = 1.0 if actual_games > line else 0.0
        push = pred.p_push(line)
        live = 1.0 - push
        p_over = pred.p_over(line)
        p_cond = p_over / live if live > 1e-9 else p_over
        samples.append(
            CalibSample(line=float(line), p_raw=float(p_cond), y_over=y, p_fav=float(p_fav))
        )


def _exp_cond(pmf: dict[int, float]) -> float:
    if not pmf:
        return 0.0
    return sum(g * p for g, p in pmf.items())


def run_calibration(
    matches: list[AtpMatch],
    *,
    warmup_until_year: int,
    best_of_only: int | None,
    lines: tuple[float, ...],
    serve_source: str | None = None,
    eval_surfaces: set[str] | None = None,
    totals_calib: str | None = None,
    totals_calib_bands: bool | None = None,
    sets_calib: str | None = None,
    sets_cond_shift: bool | None = None,
) -> tuple[_Running, dict]:
    book = SurfaceEloBook()
    serve_book = ServeStatsBook()
    fatigue = FatigueBook()
    stats = _Running()
    cache: dict[tuple, MatchProjection] = {}
    src_serve = (serve_source or TENNIS_SERVE_SOURCE).strip().lower()
    method_ou = (totals_calib or TENNIS_TOTALS_CALIB).strip().lower()
    method_sets = (sets_calib or TENNIS_SETS_CALIB).strip().lower()
    use_bands = (
        TENNIS_TOTALS_CALIB_BANDS if totals_calib_bands is None else totals_calib_bands
    )
    cond_shift = (
        TENNIS_SETS_COND_SHIFT if sets_cond_shift is None else sets_cond_shift
    )
    method_tb = TENNIS_TB_CALIB.strip().lower()
    meta = {
        "n_total": len(matches),
        "n_eval_ml": 0,
        "n_eval_games": 0,
        "n_skip_wo": 0,
        "n_serve_updates": 0,
        "cache_hits": 0,
        "warmup_until_year": warmup_until_year,
        "serve_source": src_serve,
        "totals_calib": method_ou,
        "totals_calib_bands": use_bands,
        "totals_calib_leaves": 0,
        "totals_calib_train": 0,
        "sets_calib": method_sets,
        "sets_cond_shift": cond_shift,
        "sets_calib_train": 0,
        "tb_dedicated": TENNIS_TB_DEDICATED,
        "tb_serve_scale": TENNIS_TB_SERVE_SCALE,
        "tb_calib": method_tb,
        "tb_calib_train": 0,
    }
    meta["break_back"] = TENNIS_BREAK_BACK
    meta["elo_n_prior"] = TENNIS_ELO_N_PRIOR
    meta["pmf_sigma"] = TENNIS_PMF_SIGMA
    meta["fatigue"] = TENNIS_FATIGUE
    cutoff = _warmup_until_date(warmup_until_year)
    # Snapshots warm-up : (pred_raw, actual_games, n_sets, p_fav)
    warmup_snap: list[tuple[MatchProjection, int, int, float]] = []
    tb_samples: list[TbSample] = []
    ou_cal: TotalsCalibrator | None = None
    sets_cal: SetsCalibrator | None = None
    tb_cal: TbCalibrator | None = None
    fitted = False

    need_warmup_proj = method_ou not in ("none", "off", "0", "false") or method_sets not in (
        "none",
        "off",
        "0",
        "false",
    )
    need_tb = method_tb not in ("none", "off", "0", "false")

    def _update_books(m: AtpMatch) -> None:
        book.update_after_match(m.winner_id, m.loser_id, m.surface)
        if m.serve is not None and m.serve.valid:
            serve_book.update_after_match(m)
            meta["n_serve_updates"] += 1
        if not m.score.retired and not m.score.walkover:
            fatigue.update_after_match(
                m.winner_id, m.loser_id, m.tourney_date, m.minutes
            )

    def _fit_calibrators() -> None:
        nonlocal ou_cal, sets_cal, tb_cal, fitted
        if fitted:
            return
        fitted = True
        if need_tb and tb_samples:
            tb_cal = TbCalibrator.fit(
                tb_samples,
                method="isotonic" if method_tb == "isotonic" else "platt",
                min_samples=TENNIS_TB_CALIB_MIN_SAMPLES,
            )
            meta["tb_calib_train"] = tb_cal.n_train
            meta["tb_calib_ready"] = tb_cal.leaf is not None
            cache.clear()
        if method_sets not in ("none", "off", "0", "false") and warmup_snap:
            sets_rows = [
                SetsSample(
                    p3_raw=float(pred.p_sets_3),
                    y_three=1.0 if n_sets == 3 else 0.0,
                    p_fav=p_fav,
                    games=actual,
                    n_sets=n_sets,
                    exp_games_2=_exp_cond(pred.games_pmf_2),
                    exp_games_3=_exp_cond(pred.games_pmf_3),
                )
                for pred, actual, n_sets, p_fav in warmup_snap
            ]
            sets_cal = SetsCalibrator.fit(
                sets_rows,
                method=method_sets if method_sets in ("platt", "band_offset") else "isotonic",
                cond_shift=cond_shift,
                min_samples=TENNIS_SETS_CALIB_MIN_SAMPLES,
            )
            meta["sets_calib_train"] = sets_cal.n_train
            meta["sets_band_offset"] = dict(sets_cal.band_offset)
            meta["sets_shift2"] = dict(sets_cal.shift2)
            meta["sets_shift3"] = dict(sets_cal.shift3)

        if method_ou not in ("none", "off", "0", "false") and warmup_snap:
            ou_rows: list[CalibSample] = []
            for pred, actual, _ns, p_fav in warmup_snap:
                pred2 = apply_sets_calibrator(pred, sets_cal, p_fav)
                _append_calib_samples(ou_rows, pred2, actual, lines, p_fav)
            ou_cal = TotalsCalibrator.fit(
                ou_rows,
                method="platt" if method_ou == "platt" else "isotonic",
                use_bands=use_bands,
                min_samples=TENNIS_TOTALS_CALIB_MIN_SAMPLES,
            )
            meta["totals_calib_leaves"] = len(ou_cal.leaves)
            meta["totals_calib_train"] = ou_cal.n_train

    for m in matches:
        if m.score.walkover:
            meta["n_skip_wo"] += 1
            continue

        in_eval = m.tourney_date > cutoff
        if best_of_only is not None and m.best_of != best_of_only:
            _update_books(m)
            continue

        if eval_surfaces is not None and m.surface not in eval_surfaces and in_eval:
            _update_books(m)
            continue

        if not in_eval:
            if need_tb and not m.score.retired:
                _collect_tb_samples(book, serve_book, m, src_serve, tb_samples)
            if (
                need_warmup_proj
                and m.score.usable_for_totals
                and m.score.total_games is not None
            ):
                pred, _ea, _eb = _project(
                    book, serve_book, m, src_serve, cache, meta, fatigue=fatigue
                )
                p_fav = max(pred.p_a_win, pred.p_b_win)
                warmup_snap.append(
                    (pred, int(m.score.total_games), len(m.score.sets), p_fav)
                )
            _update_books(m)
            continue

        _fit_calibrators()
        pred, _ea, _eb = _project(
            book,
            serve_book,
            m,
            src_serve,
            cache,
            meta,
            tb_cal=tb_cal,
            fatigue=fatigue,
        )
        stats.ml.add(pred.p_a_win, pred.p_a_win > 0.5)
        p_eng = pred.p_a_win_engine if pred.p_a_win_engine > 0 else pred.p_a_win
        stats.ml_engine.add(p_eng, p_eng > 0.5)
        meta["n_eval_ml"] += 1
        meta["ml_source"] = pred.ml_source

        if m.score.usable_for_totals and m.score.total_games is not None:
            p_fav = max(pred.p_a_win, pred.p_b_win)
            pred2 = apply_sets_calibrator(pred, sets_cal, p_fav)
            stats.add_games(
                pred2, m.score.total_games, lines, p_fav=p_fav, calib=ou_cal
            )
            stats.n_sets += 1
            stats.p3_real += 1.0 if len(m.score.sets) == 3 else 0.0
            stats.p3_mod += pred2.p_sets_3
            meta["n_eval_games"] += 1

        _update_books(m)

    meta["cache_size"] = len(cache)
    return stats, meta


def _fmt_ml(label: str, b: _MlBucket) -> str:
    if not b.n:
        return f"{label}: n=0"
    return (
        f"{label}: n={b.n}  acc={b.correct / b.n:.1%}  "
        f"Brier={b.brier / b.n:.4f}  logloss={b.logloss / b.n:.4f}"
    )


def _print_ou(label: str, ou: dict, lines: tuple[float, ...]) -> None:
    print(label)
    for line in lines:
        b = ou.get(line)
        if not b or b["n"] <= 0:
            continue
        n = b["n"]
        print(
            f"  O/U {line}: n={int(n)}  Brier={b['brier'] / n:.4f}  "
            f"Over réel={b['over_rate'] / n:.1%}  "
            f"Over préd. moy={b['p_over_sum'] / n:.1%}"
        )


def _print_report(stats: _Running, meta: dict, lines: tuple[float, ...]) -> None:
    src = meta.get("ml_source", "?")
    print("=== Calibration tennis ATP ===")
    print(
        f"matchs charges={meta['n_total']}  warm-up<={meta['warmup_until_year']}  "
        f"eval ML={meta['n_eval_ml']}  eval jeux={meta['n_eval_games']}  "
        f"W/O skip={meta['n_skip_wo']}  cache={meta['cache_size']} (hits={meta['cache_hits']})"
    )
    print(
        f"Split: ML={src}  |  totaux=moteur  |  serve={meta.get('serve_source')}  "
        f"(MAJ stats={meta.get('n_serve_updates', 0)})"
    )
    print(
        f"TB: dedicated={meta.get('tb_dedicated')}  scale={meta.get('tb_serve_scale')}  "
        f"calib={meta.get('tb_calib')}  train={meta.get('tb_calib_train')}  "
        f"ready={meta.get('tb_calib_ready')}"
    )
    print(
        f"VagueA: break_back={meta.get('break_back')}  elo_n_prior={meta.get('elo_n_prior')}  "
        f"pmf_sigma={meta.get('pmf_sigma')}  fatigue={meta.get('fatigue')}"
    )
    print(
        f"Sets-calib: {meta.get('sets_calib')}  cond_shift={meta.get('sets_cond_shift')}  "
        f"train={meta.get('sets_calib_train')}  "
        f"offset={meta.get('sets_band_offset')}  "
        f"shift2={meta.get('sets_shift2')}  shift3={meta.get('sets_shift3')}"
    )
    print(
        f"OU-calib: {meta.get('totals_calib')}  bands={meta.get('totals_calib_bands')}  "
        f"train={meta.get('totals_calib_train')}  leaves={meta.get('totals_calib_leaves')}"
    )
    if stats.ml.n:
        print(_fmt_ml(f"Moneyline ({src})", stats.ml))
        if src != "engine":
            print(_fmt_ml("Moneyline moteur (diag)", stats.ml_engine))
    if stats.n_games:
        print(
            f"Total jeux : n={stats.n_games}  MAE={stats.abs_err / stats.n_games:.2f}  "
            f"bias(pred-act)={stats.bias / stats.n_games:+.2f}"
        )
        if stats.n_sets:
            print(
                f"P(3 sets): réel={stats.p3_real / stats.n_sets:.1%}  "
                f"modèle={stats.p3_mod / stats.n_sets:.1%}"
            )
        if meta.get("totals_calib") not in ("none", "off", "0", "false"):
            _print_ou("O/U apres sets / avant OU-calib:", stats.ou_raw, lines)
            _print_ou("O/U final (sets+OU calib):", stats.ou, lines)
        else:
            _print_ou("O/U:", stats.ou, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration tennis ATP (Elo + total jeux)")
    parser.add_argument("--years", default=os.environ.get("TENNIS_SACKMANN_YEARS", "2018-2026"))
    parser.add_argument(
        "--warmup-until",
        type=int,
        default=int(os.environ.get("TENNIS_CALIB_WARMUP_UNTIL", "2021")),
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=int(os.environ.get("TENNIS_CALIB_BEST_OF", "3")),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--all-best-of", action="store_true")
    parser.add_argument("--serve-source", choices=("stats", "elo_proxy"), default=None)
    parser.add_argument(
        "--totals-calib",
        choices=("isotonic", "platt", "none"),
        default=None,
        help="Calibration O/U (défaut TENNIS_TOTALS_CALIB)",
    )
    parser.add_argument(
        "--sets-calib",
        choices=("isotonic", "platt", "band_offset", "none"),
        default=None,
        help="Calibration P(3 sets) (défaut TENNIS_SETS_CALIB)",
    )
    parser.add_argument(
        "--no-sets-cond-shift",
        action="store_true",
        help="Désactive les shifts E[jeux|2] / E[jeux|3]",
    )
    parser.add_argument(
        "--no-calib-bands",
        action="store_true",
        help="Désactive les bandes close/mid/chalk (O/U)",
    )
    args = parser.parse_args()

    from tennis_data import parse_year_range

    years = parse_year_range(args.years)

    if args.download:
        print(f"Téléchargement {years[0]}–{years[-1]} …")
        download_atp_years(years, force=True)

    t0 = time.perf_counter()
    matches = load_atp_matches(years, download_missing=True)
    n_serve = sum(1 for m in matches if m.serve and m.serve.valid)
    print(
        f"Chargé {len(matches)} matchs ({n_serve} avec stats service) "
        f"en {time.perf_counter() - t0:.1f}s"
    )

    lines = TENNIS_TOTAL_GAMES_LINES
    best_of = None if args.all_best_of else args.best_of
    bands = False if args.no_calib_bands else None
    cond_shift = False if args.no_sets_cond_shift else None

    t1 = time.perf_counter()
    stats, meta = run_calibration(
        matches,
        warmup_until_year=args.warmup_until,
        best_of_only=best_of,
        lines=lines,
        serve_source=args.serve_source,
        totals_calib=args.totals_calib,
        totals_calib_bands=bands,
        sets_calib=args.sets_calib,
        sets_cond_shift=cond_shift,
    )
    print(f"Calibration en {time.perf_counter() - t1:.1f}s")
    _print_report(stats, meta, lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
