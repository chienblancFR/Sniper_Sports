"""
Moteur tennis : point → jeu → set → match (best-of-3 / best-of-5).

Sorties principales :
- moneyline (P(A gagne), P(B gagne))
- distribution du total de jeux (pour O/U 21.5, 22.0, …)
- P(exactement 2 sets) / P(3 sets) en BO3 (contrôle gratuit)

Hypothèses v1 :
- points i.i.d. conditionnellement au serveur
- scoring avantage (pas de no-ad)
- tie-break classique à 6-6 (premier à 7, écart de 2)
- sets indépendants conditionnellement aux % service
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

# Totaux de jeux possibles pour un set terminé (avec TB à 6-6).
_SET_GAME_TOTALS = (6, 7, 8, 9, 10, 12, 13)


def _clamp_prob(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, float(p)))


def p_win_game_on_serve(p_point: float) -> float:
    """
    Proba de tenir son jeu (avantage) si chaque point est gagné avec proba p_point.
    Formule classique : droites + passage par les égales.
    """
    p = _clamp_prob(p_point)
    q = 1.0 - p
    straight = (p**4) * (1.0 + 4.0 * q + 10.0 * q * q)
    denom = p * p + q * q
    if denom <= 1e-15:
        return 0.5
    via_deuce = 20.0 * (p**5) * (q**3) / denom
    return straight + via_deuce


def _tb_server_is_a(point_index: int, first_server_a: bool) -> bool:
    """Serveur du point `point_index` (0-based) dans un tie-break."""
    if point_index <= 0:
        return first_server_a
    # Après le 1er point : blocs de 2 (other, first, other, …)
    block = (point_index - 1) // 2
    other_serves = block % 2 == 0
    return (not first_server_a) if other_serves else first_server_a


def tb_serve_probs(
    p_serve_a: float,
    p_serve_b: float,
    surface: str = "Hard",
) -> tuple[float, float]:
    """
    SPW dédiés au tie-break : p_tb = μ_tb + scale*(p − μ_surface).
    scale < 1 compresse l'avantage favori (TB plus coin-flip que le SPW régulier).
    """
    from tennis_params import (
        TENNIS_TB_DEDICATED,
        TENNIS_TB_MU_SHIFT,
        TENNIS_TB_SERVE_SCALE,
        mu_serve_for_surface,
        normalize_surface,
    )

    pa = _clamp_prob(p_serve_a, 0.50, 0.85)
    pb = _clamp_prob(p_serve_b, 0.50, 0.85)
    if not TENNIS_TB_DEDICATED:
        return pa, pb
    mu = mu_serve_for_surface(normalize_surface(surface))
    mu_tb = _clamp_prob(mu + float(TENNIS_TB_MU_SHIFT), 0.50, 0.85)
    scale = float(TENNIS_TB_SERVE_SCALE)

    def _adj(p: float) -> float:
        return _clamp_prob(mu_tb + scale * (p - mu), 0.50, 0.85)

    return _adj(pa), _adj(pb)


def p_win_tiebreak(
    p_serve_a: float,
    p_serve_b: float,
    *,
    first_server_a: bool = True,
) -> float:
    """
    Proba que A gagne le tie-break (premier à 7, écart de 2).
    Alternance : 1 point, puis 2 / 2 / …
    """
    from collections import defaultdict

    pa = _clamp_prob(p_serve_a)
    pb = _clamp_prob(p_serve_b)
    target = 7
    max_pts = 50
    mass: dict[tuple[int, int, int], float] = {(0, 0, 0): 1.0}
    p_a = 0.0

    for _ in range(max_pts * 2):
        nxt: dict[tuple[int, int, int], float] = defaultdict(float)
        progressed = False
        for (a, b, t), w in mass.items():
            if w <= 0:
                continue
            if (a >= target and a - b >= 2) or (b >= target and b - a >= 2):
                if a > b:
                    p_a += w
                continue
            if t >= max_pts:
                server_a = _tb_server_is_a(t, first_server_a)
                p_pt = pa if server_a else (1.0 - pb)
                p_a += w * p_pt
                continue
            server_a = _tb_server_is_a(t, first_server_a)
            p_pt = pa if server_a else (1.0 - pb)
            nxt[(a + 1, b, t + 1)] += w * p_pt
            nxt[(a, b + 1, t + 1)] += w * (1.0 - p_pt)
            progressed = True
        if not progressed:
            break
        mass = defaultdict(float)
        for (a, b, t), w in nxt.items():
            if a >= target and a - b >= 2:
                p_a += w
            elif b >= target and b - a >= 2:
                pass
            else:
                mass[(a, b, t)] += w
    return p_a


def p_win_tiebreak_match(
    p_serve_a: float,
    p_serve_b: float,
    *,
    surface: str = "Hard",
    first_server_a: bool = True,
    tb_calibrator=None,
) -> float:
    """
    P(A gagne le TB) avec SPW dédiés (+ calib optionnelle).
    """
    pa, pb = tb_serve_probs(p_serve_a, p_serve_b, surface)
    p = p_win_tiebreak(pa, pb, first_server_a=first_server_a)
    if tb_calibrator is not None:
        p = float(tb_calibrator.calibrate(p))
    return _clamp_prob(p)


@dataclass(frozen=True)
class SetDistribution:
    """Distribution jointe (vainqueur du set, total de jeux du set)."""

    # shape (n_game_bins, 2) : [:,0]=P(A, games), [:,1]=P(B, games)
    game_totals: tuple[int, ...]
    probs: np.ndarray

    @property
    def p_a_wins(self) -> float:
        return float(self.probs[:, 0].sum())

    @property
    def p_b_wins(self) -> float:
        return float(self.probs[:, 1].sum())

    def expected_games(self) -> float:
        g = np.asarray(self.game_totals, dtype=float)
        return float((g * self.probs.sum(axis=1)).sum())


def set_distribution(
    p_serve_a: float,
    p_serve_b: float,
    *,
    first_server_a: bool = True,
    surface: str = "Hard",
    tb_calibrator=None,
    break_back: float | None = None,
) -> SetDistribution:
    """
    Distribution d'un set (premier à 6, écart 2, TB à 6-6).
    Le serveur alterne à chaque jeu ; le TB utilise des SPW dédiés.
    Si break_back>0 : après un break, hold du prochain serveur × (1-ρ).
    """
    from tennis_params import TENNIS_BREAK_BACK

    hold_a0 = p_win_game_on_serve(p_serve_a)
    hold_b0 = p_win_game_on_serve(p_serve_b)
    rho = float(TENNIS_BREAK_BACK if break_back is None else break_back)
    rho = max(0.0, min(0.50, rho))
    # Index jeux → colonne 0=A, 1=B
    idx = {g: i for i, g in enumerate(_SET_GAME_TOTALS)}
    out = np.zeros((len(_SET_GAME_TOTALS), 2), dtype=float)

    # État : (games_a, games_b, server_a, prev_break) → proba
    from collections import defaultdict

    mass: dict[tuple[int, int, bool, bool], float] = {
        (0, 0, first_server_a, False): 1.0
    }

    while mass:
        nxt: dict[tuple[int, int, bool, bool], float] = defaultdict(float)
        for (ga, gb, srv_a, prev_br), w in mass.items():
            if ga == 6 and gb == 6:
                p_tb = p_win_tiebreak_match(
                    p_serve_a,
                    p_serve_b,
                    surface=surface,
                    first_server_a=srv_a,
                    tb_calibrator=tb_calibrator,
                )
                out[idx[13], 0] += w * p_tb
                out[idx[13], 1] += w * (1.0 - p_tb)
                continue
            if ga >= 6 and ga - gb >= 2:
                total = ga + gb
                if total in idx:
                    out[idx[total], 0] += w
                continue
            if gb >= 6 and gb - ga >= 2:
                total = ga + gb
                if total in idx:
                    out[idx[total], 1] += w
                continue

            hold_a = hold_a0 * (1.0 - rho) if (prev_br and srv_a) else hold_a0
            hold_b = hold_b0 * (1.0 - rho) if (prev_br and not srv_a) else hold_b0
            hold_a = max(0.01, min(0.99, hold_a))
            hold_b = max(0.01, min(0.99, hold_b))
            p_a_wins_game = hold_a if srv_a else (1.0 - hold_b)
            if srv_a:
                # A sert : A gagne = hold ; B gagne = break → prev_break pour B
                nxt[(ga + 1, gb, False, False)] += w * p_a_wins_game
                nxt[(ga, gb + 1, False, True)] += w * (1.0 - p_a_wins_game)
            else:
                # B sert : A gagne = break ; B gagne = hold
                nxt[(ga + 1, gb, True, True)] += w * p_a_wins_game
                nxt[(ga, gb + 1, True, False)] += w * (1.0 - p_a_wins_game)
        mass = {k: v for k, v in nxt.items() if v > 1e-18}

    s = out.sum()
    if s > 0:
        out /= s
    return SetDistribution(game_totals=_SET_GAME_TOTALS, probs=out)


@dataclass(frozen=True)
class MatchProjection:
    """
    Projection moneyline + totaux de jeux.

    Par défaut (via `project_matchup`) :
    - `p_a_win` / `p_b_win` = moneyline officielle (Elo, éventuellement blend)
    - `p_a_win_engine` / `p_b_win_engine` = vainqueur issu du moteur sets (diag)
    - `games_pmf` / O/U = toujours le moteur point→jeux
    """

    p_a_win: float
    p_b_win: float
    p_sets_2: float  # match en 2 sets (BO3) / « droit » BO5
    p_sets_3: float  # match en 3 sets (BO3) / 4–5 sets BO5
    expected_games: float
    # PMF sparse : total_jeux -> proba
    games_pmf: dict[int, float]
    best_of: int
    p_serve_a: float
    p_serve_b: float
    p_a_win_engine: float = 0.0
    p_b_win_engine: float = 0.0
    ml_source: str = "engine"
    # PMF conditionnelles (somme ≈ 1 chacune) — BO3 ; vides si indisponibles
    games_pmf_2: dict[int, float] = field(default_factory=dict)
    games_pmf_3: dict[int, float] = field(default_factory=dict)

    def p_over(self, line: float) -> float:
        """P(total jeux > line). Pour x.5 : strictement au-dessus."""
        return sum(p for g, p in self.games_pmf.items() if g > line)

    def p_under(self, line: float) -> float:
        return sum(p for g, p in self.games_pmf.items() if g < line)

    def p_push(self, line: float) -> float:
        """P(total == line) si ligne entière ; 0 si .5."""
        if abs(line - round(line)) > 1e-9:
            return 0.0
        key = int(round(line))
        return float(self.games_pmf.get(key, 0.0))

    def fair_odds_over(self, line: float) -> float | None:
        p = self.p_over(line)
        return (1.0 / p) if p > 1e-12 else None

    def fair_odds_under(self, line: float) -> float | None:
        # Sur ligne .5 : under = 1 - over ; sur entière : exclude push
        push = self.p_push(line)
        p = self.p_under(line)
        # Si push possible, cotes « win only » conditionnelles au non-push
        denom = p
        if push > 0:
            # marché 2-way classique : over vs under, push remboursé → fair sur (1-push)
            live = 1.0 - push
            if live <= 1e-12:
                return None
            p = p / live
        return (1.0 / p) if p > 1e-12 else None

    def summary_lines(self, lines: Iterable[float] = (20.5, 21.5, 22.5, 23.5)) -> list[str]:
        ml = (
            f"ML({self.ml_source}) A={self.p_a_win:.1%}  B={self.p_b_win:.1%}"
        )
        if self.ml_source != "engine" and self.p_a_win_engine > 0:
            ml += f"  [moteur {self.p_a_win_engine:.1%}/{self.p_b_win_engine:.1%}]"
        rows = [
            f"{ml}  E[jeux]={self.expected_games:.1f}  "
            f"2sets={self.p_sets_2:.1%}  3sets={self.p_sets_3:.1%}"
        ]
        for line in lines:
            po, pu = self.p_over(line), self.p_under(line)
            fo, fu = self.fair_odds_over(line), self.fair_odds_under(line)
            rows.append(
                f"  O/U {line}: Over {po:.1%} (@{fo:.3f})"
                f"  Under {pu:.1%} (@{fu:.3f})"
                if fo and fu
                else f"  O/U {line}: Over {po:.1%}  Under {pu:.1%}"
            )
        return rows


def match_projection(
    p_serve_a: float,
    p_serve_b: float,
    *,
    best_of: int = 3,
    first_server_a: bool = True,
    surface: str = "Hard",
    tb_calibrator=None,
) -> MatchProjection:
    """
    Projette un match best-of-`best_of` (3 ou 5).
    Le serveur du 1er jeu de chaque set alterne (approx. réaliste : vainqueur du set précédent
    ne sert pas en premier le set suivant — ici on alterne simplement set à set).
    """
    if best_of not in (3, 5):
        raise ValueError("best_of doit être 3 ou 5")
    need = best_of // 2 + 1

    # Précompute 2 distributions de set (serveur initial A ou B)
    set_a = set_distribution(
        p_serve_a,
        p_serve_b,
        first_server_a=True,
        surface=surface,
        tb_calibrator=tb_calibrator,
    )
    set_b = set_distribution(
        p_serve_a,
        p_serve_b,
        first_server_a=False,
        surface=surface,
        tb_calibrator=tb_calibrator,
    )

    from collections import defaultdict

    # État match : (sets_a, sets_b, next_set_first_server_a, games_so_far) → proba
    mass: dict[tuple[int, int, bool, int], float] = {
        (0, 0, first_server_a, 0): 1.0
    }
    games_pmf: dict[int, float] = defaultdict(float)
    games_pmf_2: dict[int, float] = defaultdict(float)
    games_pmf_3: dict[int, float] = defaultdict(float)
    p_a_win = 0.0
    p_b_win = 0.0
    p_sets_exact: dict[int, float] = defaultdict(float)

    safety = 0
    while mass and safety < 20:
        safety += 1
        nxt: dict[tuple[int, int, bool, int], float] = defaultdict(float)
        for (sa, sb, srv0, g0), w in mass.items():
            dist = set_a if srv0 else set_b
            for gi, total in enumerate(dist.game_totals):
                for winner in (0, 1):
                    pw = dist.probs[gi, winner]
                    if pw <= 0:
                        continue
                    w2 = w * pw
                    g1 = g0 + total
                    sa2 = sa + (1 if winner == 0 else 0)
                    sb2 = sb + (1 if winner == 1 else 0)
                    srv1 = not srv0
                    if sa2 >= need or sb2 >= need:
                        sets_played = sa2 + sb2
                        p_sets_exact[sets_played] += w2
                        games_pmf[g1] += w2
                        if best_of == 3:
                            if sets_played == 2:
                                games_pmf_2[g1] += w2
                            elif sets_played == 3:
                                games_pmf_3[g1] += w2
                        if sa2 > sb2:
                            p_a_win += w2
                        else:
                            p_b_win += w2
                    else:
                        nxt[(sa2, sb2, srv1, g1)] += w2
        mass = {k: v for k, v in nxt.items() if v > 1e-18}

    tot = p_a_win + p_b_win
    if tot > 0 and abs(tot - 1.0) > 1e-6:
        p_a_win /= tot
        p_b_win /= tot
        games_pmf = {g: p / tot for g, p in games_pmf.items()}
        p_sets_exact = {k: v / tot for k, v in p_sets_exact.items()}
        games_pmf_2 = {g: p / tot for g, p in games_pmf_2.items()}
        games_pmf_3 = {g: p / tot for g, p in games_pmf_3.items()}

    if best_of == 3:
        p2 = float(p_sets_exact.get(2, 0.0))
        p3 = float(p_sets_exact.get(3, 0.0))
    else:
        p2 = float(p_sets_exact.get(3, 0.0))
        p3 = float(p_sets_exact.get(4, 0.0) + p_sets_exact.get(5, 0.0))

    def _cond(pmf: dict[int, float]) -> dict[int, float]:
        s = sum(pmf.values())
        if s <= 1e-15:
            return {}
        return {g: p / s for g, p in sorted(pmf.items())}

    pmf2_c = _cond(games_pmf_2)
    pmf3_c = _cond(games_pmf_3)
    exp_g = sum(g * p for g, p in games_pmf.items())

    return MatchProjection(
        p_a_win=p_a_win,
        p_b_win=p_b_win,
        p_sets_2=p2,
        p_sets_3=p3,
        expected_games=exp_g,
        games_pmf=dict(sorted(games_pmf.items())),
        best_of=best_of,
        p_serve_a=_clamp_prob(p_serve_a),
        p_serve_b=_clamp_prob(p_serve_b),
        p_a_win_engine=p_a_win,
        p_b_win_engine=p_b_win,
        ml_source="engine",
        games_pmf_2=pmf2_c,
        games_pmf_3=pmf3_c,
    )


def reweight_sets_pmf(
    proj: MatchProjection,
    p_sets_3_new: float,
    *,
    shift_games_2: float = 0.0,
    shift_games_3: float = 0.0,
) -> MatchProjection:
    """
    Reconstruit la PMF des jeux : (1-p3)*pmf|2sets + p3*pmf|3sets.
    Les shifts (en jeux) décalent les PMF conditionnelles (entiers arrondis).
    """
    if proj.best_of != 3:
        return proj
    pmf2 = proj.games_pmf_2 or {}
    pmf3 = proj.games_pmf_3 or {}
    if not pmf2 or not pmf3:
        return proj

    p3 = max(0.0, min(1.0, float(p_sets_3_new)))
    p2 = 1.0 - p3

    def _shift(pmf: dict[int, float], delta: float) -> dict[int, float]:
        if abs(delta) < 1e-9:
            return pmf
        d = int(round(delta))
        out: dict[int, float] = {}
        for g, p in pmf.items():
            out[g + d] = out.get(g + d, 0.0) + p
        return out

    pmf2 = _shift(pmf2, shift_games_2)
    pmf3 = _shift(pmf3, shift_games_3)

    new_pmf: dict[int, float] = {}
    for g, p in pmf2.items():
        new_pmf[g] = new_pmf.get(g, 0.0) + p2 * p
    for g, p in pmf3.items():
        new_pmf[g] = new_pmf.get(g, 0.0) + p3 * p
    s = sum(new_pmf.values())
    if s > 0:
        new_pmf = {g: p / s for g, p in sorted(new_pmf.items())}
    exp_g = sum(g * p for g, p in new_pmf.items())
    return replace(
        proj,
        p_sets_2=p2,
        p_sets_3=p3,
        games_pmf=new_pmf,
        expected_games=exp_g,
        games_pmf_2=pmf2,
        games_pmf_3=pmf3,
    )


def elo_to_serve_probs(
    elo_a: float,
    elo_b: float,
    *,
    mu_serve: float = 0.64,
    elo_point_scale: float = 400.0,
    point_sensitivity: float = 0.25,
) -> tuple[float, float]:
    """
    Convertit un écart Elo surface en (p_serve_a, p_serve_b).

    Le joueur le mieux classé gagne un peu plus de points au service et
    en retour (symétrique autour de la moyenne circuit `mu_serve`).
    `point_sensitivity` contrôle l'amplitude (à calibrer en backtest).
    """
    mu = _clamp_prob(mu_serve, 0.50, 0.80)
    delta = (float(elo_a) - float(elo_b)) / float(elo_point_scale)
    # logit(mu) ± sens * delta
    def _logit(x: float) -> float:
        x = _clamp_prob(x)
        return math.log(x / (1.0 - x))

    def _sig(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    base = _logit(mu)
    p_a = _clamp_prob(_sig(base + point_sensitivity * delta))
    p_b = _clamp_prob(_sig(base - point_sensitivity * delta))
    return p_a, p_b


def project_from_elo(
    elo_a: float,
    elo_b: float,
    *,
    best_of: int = 3,
    mu_serve: float = 0.64,
    elo_point_scale: float = 400.0,
    point_sensitivity: float = 0.25,
    first_server_a: bool = True,
    surface: str = "Hard",
    tb_calibrator=None,
) -> MatchProjection:
    """Raccourci Elo surface → projection match."""
    pa, pb = elo_to_serve_probs(
        elo_a,
        elo_b,
        mu_serve=mu_serve,
        elo_point_scale=elo_point_scale,
        point_sensitivity=point_sensitivity,
    )
    return match_projection(
        pa,
        pb,
        best_of=best_of,
        first_server_a=first_server_a,
        surface=surface,
        tb_calibrator=tb_calibrator,
    )


def moneyline_from_elo(
    elo_a: float,
    elo_b: float,
    *,
    reliability: float = 1.0,
) -> tuple[float, float]:
    """P(A gagne), P(B gagne) via Elo ; shrink vers 0.5 si reliability < 1."""
    from tennis_ratings import expected_score

    p_a = expected_score(float(elo_a), float(elo_b))
    rel = max(0.0, min(1.0, float(reliability)))
    p_a = 0.5 + rel * (p_a - 0.5)
    return p_a, 1.0 - p_a


def widen_games_pmf(
    pmf: dict[int, float],
    sigma: float,
) -> dict[int, float]:
    """
    Élargit une PMF jeux par noyau discret ±0/±1/±2 pondéré par sigma.
    sigma=0 → no-op ; sigma≈1.5 → mélange doux.
    """
    if not pmf or sigma <= 1e-9:
        return pmf
    # Poids type gaussien discrétisé, intensité croissante avec sigma
    s = max(0.0, float(sigma))
    # w0 au centre, w1 à ±1, w2 à ±2 — normalisés
    w0 = 1.0
    w1 = min(1.2, 0.35 * s)
    w2 = min(0.8, 0.15 * s)
    tot_w = w0 + 2.0 * w1 + 2.0 * w2
    kernel = {0: w0 / tot_w, 1: w1 / tot_w, -1: w1 / tot_w, 2: w2 / tot_w, -2: w2 / tot_w}
    out: dict[int, float] = {}
    for g, p in pmf.items():
        for d, kd in kernel.items():
            key = int(g + d)
            if key < 12:  # floor réaliste BO3
                key = 12
            out[key] = out.get(key, 0.0) + p * kd
    ssum = sum(out.values())
    if ssum <= 0:
        return pmf
    return {g: p / ssum for g, p in sorted(out.items())}


def apply_pmf_uncertainty(
    proj: MatchProjection,
    reliability: float,
    *,
    sigma_max: float | None = None,
) -> MatchProjection:
    """Élargit games_pmf (et conditionnelles) selon (1-rel)*sigma_max."""
    from dataclasses import replace

    from tennis_params import TENNIS_PMF_SIGMA

    rel = max(0.0, min(1.0, float(reliability)))
    sig_max = float(TENNIS_PMF_SIGMA if sigma_max is None else sigma_max)
    sig = sig_max * (1.0 - rel)
    if sig <= 1e-9:
        return proj
    pmf = widen_games_pmf(proj.games_pmf, sig)
    pmf2 = widen_games_pmf(proj.games_pmf_2, sig) if proj.games_pmf_2 else proj.games_pmf_2
    pmf3 = widen_games_pmf(proj.games_pmf_3, sig) if proj.games_pmf_3 else proj.games_pmf_3
    exp_g = sum(g * p for g, p in pmf.items())
    return replace(
        proj,
        games_pmf=pmf,
        games_pmf_2=pmf2 or {},
        games_pmf_3=pmf3 or {},
        expected_games=exp_g,
    )


def apply_ml_policy(
    engine: MatchProjection,
    elo_a: float,
    elo_b: float,
    *,
    source: str | None = None,
    blend_engine: float | None = None,
    reliability: float = 1.0,
) -> MatchProjection:
    """
    Remplace la moneyline officielle selon la politique configurée.

    - elo    : 100 % Elo (défaut — meilleur Brier)
    - engine : 100 % moteur sets
    - blend  : (1-w)*Elo + w*moteur, w = TENNIS_ML_BLEND_ENGINE
    Les totaux de jeux restent ceux du moteur (éventuellement élargis à part).
    """
    from dataclasses import replace

    from tennis_params import TENNIS_ML_BLEND_ENGINE, TENNIS_ML_SOURCE

    src = (source or TENNIS_ML_SOURCE).strip().lower()
    p_elo_a, p_elo_b = moneyline_from_elo(elo_a, elo_b, reliability=reliability)
    p_eng_a = engine.p_a_win_engine if engine.p_a_win_engine > 0 else engine.p_a_win
    p_eng_b = engine.p_b_win_engine if engine.p_b_win_engine > 0 else engine.p_b_win

    if src == "engine":
        p_a, p_b = p_eng_a, p_eng_b
    elif src == "blend":
        w = float(TENNIS_ML_BLEND_ENGINE if blend_engine is None else blend_engine)
        w = max(0.0, min(1.0, w))
        p_a = (1.0 - w) * p_elo_a + w * p_eng_a
        p_b = 1.0 - p_a
        s = p_a + p_b
        if s > 0:
            p_a, p_b = p_a / s, p_b / s
    else:
        src = "elo"
        p_a, p_b = p_elo_a, p_elo_b

    return replace(
        engine,
        p_a_win=p_a,
        p_b_win=p_b,
        p_a_win_engine=p_eng_a,
        p_b_win_engine=p_eng_b,
        ml_source=src,
    )


def project_matchup(
    elo_a: float,
    elo_b: float,
    surface: str = "Hard",
    *,
    best_of: int | None = None,
    first_server_a: bool = True,
    ml_source: str | None = None,
    p_serve_a: float | None = None,
    p_serve_b: float | None = None,
    tb_calibrator=None,
    reliability: float = 1.0,
) -> MatchProjection:
    """
    Projection matchup :
    - totaux / sets = moteur point→jeux
      (p_serve_* fournis, sinon proxy Elo selon TENNIS_SERVE_SOURCE côté appelant)
    - moneyline = Elo (défaut), shrink si reliability < 1
    - TB = SPW dédiés (+ calib optionnelle)
    - PMF jeux élargie si peu de données (1-rel)
    """
    from tennis_params import (
        TENNIS_BEST_OF_DEFAULT,
        TENNIS_ELO_POINT_SCALE,
        TENNIS_POINT_SENSITIVITY,
        mu_serve_for_surface,
        normalize_surface,
    )

    surf = normalize_surface(surface)
    bo = TENNIS_BEST_OF_DEFAULT if best_of is None else best_of
    if p_serve_a is not None and p_serve_b is not None:
        engine = match_projection(
            p_serve_a,
            p_serve_b,
            best_of=bo,
            first_server_a=first_server_a,
            surface=surf,
            tb_calibrator=tb_calibrator,
        )
    else:
        engine = project_from_elo(
            elo_a,
            elo_b,
            best_of=bo,
            mu_serve=mu_serve_for_surface(surf),
            elo_point_scale=TENNIS_ELO_POINT_SCALE,
            point_sensitivity=TENNIS_POINT_SENSITIVITY,
            first_server_a=first_server_a,
            surface=surf,
            tb_calibrator=tb_calibrator,
        )
    out = apply_ml_policy(
        engine, elo_a, elo_b, source=ml_source, reliability=reliability
    )
    return apply_pmf_uncertainty(out, reliability)


def _self_check() -> None:
    """Contrôles numériques minimaux (smoke)."""
    from config_env import load_project_env

    load_project_env("tennis")

    m = match_projection(0.64, 0.64, best_of=3)
    assert abs(m.p_a_win - 0.5) < 0.02, m.p_a_win
    assert abs(m.p_b_win - 0.5) < 0.02, m.p_b_win
    assert abs(m.p_sets_2 + m.p_sets_3 - 1.0) < 1e-6
    assert abs(sum(m.games_pmf.values()) - 1.0) < 1e-5

    fav = match_projection(0.72, 0.58, best_of=3)
    dog = match_projection(0.58, 0.72, best_of=3)
    assert fav.p_a_win > 0.75
    assert dog.p_a_win < 0.25
    assert fav.expected_games < match_projection(0.64, 0.64).expected_games

    h = p_win_game_on_serve(0.65)
    assert 0.78 < h < 0.90, h

    e = project_from_elo(1700, 1500, best_of=3)
    assert e.p_a_win > 0.55

    from tennis_ratings import SurfaceEloBook, expected_score

    book = SurfaceEloBook()
    book.update_after_match("p1", "p2", "Hard")
    book.update_after_match("p1", "p2", "Hard")
    ea, eb = book.matchup_elos("p1", "p2", "Hard")
    assert ea > eb
    proj = project_matchup(ea, eb, "Hard")
    assert proj.p_a_win > 0.5
    assert proj.ml_source == "elo"
    p_elo, _ = moneyline_from_elo(ea, eb)
    assert abs(proj.p_a_win - p_elo) < 1e-9
    assert abs(proj.p_a_win - expected_score(ea, eb)) < 1e-9
    # Moteur plus extrême que Elo sur gros écarts (diag)
    assert proj.p_a_win_engine > 0.5

    print("tennis_model OK")
    for line in m.summary_lines():
        print(line)
    print("--- favori serve 0.72 vs 0.58 ---")
    for line in fav.summary_lines():
        print(line)
    print(f"--- Elo demos p1 vs p2 Hard: {ea:.0f} vs {eb:.0f} ---")
    for line in proj.summary_lines():
        print(line)


if __name__ == "__main__":
    _self_check()
