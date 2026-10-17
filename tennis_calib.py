"""
Calibration des probabilités O/U jeux (post-moteur).

Fit PIT-safe : entraîner sur le warm-up, appliquer sur l'éval / le live.
Méthodes : isotonic (PAVA) ou Platt (sigmoid sur logit).
Bandes optionnelles selon la force du favori Elo (close / mid / chalk).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


def _clamp01(p: float, eps: float = 1e-4) -> float:
    return max(eps, min(1.0 - eps, float(p)))


def _logit(p: float) -> float:
    p = _clamp01(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def favorite_band(p_fav: float) -> str:
    """Bande selon P(favori) Elo : close < 0.58 ≤ mid < 0.70 ≤ chalk."""
    p = float(p_fav)
    if p < 0.58:
        return "close"
    if p < 0.70:
        return "mid"
    return "chalk"


# ---------------------------------------------------------------------------
# Isotonic (PAVA)
# ---------------------------------------------------------------------------


@dataclass
class IsotonicCalibrator:
    """Fonction de calibration non-décroissante p_raw → p_cal."""

    x: np.ndarray  # seuils p_raw croissants
    y: np.ndarray  # p_cal associés

    def predict(self, p: float) -> float:
        if self.x.size == 0:
            return _clamp01(p)
        pv = float(p)
        if pv <= float(self.x[0]):
            return float(self.y[0])
        if pv >= float(self.x[-1]):
            return float(self.y[-1])
        # Interpolation linéaire entre blocks
        idx = int(np.searchsorted(self.x, pv, side="right") - 1)
        idx = max(0, min(idx, len(self.x) - 2))
        x0, x1 = float(self.x[idx]), float(self.x[idx + 1])
        y0, y1 = float(self.y[idx]), float(self.y[idx + 1])
        if x1 <= x0:
            return y0
        t = (pv - x0) / (x1 - x0)
        return _clamp01(y0 + t * (y1 - y0))

    @staticmethod
    def fit(p_raw: np.ndarray, y: np.ndarray, *, min_samples: int = 30) -> IsotonicCalibrator | None:
        return IsotonicCalibrator._fit_pava(p_raw, y, min_samples=min_samples)

    @staticmethod
    def _fit_pava(
        p_raw: np.ndarray, y: np.ndarray, *, min_samples: int
    ) -> IsotonicCalibrator | None:
        if p_raw.size < min_samples:
            return None
        order = np.argsort(p_raw, kind="mergesort")
        x_sorted = p_raw[order].astype(float)
        y_sorted = y[order].astype(float)
        # Blocks: list of (sum_y, weight, x_min, x_max)
        blocks: list[list[float]] = []
        for xi, yi in zip(x_sorted, y_sorted):
            blocks.append([yi, 1.0, xi, xi])
            while len(blocks) >= 2:
                b1, b2 = blocks[-2], blocks[-1]
                m1 = b1[0] / b1[1]
                m2 = b2[0] / b2[1]
                if m1 <= m2 + 1e-15:
                    break
                blocks[-2] = [
                    b1[0] + b2[0],
                    b1[1] + b2[1],
                    b1[2],
                    b2[3],
                ]
                blocks.pop()
        xs = []
        ys = []
        for sy, w, xmin, xmax in blocks:
            mean = sy / w
            xs.append(0.5 * (xmin + xmax))
            ys.append(mean)
        return IsotonicCalibrator(x=np.asarray(xs, dtype=float), y=np.asarray(ys, dtype=float))


# ---------------------------------------------------------------------------
# Platt
# ---------------------------------------------------------------------------


@dataclass
class PlattCalibrator:
    a: float
    b: float

    def predict(self, p: float) -> float:
        return _clamp01(_sigmoid(self.a * _logit(p) + self.b))

    @staticmethod
    def fit(p_raw: np.ndarray, y: np.ndarray, *, min_samples: int = 30) -> PlattCalibrator | None:
        if p_raw.size < min_samples:
            return None
        # Cible adoucie (Platt 1999)
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        if n_pos < 5 or n_neg < 5:
            return None
        t = np.where(y > 0.5, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        x = np.array([_logit(float(p)) for p in p_raw], dtype=float)
        a, b = 1.0, 0.0
        for _ in range(40):
            f = 1.0 / (1.0 + np.exp(-(a * x + b)))
            f = np.clip(f, 1e-6, 1.0 - 1e-6)
            d = f - t
            # Newton sur [a,b]
            w = f * (1.0 - f)
            # Hessian / gradient
            g_a = float(np.sum(d * x))
            g_b = float(np.sum(d))
            h_aa = float(np.sum(w * x * x)) + 1e-6
            h_ab = float(np.sum(w * x))
            h_bb = float(np.sum(w)) + 1e-6
            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-12:
                break
            da = (h_bb * g_a - h_ab * g_b) / det
            db = (h_aa * g_b - h_ab * g_a) / det
            a -= da
            b -= db
            if abs(da) + abs(db) < 1e-8:
                break
        return PlattCalibrator(a=float(a), b=float(b))


# ---------------------------------------------------------------------------
# Totals O/U
# ---------------------------------------------------------------------------


@dataclass
class CalibSample:
    line: float
    p_raw: float
    y_over: float  # 0/1
    p_fav: float  # max(p_ml_a, p_ml_b)


@dataclass
class _Leaf:
    method: str
    iso: IsotonicCalibrator | None = None
    platt: PlattCalibrator | None = None

    def predict(self, p: float) -> float:
        if self.method == "isotonic" and self.iso is not None:
            return self.iso.predict(p)
        if self.method == "platt" and self.platt is not None:
            return self.platt.predict(p)
        return _clamp01(p)


@dataclass
class TotalsCalibrator:
    """
    Calibre p_over par ligne (+ bande favori optionnelle).
    Clé : (line_rounded, band) ou (line_rounded, "all").
    """

    method: str = "isotonic"
    use_bands: bool = True
    leaves: dict[tuple[float, str], _Leaf] = field(default_factory=dict)
    n_train: int = 0

    def calibrate_over(self, line: float, p_raw: float, p_fav: float) -> float:
        if not self.leaves:
            return _clamp01(p_raw)
        line_k = round(float(line), 1)
        band = favorite_band(p_fav) if self.use_bands else "all"
        for key in ((line_k, band), (line_k, "all"), (0.0, band), (0.0, "all")):
            leaf = self.leaves.get(key)
            if leaf is not None:
                return leaf.predict(p_raw)
        return _clamp01(p_raw)

    def calibrate_under(
        self, line: float, p_over_raw: float, p_under_raw: float, p_fav: float, p_push: float = 0.0
    ) -> tuple[float, float]:
        """
        Retourne (p_over_cal, p_under_cal) renormalisés hors push.
        Le push modèle est conservé.
        """
        push = max(0.0, float(p_push))
        live = 1.0 - push
        if live <= 1e-9:
            return 0.0, 0.0
        # Travailler sur proba conditionnelle au non-push
        p_o_cond = p_over_raw / live if p_over_raw > 0 else 0.0
        p_o_cal = self.calibrate_over(line, p_o_cond, p_fav)
        p_o_cal = _clamp01(p_o_cal)
        p_u_cal = 1.0 - p_o_cal
        return p_o_cal * live, p_u_cal * live

    @staticmethod
    def fit(
        samples: list[CalibSample],
        *,
        method: str = "isotonic",
        use_bands: bool = True,
        min_samples: int = 80,
    ) -> TotalsCalibrator:
        cal = TotalsCalibrator(method=method, use_bands=use_bands, n_train=len(samples))
        if not samples:
            return cal

        buckets: dict[tuple[float, str], list[CalibSample]] = {}
        for s in samples:
            line_k = round(float(s.line), 1)
            band = favorite_band(s.p_fav) if use_bands else "all"
            buckets.setdefault((line_k, band), []).append(s)
            buckets.setdefault((line_k, "all"), []).append(s)
            buckets.setdefault((0.0, band), []).append(s)
            buckets.setdefault((0.0, "all"), []).append(s)

        for key, rows in buckets.items():
            if len(rows) < min_samples:
                continue
            p = np.array([r.p_raw for r in rows], dtype=float)
            y = np.array([r.y_over for r in rows], dtype=float)
            leaf: _Leaf | None = None
            if method == "platt":
                pl = PlattCalibrator.fit(p, y, min_samples=min_samples)
                if pl is not None:
                    leaf = _Leaf(method="platt", platt=pl)
            else:
                iso = IsotonicCalibrator._fit_pava(p, y, min_samples=min_samples)
                if iso is not None:
                    leaf = _Leaf(method="isotonic", iso=iso)
                else:
                    pl = PlattCalibrator.fit(p, y, min_samples=min_samples)
                    if pl is not None:
                        leaf = _Leaf(method="platt", platt=pl)
            if leaf is not None:
                cal.leaves[key] = leaf
        return cal


# ---------------------------------------------------------------------------
# P(3 sets) + shifts conditionnels
# ---------------------------------------------------------------------------


@dataclass
class SetsSample:
    p3_raw: float
    y_three: float  # 1 si match en 3 sets
    p_fav: float
    games: int
    n_sets: int  # 2 ou 3
    exp_games_2: float  # E_model[jeux | 2 sets]
    exp_games_3: float


@dataclass
class SetsCalibrator:
    """
    Corrige le raccourcissement des gros favoris :
    1) p3_cal = f(p3_raw)  (isotonic / platt / offset par bande)
    2) shifts optionnels sur E[jeux|2] et E[jeux|3] par bande
    """

    method: str = "isotonic"
    leaf: _Leaf | None = None
    band_offset: dict[str, float] = field(default_factory=dict)
    shift2: dict[str, float] = field(default_factory=dict)
    shift3: dict[str, float] = field(default_factory=dict)
    n_train: int = 0

    def calibrate_p3(self, p3_raw: float, p_fav: float) -> float:
        p = _clamp01(p3_raw)
        band = favorite_band(p_fav)
        if self.method == "band_offset":
            return _clamp01(p + self.band_offset.get(band, 0.0))
        if self.leaf is not None:
            return _clamp01(self.leaf.predict(p))
        return p

    def shifts_for(self, p_fav: float) -> tuple[float, float]:
        band = favorite_band(p_fav)
        return float(self.shift2.get(band, 0.0)), float(self.shift3.get(band, 0.0))

    @staticmethod
    def fit(
        samples: list[SetsSample],
        *,
        method: str = "isotonic",
        cond_shift: bool = True,
        min_samples: int = 80,
    ) -> SetsCalibrator:
        cal = SetsCalibrator(method=method, n_train=len(samples))
        if not samples:
            return cal

        # --- P(3 sets) ---
        if method == "band_offset":
            buckets: dict[str, list[SetsSample]] = defaultdict(list)
            for s in samples:
                buckets[favorite_band(s.p_fav)].append(s)
            for band, rows in buckets.items():
                if len(rows) < min_samples // 2:
                    continue
                real = sum(r.y_three for r in rows) / len(rows)
                mod = sum(r.p3_raw for r in rows) / len(rows)
                cal.band_offset[band] = real - mod
        else:
            p = np.array([s.p3_raw for s in samples], dtype=float)
            y = np.array([s.y_three for s in samples], dtype=float)
            if method == "platt":
                pl = PlattCalibrator.fit(p, y, min_samples=min_samples)
                if pl is not None:
                    cal.leaf = _Leaf(method="platt", platt=pl)
            else:
                iso = IsotonicCalibrator._fit_pava(p, y, min_samples=min_samples)
                if iso is not None:
                    cal.leaf = _Leaf(method="isotonic", iso=iso)

        # --- Shifts E[jeux|k sets] ---
        if cond_shift:
            by_band: dict[str, list[SetsSample]] = defaultdict(list)
            for s in samples:
                by_band[favorite_band(s.p_fav)].append(s)
            for band, rows in by_band.items():
                g2_real = [r.games for r in rows if r.n_sets == 2]
                g3_real = [r.games for r in rows if r.n_sets == 3]
                g2_mod = [r.exp_games_2 for r in rows if r.n_sets == 2]
                g3_mod = [r.exp_games_3 for r in rows if r.n_sets == 3]
                if len(g2_real) >= min_samples // 2 and g2_mod:
                    cal.shift2[band] = float(
                        sum(g2_real) / len(g2_real) - sum(g2_mod) / len(g2_mod)
                    )
                if len(g3_real) >= min_samples // 2 and g3_mod:
                    cal.shift3[band] = float(
                        sum(g3_real) / len(g3_real) - sum(g3_mod) / len(g3_mod)
                    )
        return cal


def apply_sets_calibrator(proj, sets_cal: SetsCalibrator | None, p_fav: float):
    """Repondère la PMF jeux via SetsCalibrator (no-op si None / hors BO3)."""
    from tennis_model import MatchProjection, reweight_sets_pmf

    if sets_cal is None or not isinstance(proj, MatchProjection):
        return proj
    if proj.best_of != 3:
        return proj
    p3 = sets_cal.calibrate_p3(proj.p_sets_3, p_fav)
    s2, s3 = sets_cal.shifts_for(p_fav)
    return reweight_sets_pmf(proj, p3, shift_games_2=s2, shift_games_3=s3)


# ---------------------------------------------------------------------------
# Tie-break : P(favori / joueur A gagne le TB)
# ---------------------------------------------------------------------------


@dataclass
class TbSample:
    p_raw: float
    y: float  # 1 si le joueur ciblé a gagné le TB


@dataclass
class TbCalibrator:
    """Corrige la surconfiance du favori au tie-break (post SPW dédié)."""

    method: str = "platt"
    leaf: _Leaf | None = None
    n_train: int = 0

    def calibrate(self, p_raw: float) -> float:
        p = _clamp01(p_raw)
        if self.leaf is None:
            return p
        return _clamp01(self.leaf.predict(p))

    @staticmethod
    def fit(
        samples: list[TbSample],
        *,
        method: str = "platt",
        min_samples: int = 80,
    ) -> TbCalibrator:
        cal = TbCalibrator(method=method, n_train=len(samples))
        if len(samples) < min_samples:
            return cal
        p = np.array([s.p_raw for s in samples], dtype=float)
        y = np.array([s.y for s in samples], dtype=float)
        if method == "isotonic":
            iso = IsotonicCalibrator._fit_pava(p, y, min_samples=min_samples)
            if iso is not None:
                cal.leaf = _Leaf(method="isotonic", iso=iso)
        else:
            pl = PlattCalibrator.fit(p, y, min_samples=min_samples)
            if pl is not None:
                cal.leaf = _Leaf(method="platt", platt=pl)
        return cal


def tb_raw_p_fav_wins(
    p_serve_winner: float,
    p_serve_loser: float,
    *,
    surface: str,
    fav_is_winner: bool,
) -> float:
    """
    P(favori Elo gagne le TB) avec SPW dédiés, moyenne sur 1er serveur.
    Sans calib (pour constituer l'échantillon de fit).
    """
    from tennis_model import p_win_tiebreak_match

    if fav_is_winner:
        pa, pb = p_serve_winner, p_serve_loser
    else:
        pa, pb = p_serve_loser, p_serve_winner
    return 0.5 * (
        p_win_tiebreak_match(pa, pb, surface=surface, first_server_a=True)
        + p_win_tiebreak_match(pa, pb, surface=surface, first_server_a=False)
    )


def blend_model_market(
    p_model: float,
    p_market: float | None,
    lam: float,
) -> tuple[float, float]:
    """
    p_final = (1-λ)*p_model + λ*p_market (Shin no-vig).
    Retourne (p_final, λ_effectif). Si p_market absent → modèle pur, λ=0.
    """
    pm = _clamp01(p_model)
    if p_market is None:
        return pm, 0.0
    w = max(0.0, min(1.0, float(lam)))
    pk = _clamp01(p_market)
    return (1.0 - w) * pm + w * pk, w


def _self_check() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, size=400)
    y = (rng.uniform(0, 1, size=400) < (p - 0.1).clip(0.05, 0.95)).astype(float)
    iso = IsotonicCalibrator._fit_pava(p, y, min_samples=50)
    assert iso is not None
    assert iso.predict(0.75) < 0.75 + 0.05

    samples = [
        CalibSample(line=22.5, p_raw=float(pi), y_over=float(yi), p_fav=0.55 + 0.2 * (i % 3))
        for i, (pi, yi) in enumerate(zip(p, y))
    ]
    cal = TotalsCalibrator.fit(samples, method="isotonic", use_bands=True, min_samples=40)
    assert cal.leaves
    po, pu = cal.calibrate_under(22.5, 0.7, 0.3, p_fav=0.62, p_push=0.0)
    assert abs(po + pu - 1.0) < 1e-6

    # Sets : chalk sous-estimé → offset positif
    sets_rows = []
    for i in range(300):
        p_fav = 0.75
        p3_raw = 0.25
        y3 = 1.0 if rng.uniform() < 0.32 else 0.0
        sets_rows.append(
            SetsSample(
                p3_raw=p3_raw,
                y_three=y3,
                p_fav=p_fav,
                games=20 if y3 == 0 else 30,
                n_sets=2 if y3 == 0 else 3,
                exp_games_2=19.5,
                exp_games_3=30.0,
            )
        )
    sc = SetsCalibrator.fit(sets_rows, method="band_offset", cond_shift=True, min_samples=40)
    assert sc.band_offset.get("chalk", 0) > 0.02
    p3c = sc.calibrate_p3(0.25, 0.75)
    assert p3c > 0.25

    print(
        "tennis_calib OK",
        f"ou_leaves={len(cal.leaves)}",
        f"p_over(0.7)->{cal.calibrate_over(22.5, 0.7, 0.62):.3f}",
        f"sets chalk offset={sc.band_offset.get('chalk', 0):+.3f} p3={p3c:.3f}",
    )


if __name__ == "__main__":
    _self_check()
