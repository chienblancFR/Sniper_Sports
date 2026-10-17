"""
Stats service / retour PIT (Sackmann).

Modèle matchup (défaut) — split 1er / 2e + effets ajustés adversaire :
  f = first-in (serveur, shrinké)
  p1 = μ1 + scale*(α1_server − γ1_returner)
  p2 = μ2 + scale*(α2_server − γ2_returner)
  p  = f*p1 + (1-f)*p2

Fallback : SPW agrégé μ + α − γ (si split off / stats incomplètes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tennis_data import AtpMatch, MatchServeStats
from tennis_params import (
    SURFACES,
    TENNIS_SERVE_EFFECT_SCALE,
    TENNIS_SERVE_ELO_SPREAD_WEIGHT,
    TENNIS_SERVE_N_PRIOR,
    TENNIS_SERVE_OPP_ADJ,
    TENNIS_SERVE_SPLIT,
    TENNIS_SERVE_SURFACE_BLEND,
    mu_1st_in_for_surface,
    mu_1st_won_for_surface,
    mu_2nd_won_for_surface,
    mu_serve_for_surface,
    normalize_surface,
)


def _clamp(p: float, lo: float = 0.50, hi: float = 0.85) -> float:
    return max(lo, min(hi, float(p)))


def _clamp_f(p: float, lo: float = 0.45, hi: float = 0.80) -> float:
    return max(lo, min(hi, float(p)))


def _clamp_p1(p: float) -> float:
    return max(0.45, min(0.92, float(p)))


def _clamp_p2(p: float) -> float:
    return max(0.30, min(0.75, float(p)))


@dataclass
class _Bag:
    """Compteurs bruts (diag + sample size + split 1er/2e)."""

    serve_won: float = 0.0
    serve_total: float = 0.0
    return_won: float = 0.0
    return_total: float = 0.0
    first_in: float = 0.0
    first_att: float = 0.0  # = serve points with known 1stIn
    first_won: float = 0.0
    first_won_att: float = 0.0  # 1stIn count for p1
    second_won: float = 0.0
    second_att: float = 0.0

    def add_serve(self, won: int, total: int) -> None:
        if total > 0:
            self.serve_won += won
            self.serve_total += total

    def add_return(self, won: int, total: int) -> None:
        if total > 0:
            self.return_won += won
            self.return_total += total

    def add_split(
        self,
        *,
        first_in: int,
        svpt: int,
        first_won: int,
        second_won: int,
        second_att: int,
    ) -> None:
        if svpt <= 0:
            return
        self.first_in += first_in
        self.first_att += svpt
        if first_in > 0:
            self.first_won += first_won
            self.first_won_att += first_in
        if second_att > 0:
            self.second_won += second_won
            self.second_att += second_att

    def spw_raw(self) -> float | None:
        if self.serve_total <= 0:
            return None
        return self.serve_won / self.serve_total

    def rpw_raw(self) -> float | None:
        if self.return_total <= 0:
            return None
        return self.return_won / self.return_total

    def first_in_raw(self) -> float | None:
        if self.first_att <= 0:
            return None
        return self.first_in / self.first_att

    def first_won_raw(self) -> float | None:
        if self.first_won_att <= 0:
            return None
        return self.first_won / self.first_won_att

    def second_won_raw(self) -> float | None:
        if self.second_att <= 0:
            return None
        return self.second_won / self.second_att


@dataclass
class _Effects:
    """
    Effets additifs (prior N(0) via précision = n_prior).
    alpha > 0 : serveur au-dessus de μ
    gamma > 0 : retourneur au-dessus de (1−μ)  → abaisse le SPW adverse
    Suffixe 1/2 : 1er / 2e balle.
    """

    alpha: float = 0.0
    gamma: float = 0.0
    prec_alpha: float = 0.0
    prec_gamma: float = 0.0
    alpha1: float = 0.0
    gamma1: float = 0.0
    prec_a1: float = 0.0
    prec_g1: float = 0.0
    alpha2: float = 0.0
    gamma2: float = 0.0
    prec_a2: float = 0.0
    prec_g2: float = 0.0


def _shrink(raw: float | None, n: float, mu: float, n_prior: float) -> float:
    if n <= 0 or raw is None:
        return mu
    return (raw * n + mu * n_prior) / (n + n_prior)


@dataclass
class ServeStatsBook:
    """Compteurs bruts + effets serveur/retourneur (global + surface)."""

    n_prior: float = TENNIS_SERVE_N_PRIOR
    surface_blend: float = TENNIS_SERVE_SURFACE_BLEND
    opp_adj: bool = TENNIS_SERVE_OPP_ADJ
    effect_scale: float = TENNIS_SERVE_EFFECT_SCALE
    use_split: bool = TENNIS_SERVE_SPLIT
    global_bag: dict[str, _Bag] = field(default_factory=dict)
    surface_bag: dict[str, dict[str, _Bag]] = field(
        default_factory=lambda: {s: {} for s in SURFACES}
    )
    global_fx: dict[str, _Effects] = field(default_factory=dict)
    surface_fx: dict[str, dict[str, _Effects]] = field(
        default_factory=lambda: {s: {} for s in SURFACES}
    )

    def _bag(self, player_id: str, surface: str | None) -> _Bag:
        if surface is None:
            return self.global_bag.setdefault(player_id, _Bag())
        s = surface if surface in self.surface_bag else "Hard"
        return self.surface_bag.setdefault(s, {}).setdefault(player_id, _Bag())

    def _fx(self, player_id: str, surface: str | None) -> _Effects:
        prior = float(self.n_prior)
        if surface is None:
            fx = self.global_fx.setdefault(player_id, _Effects())
        else:
            s = surface if surface in self.surface_fx else "Hard"
            fx = self.surface_fx.setdefault(s, {}).setdefault(player_id, _Effects())
        if fx.prec_alpha <= 0:
            fx.prec_alpha = prior
        if fx.prec_gamma <= 0:
            fx.prec_gamma = prior
        if fx.prec_a1 <= 0:
            fx.prec_a1 = prior
        if fx.prec_g1 <= 0:
            fx.prec_g1 = prior
        if fx.prec_a2 <= 0:
            fx.prec_a2 = prior
        if fx.prec_g2 <= 0:
            fx.prec_g2 = prior
        return fx

    def _blend_effects(self, player_id: str, surface: str) -> tuple[float, float]:
        s = normalize_surface(surface)
        b = max(0.0, min(1.0, float(self.surface_blend)))
        g = self._fx(player_id, None)
        surf = self._fx(player_id, s)
        alpha = b * surf.alpha + (1.0 - b) * g.alpha
        gamma = b * surf.gamma + (1.0 - b) * g.gamma
        return alpha, gamma

    def _blend_split_effects(
        self, player_id: str, surface: str
    ) -> tuple[float, float, float, float]:
        """(α1, γ1, α2, γ2) blendés surface/global."""
        s = normalize_surface(surface)
        b = max(0.0, min(1.0, float(self.surface_blend)))
        g = self._fx(player_id, None)
        surf = self._fx(player_id, s)
        a1 = b * surf.alpha1 + (1.0 - b) * g.alpha1
        g1 = b * surf.gamma1 + (1.0 - b) * g.gamma1
        a2 = b * surf.alpha2 + (1.0 - b) * g.alpha2
        g2 = b * surf.gamma2 + (1.0 - b) * g.gamma2
        return a1, g1, a2, g2

    def _first_in_shrunk(self, player_id: str, surface: str) -> float:
        s = normalize_surface(surface)
        mu_f = mu_1st_in_for_surface(s)
        n_prior = float(self.n_prior)
        b = max(0.0, min(1.0, float(self.surface_blend)))
        g = self._bag(player_id, None)
        surf = self._bag(player_id, s)
        f_g = _shrink(g.first_in_raw(), g.first_att, mu_f, n_prior)
        f_s = _shrink(surf.first_in_raw(), surf.first_att, mu_f, n_prior)
        return _clamp_f(b * f_s + (1.0 - b) * f_g)

    @staticmethod
    def _update_pair(fx_s: _Effects, fx_r: _Effects, y: float, n: int, mu: float) -> None:
        """Observation SPW/p1/p2=y sur n points : y ≈ μ + α_s − γ_r."""
        if n <= 0:
            return
        n_f = float(n)
        target_a = y - mu + fx_r.gamma
        new_pa = fx_s.prec_alpha + n_f
        fx_s.alpha = (fx_s.alpha * fx_s.prec_alpha + target_a * n_f) / new_pa
        fx_s.prec_alpha = new_pa
        target_g = mu + fx_s.alpha - y
        new_pg = fx_r.prec_gamma + n_f
        fx_r.gamma = (fx_r.gamma * fx_r.prec_gamma + target_g * n_f) / new_pg
        fx_r.prec_gamma = new_pg

    @staticmethod
    def _update_pair_split(
        fx_s: _Effects,
        fx_r: _Effects,
        y: float,
        n: int,
        mu: float,
        *,
        which: str,
    ) -> None:
        if n <= 0:
            return
        n_f = float(n)
        if which == "1":
            target_a = y - mu + fx_r.gamma1
            new_pa = fx_s.prec_a1 + n_f
            fx_s.alpha1 = (fx_s.alpha1 * fx_s.prec_a1 + target_a * n_f) / new_pa
            fx_s.prec_a1 = new_pa
            target_g = mu + fx_s.alpha1 - y
            new_pg = fx_r.prec_g1 + n_f
            fx_r.gamma1 = (fx_r.gamma1 * fx_r.prec_g1 + target_g * n_f) / new_pg
            fx_r.prec_g1 = new_pg
        else:
            target_a = y - mu + fx_r.gamma2
            new_pa = fx_s.prec_a2 + n_f
            fx_s.alpha2 = (fx_s.alpha2 * fx_s.prec_a2 + target_a * n_f) / new_pa
            fx_s.prec_a2 = new_pa
            target_g = mu + fx_s.alpha2 - y
            new_pg = fx_r.prec_g2 + n_f
            fx_r.gamma2 = (fx_r.gamma2 * fx_r.prec_g2 + target_g * n_f) / new_pg
            fx_r.prec_g2 = new_pg

    def update_from_stats(
        self,
        winner_id: str,
        loser_id: str,
        surface: str,
        serve: MatchServeStats,
    ) -> None:
        """MAJ après match (PIT : appeler seulement après la projection)."""
        if not serve.valid:
            return
        s = normalize_surface(surface)
        mu = mu_serve_for_surface(s)

        w_ret = serve.l_svpt - serve.l_serve_won
        l_ret = serve.w_svpt - serve.w_serve_won
        for bag in (self._bag(winner_id, None), self._bag(winner_id, s)):
            bag.add_serve(serve.w_serve_won, serve.w_svpt)
            bag.add_return(w_ret, serve.l_svpt)
        for bag in (self._bag(loser_id, None), self._bag(loser_id, s)):
            bag.add_serve(serve.l_serve_won, serve.l_svpt)
            bag.add_return(l_ret, serve.w_svpt)

        y_w = serve.w_serve_won / serve.w_svpt
        y_l = serve.l_serve_won / serve.l_svpt
        for scope in (None, s):
            self._update_pair(
                self._fx(winner_id, scope),
                self._fx(loser_id, scope),
                y_w,
                serve.w_svpt,
                mu,
            )
            self._update_pair(
                self._fx(loser_id, scope),
                self._fx(winner_id, scope),
                y_l,
                serve.l_svpt,
                mu,
            )

        if not (self.use_split and serve.has_split):
            return

        assert serve.w_1st_in is not None and serve.l_1st_in is not None
        assert serve.w_1st_won is not None and serve.l_1st_won is not None
        assert serve.w_2nd_won is not None and serve.l_2nd_won is not None

        for bag in (self._bag(winner_id, None), self._bag(winner_id, s)):
            bag.add_split(
                first_in=serve.w_1st_in,
                svpt=serve.w_svpt,
                first_won=serve.w_1st_won,
                second_won=serve.w_2nd_won,
                second_att=serve.w_2nd_att,
            )
        for bag in (self._bag(loser_id, None), self._bag(loser_id, s)):
            bag.add_split(
                first_in=serve.l_1st_in,
                svpt=serve.l_svpt,
                first_won=serve.l_1st_won,
                second_won=serve.l_2nd_won,
                second_att=serve.l_2nd_att,
            )

        mu1 = mu_1st_won_for_surface(s)
        mu2 = mu_2nd_won_for_surface(s)
        # Winner sert vs loser
        if serve.w_1st_in > 0:
            y1_w = serve.w_1st_won / serve.w_1st_in
            for scope in (None, s):
                self._update_pair_split(
                    self._fx(winner_id, scope),
                    self._fx(loser_id, scope),
                    y1_w,
                    serve.w_1st_in,
                    mu1,
                    which="1",
                )
        if serve.w_2nd_att > 0:
            y2_w = serve.w_2nd_won / serve.w_2nd_att
            for scope in (None, s):
                self._update_pair_split(
                    self._fx(winner_id, scope),
                    self._fx(loser_id, scope),
                    y2_w,
                    serve.w_2nd_att,
                    mu2,
                    which="2",
                )
        # Loser sert vs winner
        if serve.l_1st_in > 0:
            y1_l = serve.l_1st_won / serve.l_1st_in
            for scope in (None, s):
                self._update_pair_split(
                    self._fx(loser_id, scope),
                    self._fx(winner_id, scope),
                    y1_l,
                    serve.l_1st_in,
                    mu1,
                    which="1",
                )
        if serve.l_2nd_att > 0:
            y2_l = serve.l_2nd_won / serve.l_2nd_att
            for scope in (None, s):
                self._update_pair_split(
                    self._fx(loser_id, scope),
                    self._fx(winner_id, scope),
                    y2_l,
                    serve.l_2nd_att,
                    mu2,
                    which="2",
                )

    def update_after_match(self, match: AtpMatch) -> None:
        if match.serve is None or not match.serve.valid:
            return
        self.update_from_stats(
            match.winner_id, match.loser_id, match.surface, match.serve
        )

    def _rates_raw_shrunk(
        self, player_id: str, surface: str
    ) -> tuple[float, float, float, float]:
        """Legacy SPW/RPW shrinkés (sans découplage adversaire)."""
        s = normalize_surface(surface)
        mu_s = mu_serve_for_surface(s)
        mu_r = 1.0 - mu_s
        n_prior = float(self.n_prior)
        b = max(0.0, min(1.0, float(self.surface_blend)))

        g = self._bag(player_id, None)
        surf = self._bag(player_id, s)

        spw_g = _shrink(g.spw_raw(), g.serve_total, mu_s, n_prior)
        spw_s = _shrink(surf.spw_raw(), surf.serve_total, mu_s, n_prior)
        rpw_g = _shrink(g.rpw_raw(), g.return_total, mu_r, n_prior)
        rpw_s = _shrink(surf.rpw_raw(), surf.return_total, mu_r, n_prior)

        spw = b * spw_s + (1.0 - b) * spw_g
        rpw = b * rpw_s + (1.0 - b) * rpw_g
        n_sv = b * surf.serve_total + (1.0 - b) * g.serve_total
        n_rt = b * surf.return_total + (1.0 - b) * g.return_total
        return spw, rpw, n_sv, n_rt

    def _matchup_split(
        self, player_a: str, player_b: str, surface: str
    ) -> tuple[float, float]:
        s = normalize_surface(surface)
        mu1 = mu_1st_won_for_surface(s)
        mu2 = mu_2nd_won_for_surface(s)
        scale = max(0.0, float(self.effect_scale))
        f_a = self._first_in_shrunk(player_a, s)
        f_b = self._first_in_shrunk(player_b, s)
        a1_a, g1_a, a2_a, g2_a = self._blend_split_effects(player_a, s)
        a1_b, g1_b, a2_b, g2_b = self._blend_split_effects(player_b, s)
        p1_a = _clamp_p1(mu1 + scale * (a1_a - g1_b))
        p2_a = _clamp_p2(mu2 + scale * (a2_a - g2_b))
        p1_b = _clamp_p1(mu1 + scale * (a1_b - g1_a))
        p2_b = _clamp_p2(mu2 + scale * (a2_b - g2_a))
        p_a = _clamp(f_a * p1_a + (1.0 - f_a) * p2_a)
        p_b = _clamp(f_b * p1_b + (1.0 - f_b) * p2_b)
        return p_a, p_b

    def matchup_serve_probs(
        self, player_a: str, player_b: str, surface: str
    ) -> tuple[float, float]:
        """
        Probas de point gagné au service A vs B.
        Split : f*p1+(1-f)*p2 avec α1/γ1, α2/γ2
        Opp-adj agrégé : μ + α_A − γ_B
        Legacy : spw − (rpw_opp − μ_r)
        """
        s = normalize_surface(surface)
        mu_s = mu_serve_for_surface(s)

        if self.use_split and self.opp_adj:
            return self._matchup_split(player_a, player_b, s)

        if self.opp_adj:
            a_a, g_a = self._blend_effects(player_a, s)
            a_b, g_b = self._blend_effects(player_b, s)
            scale = max(0.0, float(self.effect_scale))
            p_a = _clamp(mu_s + scale * (a_a - g_b))
            p_b = _clamp(mu_s + scale * (a_b - g_a))
            return p_a, p_b

        mu_r = 1.0 - mu_s
        spw_a, rpw_a, _, _ = self._rates_raw_shrunk(player_a, s)
        spw_b, rpw_b, _, _ = self._rates_raw_shrunk(player_b, s)
        p_a = _clamp(spw_a - (rpw_b - mu_r))
        p_b = _clamp(spw_b - (rpw_a - mu_r))
        if abs(p_a - mu_s) < 1e-15 and abs(p_b - mu_s) < 1e-15:
            return mu_s, mu_s
        return p_a, p_b

    def matchup_serve_probs_engine(
        self,
        player_a: str,
        player_b: str,
        surface: str,
        elo_a: float,
        elo_b: float,
        *,
        elo_spread_weight: float | None = None,
    ) -> tuple[float, float]:
        """
        Serve pour le moteur totaux.
        Niveau + écart issus du matchup (opp-adj / split) ; optionnellement
        mélange d'écart avec le proxy Elo (`TENNIS_SERVE_ELO_SPREAD_WEIGHT`).
        """
        from tennis_model import elo_to_serve_probs
        from tennis_params import TENNIS_ELO_POINT_SCALE, TENNIS_POINT_SENSITIVITY

        s = normalize_surface(surface)
        mu = mu_serve_for_surface(s)
        pa_s, pb_s = self.matchup_serve_probs(player_a, player_b, s)
        w = float(
            TENNIS_SERVE_ELO_SPREAD_WEIGHT
            if elo_spread_weight is None
            else elo_spread_weight
        )
        w = max(0.0, min(1.0, w))
        if w <= 0.0:
            return pa_s, pb_s

        pa_e, pb_e = elo_to_serve_probs(
            elo_a,
            elo_b,
            mu_serve=mu,
            elo_point_scale=TENNIS_ELO_POINT_SCALE,
            point_sensitivity=TENNIS_POINT_SENSITIVITY,
        )
        mid = 0.5 * (pa_s + pb_s)
        spread_s = 0.5 * (pa_s - pb_s)
        spread_e = 0.5 * (pa_e - pb_e)
        spread = (1.0 - w) * spread_s + w * spread_e
        return _clamp(mid + spread), _clamp(mid - spread)

    def sample_size(self, player_id: str, surface: str) -> float:
        """Points de service effectifs (blend) — utile pour logs / gates."""
        _, _, n_sv, _ = self._rates_raw_shrunk(player_id, surface)
        return n_sv

    def effects(self, player_id: str, surface: str) -> tuple[float, float]:
        """(alpha, gamma) agrégés blendés — diag / tests."""
        return self._blend_effects(player_id, surface)

    def split_components(
        self, player_a: str, player_b: str, surface: str
    ) -> dict[str, float]:
        """Diag : f, p1, p2 et p pour A vs B."""
        s = normalize_surface(surface)
        mu1 = mu_1st_won_for_surface(s)
        mu2 = mu_2nd_won_for_surface(s)
        scale = max(0.0, float(self.effect_scale))
        f_a = self._first_in_shrunk(player_a, s)
        a1_a, _g1_a, a2_a, _g2_a = self._blend_split_effects(player_a, s)
        _a1_b, g1_b, _a2_b, g2_b = self._blend_split_effects(player_b, s)
        p1_a = _clamp_p1(mu1 + scale * (a1_a - g1_b))
        p2_a = _clamp_p2(mu2 + scale * (a2_a - g2_b))
        return {
            "f": f_a,
            "p1": p1_a,
            "p2": p2_a,
            "p": _clamp(f_a * p1_a + (1.0 - f_a) * p2_a),
        }


def _self_check() -> None:
    from config_env import load_project_env

    load_project_env("tennis")
    book = ServeStatsBook(
        n_prior=100.0, surface_blend=0.7, opp_adj=True, use_split=True
    )
    pa, pb = book.matchup_serve_probs("a", "b", "Hard")
    assert abs(pa - mu_serve_for_surface("Hard")) < 0.02, (pa, pb)

    # Gros 1er service pour a, 2e faible pour b
    stats = MatchServeStats(
        w_svpt=100,
        w_serve_won=78,
        l_svpt=100,
        l_serve_won=55,
        w_1st_in=70,
        w_1st_won=60,
        w_2nd_won=18,
        l_1st_in=55,
        l_1st_won=35,
        l_2nd_won=20,
    )
    assert stats.has_split
    book.update_from_stats("a", "b", "Hard", stats)
    pa2, pb2 = book.matchup_serve_probs("a", "b", "Hard")
    assert pa2 > pb2, (pa2, pb2)
    comp = book.split_components("a", "b", "Hard")
    assert comp["p1"] > comp["p2"], comp
    assert abs(comp["p"] - pa2) < 1e-9

    # Legacy agrégé
    legacy = ServeStatsBook(
        n_prior=100.0, surface_blend=0.7, opp_adj=True, use_split=False
    )
    legacy.update_from_stats("a", "b", "Hard", stats)
    pl_a, pl_b = legacy.matchup_serve_probs("a", "b", "Hard")
    assert pl_a > pl_b

    print(
        "tennis_serve OK",
        f"split {pa2:.3f}/{pb2:.3f} f={comp['f']:.3f} p1={comp['p1']:.3f} p2={comp['p2']:.3f}",
        f"agg {pl_a:.3f}/{pl_b:.3f}",
    )


if __name__ == "__main__":
    _self_check()
