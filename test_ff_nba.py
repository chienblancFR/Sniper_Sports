"""Smoke tests Four Factors → ORtg + matchup (données synthétiques)."""
from __future__ import annotations

import sys

import nba_params
import nba_sniper as nba


def test_ff_to_ortg_league_avg():
    fb = nba_params.get_ff_league_fallback()
    ortg = nba._ff_to_ortg(fb["efg_pct"], fb["tov_pct"], fb["oreb_pct"], fb["fta_rate"])
    # Doit retomber près de la moyenne ligue (~114)
    assert 110.0 < ortg < 120.0, ortg


def test_ff_to_ortg_better_shooting_raises():
    fb = nba_params.get_ff_league_fallback()
    base = nba._ff_to_ortg(fb["efg_pct"], fb["tov_pct"], fb["oreb_pct"], fb["fta_rate"])
    high = nba._ff_to_ortg(fb["efg_pct"] + 0.05, fb["tov_pct"], fb["oreb_pct"], fb["fta_rate"])
    assert high > base + 5.0, (base, high)


def test_matchup_punishes_weak_defense():
    """Attaque moyenne vs défense qui laisse eFG élevé → ORtg matchup ↑."""
    lg = {
        "off_rating": 114.0, "def_rating": 114.0, "pace": 100.0,
        **nba_params.get_ff_league_fallback(),
    }
    fb = nba_params.get_ff_league_fallback()
    rating_off = {**fb, "off_rating": 114.0, "def_rating": 114.0, "pace": 100.0, "gp": 20}
    rating_def_weak = {
        **fb,
        "opp_efg_pct": fb["efg_pct"] + 0.04,  # laisse tirer à eFG élevé
        "off_rating": 114.0, "def_rating": 120.0, "pace": 100.0, "gp": 20,
    }
    rating_def_strong = {
        **fb,
        "opp_efg_pct": fb["efg_pct"] - 0.04,
        "off_rating": 114.0, "def_rating": 108.0, "pace": 100.0, "gp": 20,
    }
    o_weak = nba._matchup_ortg_four_factors(rating_off, rating_def_weak, lg)
    o_strong = nba._matchup_ortg_four_factors(rating_off, rating_def_strong, lg)
    assert o_weak > o_strong + 2.0, (o_weak, o_strong)


def test_calculer_mu_ff_blend_moves_expectation():
    """Avec FF_BLEND=1, une défense poreuse (opp_efg haut) gonfle mu_home vs blend=0."""
    lg = {
        "off_rating": 114.0, "def_rating": 114.0, "pace": 100.0,
        **nba_params.get_ff_league_fallback(),
    }
    fb = nba_params.get_ff_league_fallback()
    home = {
        "off_rating": 114.0, "def_rating": 114.0, "pace": 100.0, "gp": 40,
        **fb,
    }
    away = {
        "off_rating": 114.0, "def_rating": 114.0, "pace": 100.0, "gp": 40,
        **fb,
        "opp_efg_pct": fb["efg_pct"] + 0.05,
        "def_rating": 118.0,
    }
    old = nba_params._tuned_cache
    try:
        nba_params.set_tuned_global_in_memory({
            **nba_params._global_entry(),
            "ff_actif": True, "ff_blend": 0.0, "hca": 0.0,
            "b2b_atk_pct": 0.0, "b2b_def_pct": 0.0, "rest_bonus_pct": 0.0, "altitude_bonus": 0.0,
        })
        mu0_h, _ = nba.calculer_mu_points(home, away, lg, hca=0.0)
        nba_params.set_tuned_global_in_memory({
            **nba_params._global_entry(),
            "ff_actif": True, "ff_blend": 1.0, "hca": 0.0,
            "b2b_atk_pct": 0.0, "b2b_def_pct": 0.0, "rest_bonus_pct": 0.0, "altitude_bonus": 0.0,
        })
        mu1_h, _ = nba.calculer_mu_points(home, away, lg, hca=0.0)
        # Chemin FF pur doit donner un mu_home différent (défense poreuse sur eFG)
        assert abs(mu1_h - mu0_h) > 0.5, (mu0_h, mu1_h)
    finally:
        nba_params._tuned_cache = old


def test_stats_decay_includes_ff():
    games = [
        {
            "date": "2024-01-02", "off_rating": 110, "def_rating": 108, "pace": 100,
            "poss": 100, "efg_pct": 0.55, "tov_pct": 0.14, "oreb_pct": 0.28, "fta_rate": 0.25,
            "opp_efg_pct": 0.52, "opp_tov_pct": 0.13, "opp_oreb_pct": 0.27, "opp_fta_rate": 0.24,
            "game_id": "g1", "team": "AAA", "opponent": "BBB", "domicile": True,
        },
        {
            "date": "2024-01-04", "off_rating": 112, "def_rating": 110, "pace": 101,
            "poss": 100, "efg_pct": 0.57, "tov_pct": 0.12, "oreb_pct": 0.30, "fta_rate": 0.26,
            "opp_efg_pct": 0.53, "opp_tov_pct": 0.14, "opp_oreb_pct": 0.26, "opp_fta_rate": 0.22,
            "game_id": "g2", "team": "AAA", "opponent": "CCC", "domicile": False,
        },
    ]
    s = nba._stats_brutes_decay_equipe(games, "2024-01-05", half_life_jours=0)
    assert s is not None
    assert "efg_pct" in s and abs(s["efg_pct"] - 0.56) < 0.01
    assert "opp_efg_pct" in s


def main():
    tests = [
        test_ff_to_ortg_league_avg,
        test_ff_to_ortg_better_shooting_raises,
        test_matchup_punishes_weak_defense,
        test_calculer_mu_ff_blend_moves_expectation,
        test_stats_decay_includes_ff,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"OK  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}", file=sys.stderr)
    if failed:
        raise SystemExit(1)
    print(f"\n{len(tests)} tests Four Factors NBA OK")


if __name__ == "__main__":
    main()
