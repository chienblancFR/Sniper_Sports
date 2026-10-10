"""Smoke tests gate lineups ESPN + proxy impact absences (sans réseau)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import nba_sniper as nba


def test_espn_abbrev_mapping():
    assert nba._espn_abbrev_to_nba("GS") == "GSW"
    assert nba._espn_abbrev_to_nba("NO") == "NOP"
    assert nba._espn_abbrev_to_nba("DAL") == "DAL"


def test_norm_player_name():
    assert nba._norm_player_name("P.J. Washington") == nba._norm_player_name("PJ Washington")
    assert nba._norm_player_name("Nikola Jokić") != ""


def test_lineups_confirmes_starters():
    tip = datetime.now(timezone.utc) + timedelta(hours=5)
    info = {
        "fetch_ok": True,
        "injuries_fetched": True,
        "starters_home": 5,
        "starters_away": 5,
    }
    assert nba.lineups_sont_confirmes_nba(info, tip) is True


def test_lineups_confirmes_fenetre_minutes():
    tip = datetime.now(timezone.utc) + timedelta(minutes=30)
    info = {
        "fetch_ok": True,
        "injuries_fetched": True,
        "starters_home": 0,
        "starters_away": 0,
    }
    assert nba.lineups_sont_confirmes_nba(info, tip) is True


def test_lineups_pas_confirmes_trop_tot():
    tip = datetime.now(timezone.utc) + timedelta(hours=6)
    info = {
        "fetch_ok": True,
        "injuries_fetched": True,
        "starters_home": 0,
        "starters_away": 0,
    }
    assert nba.lineups_sont_confirmes_nba(info, tip) is False


def test_lineups_fetch_fail():
    tip = datetime.now(timezone.utc) + timedelta(minutes=10)
    assert nba.lineups_sont_confirmes_nba({"fetch_ok": False}, tip) is False


def test_impact_absence_star():
    stats = {
        nba._norm_player_name("Star Player"): {
            "name": "Star Player", "team": "BOS", "min": 36.0, "pts": 28.0, "gp": 40,
        }
    }
    # (36/48)*(28-8)*1.0 = 15.0
    impact, _ = nba._impact_absence_joueur("Star Player", 1.0, stats)
    assert abs(impact - 15.0) < 1e-6, impact


def test_impact_ignore_bench():
    stats = {
        nba._norm_player_name("Bench"): {
            "name": "Bench", "team": "BOS", "min": 12.0, "pts": 6.0, "gp": 40,
        }
    }
    impact, _ = nba._impact_absence_joueur("Bench", 1.0, stats)
    assert impact == 0.0


def test_calculer_impact_absences_sides():
    old = nba.NBA_ABSENCES_ACTIF
    nba.NBA_ABSENCES_ACTIF = True
    try:
        stats = {
            nba._norm_player_name("Big Star"): {
                "name": "Big Star", "team": "LAL", "min": 34.0, "pts": 26.0, "gp": 50,
            }
        }
        absences = {"LAL": [("Big Star", 1.0, "OUT")], "BOS": []}
        d_h, d_a, detail = nba.calculer_impact_absences_nba("BOS", "LAL", absences, player_stats=stats)
        # Away (LAL) missing star → mu_away down, mu_home up slightly
        assert d_a < 0 and d_h > 0, (d_h, d_a)
        assert "LAL:Big Star" in detail
    finally:
        nba.NBA_ABSENCES_ACTIF = old


def test_ajuster_probas_apres_absences():
    probas = {
        "mu_home": 110.0, "mu_away": 108.0,
        "sigma_home": 12.0, "sigma_away": 12.0, "rho": 0.2,
        "moneyline": {"domicile": 0.55, "exterieur": 0.45},
        "spread": {}, "total": {},
        "rating_home": {"gp": 10}, "rating_away": {"gp": 10},
    }
    out = nba.ajuster_probas_apres_absences_nba(probas, -5.0, 2.0, lignes_spread=[-3.5], lignes_total=[220.5])
    assert out["mu_home"] == 105.0
    assert out["mu_away"] == 110.0
    assert "moneyline" in out and -3.5 in out["spread"]


def main():
    tests = [
        test_espn_abbrev_mapping,
        test_norm_player_name,
        test_lineups_confirmes_starters,
        test_lineups_confirmes_fenetre_minutes,
        test_lineups_pas_confirmes_trop_tot,
        test_lineups_fetch_fail,
        test_impact_absence_star,
        test_impact_ignore_bench,
        test_calculer_impact_absences_sides,
        test_ajuster_probas_apres_absences,
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
    print(f"\n{len(tests)} tests lineups/absences NBA OK")


if __name__ == "__main__":
    main()
