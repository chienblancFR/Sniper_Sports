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


def test_projection_prior_only_debut_saison():
    """GP=0 → projection basée N-1 (et présaison si dispo)."""
    prior = {"name": "Rookie Grow", "team": "BOS", "min": 12.0, "pts": 5.0, "gp": 70}
    pre = {"name": "Rookie Grow", "team": "BOS", "min": 28.0, "pts": 16.0, "gp": 4}
    proj = nba._projeter_stats_joueur(None, prior, pre, 15, 0.35, 0.55)
    assert proj is not None
    # 0.55*28 + 0.45*12 = 15.4+5.4 = 20.8
    assert abs(proj["min"] - 20.8) < 0.05, proj
    assert proj["source"] == "projection"
    assert proj["gp"] == 0


def test_projection_blend_monte_avec_gp():
    """Un bench N-1 qui joue gros minutes en N → MPG projection monte avec GP."""
    prior = {"name": "Breakout", "team": "SAS", "min": 11.0, "pts": 4.0, "gp": 60}
    cur = {"name": "Breakout", "team": "SAS", "min": 30.0, "pts": 15.0, "gp": 8}
    proj0 = nba._projeter_stats_joueur(None, prior, None, 15, 0.35, 0.55)
    proj8 = nba._projeter_stats_joueur(cur, prior, None, 15, 0.35, 0.55)
    assert proj0["min"] < 15
    assert proj8["min"] > proj0["min"]
    assert proj8["blend_w"] == round(8 / 15, 3)
    assert proj8["source"] == "blend"


def test_projection_transfert_downweight_prior():
    prior = {"name": "Tradee", "team": "LAL", "min": 34.0, "pts": 22.0, "gp": 70}
    cur = {"name": "Tradee", "team": "BOS", "min": 0.0, "pts": 0.0, "gp": 0}
    # gp=0 + team change : prior downweighted, pas de cur utile → projection réduite
    proj = nba._projeter_stats_joueur(None, prior, {"name": "Tradee", "team": "BOS", "min": 20.0, "pts": 10.0, "gp": 3}, 15, 0.35, 0.55)
    assert proj["team_changed"] is True
    # prior_min effective 34*0.35=11.9 ; blend 0.55*20 + 0.45*11.9 = 11+5.355 = 16.355
    assert abs(proj["min"] - 16.355) < 0.05, proj


def test_projection_sps_overlay_scoring():
    """SPS (PTS/36) ajuste le PPG via les minutes projetées, sans changer MPG."""
    prior = {"name": "Star", "team": "OKC", "min": 36.0, "pts": 30.0, "gp": 70}
    sps = {"name": "Star", "pts36": 28.0}  # 28 pts / 36 min
    # Sans SPS : PPG = 30
    base = nba._projeter_stats_joueur(None, prior, None, 15, 0.35, 0.55)
    # Avec SPS weight 0.45 : sps_ppg = (36/36)*28 = 28 → 0.55*30 + 0.45*28 = 29.1
    proj = nba._projeter_stats_joueur(
        None, prior, None, 15, 0.35, 0.55, sps=sps, sps_weight=0.45,
    )
    assert abs(base["min"] - 36.0) < 1e-6
    assert abs(proj["min"] - 36.0) < 1e-6  # minutes inchangées
    assert abs(proj["pts"] - 29.1) < 0.05, proj
    assert proj["sps_used"] is True
    assert proj["sps_pts36"] == 28.0


def test_matchs_depuis_odds_cache_fenetre():
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 10, 21, 18, 0, tzinfo=timezone.utc)
    tip_ok = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tip_loin = (now + timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cache = {
        ("BOS", "NYK"): {"commence_time": tip_ok, "event_id": "evt1"},
        ("LAL", "DEN"): {"commence_time": tip_loin, "event_id": "evt2"},
    }
    out = nba._matchs_depuis_odds_cache(cache, maintenant=now)
    assert len(out) == 1
    assert out[0]["domicile"] == "BOS" and out[0]["exterieur"] == "NYK"
    assert out[0]["game_id"] == "evt1"


def test_parse_bref_sps_html_minimal():
    html = """
    <table id="stats"><tbody>
    <tr>
      <td data-stat="player"><a>Test Player</a></td>
      <td data-stat="type">Projected</td>
      <td data-stat="pts_per_mp">22.5</td>
    </tr>
    <tr>
      <td data-stat="player">Skip Me</td>
      <td data-stat="type">Actual</td>
      <td data-stat="pts_per_mp">99.0</td>
    </tr>
    </tbody></table>
    """
    by_name = nba._parse_bref_sps_html(html)
    assert nba._norm_player_name("Test Player") in by_name
    assert by_name[nba._norm_player_name("Test Player")]["pts36"] == 22.5
    assert nba._norm_player_name("Skip Me") not in by_name


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
        test_projection_prior_only_debut_saison,
        test_projection_blend_monte_avec_gp,
        test_projection_transfert_downweight_prior,
        test_projection_sps_overlay_scoring,
        test_matchs_depuis_odds_cache_fenetre,
        test_parse_bref_sps_html_minimal,
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
