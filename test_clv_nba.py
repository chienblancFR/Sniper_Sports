#!/usr/bin/env python3
"""Smoke tests matching/CLV/règlement NBA (lancer : python test_clv_nba.py)."""
from __future__ import annotations

import sys

from nba_sniper import (
    _extraire_cote_clv_nba,
    _parse_pinnacle_game_nba,
    _resoudre_tricode_depuis_nom_odds,
    regler_pari_nba,
)


def test_tricode_nom_complet():
    assert _resoudre_tricode_depuis_nom_odds("Los Angeles Lakers") == "LAL"
    assert _resoudre_tricode_depuis_nom_odds("Boston Celtics") == "BOS"


def test_tricode_alias_clippers():
    """Los Angeles Clippers vs LA Clippers — alias observé côté Odds API."""
    assert _resoudre_tricode_depuis_nom_odds("Los Angeles Clippers") == "LAC"
    assert _resoudre_tricode_depuis_nom_odds("LA Clippers") == "LAC"


def test_tricode_casse_insensible():
    assert _resoudre_tricode_depuis_nom_odds("boston celtics") == "BOS"


def test_tricode_inconnu():
    assert _resoudre_tricode_depuis_nom_odds("Équipe Fantôme") is None
    assert _resoudre_tricode_depuis_nom_odds(None) is None


def test_parse_pinnacle_game_complet():
    game = {
        "home_team": "Boston Celtics",
        "away_team": "Los Angeles Lakers",
        "bookmakers": [{
            "key": "pinnacle",
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Boston Celtics", "price": 1.65},
                    {"name": "Los Angeles Lakers", "price": 2.35},
                ]},
                {"key": "spreads", "outcomes": [
                    {"name": "Boston Celtics", "point": -4.5, "price": 1.91},
                    {"name": "Los Angeles Lakers", "point": 4.5, "price": 1.91},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": 224.5, "price": 1.87},
                    {"name": "Under", "point": 224.5, "price": 1.95},
                ]},
            ],
        }],
    }
    parsed = _parse_pinnacle_game_nba(game)
    assert parsed is not None
    assert parsed["home"] == "BOS"
    assert parsed["away"] == "LAL"
    assert parsed["cote_1"] == 1.65
    assert parsed["cote_2"] == 2.35
    assert parsed["spreads"][-4.5] == {"home": 1.91, "away": 1.91}
    assert parsed["totals"][224.5] == {"over": 1.87, "under": 1.95}


def test_parse_pinnacle_game_equipe_non_resolue():
    game = {
        "home_team": "Équipe Fantôme",
        "away_team": "Los Angeles Lakers",
        "bookmakers": [],
    }
    assert _parse_pinnacle_game_nba(game) is None


def test_parse_pinnacle_game_ignore_bookmaker_non_pinnacle():
    """Aucune cote Pinnacle h2h (que du betfair) → None (pas de ML = pas de match exploitable)."""
    game = {
        "home_team": "Boston Celtics",
        "away_team": "Los Angeles Lakers",
        "bookmakers": [{
            "key": "betfair",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Boston Celtics", "price": 9.99},
            ]}],
        }],
    }
    assert _parse_pinnacle_game_nba(game) is None


def test_parse_pinnacle_game_h2h_seul_sans_spreads_totals():
    """Pinnacle h2h présent mais spreads/totals absents → dict valide, marchés vides."""
    game = {
        "home_team": "Boston Celtics",
        "away_team": "Los Angeles Lakers",
        "bookmakers": [{
            "key": "pinnacle",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Boston Celtics", "price": 1.65},
                {"name": "Los Angeles Lakers", "price": 2.35},
            ]}],
        }],
    }
    parsed = _parse_pinnacle_game_nba(game)
    assert parsed is not None
    assert parsed["cote_1"] == 1.65 and parsed["cote_2"] == 2.35
    assert parsed["spreads"] == {} and parsed["totals"] == {}


def test_regler_moneyline():
    gagne, push = regler_pari_nba("Victoire Boston Celtics", "Boston Celtics", "Los Angeles Lakers", 110, 105)
    assert gagne and not push
    gagne, push = regler_pari_nba("Victoire Los Angeles Lakers", "Boston Celtics", "Los Angeles Lakers", 110, 105)
    assert not gagne and not push


def test_regler_spread_cover_domicile():
    """Boston -4.5 : marge 6 pts > 4.5 → couvert."""
    gagne, push = regler_pari_nba("Boston Celtics -4.5", "Boston Celtics", "Los Angeles Lakers", 110, 104)
    assert gagne and not push


def test_regler_spread_non_couvert_visiteur():
    gagne, push = regler_pari_nba("Los Angeles Lakers +4.5", "Boston Celtics", "Los Angeles Lakers", 110, 104)
    assert not gagne and not push


def test_regler_spread_push_ligne_entiere():
    """Marge exacte 5 sur ligne -5 → push."""
    gagne, push = regler_pari_nba("Boston Celtics -5", "Boston Celtics", "Los Angeles Lakers", 110, 105)
    assert push and not gagne


def test_regler_total_over_under_push():
    gagne, push = regler_pari_nba("OVER 224.5", "Boston Celtics", "Los Angeles Lakers", 115, 110)
    assert gagne and not push
    gagne, push = regler_pari_nba("UNDER 224.5", "Boston Celtics", "Los Angeles Lakers", 115, 110)
    assert not gagne and not push
    gagne, push = regler_pari_nba("OVER 225", "Boston Celtics", "Los Angeles Lakers", 115, 110)
    assert push and not gagne


def test_extraire_cote_clv_moneyline():
    cache = {("BOS", "LAL"): {
        "home": "BOS", "away": "LAL", "cote_1": 1.70, "cote_2": 2.20,
        "spreads": {-4.5: {"home": 1.90, "away": 1.92}}, "totals": {224.5: {"over": 1.85, "under": 1.97}},
    }}
    val = _extraire_cote_clv_nba("BOS", "LAL", "Victoire BOS", "1.65", cache)
    assert val == "1.7"


def test_extraire_cote_clv_total():
    cache = {("BOS", "LAL"): {
        "home": "BOS", "away": "LAL",
        "spreads": {}, "totals": {224.5: {"over": 1.85, "under": 1.97}},
    }}
    val = _extraire_cote_clv_nba("BOS", "LAL", "OVER 224.5", "1.87", cache)
    assert val == "1.85"


def test_extraire_cote_clv_spread_absente_fallback():
    cache = {("BOS", "LAL"): {"home": "BOS", "away": "LAL", "spreads": {}, "totals": {}}}
    val = _extraire_cote_clv_nba("BOS", "LAL", "BOS -4.5", "1.91", cache)
    assert val == "1.91"


def main():
    tests = [
        test_tricode_nom_complet,
        test_tricode_alias_clippers,
        test_tricode_casse_insensible,
        test_tricode_inconnu,
        test_parse_pinnacle_game_complet,
        test_parse_pinnacle_game_equipe_non_resolue,
        test_parse_pinnacle_game_ignore_bookmaker_non_pinnacle,
        test_parse_pinnacle_game_h2h_seul_sans_spreads_totals,
        test_regler_moneyline,
        test_regler_spread_cover_domicile,
        test_regler_spread_non_couvert_visiteur,
        test_regler_spread_push_ligne_entiere,
        test_regler_total_over_under_push,
        test_extraire_cote_clv_moneyline,
        test_extraire_cote_clv_total,
        test_extraire_cote_clv_spread_absente_fallback,
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
    print(f"\n{len(tests)} tests CLV NBA OK")


if __name__ == "__main__":
    main()
