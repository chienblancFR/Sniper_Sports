#!/usr/bin/env python3
"""Smoke tests matching CLV (lancer : python test_clv_foot.py)."""
from __future__ import annotations

import sys

from sniper_bot_foot import (
    _equipe_clv_match,
    _formater_pari_clv,
    _fusionner_marches_pinnacle,
    _ligne_clv_exacte,
    _normaliser_equipe_clv,
    _resoudre_pari_clv,
    _trouver_event_clv,
    _trouver_outcome_clv,
)


def _market_totals(outcomes):
    return {"key": "totals", "outcomes": outcomes}


def _market_spreads(outcomes):
    return {"key": "spreads", "outcomes": outcomes}


def test_normaliser_orgryte():
    assert _normaliser_equipe_clv("Örgryte IS") == _normaliser_equipe_clv("Orgryte IS")


def test_equipe_mjallby_match():
    assert _equipe_clv_match("Mjällby AIF", "Mjällby AIF", "Mjällby AIF", "AIK")
    assert _equipe_clv_match("Mjällby AIF", "Mjallby", "Mjällby AIF", "AIK")


def test_formater_pari_totaux():
    assert _formater_pari_clv("Under", 2.75, "totals") == "Under 2.75"
    assert _formater_pari_clv("Over", 3.25, "totals") == "Over 3.25"


def test_formater_pari_ah():
    assert _formater_pari_clv("Malmo FF", -0.5, "spreads") == "Malmo FF (-0.5)"


def test_ligne_clv_exacte_quarts():
    assert _ligne_clv_exacte(2.75, 2.7500001)
    assert _ligne_clv_exacte(0.25, 0.25)
    assert not _ligne_clv_exacte(2.75, 3.0)


def test_totals_ligne_exacte_parmi_plusieurs():
    """Under 2.75 trouvé même si Pinnacle propose aussi 2.5 et 3."""
    market = _market_totals([
        {"name": "Under", "point": 2.5, "price": 2.10},
        {"name": "Over", "point": 2.5, "price": 1.75},
        {"name": "Under", "point": 2.75, "price": 1.84},
        {"name": "Over", "point": 2.75, "price": 1.98},
        {"name": "Under", "point": 3.0, "price": 1.65},
        {"name": "Over", "point": 3.0, "price": 2.25},
    ])
    o = _trouver_outcome_clv(market, "Under", 2.75, "totals", "Kalmar FF", "Orgryte IS")
    assert o is not None
    assert float(o["point"]) == 2.75
    assert float(o["price"]) == 1.84


def test_spread_mjallby_fallback_moins_05():
    """AH foot : -0.5 retiré, repli sur -0.25 (1 ligne principale API)."""
    market = _market_spreads([
        {"name": "AIK", "point": 0.25, "price": 1.97},
        {"name": "Mjällby AIF", "point": -0.25, "price": 1.90},
    ])
    o = _trouver_outcome_clv(
        market, "Mjällby AIF", -0.5, "spreads", "Mjällby AIF", "AIK",
    )
    assert o is not None
    assert float(o["point"]) == -0.25
    assert float(o["price"]) == 1.90


def test_fusion_alternate_totaux_under_3():
    """Bulk = 3.25 seulement ; alternate_totals contient Under 3."""
    pinnacle = {
        'key': 'pinnacle',
        'markets': [
            {'key': 'totals', 'outcomes': [
                {'name': 'Over', 'point': 3.25, 'price': 1.99},
                {'name': 'Under', 'point': 3.25, 'price': 1.89},
            ]},
            {'key': 'alternate_totals', 'outcomes': [
                {'name': 'Under', 'point': 3.0, 'price': 2.17},
                {'name': 'Over', 'point': 3.0, 'price': 1.72},
                {'name': 'Under', 'point': 3.25, 'price': 1.89},
            ]},
        ],
    }
    market = _fusionner_marches_pinnacle(pinnacle, 'totals')
    o = _trouver_outcome_clv(market, 'Under', 3.0, 'totals', 'Aalesund', 'Molde')
    assert o is not None
    assert float(o['point']) == 3.0
    assert float(o['price']) == 2.17


def test_fusion_alternate_spread_moins_05():
    pinnacle = {
        'key': 'pinnacle',
        'markets': [
            {'key': 'spreads', 'outcomes': [
                {'name': 'AIK', 'point': 0.25, 'price': 1.97},
                {'name': 'Mjällby AIF', 'point': -0.25, 'price': 1.90},
            ]},
            {'key': 'alternate_spreads', 'outcomes': [
                {'name': 'Mjällby AIF', 'point': -0.5, 'price': 2.10},
                {'name': 'AIK', 'point': 0.5, 'price': 1.75},
            ]},
        ],
    }
    market = _fusionner_marches_pinnacle(pinnacle, 'spreads')
    o = _trouver_outcome_clv(
        market, 'Mjällby AIF', -0.5, 'spreads', 'Mjällby AIF', 'AIK',
    )
    assert o is not None
    assert float(o['point']) == -0.5


def test_event_kalmar_orgryte_ligne_exacte_disponible():
    data = [{
        "home_team": "Kalmar",
        "away_team": "Orgryte IS",
        "bookmakers": [{
            "key": "pinnacle",
            "markets": [_market_totals([
                {"name": "Under", "point": 2.5, "price": 2.01},
                {"name": "Over", "point": 2.5, "price": 1.85},
                {"name": "Under", "point": 2.75, "price": 1.92},
                {"name": "Over", "point": 2.75, "price": 1.90},
            ])],
        }],
    }]
    event, err = _trouver_event_clv(data, "Kalmar FF", "Örgryte IS")
    assert err is None
    assert event is not None
    outcome, _, _, err2 = _resoudre_pari_clv(
        data, "Kalmar FF", "Örgryte IS", "Under", 2.75, "totals",
    )
    assert err2 is None
    assert outcome is not None
    assert float(outcome["price"]) == 1.92


def test_goteborg_aik_ligne_exacte():
    data = [{
        "home_team": "IFK Goteborg",
        "away_team": "AIK Stockholm",
        "bookmakers": [{
            "key": "pinnacle",
            "markets": [_market_totals([
                {"name": "Under", "point": 2.75, "price": 1.93},
                {"name": "Over", "point": 2.75, "price": 1.93},
            ])],
        }],
    }]
    outcome, _, _, err = _resoudre_pari_clv(
        data, "IFK Goteborg", "AIK Stockholm", "Under", 2.75, "totals",
    )
    assert err is None
    assert float(outcome["price"]) == 1.93


def main():
    tests = [
        test_normaliser_orgryte,
        test_equipe_mjallby_match,
        test_formater_pari_totaux,
        test_formater_pari_ah,
        test_ligne_clv_exacte_quarts,
        test_totals_ligne_exacte_parmi_plusieurs,
        test_spread_mjallby_fallback_moins_05,
        test_fusion_alternate_totaux_under_3,
        test_fusion_alternate_spread_moins_05,
        test_event_kalmar_orgryte_ligne_exacte_disponible,
        test_goteborg_aik_ligne_exacte,
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
    print(f"\n{len(tests)} tests CLV OK")


if __name__ == "__main__":
    main()
