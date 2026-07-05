#!/usr/bin/env python3
"""Smoke tests matching CLV (lancer : python test_clv_foot.py)."""
from __future__ import annotations

import sys

from sniper_bot_foot import (
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


def test_totals_ligne_deplacee():
    market = _market_totals([
        {"name": "Under", "point": 2.5, "price": 1.91},
        {"name": "Over", "point": 2.5, "price": 1.95},
    ])
    o = _trouver_outcome_clv(market, "Under", 2.75, "totals", "Kalmar FF", "Orgryte IS")
    assert o is not None
    assert float(o["point"]) == 2.5


def test_event_kalmar_orgryte():
    data = [{
        "home_team": "Kalmar",
        "away_team": "Orgryte IS",
        "bookmakers": [{
            "key": "pinnacle",
            "markets": [_market_totals([
                {"name": "Under", "point": 2.5, "price": 2.01},
                {"name": "Over", "point": 2.5, "price": 1.85},
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


def test_goteborg_aik():
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
        test_totals_ligne_deplacee,
        test_event_kalmar_orgryte,
        test_goteborg_aik,
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
