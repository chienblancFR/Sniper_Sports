"""Tests sélection 1 pari / marché / match."""
from __future__ import annotations

import sys

import nba_sniper as nba


def _cand(marche, typ, edge):
    return {
        "marche": marche,
        "type": typ,
        "inv": {"edge": edge, "mise": 10, "pct_bankroll": 1},
        "cote_book": 1.9,
        "cote_vraie": 1.8,
    }


def test_selection_un_par_marche():
    old = nba.NBA_TOUS_CANDIDATS_ACTIF
    nba.NBA_TOUS_CANDIDATS_ACTIF = True
    try:
        cands = [
            _cand("TOTAL", "UNDER 237", 6.0),
            _cand("TOTAL", "UNDER 239", 10.0),
            _cand("TOTAL", "OVER 237", 5.0),
            _cand("SPREAD", "CLE +10.5", 9.0),
            _cand("SPREAD", "CLE +15.5", 16.0),
            _cand("ML", "Victoire CLE", 12.0),
            _cand("ML", "Victoire ORL", 4.0),
        ]
        out = nba._selectionner_paris_nba(cands)
        assert len(out) == 3, out
        by_m = {c["marche"]: c for c in out}
        assert by_m["ML"]["type"] == "Victoire CLE"
        assert by_m["SPREAD"]["type"] == "CLE +15.5"
        assert by_m["TOTAL"]["type"] == "UNDER 239"
    finally:
        nba.NBA_TOUS_CANDIDATS_ACTIF = old


def test_selection_max_edge_seul():
    old = nba.NBA_TOUS_CANDIDATS_ACTIF
    nba.NBA_TOUS_CANDIDATS_ACTIF = False
    try:
        cands = [
            _cand("ML", "Victoire A", 5.0),
            _cand("TOTAL", "UNDER 220", 12.0),
            _cand("SPREAD", "A +3", 8.0),
        ]
        out = nba._selectionner_paris_nba(cands)
        assert len(out) == 1
        assert out[0]["type"] == "UNDER 220"
    finally:
        nba.NBA_TOUS_CANDIDATS_ACTIF = old


def test_marche_depuis_type():
    assert nba._marche_depuis_type_pari_nba("Victoire NYK") == "ML"
    assert nba._marche_depuis_type_pari_nba("UNDER 237") == "TOTAL"
    assert nba._marche_depuis_type_pari_nba("CLE +10.5") == "SPREAD"


def test_cle_dedup_marche():
    assert nba._cle_dedup_marche_nba("20261015_NOP_DAL", "TOTAL") == "20261015_NOP_DAL|TOTAL"
    assert nba._cle_dedup_marche_nba("g", "spread") == "g|SPREAD"


def main():
    tests = [
        test_selection_un_par_marche,
        test_selection_max_edge_seul,
        test_marche_depuis_type,
        test_cle_dedup_marche,
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
    print(f"\n{len(tests)} tests sélection NBA OK")


if __name__ == "__main__":
    main()
