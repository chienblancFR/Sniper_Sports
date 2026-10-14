"""Gardes CLV NBA : plausibilité + gel post-tip."""
from datetime import datetime, timedelta, timezone
from unittest import mock

import nba_sniper as sniper


def test_plausible_ok():
    assert sniper._cote_clv_plausible_nba(1.90, 1.95)
    assert sniper._cote_clv_plausible_nba(2.87, 3.72)  # ~30 %


def test_plausible_rejette_blowout():
    assert not sniper._cote_clv_plausible_nba(3.01, 1.25)  # +140 % CLV affiché
    assert not sniper._cote_clv_plausible_nba(4.85, 10.9)
    assert not sniper._cote_clv_plausible_nba(3.07, 5.24)


def test_match_deja_tippe():
    tip = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    cotes = {"commence_time": tip, "cote_1": 1.5, "cote_2": 2.5}
    with mock.patch.object(sniper, "get_odds_for_match_nba", return_value=cotes):
        assert sniper._match_deja_tippe_nba("OKC", "IND") is True

    tip_futur = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    cotes2 = {"commence_time": tip_futur}
    with mock.patch.object(sniper, "get_odds_for_match_nba", return_value=cotes2):
        assert sniper._match_deja_tippe_nba("OKC", "IND") is False


def test_extraire_ml_exact():
    cotes = {"cote_1": 1.5, "cote_2": 2.8, "spreads": {}, "totals": {}}
    with mock.patch.object(sniper, "get_odds_for_match_nba", return_value=cotes):
        assert sniper._extraire_cote_clv_nba("OKC", "IND", "Victoire OKC", "2.0") == "1.5"
        assert sniper._extraire_cote_clv_nba("OKC", "IND", "Victoire IND", "2.0") == "2.8"


if __name__ == "__main__":
    for t in (
        test_plausible_ok,
        test_plausible_rejette_blowout,
        test_match_deja_tippe,
        test_extraire_ml_exact,
    ):
        t()
        print(f"OK  {t.__name__}")
    print("4 tests CLV guard NBA OK")
