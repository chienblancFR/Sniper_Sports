"""Smoke tests AdjORtg/AdjDRtg (SoS) — données synthétiques, pas d'appel nba_api."""
from __future__ import annotations

import sys

import nba_params
import nba_sniper as nba


def _game(team, date, opponent, ortg, drtg, domicile=True, poss=100.0, pace=100.0):
    return {
        "team": team,
        "date": date,
        "opponent": opponent,
        "off_rating": ortg,
        "def_rating": drtg,
        "domicile": domicile,
        "poss": poss,
        "pace": pace,
        "game_id": f"{date}-{team}-{opponent}",
    }


def _par_equipe_toy():
    """AAA a gonflé son ORtg contre BBB (mauvaise défense) ; CCC a un ORtg
    similaire contre DDD (bonne défense). Après SoS, AdjO(AAA) < AdjO(CCC)."""
    games = [
        # BBB = passoire (DRtg élevé)
        _game("AAA", "2024-01-02", "BBB", 125.0, 110.0, True),
        _game("BBB", "2024-01-02", "AAA", 110.0, 125.0, False),
        _game("AAA", "2024-01-04", "BBB", 124.0, 112.0, False),
        _game("BBB", "2024-01-04", "AAA", 112.0, 124.0, True),
        # DDD = mur (DRtg bas)
        _game("CCC", "2024-01-02", "DDD", 118.0, 108.0, True),
        _game("DDD", "2024-01-02", "CCC", 108.0, 118.0, False),
        _game("CCC", "2024-01-04", "DDD", 119.0, 107.0, False),
        _game("DDD", "2024-01-04", "CCC", 107.0, 119.0, True),
        # Remplissage pour stabiliser les 4 équipes (matchs croisés légers)
        _game("BBB", "2024-01-06", "DDD", 105.0, 115.0, True),
        _game("DDD", "2024-01-06", "BBB", 115.0, 105.0, False),
        _game("AAA", "2024-01-06", "CCC", 112.0, 112.0, True),
        _game("CCC", "2024-01-06", "AAA", 112.0, 112.0, False),
    ]
    return nba._grouper_records_par_equipe(games)


def test_sos_deflate_offense_vs_weak_defense():
    nba._sos_snap_cache.clear()
    par = _par_equipe_toy()
    date_ref = "2024-01-07"
    snap_brut = nba._snapshot_ligue_decay(par, date_ref, half_life_jours=0)
    lg = nba._moyennes_ligue_depuis_snapshot(snap_brut)
    assert lg is not None

    # Force SoS on pour le test même si env le coupe
    old = nba_params._tuned_cache
    nba_params.set_tuned_global_in_memory({**(nba_params._global_entry()), "sos_actif": True, "sos_iters": 20})
    try:
        snap_adj = nba._snapshot_ligue_sos(par, date_ref, 0, lg, n_iter=20)
        raw_aaa = snap_brut["AAA"]["off_rating"]
        raw_ccc = snap_brut["CCC"]["off_rating"]
        adj_aaa = snap_adj["AAA"]["off_rating"]
        adj_ccc = snap_adj["CCC"]["off_rating"]
        # AAA a un ORtg brut plus haut (a joué BBB), CCC plus bas (a joué DDD)
        assert raw_aaa > raw_ccc, (raw_aaa, raw_ccc)
        # Après SoS, l'écart se réduit ou s'inverse : Adj(AAA) ne doit plus
        # dominer autant — au minimum Adj(AAA) - Adj(CCC) < raw_aaa - raw_ccc
        gap_raw = raw_aaa - raw_ccc
        gap_adj = adj_aaa - adj_ccc
        assert gap_adj < gap_raw - 0.5, (gap_raw, gap_adj, adj_aaa, adj_ccc)
    finally:
        nba_params._tuned_cache = old
        nba._sos_snap_cache.clear()


def test_sos_pit_ignore_future_games():
    nba._sos_snap_cache.clear()
    par = _par_equipe_toy()
    # Avant le 6 janvier : pas encore les matchs AAA-CCC / BBB-DDD
    date_ref = "2024-01-05"
    snap_brut = nba._snapshot_ligue_decay(par, date_ref, 0)
    lg = nba._moyennes_ligue_depuis_snapshot(snap_brut)
    old = nba_params._tuned_cache
    nba_params.set_tuned_global_in_memory({**(nba_params._global_entry()), "sos_actif": True, "sos_iters": 10})
    try:
        snap = nba._snapshot_ligue_sos(par, date_ref, 0, lg, n_iter=10)
        # AAA n'a joué que BBB avant le 5 → gp == 2
        assert snap["AAA"]["gp"] == 2
        assert snap["CCC"]["gp"] == 2
        # Aucune donnée du 6 ne doit entrer : si on ajoute un match futur
        # artificiel avec ORtg absurde, le rating avant le 5 ne bouge pas
        par2 = {t: list(gs) for t, gs in par.items()}
        par2["AAA"].append(_game("AAA", "2024-01-10", "BBB", 200.0, 80.0, True))
        par2["BBB"].append(_game("BBB", "2024-01-10", "AAA", 80.0, 200.0, False))
        for t in par2:
            par2[t].sort(key=lambda g: (g["date"], g["game_id"]))
        nba._sos_snap_cache.clear()
        snap2 = nba._snapshot_ligue_sos(par2, date_ref, 0, lg, n_iter=10)
        assert abs(snap2["AAA"]["off_rating"] - snap["AAA"]["off_rating"]) < 1e-9
        assert snap2["AAA"]["gp"] == 2
    finally:
        nba_params._tuned_cache = old
        nba._sos_snap_cache.clear()


def test_sos_off_returns_brut():
    nba._sos_snap_cache.clear()
    par = _par_equipe_toy()
    date_ref = "2024-01-07"
    snap_brut = nba._snapshot_ligue_decay(par, date_ref, 0)
    lg = nba._moyennes_ligue_depuis_snapshot(snap_brut)
    old = nba_params._tuned_cache
    nba_params.set_tuned_global_in_memory({**(nba_params._global_entry()), "sos_actif": False})
    try:
        snap = nba._snapshot_ligue_sos(par, date_ref, 0, lg)
        assert abs(snap["AAA"]["off_rating"] - snap_brut["AAA"]["off_rating"]) < 1e-9
    finally:
        nba_params._tuned_cache = old
        nba._sos_snap_cache.clear()


def main():
    tests = [
        test_sos_deflate_offense_vs_weak_defense,
        test_sos_pit_ignore_future_games,
        test_sos_off_returns_brut,
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
    print(f"\n{len(tests)} tests SoS NBA OK")


if __name__ == "__main__":
    main()
