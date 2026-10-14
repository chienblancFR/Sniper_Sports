"""Smoke : miroir static PA du journal NBA (~/ vs ~/data/)."""
import os
import tempfile
from pathlib import Path
from unittest import mock

import nba_sniper as sniper


def test_chemin_miroir_static():
    p = sniper._chemin_miroir_static_nba()
    assert p.endswith("journal_trading_nba.csv")
    assert os.path.dirname(p) == os.path.expanduser("~")


def test_publier_copie_vers_home():
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td) / "data"
        data_dir.mkdir()
        canon = data_dir / "journal_trading_nba.csv"
        canon.write_text("Date,Pari\n2026-01-01,Victoire NYK\n", encoding="utf-8")
        legacy = Path(td) / "journal_trading_nba.csv"

        with mock.patch.object(sniper, "PA_DATA_DIR", str(data_dir)), \
             mock.patch.object(sniper, "FICHIER_JOURNAL_NBA", str(canon)), \
             mock.patch.object(sniper, "_chemin_miroir_static_nba", return_value=str(legacy)), \
             mock.patch.object(sniper, "log_nba"):
            sniper.publier_journal_dashboard_nba()

        assert legacy.is_file()
        assert "Victoire NYK" in legacy.read_text(encoding="utf-8")


if __name__ == "__main__":
    test_chemin_miroir_static()
    print("OK  test_chemin_miroir_static")
    test_publier_copie_vers_home()
    print("OK  test_publier_copie_vers_home")
    print("2 tests miroir journal NBA OK")
