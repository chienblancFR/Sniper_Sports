"""Archive le journal paper NBA (présaison) et repart à zéro.

Usage (local ou PythonAnywhere) :
  python reset_journal_nba.py
  python reset_journal_nba.py --yes   # sans confirmation

Effets :
  - copie journal + miroir ~/ + mémoire dédup vers nba_saison_backups/
  - recrée journal vide (header seul) + miroir static
  - vide alertes_nba_envoyees.txt
  - bankroll paper revient à NBA_BANKROLL au prochain scan
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime, timezone

import nba_sniper as nba


def _backup_dir() -> str:
    root = nba._DIR_RUNTIME_NBA
    d = os.path.join(root, "nba_saison_backups")
    os.makedirs(d, exist_ok=True)
    return d


def reset_journal_nba(confirm: bool = True) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = _backup_dir()
    journal = nba.FICHIER_JOURNAL_NBA
    memoire = nba.FICHIER_MEMOIRE_NBA
    miroir = nba._chemin_miroir_static_nba()

    n_lignes = 0
    if os.path.isfile(journal):
        with open(journal, encoding="utf-8") as f:
            n_lignes = max(0, sum(1 for _ in f) - 1)

    print(f"Journal  : {journal} ({n_lignes} paris)")
    print(f"Mémoire  : {memoire}")
    print(f"Miroir   : {miroir}")
    print(f"Backup → : {backup_dir}")

    if confirm:
        rep = input("Archiver et vider le journal NBA ? [y/N] ").strip().lower()
        if rep not in ("y", "yes", "o", "oui"):
            print("Annulé.")
            return 1

    archived = []
    for src, tag in (
        (journal, "journal"),
        (miroir, "miroir"),
        (memoire, "memoire"),
    ):
        if not os.path.isfile(src):
            continue
        dest = os.path.join(backup_dir, f"{tag}_preseason_{stamp}{os.path.splitext(src)[1]}")
        shutil.copy2(src, dest)
        archived.append(dest)
        print(f"  📦 {dest}")

    # Journal vide
    parent = os.path.dirname(journal) or "."
    os.makedirs(parent, exist_ok=True)
    with open(journal, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=nba.JOURNAL_COLONNES_NBA).writeheader()
    nba.publier_journal_dashboard_nba()

    # Mémoire dédup vide
    with open(memoire, "w", encoding="utf-8") as f:
        f.write("")

    print(f"✅ Reset OK — {n_lignes} pari(s) archivé(s), journal vide + mémoire clear.")
    print("   Reload Always-on sniper NBA. Dashboard : bankroll = NBA_BANKROLL.")
    if archived:
        print(f"   Archives : {backup_dir}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Reset journal paper NBA (archive présaison)")
    p.add_argument("--yes", "-y", action="store_true", help="Sans confirmation")
    args = p.parse_args()
    sys.exit(reset_journal_nba(confirm=not args.yes))


if __name__ == "__main__":
    main()
