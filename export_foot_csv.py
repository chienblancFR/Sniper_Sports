#!/usr/bin/env python3
"""
Export paris_log → historique_sniper.csv (dashboard foot).
Aucune dépendance au bot — utilisable sur PythonAnywhere même si odds_devig/foot_params manquent.

Usage (Bash PA, dossier du projet) :
  python export_foot_csv.py
  python export_foot_csv.py --db sniper_data.db --out /home/chienblanc/data/historique_sniper.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys

PA_DATA_DIR = "/home/chienblanc/data"
DEFAULT_DB = "sniper_data.db"
DEFAULT_CSV = (
    f"{PA_DATA_DIR}/historique_sniper.csv"
    if os.path.isdir(PA_DATA_DIR)
    else "historique_sniper.csv"
)

_BASE_COLS = [
    ("id_match", "ID_Match"),
    ("equipe", "Equipe"),
    ("handicap", "Handicap"),
    ("cote_prise", "Cote_Prise"),
    ("mise", "Mise"),
    ("cote_cloture", "Cote_Cloture"),
    ("edge_detecte", "Edge"),
    ("p_modele", "Prob_Modele"),
    ("clv", "CLV"),
    ("statut", "Statut"),
    ("resultat", "Profit_Unites"),
    ("ligue", "Ligue"),
    ("is_lineup_official", "Compo_Officielle"),
]
_EXTRA_COLS = [
    ("equipe_dom", "Equipe_Dom"),
    ("equipe_ext", "Equipe_Ext"),
    ("kickoff", "Kickoff"),
]


def _colonnes_disponibles(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("PRAGMA table_info(paris_log)")
    return {row[1] for row in cur.fetchall()}


def export_csv(db_path: str, csv_path: str) -> int:
    if not os.path.isfile(db_path):
        print(f"❌ Base introuvable : {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    try:
        dispo = _colonnes_disponibles(conn)
        if "id_match" not in dispo:
            print("❌ Table paris_log absente ou vide schéma.", file=sys.stderr)
            return 1

        select_parts = []
        headers = []
        for col, alias in _BASE_COLS:
            if col in dispo:
                select_parts.append(f"{col} AS {alias}")
                headers.append(alias)
        for col, alias in _EXTRA_COLS:
            if col in dispo:
                select_parts.append(f"{col} AS {alias}")
                headers.append(alias)
        if "timestamp" in dispo:
            select_parts.append("timestamp AS Date")
            headers.append("Date")

        sql = f"SELECT {', '.join(select_parts)} FROM paris_log ORDER BY timestamp DESC"
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    idx_statut = headers.index("Statut") if "Statut" in headers else -1
    pending = sum(
        1 for r in rows
        if idx_statut >= 0 and r[idx_statut] == "PENDING"
    )
    print(f"✅ {len(rows)} paris exportés ({pending} PENDING) → {csv_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export paris_log → historique_sniper.csv")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite bot (défaut: {DEFAULT_DB})")
    parser.add_argument("--out", default=DEFAULT_CSV, help=f"CSV de sortie (défaut: {DEFAULT_CSV})")
    args = parser.parse_args()
    return export_csv(args.db, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
