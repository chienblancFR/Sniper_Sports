#!/usr/bin/env python3
"""Lanceur PythonAnywhere — bouton « Run » sans ModuleNotFoundError.

Le Run PA utilise le Python système (souvent sans nba_api). Ce wrapper
appelle le venv ``~/venv-nba`` en subprocess (console reste ouverte).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_VENV_CANDIDATS = [
    Path(os.environ.get("NBA_VENV_PYTHON", "")),
    Path.home() / "venv-nba" / "bin" / "python",
    Path("/home/chienblanc/venv-nba/bin/python"),
]


def _python_venv() -> Path:
    for p in _VENV_CANDIDATS:
        if p and str(p).strip() and p.is_file():
            return p
    raise SystemExit(
        "venv-nba introuvable. Crée-le une fois en Bash :\n"
        "  python3.10 -m venv /home/chienblanc/venv-nba\n"
        "  /home/chienblanc/venv-nba/bin/pip install nba_api==1.11.4 python-dotenv==1.0.1\n"
    )


def main() -> int:
    venv_py = _python_venv()
    script = Path(__file__).resolve().parent / "nba_sniper.py"
    if not script.is_file():
        script = Path("/home/chienblanc/nba_sniper.py")
    if not script.is_file():
        raise SystemExit(f"nba_sniper.py introuvable (cherché à côté de {__file__})")
    print(f"▶ lance {script.name} via {venv_py}", flush=True)
    return subprocess.call([str(venv_py), str(script), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
