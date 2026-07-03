"""Chargement centralisé des variables d'environnement (common + sport)."""
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
ENV_DIR = _ROOT / "env"
LEGACY_ENV = _ROOT / "identifiants_différent_api.env"


def load_project_env(sport: str | None = None) -> None:
    """
    Ordre de priorité (dernier gagne) :
    1. env/common.env
    2. env/{sport}.env
    3. identifiants_différent_api.env (legacy, si common absent)
    4. .env à la racine (override local optionnel)
    """
    loaded = False
    common = ENV_DIR / "common.env"
    if common.is_file():
        load_dotenv(common)
        loaded = True
    if sport:
        sport_file = ENV_DIR / f"{sport}.env"
        if sport_file.is_file():
            load_dotenv(sport_file)
            loaded = True
    if not loaded and LEGACY_ENV.is_file():
        load_dotenv(LEGACY_ENV)
    load_dotenv(_ROOT / ".env")


def env_files_hint(sport: str | None = None) -> str:
    """Chemin(s) attendus pour les messages d'erreur / logs."""
    parts = ["env/common.env"]
    if sport:
        parts.append(f"env/{sport}.env")
    return " + ".join(parts)
