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


def ensure_sport_env_key(sport: str, key: str, value: str) -> bool:
    """
    Ajoute ou complète une clé vide dans env/{sport}.env.
    Ne remplace jamais une valeur déjà renseignée (secrets / toggles user).
    Retourne True si le fichier a été créé ou modifié.
    """
    if not value or not key:
        return False

    path = ENV_DIR / f"{sport}.env"
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        example = ENV_DIR / f"{sport}.env.example"
        lines = example.read_text(encoding="utf-8").splitlines() if example.is_file() else []

    prefix = f"{key}="
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        current = stripped.split("=", 1)[1].strip()
        if current:
            return False
        lines[i] = f"{key}={value}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return True

    insert_at = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("# CSV MoneyPuck"):
            insert_at = i + 1
            break
    lines.insert(insert_at, f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return True
