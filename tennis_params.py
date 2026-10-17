"""
Paramètres tennis (surchargeables via env/tennis.env).

Cibles v1 : moneyline + total de jeux (O/U). Circuit ATP best-of-3 en priorité.
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
# Journal Streamlit / PA (même convention NHL-NBA)
PA_DATA_DIR = "/home/chienblanc/data"


def _f(key: str, default: str) -> float:
    return float(os.environ.get(key, default))


def _i(key: str, default: str) -> int:
    return int(os.environ.get(key, default))


def _b(key: str, default: str | bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        if isinstance(default, bool):
            return default
        raw = str(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")

# --- Elo surface ---
TENNIS_ELO_BASE = _f("TENNIS_ELO_BASE", "1500")
TENNIS_ELO_K = _f("TENNIS_ELO_K", "32")
# Pondération Elo « toutes surfaces » mélangé à l'Elo surface du match
TENNIS_ELO_SURFACE_BLEND = _f("TENNIS_ELO_SURFACE_BLEND", "0.70")

# --- Point / service (moyennes circuit, à recalibrer sur historique) ---
# Fraction de points gagnés au service (ATP approx.)
TENNIS_MU_SERVE_HARD = _f("TENNIS_MU_SERVE_HARD", "0.64")
TENNIS_MU_SERVE_CLAY = _f("TENNIS_MU_SERVE_CLAY", "0.62")
TENNIS_MU_SERVE_GRASS = _f("TENNIS_MU_SERVE_GRASS", "0.66")
TENNIS_MU_SERVE_CARPET = _f("TENNIS_MU_SERVE_CARPET", "0.65")
# Échelle Elo → écart de points gagnés au service
TENNIS_ELO_POINT_SCALE = _f("TENNIS_ELO_POINT_SCALE", "400")
# Proxy Elo→service (ablation / fallback si TENNIS_SERVE_SOURCE=elo_proxy)
TENNIS_POINT_SENSITIVITY = _f("TENNIS_POINT_SENSITIVITY", "0.55")

# --- Stats service PIT (Sackmann w_svpt / 1stWon+2ndWon) ---
# stats = compteurs réels (défaut) ; elo_proxy = mapping Elo seul
TENNIS_SERVE_SOURCE = os.environ.get("TENNIS_SERVE_SOURCE", "stats").strip().lower()
# Points équivalents = précision prior ridge (α/γ ~ 0). 40 calibre mieux l'écart que 150
TENNIS_SERVE_N_PRIOR = _f("TENNIS_SERVE_N_PRIOR", "40")
# Part surface vs carrière
TENNIS_SERVE_SURFACE_BLEND = _f("TENNIS_SERVE_SURFACE_BLEND", "0.70")
# true = p = μ + α_serveur − γ_retourneur (ajusté adversaire, PIT)
TENNIS_SERVE_OPP_ADJ = _b("TENNIS_SERVE_OPP_ADJ", True)
# Amplifie (α − γ) ; 2.0 ≈ biais totaux ~0 sur ATP 2022-26 (grille)
TENNIS_SERVE_EFFECT_SCALE = _f("TENNIS_SERVE_EFFECT_SCALE", "2.0")
# Filet Elo sur l'écart (0 = opp-adj pur une fois scale/prior calés)
TENNIS_SERVE_ELO_SPREAD_WEIGHT = _f("TENNIS_SERVE_ELO_SPREAD_WEIGHT", "0.0")

# --- Split 1er / 2e service : p = f*p1 + (1-f)*p2 ---
TENNIS_SERVE_SPLIT = _b("TENNIS_SERVE_SPLIT", True)
# Moyennes circuit (ATP 2018-26) — first-in / win on 1st / win on 2nd
TENNIS_MU_1ST_IN_HARD = _f("TENNIS_MU_1ST_IN_HARD", "0.621")
TENNIS_MU_1ST_IN_CLAY = _f("TENNIS_MU_1ST_IN_CLAY", "0.627")
TENNIS_MU_1ST_IN_GRASS = _f("TENNIS_MU_1ST_IN_GRASS", "0.627")
TENNIS_MU_1ST_IN_CARPET = _f("TENNIS_MU_1ST_IN_CARPET", "0.624")
TENNIS_MU_1ST_WON_HARD = _f("TENNIS_MU_1ST_WON_HARD", "0.727")
TENNIS_MU_1ST_WON_CLAY = _f("TENNIS_MU_1ST_WON_CLAY", "0.687")
TENNIS_MU_1ST_WON_GRASS = _f("TENNIS_MU_1ST_WON_GRASS", "0.741")
TENNIS_MU_1ST_WON_CARPET = _f("TENNIS_MU_1ST_WON_CARPET", "0.720")
TENNIS_MU_2ND_WON_HARD = _f("TENNIS_MU_2ND_WON_HARD", "0.507")
TENNIS_MU_2ND_WON_CLAY = _f("TENNIS_MU_2ND_WON_CLAY", "0.504")
TENNIS_MU_2ND_WON_GRASS = _f("TENNIS_MU_2ND_WON_GRASS", "0.520")
TENNIS_MU_2ND_WON_CARPET = _f("TENNIS_MU_2ND_WON_CARPET", "0.510")

# --- Moneyline vs totaux (split) ---
# elo = expected_score Elo (défaut, meilleur Brier) ; engine = moteur sets ; blend = mélange
TENNIS_ML_SOURCE = os.environ.get("TENNIS_ML_SOURCE", "elo").strip().lower()
# Poids du moteur dans le blend : p = (1-w)*Elo + w*engine (ignoré si source != blend)
TENNIS_ML_BLEND_ENGINE = _f("TENNIS_ML_BLEND_ENGINE", "0.0")

# --- Blend modèle × marché sharp (Pinnacle no-vig, sniper live) ---
# p_final = (1-λ)*p_model + λ*p_market_shin
TENNIS_MARKET_BLEND = _b("TENNIS_MARKET_BLEND", True)
# λ élevé = confiance marché (réduit les faux edges) ; 0 = modèle pur
TENNIS_MARKET_BLEND_ML = _f("TENNIS_MARKET_BLEND_ML", "0.70")
TENNIS_MARKET_BLEND_OU = _f("TENNIS_MARKET_BLEND_OU", "0.70")

# --- Marchés / ops (scanner plus tard) ---
TENNIS_BEST_OF_DEFAULT = _i("TENNIS_BEST_OF_DEFAULT", "3")
TENNIS_EDGE_MIN = _f("TENNIS_EDGE_MIN", "0.03")
TENNIS_DRY_RUN = _b("TENNIS_DRY_RUN", "true")
TENNIS_BANKROLL = _f("TENNIS_BANKROLL", "1000")
# Scan Odds API : tip dans ≤ N h (observation ; edges hors prise loggés)
TENNIS_SCAN_HEURES_AVANCE = _f("TENNIS_SCAN_HEURES_AVANCE", "18")
# Prise paper : tip entre MIN et MAX heures (sweet spot ~3–6 h)
TENNIS_PRISE_HEURES_MAX = _f("TENNIS_PRISE_HEURES_MAX", "6")
TENNIS_PRISE_HEURES_MIN = _f("TENNIS_PRISE_HEURES_MIN", "3")
# Filet de sécurité : jamais de prise dans les N min avant tip
TENNIS_MINUTES_AVANT_TIP = _f("TENNIS_MINUTES_AVANT_TIP", "10")

# Journal paper → dashboard Streamlit (URL PA /data/…)
TENNIS_JOURNAL_NOM = os.environ.get("TENNIS_JOURNAL_FILE", "journal_trading_tennis.csv").strip() or (
    "journal_trading_tennis.csv"
)
_journal_override = os.environ.get("TENNIS_JOURNAL_CSV", "").strip()
if _journal_override:
    TENNIS_JOURNAL_PATH = Path(_journal_override)
elif os.path.isdir(PA_DATA_DIR):
    TENNIS_JOURNAL_PATH = Path(PA_DATA_DIR) / Path(TENNIS_JOURNAL_NOM).name
else:
    TENNIS_JOURNAL_PATH = _ROOT / Path(TENNIS_JOURNAL_NOM).name

# Cache Elo/serve + refresh Sackmann auto (sniper Always-on)
TENNIS_STATE_CACHE_HOURS = _f("TENNIS_STATE_CACHE_HOURS", "24")
# Dans la boucle sniper : recharger état / CSV si > N h
TENNIS_STATE_RELOAD_HOURS = _f("TENNIS_STATE_RELOAD_HOURS", "12")
# Re-télécharge le CSV année courante s'il est plus vieux que N h
TENNIS_SACKMANN_AUTO_REFRESH = _b("TENNIS_SACKMANN_AUTO_REFRESH", True)
TENNIS_SACKMANN_REFRESH_HOURS = _f("TENNIS_SACKMANN_REFRESH_HOURS", "12")

# Lignes O/U jeux à évaluer par défaut
TENNIS_TOTAL_GAMES_LINES = tuple(
    float(x.strip())
    for x in os.environ.get(
        "TENNIS_TOTAL_GAMES_LINES", "20.5,21.5,22.0,22.5,23.5"
    ).split(",")
    if x.strip()
)

# --- Calibration O/U (tennis_calib.py) ---
# none | isotonic | platt — fit sur warm-up, appliqué à l'éval / live
TENNIS_TOTALS_CALIB = os.environ.get("TENNIS_TOTALS_CALIB", "isotonic").strip().lower()
TENNIS_TOTALS_CALIB_BANDS = _b("TENNIS_TOTALS_CALIB_BANDS", True)
TENNIS_TOTALS_CALIB_MIN_SAMPLES = _i("TENNIS_TOTALS_CALIB_MIN_SAMPLES", "80")

# --- Calibration P(3 sets) / longueur conditionnelle ---
# none | isotonic | platt | band_offset
TENNIS_SETS_CALIB = os.environ.get("TENNIS_SETS_CALIB", "isotonic").strip().lower()
# Ajuste E[jeux|2] et E[jeux|3] par bande favori (après repondération p3)
TENNIS_SETS_COND_SHIFT = _b("TENNIS_SETS_COND_SHIFT", True)
TENNIS_SETS_CALIB_MIN_SAMPLES = _i("TENNIS_SETS_CALIB_MIN_SAMPLES", "80")

# --- Tie-break dédié (SPW compressé + calib P(win TB)) ---
# TB ≠ jeu normal : l'écart SPW régulier surestime le favori (~60% vs ~55%)
TENNIS_TB_DEDICATED = _b("TENNIS_TB_DEDICATED", True)
# p_tb = μ_tb + scale*(p − μ_surface) ; 0.5 ≈ Brier TB min sur ATP 2018-26
TENNIS_TB_SERVE_SCALE = _f("TENNIS_TB_SERVE_SCALE", "0.50")
# Décale la moyenne service en TB (0 = même μ surface)
TENNIS_TB_MU_SHIFT = _f("TENNIS_TB_MU_SHIFT", "0.0")
# none | isotonic | platt — fit warm-up sur P(favori gagne le TB)
TENNIS_TB_CALIB = os.environ.get("TENNIS_TB_CALIB", "platt").strip().lower()
TENNIS_TB_CALIB_MIN_SAMPLES = _i("TENNIS_TB_CALIB_MIN_SAMPLES", "80")

# --- Vague A4 : dépendance jeu-à-jeu (break-back) ---
# Après un break, hold du prochain serveur × (1-ρ) ; 0 = i.i.d.
TENNIS_BREAK_BACK = _f("TENNIS_BREAK_BACK", "0.08")

# --- Vague A5 : fiabilité Elo + élargissement PMF ---
# rel = n_eff / (n_eff + prior) ; shrink ML vers 0.5 si peu de matchs
TENNIS_ELO_N_PRIOR = _f("TENNIS_ELO_N_PRIOR", "15")
# Largeur PMF jeux (en jeux) × (1-rel) ; 0 = off
TENNIS_PMF_SIGMA = _f("TENNIS_PMF_SIGMA", "1.5")

# --- Vague A6 : fatigue (repos + minutes match précédent) ---
TENNIS_FATIGUE = _b("TENNIS_FATIGUE", True)
# pen ≈ scale * max(0, last_min - thresh) * max(0, rest_days - days) / 60
TENNIS_FATIGUE_SCALE = _f("TENNIS_FATIGUE_SCALE", "0.015")
TENNIS_FATIGUE_MIN_THRESH = _f("TENNIS_FATIGUE_MIN_THRESH", "120")
TENNIS_FATIGUE_REST_DAYS = _f("TENNIS_FATIGUE_REST_DAYS", "2")

# --- Vague A7 : CLV paper (pré-tip only, comme NHL/NBA) ---
TENNIS_CLV_TRACK = _b("TENNIS_CLV_TRACK", True)
# Écart max relatif prise→close ; au-delà = blowout live ignoré / réparé
TENNIS_CLV_MAX_DEV_REL = _f("TENNIS_CLV_MAX_DEV_REL", "0.35")


SURFACES = ("Hard", "Clay", "Grass", "Carpet")


def mu_serve_for_surface(surface: str) -> float:
    s = (surface or "Hard").strip().capitalize()
    if s == "Clay":
        return TENNIS_MU_SERVE_CLAY
    if s == "Grass":
        return TENNIS_MU_SERVE_GRASS
    if s == "Carpet":
        return TENNIS_MU_SERVE_CARPET
    return TENNIS_MU_SERVE_HARD


def mu_1st_in_for_surface(surface: str) -> float:
    s = (surface or "Hard").strip().capitalize()
    if s == "Clay":
        return TENNIS_MU_1ST_IN_CLAY
    if s == "Grass":
        return TENNIS_MU_1ST_IN_GRASS
    if s == "Carpet":
        return TENNIS_MU_1ST_IN_CARPET
    return TENNIS_MU_1ST_IN_HARD


def mu_1st_won_for_surface(surface: str) -> float:
    s = (surface or "Hard").strip().capitalize()
    if s == "Clay":
        return TENNIS_MU_1ST_WON_CLAY
    if s == "Grass":
        return TENNIS_MU_1ST_WON_GRASS
    if s == "Carpet":
        return TENNIS_MU_1ST_WON_CARPET
    return TENNIS_MU_1ST_WON_HARD


def mu_2nd_won_for_surface(surface: str) -> float:
    s = (surface or "Hard").strip().capitalize()
    if s == "Clay":
        return TENNIS_MU_2ND_WON_CLAY
    if s == "Grass":
        return TENNIS_MU_2ND_WON_GRASS
    if s == "Carpet":
        return TENNIS_MU_2ND_WON_CARPET
    return TENNIS_MU_2ND_WON_HARD


def normalize_surface(raw: str | None) -> str:
    """Normalise les libellés Sackmann / Odds API → Hard|Clay|Grass|Carpet."""
    if not raw:
        return "Hard"
    s = raw.strip().lower()
    if "clay" in s or "terre" in s:
        return "Clay"
    if "grass" in s or "gazon" in s:
        return "Grass"
    if "carpet" in s:
        return "Carpet"
    return "Hard"
