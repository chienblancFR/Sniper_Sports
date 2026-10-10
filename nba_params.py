"""Paramètres du moteur mathématique NBA — defaults + surcharges walk-forward.

Mirror de `foot_params.py` : les valeurs de base sont des heuristiques
documentées (littérature NBA analytics / ordres de grandeur usuels), pas
encore calibrées sur données. `nba_params_tuned.json` (généré par
`backtest_nba.py --tune`, phase ultérieure du plan) les remplace sans
toucher au code, avec deux blocs :

- ``_global`` : n_prior, demi-vie de décroissance, sigma/rho par défaut, etc.
- ``teams`` : surcharges par équipe (HCA, sigma), clé = tricode 3 lettres.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_PARAMS_FILE = Path(os.environ.get("NBA_PARAMS_TUNED_FILE", "nba_params_tuned.json"))

# Shrinkage bayésien vers la moyenne ligue : n_prior = poids (en "matchs
# équivalents") de la moyenne ligue dans le blend ; plus fort = régression
# plus agressive tant que l'équipe a peu de matchs decay-pondérés.
N_PRIOR_DEFAULT = int(os.environ.get("NBA_N_PRIOR_DEFAULT", "15"))
# Décroissance temporelle des ratings (poids = 0.5 ^ (age_jours / demi_vie)).
RATING_HALF_LIFE_JOURS_DEFAULT = float(os.environ.get("NBA_RATING_HALF_LIFE_JOURS", "45"))
# GP (comptés, non pondérés) pour atteindre la confiance pleine dans la
# saison en cours vs le prior saison N-1 (pattern NHL_BLEND_GP_PLEIN).
BLEND_GP_PLEIN_DEFAULT = float(os.environ.get("NBA_BLEND_GP_PLEIN", "20"))

# Ajustement adversaire (AdjORtg/AdjDRtg) — itération KenPom additive, PIT-safe.
# Désactiver (`false`) pour revenir aux ORtg/DRtg bruts (ablation / debug).
def _env_bool_params(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes", "on")


SOS_ACTIF_DEFAULT = _env_bool_params("NBA_SOS_ACTIF", True)
# Nombre d'itérations Jacobi (10-20 suffit en pratique pour 30 équipes).
SOS_ITERS_DEFAULT = int(os.environ.get("NBA_SOS_ITERS", "15"))

# Four Factors (Dean Oliver) → ORtg/DRtg via régression linéaire (R²≈0.95
# sur box scores 2021-26). Le matchup croise Off FF équipe × Def FF adversaire
# (opp_*), puis blend avec le chemin AdjORtg/AdjDRtg existant.
FF_ACTIF_DEFAULT = _env_bool_params("NBA_FF_ACTIF", True)
# Poids du chemin Four Factors dans mu (0 = pure Adj×cross, 1 = pur FF matchup).
FF_BLEND_DEFAULT = float(os.environ.get("NBA_FF_BLEND", "0.50"))
# Coefs OLS ORtg ~ intercept + b_efg*eFG + b_tov*TOV + b_orb*OREB + b_fta*FTA
# (fit 5 saisons, n≈12k ; même mapping pour DRtg via opp_*).
FF_COEF_INTERCEPT = float(os.environ.get("NBA_FF_COEF_INTERCEPT", "27.0459"))
FF_COEF_EFG = float(os.environ.get("NBA_FF_COEF_EFG", "152.1871"))
FF_COEF_TOV = float(os.environ.get("NBA_FF_COEF_TOV", "-129.1025"))
FF_COEF_ORB = float(os.environ.get("NBA_FF_COEF_ORB", "62.1513"))
FF_COEF_FTA = float(os.environ.get("NBA_FF_COEF_FTA", "16.9229"))
# Moyennes ligue fallback Four Factors (fractions 0-1, stats.nba.com).
FF_LG_EFG_DEFAULT = float(os.environ.get("NBA_FF_LG_EFG", "0.5437"))
FF_LG_TOV_DEFAULT = float(os.environ.get("NBA_FF_LG_TOV", "0.1403"))
FF_LG_ORB_DEFAULT = float(os.environ.get("NBA_FF_LG_ORB", "0.2852"))
FF_LG_FTA_DEFAULT = float(os.environ.get("NBA_FF_LG_FTA", "0.2562"))

# Home court advantage : fraction appliquée symétriquement sur ORtg (+) et
# DRtg (-) de l'équipe à domicile ; ~0.015 ≈ 2.5-3 pts d'avantage net typique.
HCA_DEFAULT = float(os.environ.get("NBA_HCA_DEFAULT", "0.015"))
# Pénalité back-to-back (2e soir consécutif) : attaque en baisse, défense en hausse.
B2B_ATK_PCT_DEFAULT = float(os.environ.get("NBA_B2B_ATK_PCT", "0.015"))
B2B_DEF_PCT_DEFAULT = float(os.environ.get("NBA_B2B_DEF_PCT", "0.015"))
# Bonus attaque si repos >= 2 jours (pas de B2B).
REST_BONUS_PCT_DEFAULT = float(os.environ.get("NBA_REST_BONUS_PCT", "0.005"))
# Bonus altitude (Denver ~1600m) pour l'équipe à domicile + malus visiteur.
ALTITUDE_BONUS_DEFAULT = float(os.environ.get("NBA_ALTITUDE_BONUS", "0.01"))
ALTITUDE_TEAMS = {
    t.strip().upper() for t in os.environ.get("NBA_ALTITUDE_TEAMS", "DEN").split(",") if t.strip()
}

# Distribution bivariée Normale (marge / total) — écart-type par équipe
# (points/match) et corrélation entre scores domicile/extérieur (positive :
# effet pace partagé — un match rapide gonfle les deux scores ensemble).
SIGMA_TEAM_DEFAULT = float(os.environ.get("NBA_SIGMA_TEAM_DEFAULT", "12.5"))
RHO_SCORES_DEFAULT = float(os.environ.get("NBA_RHO_SCORES_DEFAULT", "0.20"))

# Fallback moyenne ligue (ORtg/DRtg pts/100 poss, Pace poss/48) si aucune
# équipe n'a encore de match cette saison (tout début de saison régulière).
LEAGUE_AVG_RTG_DEFAULT = float(os.environ.get("NBA_LEAGUE_AVG_RTG_DEFAULT", "114.0"))
LEAGUE_AVG_PACE_DEFAULT = float(os.environ.get("NBA_LEAGUE_AVG_PACE_DEFAULT", "99.0"))

_tuned_cache: dict | None = None


def _load_tuned() -> dict:
    global _tuned_cache
    if _tuned_cache is not None:
        return _tuned_cache
    if _PARAMS_FILE.is_file():
        try:
            with open(_PARAMS_FILE, encoding="utf-8") as f:
                _tuned_cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            _tuned_cache = {}
    else:
        _tuned_cache = {}
    return _tuned_cache


def reload_tuned_params() -> None:
    """Force le rechargement du JSON (après --tune ou édition manuelle)."""
    global _tuned_cache
    _tuned_cache = None
    _load_tuned()


def set_tuned_global_in_memory(global_params: dict) -> None:
    """Surcharge `_global` en mémoire uniquement (pas d'écriture disque).

    Utilisé par `backtest_nba.py --tune` pour évaluer des centaines de
    candidats sans réécrire `nba_params_tuned.json` à chaque itération.
    """
    global _tuned_cache
    base = dict(_load_tuned())
    base["_global"] = dict(global_params)
    _tuned_cache = base


def _global_entry() -> dict:
    return _load_tuned().get("_global", {})


def _team_entry(team: str) -> dict:
    return _load_tuned().get("teams", {}).get(team, {})


def get_n_prior() -> float:
    g = _global_entry()
    return float(g["n_prior"]) if "n_prior" in g else float(N_PRIOR_DEFAULT)


def get_rating_half_life_jours() -> float:
    g = _global_entry()
    return float(g["rating_half_life_jours"]) if "rating_half_life_jours" in g else RATING_HALF_LIFE_JOURS_DEFAULT


def get_blend_gp_plein() -> float:
    g = _global_entry()
    return float(g["blend_gp_plein"]) if "blend_gp_plein" in g else BLEND_GP_PLEIN_DEFAULT


def get_sos_actif() -> bool:
    g = _global_entry()
    if "sos_actif" in g:
        return bool(g["sos_actif"])
    return SOS_ACTIF_DEFAULT


def get_sos_iters() -> int:
    g = _global_entry()
    if "sos_iters" in g:
        return max(1, int(g["sos_iters"]))
    return max(1, SOS_ITERS_DEFAULT)


def get_ff_actif() -> bool:
    g = _global_entry()
    if "ff_actif" in g:
        return bool(g["ff_actif"])
    return FF_ACTIF_DEFAULT


def get_ff_blend() -> float:
    g = _global_entry()
    raw = float(g["ff_blend"]) if "ff_blend" in g else FF_BLEND_DEFAULT
    return min(max(raw, 0.0), 1.0)


def get_ff_coefs() -> dict:
    """Coefs OLS ORtg = intercept + efg*b_efg + tov*b_tov + orb*b_orb + fta*b_fta."""
    g = _global_entry()
    return {
        "intercept": float(g.get("ff_coef_intercept", FF_COEF_INTERCEPT)),
        "efg": float(g.get("ff_coef_efg", FF_COEF_EFG)),
        "tov": float(g.get("ff_coef_tov", FF_COEF_TOV)),
        "orb": float(g.get("ff_coef_orb", FF_COEF_ORB)),
        "fta": float(g.get("ff_coef_fta", FF_COEF_FTA)),
    }


def get_ff_league_fallback() -> dict:
    g = _global_entry()
    return {
        "efg_pct": float(g.get("ff_lg_efg", FF_LG_EFG_DEFAULT)),
        "tov_pct": float(g.get("ff_lg_tov", FF_LG_TOV_DEFAULT)),
        "oreb_pct": float(g.get("ff_lg_orb", FF_LG_ORB_DEFAULT)),
        "fta_rate": float(g.get("ff_lg_fta", FF_LG_FTA_DEFAULT)),
        "opp_efg_pct": float(g.get("ff_lg_efg", FF_LG_EFG_DEFAULT)),
        "opp_tov_pct": float(g.get("ff_lg_tov", FF_LG_TOV_DEFAULT)),
        "opp_oreb_pct": float(g.get("ff_lg_orb", FF_LG_ORB_DEFAULT)),
        "opp_fta_rate": float(g.get("ff_lg_fta", FF_LG_FTA_DEFAULT)),
    }


def get_hca(team: str | None = None) -> float:
    if team:
        t = _team_entry(team)
        if "hca" in t:
            return float(t["hca"])
    g = _global_entry()
    return float(g["hca"]) if "hca" in g else HCA_DEFAULT


def get_b2b_atk_pct() -> float:
    g = _global_entry()
    return float(g["b2b_atk_pct"]) if "b2b_atk_pct" in g else B2B_ATK_PCT_DEFAULT


def get_b2b_def_pct() -> float:
    g = _global_entry()
    return float(g["b2b_def_pct"]) if "b2b_def_pct" in g else B2B_DEF_PCT_DEFAULT


def get_rest_bonus_pct() -> float:
    g = _global_entry()
    return float(g["rest_bonus_pct"]) if "rest_bonus_pct" in g else REST_BONUS_PCT_DEFAULT


def get_altitude_bonus() -> float:
    g = _global_entry()
    return float(g["altitude_bonus"]) if "altitude_bonus" in g else ALTITUDE_BONUS_DEFAULT


def is_altitude_team(team: str) -> bool:
    return (team or "").strip().upper() in ALTITUDE_TEAMS


def get_sigma_team(team: str | None = None) -> float:
    if team:
        t = _team_entry(team)
        if "sigma" in t:
            return float(t["sigma"])
    g = _global_entry()
    return float(g["sigma_team"]) if "sigma_team" in g else SIGMA_TEAM_DEFAULT


def get_rho_scores() -> float:
    g = _global_entry()
    return float(g["rho_scores"]) if "rho_scores" in g else RHO_SCORES_DEFAULT


def get_league_avg_fallback() -> dict:
    g = _global_entry()
    return {
        "off_rating": float(g.get("league_avg_rtg", LEAGUE_AVG_RTG_DEFAULT)),
        "def_rating": float(g.get("league_avg_rtg", LEAGUE_AVG_RTG_DEFAULT)),
        "pace": float(g.get("league_avg_pace", LEAGUE_AVG_PACE_DEFAULT)),
    }


def save_tuned_params(
    results: dict,
    source: str = "backtest_walkforward",
    merge_existing: bool = True,
) -> Path:
    """Écrit `nba_params_tuned.json` (blocs `_global` / `teams`).

    `merge_existing=True` : conserve les clés déjà calibrées (tuning
    incrémental, ex. `_global` puis `teams` dans des runs séparés).
    """
    existing = _load_tuned() if merge_existing else {}
    merged = {k: v for k, v in existing.items() if k != "_meta"}
    for key, value in results.items():
        if key == "teams" and isinstance(value, dict) and isinstance(merged.get("teams"), dict):
            merged["teams"] = {**merged["teams"], **value}
        else:
            merged[key] = value

    payload = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
        },
        **merged,
    }
    with open(_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    reload_tuned_params()
    return _PARAMS_FILE
