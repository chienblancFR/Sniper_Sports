"""
Paramètres Dixon-Coles / xG par ligue — defaults + surcharges walk-forward.

Les valeurs de base viennent de la littérature / heuristiques manuelles.
`foot_params_tuned.json` (généré par backtest --tune) les remplace par ligue
sans toucher au code.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

_PARAMS_FILE = Path(os.environ.get("FOOT_PARAMS_TUNED_FILE", "foot_params_tuned.json"))

N_PRIOR_DEFAULT = 8
RHO_DEFAULT = -0.12
XG_HALF_LIFE_DEFAULT = 46.0
DC_HALF_LIFE_DEFAULT = 90.0

N_PRIOR_PAR_LIGUE: dict[int, int] = {
    140: 7, 39: 7, 135: 7, 40: 7, 78: 8, 61: 8, 141: 8,
    88: 9, 94: 9, 203: 9, 71: 9, 136: 10, 253: 10, 144: 11, 113: 12, 103: 12,
}

RHO_PAR_LIGUE: dict[int, float] = {
    78: -0.16, 88: -0.15, 39: -0.13, 40: -0.12, 61: -0.12, 141: -0.12,
    136: -0.11, 140: -0.10, 94: -0.10, 135: -0.09, 203: -0.08, 71: -0.08,
    113: -0.11, 103: -0.10, 144: -0.12, 253: -0.09,
}

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


def _ligue_entry(ligue_id: int) -> dict:
    return _load_tuned().get(str(ligue_id), {})


def get_n_prior(ligue_id: int) -> int:
    entry = _ligue_entry(ligue_id)
    if "n_prior" in entry:
        return int(entry["n_prior"])
    return N_PRIOR_PAR_LIGUE.get(ligue_id, N_PRIOR_DEFAULT)


def get_rho_fallback(ligue_id: int) -> float:
    entry = _ligue_entry(ligue_id)
    if "rho" in entry:
        return float(entry["rho"])
    return RHO_PAR_LIGUE.get(ligue_id, RHO_DEFAULT)


def get_xg_half_life_days(ligue_id: int) -> float:
    entry = _ligue_entry(ligue_id)
    if "xg_half_life_days" in entry:
        return float(entry["xg_half_life_days"])
    return XG_HALF_LIFE_DEFAULT


def get_dc_half_life_days(ligue_id: int) -> float:
    entry = _ligue_entry(ligue_id)
    if "dc_half_life_days" in entry:
        return float(entry["dc_half_life_days"])
    return DC_HALF_LIFE_DEFAULT


def xg_decay_rate(ligue_id: int) -> float:
    """λ decay = ln(2) / demi-vie (jours)."""
    return math.log(2) / max(get_xg_half_life_days(ligue_id), 1.0)


def save_tuned_params(
    results: dict,
    source: str = "backtest_walkforward",
    metric: str = "mean_log_score_prob",
    merge_existing: bool = True,
) -> Path:
    """Écrit foot_params_tuned.json (clés ligue_id en string).

    merge_existing=True : conserve les ligues déjà calibrées (tuning ligue par ligue).
    """
    existing = _load_tuned() if merge_existing else {}
    merged = {
        k: v for k, v in existing.items()
        if k != "_meta" and not str(k).startswith("_")
    }
    merged.update({str(k): v for k, v in results.items() if not str(k).startswith("_")})

    payload = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "metric": metric,
            "ligues_tuned": len(merged),
        },
        **merged,
    }
    with open(_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    reload_tuned_params()
    return _PARAMS_FILE
