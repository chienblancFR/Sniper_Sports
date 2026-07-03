"""
Dévigorage 2-way (handicap / totaux) — méthode de Shin (1992).

Remplace le devig proportionnel (cote × overround) qui sous-estime
légèrement les favoris et surestime les outsiders.
"""

from __future__ import annotations

import math


def _parse_cotes(cote_a: float, cote_b: float) -> tuple[float, float] | None:
    try:
        a, b = float(cote_a), float(cote_b)
    except (TypeError, ValueError):
        return None
    if a <= 1.0 or b <= 1.0:
        return None
    return a, b


def proba_no_vig_proportionnel_2way(cote_a: float, cote_b: float) -> tuple[float | None, float | None]:
    """Probabilités fair par normalisation multiplicative (legacy)."""
    parsed = _parse_cotes(cote_a, cote_b)
    if parsed is None:
        return None, None
    a, b = parsed
    inv_a, inv_b = 1.0 / a, 1.0 / b
    total = inv_a + inv_b
    if total <= 0:
        return None, None
    return inv_a / total, inv_b / total


def proba_no_vig_shin_2way(cote_a: float, cote_b: float) -> tuple[float | None, float | None]:
    """
    Probabilités fair via Shin (1992) — modèle insider/trader.

    p_i = (sqrt(z² + 4(1-z)π_i²) - z) / (2(1-z))
    z ∈ [0, 1) résolu par bisection pour Σ p_i = 1.
    """
    parsed = _parse_cotes(cote_a, cote_b)
    if parsed is None:
        return None, None
    implied = [1.0 / parsed[0], 1.0 / parsed[1]]
    if sum(implied) <= 1.0:
        return proba_no_vig_proportionnel_2way(cote_a, cote_b)

    z_lo, z_hi = 0.0, 0.999
    for _ in range(80):
        z = (z_lo + z_hi) / 2.0
        denom = 2.0 * (1.0 - z)
        if denom <= 1e-12:
            break
        total = 0.0
        for pi in implied:
            total += (math.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi) - z) / denom
        if total > 1.0:
            z_lo = z
        else:
            z_hi = z

    z = (z_lo + z_hi) / 2.0
    denom = 2.0 * (1.0 - z)
    if denom <= 1e-12:
        return proba_no_vig_proportionnel_2way(cote_a, cote_b)

    probs = [
        (math.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi) - z) / denom
        for pi in implied
    ]
    s = sum(probs)
    if s <= 0:
        return None, None
    return probs[0] / s, probs[1] / s


def cote_fair_2way(cote_prise: float, cote_partenaire: float, method: str = "shin") -> float | None:
    """Cote décimale fair (no-vig) pour le côté `cote_prise`."""
    if method == "proportionnel":
        p_a, _ = proba_no_vig_proportionnel_2way(cote_prise, cote_partenaire)
    else:
        p_a, _ = proba_no_vig_shin_2way(cote_prise, cote_partenaire)
    if p_a is None or p_a <= 0:
        return None
    return 1.0 / p_a
