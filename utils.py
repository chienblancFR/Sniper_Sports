"""
utils.py — Fonctions utilitaires partagées par le dashboard Streamlit.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ──────────────────────────────────────────────────────────────
# 🔒  AUTHENTIFICATION
# ──────────────────────────────────────────────────────────────
def verifier_authentification():
    """
    Affiche un écran de connexion par mot de passe.
    Le mot de passe est stocké dans st.secrets["MOT_DE_PASSE"]
    ou dans la variable d'environnement DASHBOARD_PASSWORD.
    """
    import os
    mot_de_passe_attendu = None

    try:
        mot_de_passe_attendu = st.secrets["MOT_DE_PASSE"]
    except Exception:
        mot_de_passe_attendu = os.environ.get("DASHBOARD_PASSWORD", "sniper2025")

    if "authentifie" not in st.session_state:
        st.session_state["authentifie"] = False

    if not st.session_state["authentifie"]:
        st.markdown("## 🔒 Accès Restreint")
        mdp = st.text_input("Mot de passe :", type="password", key="mdp_input")
        if st.button("Connexion"):
            if mdp == mot_de_passe_attendu:
                st.session_state["authentifie"] = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
        st.stop()


# ──────────────────────────────────────────────────────────────
# 🧹  NETTOYAGE DES DONNÉES
# ──────────────────────────────────────────────────────────────
def nettoyer_colonnes_numeriques(df: pd.DataFrame, colonnes: list) -> pd.DataFrame:
    """Convertit les colonnes spécifiées en float, remplace les erreurs par NaN."""
    for col in colonnes:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def convertir_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte et convertit la colonne de date (Date, date, kickoff…) en datetime naive."""
    candidates = ['Date', 'date', 'Kickoff', 'kickoff', 'datetime']
    for col in candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Supprimer le fuseau horaire si présent pour uniformiser les comparaisons
            if hasattr(df[col].dt, 'tz') and df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
            if col != 'Date':
                df.rename(columns={col: 'Date'}, inplace=True)
            break
    return df


# ──────────────────────────────────────────────────────────────
# 🛑  ALERTES DE CHARGEMENT
# ──────────────────────────────────────────────────────────────
def afficher_alertes_chargement(statut: str, df: pd.DataFrame, msg_succes: str = ""):
    """
    Affiche un message selon le statut du chargement.
    Ne fait rien si statut == 'ok' et df non vide.
    """
    if statut == "error":
        st.error("⚠️ Impossible de charger les données. Vérifiez l'URL source ou la connexion.")
        st.stop()
    elif statut == "missing":
        st.warning("⚠️ Fichier de données introuvable (URL distante et fichier local).")
        st.stop()
    elif statut == "empty" or (statut == "ok" and df.empty):
        if msg_succes:
            st.info(msg_succes)
        else:
            st.info("Aucune donnée disponible pour le moment.")
            st.stop()
    # statut == "ok" et df non vide : on ne fait rien, la page s'affiche normalement


# ──────────────────────────────────────────────────────────────
# 🔍  FILTRES SIDEBAR
# ──────────────────────────────────────────────────────────────
def filtre_temporel_sidebar(df: pd.DataFrame, key_prefix: str = "live") -> pd.DataFrame:
    """
    Ajoute un filtre temporel (7j / 30j / 90j / Tout) dans la sidebar.
    key_prefix évite les collisions entre onglets Live / Back-test.
    """
    if df.empty or 'Date' not in df.columns:
        return df

    st.sidebar.subheader("📅 Période")
    periode = st.sidebar.radio(
        "Afficher les",
        options=["7 derniers jours", "30 derniers jours", "90 derniers jours", "Tout"],
        index=3,
        key=f"{key_prefix}_filtre_periode",
    )

    if periode != "Tout":
        jours = {"7 derniers jours": 7, "30 derniers jours": 30, "90 derniers jours": 90}[periode]
        date_min = pd.Timestamp.now() - pd.Timedelta(days=jours)
        df = df[df['Date'] >= date_min]

    return df


def filtre_ligue_sidebar(
    df: pd.DataFrame,
    key: str = "live_ligue",
    label_toutes: str = "Toutes les Ligues",
) -> pd.DataFrame:
    """Filtre par compétition dans la sidebar."""
    if df.empty or "Nom_Ligue" not in df.columns:
        return df
    ligues_dispo = sorted(df["Nom_Ligue"].unique().tolist())
    ligue_choisie = st.sidebar.selectbox(
        "🏆 Compétition :", [label_toutes] + ligues_dispo, key=key
    )
    if ligue_choisie != label_toutes:
        df = df[df["Nom_Ligue"] == ligue_choisie]
    return df


def filtre_marche_sidebar(
    df: pd.DataFrame,
    key: str = "live_marche",
    label_tous: str = "Tous les Marchés",
) -> pd.DataFrame:
    """Filtre par type de marché dans la sidebar."""
    if df.empty or "Type_Marche" not in df.columns:
        return df
    marches_dispo = sorted(df["Type_Marche"].unique().tolist())
    marche_choisi = st.sidebar.selectbox(
        f"📊 Marché ciblé :", [label_tous] + marches_dispo, key=key
    )
    if marche_choisi != label_tous:
        df = df[df["Type_Marche"] == marche_choisi]
    return df


# ──────────────────────────────────────────────────────────────
# 📉  CALCUL DU DRAWDOWN
# ──────────────────────────────────────────────────────────────
def calculer_max_drawdown(df: pd.DataFrame, col_profit: str, capital_initial: float):
    """
    Calcule le max drawdown en % et ajoute une colonne 'Bankroll' au DataFrame.
    Retourne (df_enrichi, max_drawdown_pct).
    """
    df = df.copy()
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)

    df['Bankroll'] = capital_initial + df[col_profit].cumsum()
    peak = df['Bankroll'].cummax()
    drawdown = (peak - df['Bankroll']) / peak
    max_dd_pct = drawdown.max() * 100 if not drawdown.empty else 0.0
    return df, max_dd_pct


def preparer_df_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise le CSV back-test : Profit_Unites = resultat × mise, tri chronologique.
    Colonne date_utc optionnelle (CSV regénéré après --report).
    """
    if df.empty:
        return df
    df = df.copy()
    cols_num = ['resultat', 'mise', 'clv', 'ev_modele', 'kelly', 'cote_h24', 'cote_cloture', 'h_val']
    df = nettoyer_colonnes_numeriques(df, cols_num)
    df['Profit_Unites'] = df['resultat'] * df['mise']
    if 'date_utc' in df.columns:
        df['Date'] = pd.to_datetime(df['date_utc'], errors='coerce', utc=True)
        if hasattr(df['Date'].dt, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        df = df.sort_values('Date', na_position='last').reset_index(drop=True)
    else:
        df = df.sort_values('fixture_id').reset_index(drop=True)
    return df


def calculer_drawdown_serie(profits, capital_initial: float = 100.0) -> float:
    """Max drawdown % — même logique que generer_rapport() dans backtest_football.py."""
    bankroll = capital_initial
    peak = bankroll
    max_dd = 0.0
    for p in profits:
        bankroll += float(p)
        if bankroll > peak:
            peak = bankroll
        if peak > 0:
            max_dd = max(max_dd, (peak - bankroll) / peak)
    return max_dd * 100


# ──────────────────────────────────────────────────────────────
# 🎯  OBJECTIFS CLV vs PINNACLE (AH — long terme)
# ──────────────────────────────────────────────────────────────
# « Battre Pinnacle significativement » = CLV ≥ cible ligue ET t ≥ 2 ET n ≥ min_n
CLV_CIBLE_PORTFOLIO_AH = 0.0050       # +0.50% portfolio AH multi-ligues
CLV_T_STAT_SIGNIFICATIF = 2.0
CLV_T_STAT_CONSERVATEUR = 2.5
CLV_MIN_N_LIGUE_AH = 120
CLV_MIN_N_PORTFOLIO_AH = 1500

# ligue_id → (cible CLV, n min AH, tier marché)
# Top = Big 5 + PL (marchés très efficient, cible plus basse)
# Mid = ligues secondaires européennes à volume correct
# Niche = ligues plus petites / volatiles
CIBLES_CLV_AH = {
    39:  (0.0035, 150, "Top"),     # Premier League
    140: (0.0035, 150, "Top"),     # La Liga
    78:  (0.0035, 150, "Top"),     # Bundesliga
    135: (0.0035, 150, "Top"),     # Serie A
    61:  (0.0035, 150, "Top"),     # Ligue 1
    88:  (0.0050, 120, "Mid"),     # Eredivisie
    94:  (0.0050, 120, "Mid"),     # Primeira Liga
    40:  (0.0045, 150, "Mid"),     # Championship
    136: (0.0050, 120, "Mid"),     # Serie B
    141: (0.0050, 120, "Mid"),     # LaLiga 2
    253: (0.0045, 100, "Niche"),   # MLS
    203: (0.0050, 100, "Niche"),   # Süper Lig
    113: (0.0050, 80,  "Niche"),   # Allsvenskan
    71:  (0.0050, 100, "Niche"),   # Série A Brésil
    103: (0.0050, 80,  "Niche"),   # Eliteserien
    144: (0.0050, 100, "Niche"),   # Jupiler Pro
}
CIBLE_CLV_AH_DEFAULT = (0.0050, 120, "Mid")

# Seuils EV minimum AH en backtest (P1 volume) — Top = marchés sharp
EV_MIN_SPREADS_TIER = {
    "Top": 0.07,
    "Mid": 0.06,
    "Niche": 0.05,
}


def get_ev_min_spreads_ligue(ligue_id, default: float = 0.05) -> float:
    """EV min AH selon tier ligue (backtest / réduction volume)."""
    _, _, tier = get_cible_clv_ligue(ligue_id)
    return EV_MIN_SPREADS_TIER.get(tier, default)


def get_cible_clv_ligue(ligue_id) -> tuple[float, int, str]:
    """Retourne (cible_clv, min_n_ah, tier) pour une ligue."""
    return CIBLES_CLV_AH.get(int(ligue_id), CIBLE_CLV_AH_DEFAULT)


def t_stat_clv(clv_vals) -> float:
    """t-stat CLV moyen (H-24 vs clôture Pinnacle)."""
    import numpy as np
    vals = [float(x) for x in clv_vals if x is not None]
    if len(vals) < 2:
        return float("nan")
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1))
    if s == 0:
        return float("nan")
    return m / (s / np.sqrt(len(vals)))


def clv_min_detectable(clv_vals, t_crit: float = CLV_T_STAT_SIGNIFICATIF) -> float:
    """CLV minimum détectable à n actuel (seuil statistique local)."""
    import numpy as np
    vals = [float(x) for x in clv_vals if x is not None]
    n = len(vals)
    if n < 2:
        return float("nan")
    std = float(np.std(vals, ddof=1))
    return t_crit * std / np.sqrt(n)


def evaluer_statut_clv_ligue(
    clv: float,
    t: float,
    n: int,
    ligue_id: int,
    t_min: float = CLV_T_STAT_SIGNIFICATIF,
) -> tuple[str, str]:
    """
    Statut long terme vs Pinnacle pour une ligue (AH).
    Retourne (emoji, libellé court).
    """
    import math
    cible, min_n, _tier = get_cible_clv_ligue(ligue_id)
    t_ok = t is not None and not (isinstance(t, float) and math.isnan(t))
    if n < max(30, min_n // 3):
        return "⏳", f"échantillon faible ({n}/{min_n})"
    if clv >= cible and t_ok and t >= t_min - 1e-6 and n >= min_n:
        return "✅", "bat Pinnacle (significatif)"
    if clv >= cible and n < min_n:
        return "🔄", f"CLV ok ({n}/{min_n} paris)"
    if clv >= cible * 0.7 and clv > 0 and t_ok and t >= 1.65:
        if n < min_n:
            return "🔄", f"edge probable ({n}/{min_n} paris)"
        return "🔄", "CLV ok, t sous seuil — confirmer"
    if clv > 0 and n >= min_n:
        return "⚠️", f"sous cible {cible:.2%}"
    if clv <= 0 and n >= max(50, min_n // 2):
        return "❌", "pas d'edge CLV"
    return "⏳", "en cours"


def construire_tableau_objectifs_clv_ah(
    signaux=None,
    df: pd.DataFrame | None = None,
    noms_ligue: dict | None = None,
) -> pd.DataFrame:
    """
    Tableau objectifs CLV AH par ligue (backtest tuples ou DataFrame CSV).
    Colonnes : ligue_id, ligue, tier, n_ah, clv_ah, t_stat, cible_clv,
               min_n, clv_min_sig, statut, libelle.
    """
    import numpy as np

    rows_raw: dict[int, dict] = {}
    if signaux is not None:
        for s in signaux:
            if s[3] != "spreads":
                continue
            lid = int(s[1])
            bucket = rows_raw.setdefault(lid, {"n": 0, "clvs": []})
            bucket["n"] += 1
            if s[14] is not None:
                bucket["clvs"].append(float(s[14]))
    elif df is not None and not df.empty:
        sub = df[df["market"] == "spreads"] if "market" in df.columns else df
        for lid, g in sub.groupby("ligue_id"):
            clvs = [float(c) for c in g["clv"].dropna().tolist()]
            rows_raw[int(lid)] = {"n": len(g), "clvs": clvs}
    else:
        return pd.DataFrame()

    noms_ligue = noms_ligue or {}
    out = []
    for lid in sorted(set(CIBLES_CLV_AH.keys()) | set(rows_raw.keys())):
        bucket = rows_raw.get(lid, {"n": 0, "clvs": []})
        clvs = bucket["clvs"]
        n = bucket["n"]
        cible, min_n, tier = get_cible_clv_ligue(lid)
        clv_m = float(np.mean(clvs)) if clvs else float("nan")
        t = t_stat_clv(clvs) if clvs else float("nan")
        clv_sig = clv_min_detectable(clvs) if clvs else float("nan")
        statut, lib = evaluer_statut_clv_ligue(
            clv_m if clvs else 0.0, t, n, lid,
        )
        nom = noms_ligue.get(lid, str(lid))
        out.append({
            "ligue_id": lid,
            "ligue": nom,
            "tier": tier,
            "n_ah": n,
            "clv_ah": clv_m,
            "t_stat": t,
            "cible_clv": cible,
            "min_n": min_n,
            "clv_min_sig": clv_sig,
            "statut": statut,
            "libelle": lib,
        })
    return pd.DataFrame(out)


def formater_objectifs_clv_ah_texte(df_obj: pd.DataFrame) -> str:
    """Bloc texte pour le rapport CLI backtest."""
    if df_obj.empty:
        return "  (aucune donnée AH)"
    lines = [
        f"  {'Ligue':<18} {'Tier':<6} {'N':>5} {'CLV':>7} {'t':>5} "
        f"{'Cible':>7} {'Min_n':>5}  {'Statut':<4} Détail",
        f"  {'─'*18} {'─'*6} {'─'*5} {'─'*7} {'─'*5} {'─'*7} {'─'*5}  {'─'*4} {'─'*20}",
    ]
    for _, r in df_obj.sort_values(["statut", "ligue"]).iterrows():
        clv_s = f"{r['clv_ah']:+.2%}" if r["n_ah"] else "  n/a"
        t_s = f"{r['t_stat']:+.1f}" if r["n_ah"] >= 2 and not pd.isna(r["t_stat"]) else " n/a"
        lines.append(
            f"  {str(r['ligue']):<18} {r['tier']:<6} {int(r['n_ah']):>5} "
            f"{clv_s:>7} {t_s:>5} {r['cible_clv']:>+6.2%} {int(r['min_n']):>5}  "
            f"{r['statut']:<4} {r['libelle']}"
        )
    n_ok = (df_obj["statut"] == "✅").sum()
    n_tot = len(df_obj)
    lines.append(
        f"\n  Portfolio cible : CLV AH ≥ {CLV_CIBLE_PORTFOLIO_AH:.2%}, "
        f"t ≥ {CLV_T_STAT_SIGNIFICATIF:.1f}, n ≥ {CLV_MIN_N_PORTFOLIO_AH}"
    )
    lines.append(f"  Ligues validées : {n_ok}/{n_tot}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 🎯  CALIBRATION & BRIER (journal live / backtest)
# ──────────────────────────────────────────────────────────────
TRANCHES_EDGE_CALIBRATION = [
    (2.0, 4.0, "2-4%"),
    (4.0, 6.0, "4-6%"),
    (6.0, 9.0, "6-9%"),
    (9.0, 15.0, "9-15%"),
    (15.0, 30.0, "15%+"),
]


def preparer_calibration_journal(
    df: pd.DataFrame,
    col_cote: str = "Vraie_Cote_Bot",
    col_statut: str = "Statut",
) -> pd.DataFrame:
    """Paris clôturés avec probabilité modèle (1 / cote fair) et issue binaire."""
    if df.empty or col_cote not in df.columns or col_statut not in df.columns:
        return pd.DataFrame()
    out = df[df[col_statut].isin(["GAGNÉ", "PERDU"])].copy()
    out["cote_modele"] = pd.to_numeric(out[col_cote], errors="coerce")
    out = out[out["cote_modele"] > 1.0]
    out["p_model"] = 1.0 / out["cote_modele"]
    out["outcome"] = (out[col_statut] == "GAGNÉ").astype(float)
    if "Edge(%)" in out.columns:
        out["edge_pct"] = pd.to_numeric(out["Edge(%)"], errors="coerce")
    return out


def preparer_calibration_foot(
    df: pd.DataFrame,
    col_prob: str = "Prob_Modele",
    col_statut: str = "Statut",
    col_edge: str = "Edge",
) -> pd.DataFrame:
    """
    Journal foot (historique_sniper.csv / paris_log export) :
    p_model = Prob_Modele (probabilité implicite 1/cote_fair),
    outcome = 1 si profit positif (WON ou HALF-WON).
    Exclut les lignes historiques où Prob_Modele stockait par erreur l'EV (~0.05-0.15).
    """
    if df.empty or col_prob not in df.columns or col_statut not in df.columns:
        return pd.DataFrame()
    termines = df[df[col_statut].isin(["WON", "HALF-WON", "LOST", "HALF-LOST"])].copy()
    termines["p_model"] = pd.to_numeric(termines[col_prob], errors="coerce")
    termines = termines[termines["p_model"].between(0.15, 0.95)]
    termines["outcome"] = termines[col_statut].isin(["WON", "HALF-WON"]).astype(float)
    if col_edge in termines.columns:
        termines["edge_pct"] = pd.to_numeric(termines[col_edge], errors="coerce") * 100
    return termines


def calculer_brier_score(df_cal: pd.DataFrame) -> float | None:
    if df_cal.empty:
        return None
    return float(((df_cal["p_model"] - df_cal["outcome"]) ** 2).mean())


def calibration_par_probabilite(
    df_cal: pd.DataFrame,
    bin_width: float = 0.05,
    p_min: float = 0.35,
    p_max: float = 0.70,
) -> pd.DataFrame:
    """Reliability diagram : prob prédite vs fréquence réelle par tranche."""
    if df_cal.empty:
        return pd.DataFrame()
    rows = []
    p = p_min
    while p < p_max:
        hi = min(p + bin_width, p_max)
        sub = df_cal[(df_cal["p_model"] >= p) & (df_cal["p_model"] < hi)]
        if not sub.empty:
            rows.append({
                "Tranche": f"{p:.0%}-{hi:.0%}",
                "p_pred": sub["p_model"].mean(),
                "p_reel": sub["outcome"].mean(),
                "N": len(sub),
            })
        p = hi
    return pd.DataFrame(rows)


def calibration_par_edge(
    df_cal: pd.DataFrame,
    tranches: list | None = None,
) -> pd.DataFrame:
    """Win rate réel vs edge détecté (comme backtest foot)."""
    if df_cal.empty or "edge_pct" not in df_cal.columns:
        return pd.DataFrame()
    tranches = tranches or TRANCHES_EDGE_CALIBRATION
    rows = []
    for lo, hi, label in tranches:
        if label.endswith("+"):
            sub = df_cal[df_cal["edge_pct"] >= lo]
        else:
            sub = df_cal[(df_cal["edge_pct"] >= lo) & (df_cal["edge_pct"] < hi)]
        if sub.empty:
            continue
        rows.append({
            "Tranche": label,
            "Win_Rate_Reel": sub["outcome"].mean() * 100,
            "Edge_Moyen": sub["edge_pct"].mean(),
            "N": len(sub),
        })
    return pd.DataFrame(rows)


def formater_rapport_calibration_texte(
    df_cal: pd.DataFrame,
    min_paris: int = 5,
    titre: str = "CALIBRATION",
) -> str:
    """Rapport texte pour CLI / logs."""
    n = len(df_cal)
    if n < min_paris:
        return f"Calibration : {n} pari(s) clôturé(s) — minimum {min_paris} requis."
    brier = calculer_brier_score(df_cal)
    bss = None
    baseline = float((df_cal["p_model"] * (1 - df_cal["p_model"])).mean())
    if baseline > 0:
        bss = 1.0 - (brier / baseline)
    lines = [
        f"{'-' * 50}",
        f"  {titre} — {n} paris clôturés",
        f"{'-' * 50}",
        f"  Brier score : {brier:.4f}  (0 = parfait, ~0.25 = coin flip 50/50)",
    ]
    if bss is not None:
        lines.append(f"  BSS (vs baseline p*(1-p)) : {bss:+.3f}")
    df_prob = calibration_par_probabilite(df_cal)
    if not df_prob.empty:
        lines.append(f"\n  Par probabilité modèle :")
        for _, r in df_prob.iterrows():
            lines.append(
                f"    {r['Tranche']:<12} n={int(r['N']):>3}  "
                f"prédit={r['p_pred']:.1%}  réel={r['p_reel']:.1%}"
            )
    df_edge = calibration_par_edge(df_cal)
    if not df_edge.empty:
        lines.append(f"\n  Par tranche d'edge :")
        for _, r in df_edge.iterrows():
            lines.append(
                f"    Edge {r['Tranche']:<8} n={int(r['N']):>3}  "
                f"win={r['Win_Rate_Reel']:.1f}%  edge_moy={r['Edge_Moyen']:+.1f}%"
            )
    return "\n".join(lines)


def creer_graphique_calibration_prob(df_bins: pd.DataFrame) -> go.Figure:
    """Courbe fiabilité : prob prédite vs observée + diagonale parfaite."""
    fig = go.Figure()
    if df_bins.empty:
        return fig
    fig.add_trace(go.Scatter(
        x=df_bins["p_pred"] * 100,
        y=df_bins["p_reel"] * 100,
        mode="lines+markers",
        name="Observé",
        line=dict(color="#00BFFF", width=2.5),
        marker=dict(size=10),
        text=[f"n={n}" for n in df_bins["N"]],
        textposition="top center",
    ))
    lo = max(0, df_bins["p_pred"].min() * 100 - 5)
    hi = min(100, df_bins["p_pred"].max() * 100 + 5)
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        name="Calibration parfaite",
        line=dict(color="#FFD700", dash="dot", width=1.5),
    ))
    fig.update_layout(
        xaxis_title="Probabilité modèle (%)",
        yaxis_title="Fréquence de victoire réelle (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    appliquer_theme_dark(fig)
    return fig


def creer_graphique_calibration_edge(df_calib: pd.DataFrame) -> go.Figure:
    """Barres win rate réel par tranche d'edge."""
    fig = go.Figure()
    if df_calib.empty:
        return fig
    fig.add_trace(go.Bar(
        x=df_calib["Tranche"],
        y=df_calib["Win_Rate_Reel"],
        marker_color="#00BFFF",
        name="Win rate réel (%)",
        text=[
            f"{v:.1f}%<br>n={n}<br>edge {e:+.1f}%"
            for v, n, e in zip(df_calib["Win_Rate_Reel"], df_calib["N"], df_calib["Edge_Moyen"])
        ],
        textposition="outside",
    ))
    fig.add_hline(
        y=50, line_dash="dot", line_color="#FFD700",
        annotation_text="50% (neutre)", annotation_position="bottom right",
    )
    fig.update_layout(
        yaxis_title="Fréquence de victoire (%)",
        xaxis_title="Tranche d'edge détecté",
        yaxis_range=[0, max(100, df_calib["Win_Rate_Reel"].max() + 10)],
        showlegend=False,
    )
    appliquer_theme_dark(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# 📈  GRAPHIQUES PLOTLY
# ──────────────────────────────────────────────────────────────
def appliquer_theme_dark(fig: go.Figure):
    """Applique le thème sombre cohérent à toutes les figures Plotly."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF', size=12),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.08)',
            linecolor='rgba(255,255,255,0.2)',
            tickcolor='rgba(255,255,255,0.3)',
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.08)',
            linecolor='rgba(255,255,255,0.2)',
            tickcolor='rgba(255,255,255,0.3)',
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )


def creer_graphique_bankroll(
    df: pd.DataFrame,
    hover_data: list = None,
    couleur: str = '#00FF00',
    unite: str = 'Unités'
) -> go.Figure:
    """
    Courbe de croissance de la bankroll (valeur cumulée du P&L).
    Attend une colonne 'Bankroll' dans df (générée par calculer_max_drawdown).
    """
    fig = go.Figure()

    x_vals = df['Date'] if 'Date' in df.columns else df.index

    # Zone de remplissage sous la courbe
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=df['Bankroll'],
        mode='lines',
        fill='tozeroy',
        line=dict(color=couleur, width=2.5),
        fillcolor=f'rgba({_hex_to_rgb(couleur)},0.10)',
        name=f'Capital ({unite})',
        hovertemplate=(
            '<b>%{x|%d/%m/%Y}</b><br>'
            f'Capital : %{{y:.2f}} {unite}<extra></extra>'
        )
    ))

    fig.update_layout(
        yaxis_title=f"Capital ({unite})",
        xaxis_title="",
        showlegend=False,
        hovermode='x unified',
    )
    appliquer_theme_dark(fig)
    return fig


def creer_graphique_pl_marche(
    df: pd.DataFrame,
    col_x: str,
    titre_x: str = "Catégorie",
    titre_y: str = "P&L"
) -> go.Figure:
    """
    Graphique à barres : P&L par catégorie (marché, ligue…).
    Attend les colonnes P_and_L et Volume dans df.
    """
    couleurs = ['#00FF00' if v >= 0 else '#FF4500' for v in df['P_and_L']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[col_x],
        y=df['P_and_L'],
        marker_color=couleurs,
        text=[f"{v:+.2f} u<br>{n} paris" for v, n in zip(df['P_and_L'], df['Volume'])],
        textposition='outside',
        name='P&L'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="#FFFFFF", opacity=0.4)
    fig.update_layout(
        xaxis_title=titre_x,
        yaxis_title=titre_y,
        showlegend=False,
    )
    appliquer_theme_dark(fig)
    return fig


def creer_graphique_clv_cumule(
    df_clv: pd.DataFrame,
    col_marche: str = "Type_Marche",
    couleurs: dict | None = None,
) -> go.Figure:
    """CLV moyen cumulé par marché — axe X = Date si disponible."""
    fig = go.Figure()
    couleurs_defaut = {"Totals (Buts)": "#FF4500", "Handicap Asiatique": "#00BFFF"}
    couleurs_map = {**couleurs_defaut, **(couleurs or {})}
    palette = ["#00BFFF", "#FF4500", "#00FF00", "#FFD700", "#DA70D6", "#FF69B4"]
    a_dates = "Date" in df_clv.columns and df_clv["Date"].notna().any()

    for i, marche in enumerate(df_clv[col_marche].unique()):
        df_cat = df_clv[df_clv[col_marche] == marche].sort_values("Date").reset_index(drop=True)
        df_cat["CLV_Pct"] = df_cat["CLV"] * 100
        df_cat["CLV_Moy_Cumulee"] = df_cat["CLV_Pct"].expanding().mean()
        axe_x = df_cat["Date"] if a_dates else df_cat.index
        couleur = couleurs_map.get(marche) or palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=axe_x,
            y=df_cat["CLV_Moy_Cumulee"],
            mode="lines",
            name=marche,
            line=dict(color=couleur, width=2.5),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#FFFFFF", opacity=0.5)
    fig.update_layout(
        yaxis_title="Beat The Close moyen (%)",
        xaxis_title="Date du pari" if a_dates else "Volume de paris",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    appliquer_theme_dark(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# 🔧  HELPERS INTERNES
# ──────────────────────────────────────────────────────────────
def _hex_to_rgb(hex_color: str) -> str:
    """Convertit '#RRGGBB' en 'R,G,B' pour les rgba() Plotly."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r},{g},{b}"
