import os

import pandas as pd
import streamlit as st

from utils import (
    afficher_alertes_chargement,
    calculer_brier_score,
    calculer_max_drawdown,
    calibration_par_edge,
    calibration_par_probabilite,
    convertir_dates,
    creer_graphique_bankroll,
    creer_graphique_calibration_edge,
    creer_graphique_calibration_prob,
    creer_graphique_clv_cumule,
    creer_graphique_pl_marche,
    filtre_marche_sidebar,
    filtre_temporel_sidebar,
    nettoyer_colonnes_numeriques,
    preparer_calibration_journal,
    verifier_authentification,
)

# ==========================================
# ⚙️ CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="NHL Quant Dashboard", page_icon="🏒", layout="wide")

verifier_authentification()

st.title("🏒 Centre de Commandement : Sniper NHL Oméga")

URL_JOURNAL = "https://chienblanc.pythonanywhere.com/data/journal_trading_nhl_SEC2026xOmG.csv"
FICHIER_LOCAL = "journal_trading_nhl_SEC2026xOmG.csv"
CAPITAL_INITIAL = 1000.0


def _charger_csv(url: str, fichier_local: str):
    """Tente l'URL distante, puis le fichier local."""
    try:
        df = pd.read_csv(url)
        if not df.empty:
            return df, "ok", f"PA · {fichier_local}"
    except Exception:
        pass

    local_path = os.path.join(os.getcwd(), fichier_local)
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            if not df.empty:
                return df, "ok", f"local · {fichier_local}"
            return df, "empty", f"local · {fichier_local} (vide)"
        except Exception:
            return pd.DataFrame(), "error", f"local · {fichier_local}"

    return pd.DataFrame(), "missing", "introuvable"


def extraire_type_pari(pari_str):
    pari_upper = str(pari_str).upper()
    if "OVER" in pari_upper or "UNDER" in pari_upper:
        return "Over/Under (Totaux)"
    if "PUCK LINE" in pari_upper or "HANDICAP" in pari_upper:
        return "Puck Line / Handicap"
    return "Moneyline"


def enrichir_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "Pari" in df.columns:
        df["Type_Marche"] = df["Pari"].apply(extraire_type_pari)
    if "Cote_Prise" in df.columns and "Cote_CLV" in df.columns:
        mask = (df["Cote_Prise"] > 1) & (df["Cote_CLV"] > 1)
        df["CLV"] = None
        df.loc[mask, "CLV"] = df.loc[mask, "Cote_Prise"] / df.loc[mask, "Cote_CLV"] - 1
    return df


@st.cache_data(ttl=60)
def load_data():
    df, statut, source = _charger_csv(URL_JOURNAL, FICHIER_LOCAL)
    if statut != "ok":
        return df, statut, source

    colonnes_numeriques = [
        "Vraie_Cote_Bot", "Cote_Prise", "Cote_CLV",
        "Edge(%)", "Risque(%)", "Mise_€", "P&L",
    ]
    df = nettoyer_colonnes_numeriques(df, colonnes_numeriques)
    df = convertir_dates(df)
    df = enrichir_dataframe(df)
    return df, "ok", source


# ==========================================
# 🎛️ SIDEBAR — CONTRÔLES
# ==========================================
st.sidebar.header("⚙️ Contrôles")
if st.sidebar.button("🔄 Rafraîchir les données", use_container_width=True):
    load_data.clear()
    st.rerun()

df, statut_chargement, source = load_data()
st.caption(f"📡 Live : **{source}** (cache 60 s)")

afficher_alertes_chargement(
    statut_chargement, df,
    msg_succes="🏒 Le radar NHL Oméga est armé. En attente des premières transactions..."
)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtres Live")

df_live = filtre_temporel_sidebar(df, key_prefix="nhl")
df_live = filtre_marche_sidebar(
    df_live, key="nhl_marche", label_tous="Tous les Marchés"
)

if not df_live.empty and "Visiteur" in df_live.columns and "Local" in df_live.columns:
    equipes = sorted(set(df_live["Visiteur"].dropna()) | set(df_live["Local"].dropna()))
    equipe_choisie = st.sidebar.selectbox(
        "🏒 Équipe :", ["Toutes les Équipes"] + equipes, key="nhl_equipe"
    )
    if equipe_choisie != "Toutes les Équipes":
        df_live = df_live[
            (df_live["Visiteur"] == equipe_choisie) | (df_live["Local"] == equipe_choisie)
        ]

if "Statut" not in df_live.columns and not df_live.empty:
    st.error("Format CSV live invalide (colonne `Statut` manquante).")
    st.stop()
if df_live.empty:
    if df.empty:
        st.stop()
    st.info("Aucune transaction ne correspond aux filtres sélectionnés.")
    st.stop()

# ==========================================
# 📊 KPI
# ==========================================
df_termines = df_live[df_live["Statut"].isin(["GAGNÉ", "PERDU"])].copy()
df_attente = df_live[df_live["Statut"] == "EN ATTENTE"].copy()

total_pl = df_termines["P&L"].sum() if not df_termines.empty else 0.0
capital_actuel = CAPITAL_INITIAL + total_pl
total_mise = df_termines["Mise_€"].sum() if not df_termines.empty else 0.0
roi = (total_pl / total_mise * 100) if total_mise > 0 else 0
winrate = (
    len(df_termines[df_termines["Statut"] == "GAGNÉ"]) / len(df_termines) * 100
    if not df_termines.empty else 0
)

max_dd_pct = 0.0
if not df_termines.empty:
    df_termines, max_dd_pct = calculer_max_drawdown(df_termines, "P&L", CAPITAL_INITIAL)

df_clv_ok = df_termines[(df_termines["CLV"].notna()) & (df_termines["CLV"] != 0)] if not df_termines.empty else pd.DataFrame()
clv_moy = df_clv_ok["CLV"].mean() * 100 if not df_clv_ok.empty else None
edge_col = "Edge(%)" if "Edge(%)" in df_termines.columns else None
df_edge_ok = df_termines[df_termines[edge_col].notna()] if edge_col and not df_termines.empty else pd.DataFrame()
edge_moy = df_edge_ok[edge_col].mean() if not df_edge_ok.empty else None

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Capital Actuel", f"{capital_actuel:.2f} €", f"{total_pl:+.2f} €")
c2.metric("ROI", f"{roi:.2f} %")
c3.metric("Winrate", f"{winrate:.1f} %")
c4.metric("Max Drawdown", f"{max_dd_pct:.1f} %")
c5.metric("Ordres en Cours", f"{len(df_attente)}")
c6.metric(
    "CLV moyen",
    f"{clv_moy:+.2f} %" if clv_moy is not None else "N/A",
    help=f"Sur {len(df_clv_ok)} paris clôturés avec CLV",
)
c7.metric(
    "Edge moyen",
    f"{edge_moy:+.2f} %" if edge_moy is not None else "N/A",
    help=f"Sur {len(df_edge_ok)} paris avec edge enregistré",
)

st.markdown("---")

# ==========================================
# 📈 BANKROLL
# ==========================================
st.subheader("📈 Évolution du Capital (Sniper NHL Oméga)")
if not df_termines.empty:
    fig_bankroll = creer_graphique_bankroll(
        df_termines,
        hover_data=["Date", "Visiteur", "Local", "Pari", "P&L"],
        couleur="#00FF00",
        unite="€",
    )
    st.plotly_chart(fig_bankroll, use_container_width=True)
else:
    st.info("La courbe de croissance s'affichera dès qu'un match sera clôturé.")

col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.subheader("🎯 Rentabilité et Volume par Type de Pari")
    if not df_termines.empty:
        pl_detail = df_termines.groupby("Type_Marche").agg(
            P_and_L=("P&L", "sum"),
            Volume=("P&L", "count"),
        ).reset_index()
        fig_segment = creer_graphique_pl_marche(
            pl_detail,
            col_x="Type_Marche",
            titre_x="Marché",
            titre_y="Profit / Perte (€)",
        )
        st.plotly_chart(fig_segment, use_container_width=True)
    else:
        st.write("Données insuffisantes.")

with col_droite:
    st.subheader("⚖️ Validation Mathématique : CLV par Marché")
    if not df_clv_ok.empty:
        fig_clv = creer_graphique_clv_cumule(df_clv_ok)
        st.plotly_chart(fig_clv, use_container_width=True)
    elif not df_termines.empty:
        st.write("En attente de données de clôture Pinnacle.")
    else:
        st.write("Données insuffisantes.")

# ==========================================
# 🎯 CALIBRATION & BRIER
# ==========================================
st.markdown("---")
st.subheader("🎯 Calibration du modèle (Brier)")
st.caption(
    "Diagnostic post-match : probabilité fair du bot (1 / Vraie_Cote_Bot) vs résultats réels. "
    "N'influence pas les paris live."
)

NHL_CALIB_MIN = 5
df_cal = preparer_calibration_journal(df_termines)

if len(df_cal) >= NHL_CALIB_MIN:
    brier = calculer_brier_score(df_cal)
    b1, b2, b3 = st.columns(3)
    b1.metric("Brier score", f"{brier:.4f}", help="0 = parfait · ~0.25 = aléatoire sur 50/50")
    b2.metric("Paris analysés", len(df_cal))
    b3.metric(
        "Prob. moyenne modèle",
        f"{df_cal['p_model'].mean() * 100:.1f} %",
        help="Moyenne des probabilités implicites au moment du pari",
    )

    col_cal_prob, col_cal_edge = st.columns(2)

    with col_cal_prob:
        st.markdown("**Fiabilité par probabilité modèle**")
        df_bins = calibration_par_probabilite(df_cal)
        if not df_bins.empty:
            st.plotly_chart(creer_graphique_calibration_prob(df_bins), use_container_width=True)
        else:
            st.info("Pas assez de paris dans les tranches de probabilité.")

    with col_cal_edge:
        st.markdown("**Win rate par tranche d'edge**")
        df_edge_cal = calibration_par_edge(df_cal)
        if not df_edge_cal.empty:
            st.plotly_chart(creer_graphique_calibration_edge(df_edge_cal), use_container_width=True)
        else:
            st.info("Colonne Edge(%) absente ou tranches vides.")

    filtre_cal = st.selectbox(
        "Filtrer la calibration par marché",
        ["Tous"] + sorted(df_cal["Type_Marche"].dropna().unique().tolist())
        if "Type_Marche" in df_cal.columns else ["Tous"],
        key="nhl_cal_marche",
    )
    if filtre_cal != "Tous" and "Type_Marche" in df_cal.columns:
        df_cal_f = df_cal[df_cal["Type_Marche"] == filtre_cal]
        if len(df_cal_f) >= 3:
            st.caption(
                f"**{filtre_cal}** — Brier {calculer_brier_score(df_cal_f):.4f} "
                f"({len(df_cal_f)} paris)"
            )
        else:
            st.caption(f"**{filtre_cal}** — trop peu de paris pour un sous-échantillon fiable.")
else:
    st.info(
        f"En attente de **{NHL_CALIB_MIN} paris clôturés** minimum "
        f"({len(df_cal)} actuellement). Le diagnostic s'affichera au fil de la saison."
    )

# ==========================================
# 📡 PARIS EN ATTENTE
# ==========================================
st.markdown("---")
st.subheader("📡 Radar NHL : Signaux Actifs / En attente de dénouement")

if not df_attente.empty:
    df_att_display = df_attente.copy()
    colonnes_attente = [
        "Date", "Visiteur", "Local", "Pari", "Type_Marche",
        "Cote_Prise", "Mise_€", "Edge(%)",
    ]
    if "CLV" in df_att_display.columns:
        colonnes_attente.append("CLV")
        df_att_display["CLV"] = (df_att_display["CLV"] * 100).round(2)
        df_att_display = df_att_display.rename(columns={"CLV": "CLV (%)"})
        colonnes_attente[-1] = "CLV (%)"

    colonnes_attente = [c for c in colonnes_attente if c in df_att_display.columns]

    def style_clv(val):
        try:
            v = float(val)
            if v > 1:
                return "color: #00FF00; font-weight: bold"
            if v < -1:
                return "color: #FF4500; font-weight: bold"
            return "color: #FFD700"
        except Exception:
            return ""

    df_sorted = df_att_display[colonnes_attente].sort_values("Date")
    if "CLV (%)" in colonnes_attente:
        st.dataframe(
            df_sorted.style.map(style_clv, subset=["CLV (%)"]),
            use_container_width=True,
        )
    else:
        st.dataframe(df_sorted, use_container_width=True)
else:
    st.success("Aucun ordre en cours sur les marchés. Le Sniper est en veille.")

# ==========================================
# 📰 JOURNAL DE BORD
# ==========================================
st.markdown("---")
st.subheader("📰 Journal de bord : 10 dernières rencontres clôturées")

if not df_termines.empty:
    df_derniers = df_termines.sort_values("Date", ascending=False).head(10)
    colonnes_hist = [
        "Date", "Visiteur", "Local", "Pari", "Type_Marche",
        "Cote_Prise", "Mise_€", "Statut", "P&L",
    ]
    colonnes_hist = [c for c in colonnes_hist if c in df_derniers.columns]
    df_derniers = df_derniers[colonnes_hist].copy()
    if "P&L" in df_derniers.columns:
        df_derniers["P&L"] = df_derniers["P&L"].round(2)

    def style_statut(val):
        if val == "GAGNÉ":
            return "color: #00FF00; font-weight: bold"
        if val == "PERDU":
            return "color: #FF4500; font-weight: bold"
        return ""

    st.dataframe(
        df_derniers.style.map(style_statut, subset=["Statut"]),
        use_container_width=True,
    )
else:
    st.write("Aucun pari terminé pour le moment.")
