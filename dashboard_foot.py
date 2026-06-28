import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import (
    verifier_authentification,
    nettoyer_colonnes_numeriques,
    convertir_dates,
    afficher_alertes_chargement,
    filtre_temporel_sidebar,
    calculer_max_drawdown,
    creer_graphique_bankroll,
    creer_graphique_pl_marche,
    appliquer_theme_dark,
)

# ==========================================
# ⚙️ CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="Football Quant Dashboard", page_icon="⚽", layout="wide")

# ==========================================
# 🛑 ÉCRAN DE SÉCURITÉ
# ==========================================
verifier_authentification()

st.title("⚽ Centre de Commandement : Sniper Football")

URL_FOOT = "https://chienblanc.pythonanywhere.com/data/historique_sniper.csv"
CAPITAL_INITIAL = 100.0

# ==========================================
# 📥 CHARGEMENT & NETTOYAGE DES DONNÉES
# ==========================================
@st.cache_data(ttl=60)
def load_football_data():
    try:
        df = pd.read_csv(URL_FOOT)
    except Exception:
        return pd.DataFrame(), "error"

    if df.empty:
        return df, "empty"

    colonnes_numeriques = ["Cote_Prise", "Mise", "Cote_Cloture", "Edge", "Prob_Modele", "CLV", "Profit_Unites"]
    df = nettoyer_colonnes_numeriques(df, colonnes_numeriques)
    df = convertir_dates(df)

    def extraire_marche(ligue_str):
        if "[totals]" in str(ligue_str):
            return "Totals (Buts)"
        return "Handicap Asiatique"

    df['Type_Marche'] = df['Ligue'].apply(extraire_marche)
    df['Nom_Ligue'] = df['Ligue'].apply(lambda x: str(x).split(" [")[0])

    return df, "ok"

df, statut_chargement = load_football_data()

# ==========================================
# 🛑 AFFICHAGE SI BASE VIDE OU ERREUR
# ==========================================
afficher_alertes_chargement(
    statut_chargement, df,
    msg_succes="⚽ Le radar Football V25 est armé. En attente des premières transactions..."
)

# ==========================================
# 🔍 BARRE DE FILTRES DYNAMIQUES (SIDEBAR)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtres Écran Principal")

df = filtre_temporel_sidebar(df)

# Filtre par Compétition
if not df.empty and "Nom_Ligue" in df.columns:
    ligues_dispo = sorted(df["Nom_Ligue"].unique().tolist())
    ligue_choisie = st.sidebar.selectbox("🏆 Compétition :", ["Toutes les Ligues"] + ligues_dispo)
    if ligue_choisie != "Toutes les Ligues":
        df = df[df["Nom_Ligue"] == ligue_choisie]

# Filtre par type de Marché
if not df.empty and "Type_Marche" in df.columns:
    marches_dispo = sorted(df["Type_Marche"].unique().tolist())
    marche_choisi = st.sidebar.selectbox("📊 Marché ciblé :", ["Tous les Marchés"] + marches_dispo)
    if marche_choisi != "Tous les Marchés":
        df = df[df["Type_Marche"] == marche_choisi]

# ==========================================
# 📊 CALCUL DES KPI
# ==========================================
df_termines = df[df['Statut'].isin(['WON', 'HALF-WON', 'VOID', 'HALF-LOST', 'LOST'])].copy()
df_attente = df[df['Statut'] == 'PENDING'].copy()

total_pl = df_termines['Profit_Unites'].sum() if not df_termines.empty else 0.0
capital_actuel = CAPITAL_INITIAL + total_pl
total_mise = df_termines['Mise'].sum() if not df_termines.empty else 0.0
roi = (total_pl / total_mise * 100) if total_mise > 0 else 0

# Winrate adapté au foot (HALF-WON et HALF-LOST comptent)
rec_gagnants = len(df_termines[df_termines['Statut'].isin(['WON', 'HALF-WON'])])
rec_perdants = len(df_termines[df_termines['Statut'].isin(['LOST', 'HALF-LOST'])])
total_tranches = rec_gagnants + rec_perdants
winrate = (rec_gagnants / total_tranches * 100) if total_tranches > 0 else 0

max_dd_pct = 0.0
if not df_termines.empty:
    df_termines, max_dd_pct = calculer_max_drawdown(df_termines, 'Profit_Unites', CAPITAL_INITIAL)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Capital Actuel", f"{capital_actuel:.2f} U", f"{total_pl:.2f} U")
col2.metric("ROI", f"{roi:.2f} %")
col3.metric("Winrate (Hors Void)", f"{winrate:.1f} %")
col4.metric("Max Drawdown", f"{max_dd_pct:.1f} %")
col5.metric("Ordres en Cours", f"{len(df_attente)}")

st.markdown("---")

# ==========================================
# 📈 GRAPHIQUE 1 : COURBE DE CROISSANCE GLOBALE
# ==========================================
st.subheader("📈 Évolution du Capital (Stratégie Hybride Foot)")

if not df_termines.empty:
    fig_bankroll = creer_graphique_bankroll(
        df_termines,
        hover_data=['Date', 'Equipe', 'Profit_Unites', 'Nom_Ligue'],
        couleur='#00FF00',
        unite='Unités'
    )
    st.plotly_chart(fig_bankroll, use_container_width=True)
else:
    st.info("La courbe de croissance s'affichera dès qu'un match sera clôturé.")

col_gauche, col_droite = st.columns(2)

# ==========================================
# 🥧 GRAPHIQUE 2 : RENTABILITÉ ET VOLUME PAR MARCHÉ
# ==========================================
with col_gauche:
    st.subheader("🎯 Rentabilité et Volume par Type de Pari")
    if not df_termines.empty:
        pl_detail = df_termines.groupby('Type_Marche').agg(
            P_and_L=('Profit_Unites', 'sum'),
            Volume=('Profit_Unites', 'count')
        ).reset_index()

        fig_segment = creer_graphique_pl_marche(
            pl_detail,
            col_x='Type_Marche',
            titre_x="Marché",
            titre_y="Profit / Perte (U)"
        )
        st.plotly_chart(fig_segment, use_container_width=True)
    else:
        st.write("Données insuffisantes.")

# ==========================================
# 🎯 GRAPHIQUE 3 : CLV SEGMENTÉE (DÉTECTEUR D'EV)
# ==========================================
with col_droite:
    st.subheader("⚖️ Validation Mathématique : CLV par Marché")
    if not df_termines.empty:
        df_clv = df_termines[df_termines['CLV'] != 0.0].copy()

        if not df_clv.empty:
            fig_clv = go.Figure()
            couleurs = {"Totals (Buts)": "#FF4500", "Handicap Asiatique": "#00BFFF"}

            for marche in df_clv['Type_Marche'].unique():
                df_cat = df_clv[df_clv['Type_Marche'] == marche].copy()
                df_cat = df_cat.sort_values(by="Date").reset_index(drop=True)
                df_cat['CLV_Pct'] = df_cat['CLV'] * 100
                df_cat['CLV_Moyenne_Cumulee'] = df_cat['CLV_Pct'].expanding().mean()

                fig_clv.add_trace(go.Scatter(
                    x=df_cat.index, y=df_cat['CLV_Moyenne_Cumulee'],
                    mode='lines', name=marche,
                    line=dict(color=couleurs.get(marche, "#FFFFFF"), width=2.5)
                ))

            fig_clv.add_hline(y=0, line_dash="dash", line_color="#FFFFFF", opacity=0.5)
            fig_clv.update_layout(
                yaxis_title="Beat The Close moyen (%)",
                xaxis_title="Volume de Paris par marché",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            appliquer_theme_dark(fig_clv)
            st.plotly_chart(fig_clv, use_container_width=True)
        else:
            st.write("En attente de données de clôture Pinnacle.")
    else:
        st.write("Données insuffisantes.")

# ==========================================
# 📋 TABLEAU 1 : PARIS EN ATTENTE
# ==========================================
st.markdown("---")
st.subheader("📡 Radar Football : Signaux Actifs / En attente de dénouement")

if not df_attente.empty:
    colonnes_attente = ["Date", "Nom_Ligue", "Equipe", "Handicap", "Cote_Prise", "Mise", "Edge"]
    st.dataframe(df_attente[colonnes_attente].sort_values(by="Date"), use_container_width=True)
else:
    st.success("Aucun ordre en cours sur les marchés. Le Sniper est en veille.")

# ==========================================
# 📋 TABLEAU 2 : DERNIERS RÉSULTATS DE LA NUIT
# ==========================================
st.markdown("---")
st.subheader("📰 Journal de bord : 10 dernières rencontres clôturées")

if not df_termines.empty:
    df_derniers = df_termines.sort_values(by="Date", ascending=False).head(10)
    colonnes_hist = ["Date", "Nom_Ligue", "Equipe", "Handicap", "Cote_Prise", "Mise", "Statut", "Profit_Unites"]
    df_derniers = df_derniers[colonnes_hist].copy()

    def style_statut(val):
        if val in ['WON', 'HALF-WON']: return 'color: #00FF00; font-weight: bold'
        elif val in ['LOST', 'HALF-LOST']: return 'color: #FF4500; font-weight: bold'
        return 'color: #FFFFFF; opacity: 0.5'

    st.dataframe(df_derniers.style.map(style_statut, subset=['Statut']), use_container_width=True)
