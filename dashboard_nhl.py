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
st.set_page_config(page_title="NHL Quant Dashboard", page_icon="🏒", layout="wide")

# ==========================================
# 🛑 ÉCRAN DE SÉCURITÉ
# ==========================================
verifier_authentification()

st.title("🏒 Centre de Commandement : Sniper NHL Oméga")

URL_JOURNAL = "https://chienblanc.pythonanywhere.com/data/journal_trading_nhl_SEC2026xOmG.csv"
CAPITAL_INITIAL = 1000.0

# ==========================================
# 📥 CHARGEMENT DES DONNÉES DISTANTES
# ==========================================
@st.cache_data(ttl=60)
def load_data():
    try:
        df = pd.read_csv(URL_JOURNAL)
    except Exception:
        return pd.DataFrame(), "error"

    if df.empty:
        return df, "empty"

    colonnes_numeriques = ["Vraie_Cote_Bot", "Cote_Prise", "Cote_CLV", "Edge(%)", "Risque(%)", "Mise_€", "P&L"]
    df = nettoyer_colonnes_numeriques(df, colonnes_numeriques)
    df = convertir_dates(df)
    return df, "ok"

df, statut_chargement = load_data()

# ==========================================
# 🛑 AFFICHAGE SI AUCUNE DONNÉE OU ERREUR
# ==========================================
afficher_alertes_chargement(
    statut_chargement, df,
    msg_succes="🎯 Le radar Oméga est armé. En attente de la première transaction de la saison en octobre..."
)

# ==========================================
# 🔍 BARRE DE FILTRES DYNAMIQUES (SIDEBAR)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtres Écran Principal")

df = filtre_temporel_sidebar(df)

# Filtre par Type de Pari
if not df.empty and "Pari" in df.columns:
    def extraire_type_pari(pari_str):
        pari_upper = str(pari_str).upper()
        if "OVER" in pari_upper or "UNDER" in pari_upper:
            return "Over/Under (Totaux)"
        if "PUCK LINE" in pari_upper or "HANDICAP" in pari_upper:
            return "Puck Line / Handicap"
        return "Moneyline"

    df['Type_Pari'] = df['Pari'].apply(extraire_type_pari)
    types_dispo = sorted(df['Type_Pari'].unique().tolist())
    type_choisi = st.sidebar.selectbox("📊 Type de pari :", ["Tous les Types"] + types_dispo)
    if type_choisi != "Tous les Types":
        df = df[df['Type_Pari'] == type_choisi]

# ==========================================
# 📊 CALCUL DES KPI
# ==========================================
df_termines = df[df['Statut'].isin(['GAGNÉ', 'PERDU'])].copy()
df_attente = df[df['Statut'] == 'EN ATTENTE'].copy()

total_pl = df_termines['P&L'].sum() if not df_termines.empty else 0.0
capital_actuel = CAPITAL_INITIAL + total_pl
winrate = (len(df_termines[df_termines['Statut'] == 'GAGNÉ']) / len(df_termines) * 100) if not df_termines.empty else 0
total_mise = df_termines['Mise_€'].sum() if not df_termines.empty else 0.0
roi = (total_pl / total_mise * 100) if total_mise > 0 else 0

max_dd_pct = 0.0
if not df_termines.empty:
    df_termines, max_dd_pct = calculer_max_drawdown(df_termines, 'P&L', CAPITAL_INITIAL)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Capital Actuel", f"{capital_actuel:.2f} €", f"{total_pl:.2f} €")
col2.metric("Retour sur Investissement (ROI)", f"{roi:.2f} %")
col3.metric("Taux de Réussite (Winrate)", f"{winrate:.1f} %")
col4.metric("Max Drawdown", f"{max_dd_pct:.1f} %")
col5.metric("Paris en Attente", f"{len(df_attente)}")

st.markdown("---")

# ==========================================
# 📈 GRAPHIQUE 1 : COURBE DE CROISSANCE
# ==========================================
st.subheader("📈 Évolution de la Bankroll")

if not df_termines.empty:
    fig_bankroll = creer_graphique_bankroll(
        df_termines,
        hover_data=['Date', 'Visiteur', 'Local', 'Pari', 'P&L'],
        couleur='#00ff00',
        unite='€'
    )
    st.plotly_chart(fig_bankroll, use_container_width=True)
else:
    st.info("La courbe de croissance s'affichera dès qu'un pari sera classé comme GAGNÉ ou PERDU.")

col_gauche, col_droite = st.columns(2)

# ==========================================
# 🥧 GRAPHIQUE 2 : RENTABILITÉ PAR MARCHÉ
# ==========================================
with col_gauche:
    st.subheader("🎯 Rentabilité par type de pari")
    if not df_termines.empty:
        df_termines['Marche'] = df_termines['Pari'].apply(
            lambda x: x.split(' ')[0] if isinstance(x, str) else 'Inconnu'
        )
        pl_par_marche = df_termines.groupby('Marche').agg(
            P_and_L=('P&L', 'sum'),
            Volume=('P&L', 'count')
        ).reset_index()

        fig_marche = creer_graphique_pl_marche(
            pl_par_marche,
            col_x='Marche',
            titre_x="Type de Marché",
            titre_y="Profit / Perte (€)"
        )
        st.plotly_chart(fig_marche, use_container_width=True)
    else:
        st.write("Aucune statistique par marché disponible pour le moment.")

# ==========================================
# 🎯 GRAPHIQUE 3 : ANALYSE CLV
# ==========================================
with col_droite:
    st.subheader("⚖️ Tracking CLV (Closing Line Value)")
    if not df_termines.empty:
        df_clv = df_termines[(df_termines['Cote_Prise'] > 0) & (df_termines['Cote_CLV'] > 0)].copy()
        fig_clv = go.Figure()
        fig_clv.add_trace(go.Scatter(
            x=df_clv.index, y=df_clv['Cote_Prise'],
            mode='markers', name='Cote Bot',
            marker=dict(color='blue', size=8)
        ))
        fig_clv.add_trace(go.Scatter(
            x=df_clv.index, y=df_clv['Cote_CLV'],
            mode='lines', name='Cote Pinnacle',
            line=dict(color='red', width=2, dash='dot')
        ))
        appliquer_theme_dark(fig_clv)
        st.plotly_chart(fig_clv, use_container_width=True)
    else:
        st.write("L'analyse graphique de la CLV s'affichera après les premiers matchs.")

# ==========================================
# 📋 TABLEAU : PARIS EN ATTENTE
# ==========================================
st.markdown("---")
st.subheader("📡 Radar : Transactions en cours")

if not df_attente.empty:
    colonnes_a_afficher = ["Date", "Visiteur", "Local", "Pari", "Cote_Prise", "Cote_CLV", "Edge(%)", "Mise_€"]
    st.dataframe(df_attente[colonnes_a_afficher], use_container_width=True)
else:
    st.success("Aucun ordre en attente sur le marché. Le robot scrute les lignes...")
