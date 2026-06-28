import streamlit as st
import pandas as pd
import plotly.express as px
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
st.set_page_config(page_title="MLB Quant Dashboard", page_icon="⚾", layout="wide")

# ==========================================
# 🛑 ÉCRAN DE SÉCURITÉ
# ==========================================
verifier_authentification()

st.title("⚾ Centre de Commandement : Sniper MLB")

URL_FG = "https://chienblanc.pythonanywhere.com/data/sniper_history_SEC2026.csv"
URL_F5 = "https://chienblanc.pythonanywhere.com/data/sniper_history_f5_SEC2026.csv"
CAPITAL_INITIAL = 100.0

# ==========================================
# 📥 CHARGEMENT & FUSION DES DONNÉES
# ==========================================
@st.cache_data(ttl=60)
def load_and_merge_data():
    dfs = []
    statut = "ok"

    try:
        df_fg = pd.read_csv(URL_FG)
        if not df_fg.empty:
            df_fg['Segment'] = 'Full Game'
            dfs.append(df_fg)
    except Exception:
        statut = "error"

    try:
        df_f5 = pd.read_csv(URL_F5)
        if not df_f5.empty:
            df_f5['Segment'] = 'F5'
            dfs.append(df_f5)
    except Exception:
        if statut != "error":
            statut = "error"

    if not dfs:
        return pd.DataFrame(), statut if statut == "error" else "empty"

    df = pd.concat(dfs, ignore_index=True)

    colonnes_numeriques = ["Cote", "Mise", "Cote_Fermeture", "CLV_Fermeture"]
    df = nettoyer_colonnes_numeriques(df, colonnes_numeriques)
    df = convertir_dates(df)

    def definir_type_pari(pari_str):
        pari_upper = str(pari_str).upper()
        if "OVER" in pari_upper or "UNDER" in pari_upper:
            return "Over/Under"
        return "Moneyline"

    df['Type_Pari'] = df['Pari'].apply(definir_type_pari)

    def calc_pl(row):
        if row['Result'] == '✅ GAGNÉ': return float(row['Mise']) * (float(row['Cote']) - 1)
        elif row['Result'] == '❌ PERDU': return -float(row['Mise'])
        return 0.0

    df['P&L'] = df.apply(calc_pl, axis=1)

    return df, "ok"

df, statut_chargement = load_and_merge_data()

# ==========================================
# 🛑 AFFICHAGE SI AUCUNE DONNÉE OU ERREUR
# ==========================================
afficher_alertes_chargement(
    statut_chargement, df,
    msg_succes="⚾ Le radar MLB est armé. En attente de la première transaction..."
)

# ==========================================
# 🔍 BARRE DE FILTRES DYNAMIQUES (SIDEBAR)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtres Écran Principal")

df = filtre_temporel_sidebar(df)

# Filtre par Équipe
if not df.empty and "Match" in df.columns:
    equipes = set()
    for match in df["Match"].dropna().unique():
        if " @ " in match:
            teams = match.split(" @ ")
            equipes.add(teams[0].strip())
            equipes.add(teams[1].strip())

    equipe_choisie = st.sidebar.selectbox("⚾ Équipe spécifique :", ["Toutes les Équipes"] + sorted(equipes))
    if equipe_choisie != "Toutes les Équipes":
        df = df[df["Match"].str.contains(equipe_choisie, na=False)]

# Filtre par Marché (Segment + Type_Pari)
if not df.empty and "Segment" in df.columns and "Type_Pari" in df.columns:
    df['Catégorie_Temp'] = df['Segment'] + " (" + df['Type_Pari'] + ")"
    categories_dispo = sorted(df['Catégorie_Temp'].unique().tolist())
    marche_choisi = st.sidebar.selectbox("📊 Marché ciblé :", ["Tous les Marchés"] + categories_dispo)
    if marche_choisi != "Tous les Marchés":
        df = df[df['Catégorie_Temp'] == marche_choisi]
    df = df.drop(columns=['Catégorie_Temp'])

# ==========================================
# 📊 CALCUL DES KPI
# ==========================================
df_termines = df[df['Result'].isin(['✅ GAGNÉ', '❌ PERDU'])].copy()
df_attente = df[df['Result'] == 'En attente'].copy()

total_pl = df_termines['P&L'].sum() if not df_termines.empty else 0.0
capital_actuel = CAPITAL_INITIAL + total_pl
winrate = (len(df_termines[df_termines['Result'] == '✅ GAGNÉ']) / len(df_termines) * 100) if not df_termines.empty else 0
total_mise = df_termines['Mise'].sum() if not df_termines.empty else 0.0
roi = (total_pl / total_mise * 100) if total_mise > 0 else 0

max_dd_pct = 0.0
if not df_termines.empty:
    df_termines, max_dd_pct = calculer_max_drawdown(df_termines, 'P&L', CAPITAL_INITIAL)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Capital Actuel", f"{capital_actuel:.2f} U", f"{total_pl:.2f} U")
col2.metric("ROI", f"{roi:.2f} %")
col3.metric("Winrate", f"{winrate:.1f} %")
col4.metric("Max Drawdown", f"{max_dd_pct:.1f} %")
col5.metric("Paris en Attente", f"{len(df_attente)}")

st.markdown("---")

# ==========================================
# 📈 GRAPHIQUE 1 : COURBE DE CROISSANCE GLOBALE
# ==========================================
st.subheader("📈 Évolution de la Bankroll (Toutes tables confondues)")

if not df_termines.empty:
    fig_bankroll = creer_graphique_bankroll(
        df_termines,
        hover_data=['Date', 'Match', 'Pari', 'P&L', 'Segment'],
        couleur='#00BFFF',
        unite='Unités'
    )
    st.plotly_chart(fig_bankroll, use_container_width=True)
else:
    st.info("La courbe de croissance s'affichera dès qu'un pari sera classé comme GAGNÉ ou PERDU.")

col_gauche, col_droite = st.columns(2)

# ==========================================
# 🥧 GRAPHIQUE 2 : RENTABILITÉ ET VOLUME PAR MARCHÉ
# ==========================================
with col_gauche:
    st.subheader("🎯 Rentabilité et Volume")
    if not df_termines.empty:
        df_termines['Catégorie'] = df_termines['Segment'] + " (" + df_termines['Type_Pari'] + ")"
        pl_detail = df_termines.groupby('Catégorie').agg(
            P_and_L=('P&L', 'sum'),
            Volume=('P&L', 'count')
        ).reset_index()

        fig_segment = creer_graphique_pl_marche(
            pl_detail,
            col_x='Catégorie',
            titre_x="Type de Marché",
            titre_y="Profit & Loss (Unités)"
        )
        st.plotly_chart(fig_segment, use_container_width=True)
    else:
        st.write("Données insuffisantes.")

# ==========================================
# 🎯 GRAPHIQUE 3 : CLV SEGMENTÉE (DÉTECTEUR D'EV)
# ==========================================
with col_droite:
    st.subheader("⚖️ Tracking CLV par Marché")
    if not df_termines.empty and 'CLV_Fermeture' in df_termines.columns:
        df_clv = df_termines[df_termines['CLV_Fermeture'] != 0.0].copy()

        if not df_clv.empty:
            if 'Catégorie' not in df_clv.columns:
                df_clv['Catégorie'] = df_clv['Segment'] + " (" + df_clv['Type_Pari'] + ")"

            fig_clv = go.Figure()
            couleurs = {
                "F5 (Over/Under)": "#00FF00",
                "Full Game (Moneyline)": "#00BFFF",
                "Full Game (Over/Under)": "#FF4500",
                "F5 (Moneyline)": "#FFD700"
            }

            for categorie in df_clv['Catégorie'].unique():
                df_cat = df_clv[df_clv['Catégorie'] == categorie].copy()
                df_cat = df_cat.sort_values(by="Date").reset_index(drop=True)
                df_cat['CLV_Moyenne_Cumulee'] = df_cat['CLV_Fermeture'].expanding().mean()

                fig_clv.add_trace(go.Scatter(
                    x=df_cat.index, y=df_cat['CLV_Moyenne_Cumulee'],
                    mode='lines', name=categorie,
                    line=dict(color=couleurs.get(categorie, "#FFFFFF"), width=2.5)
                ))

            fig_clv.add_hline(
                y=0, line_dash="dash", line_color="#FFFFFF", opacity=0.5,
                annotation_text=" Ligne de Rentabilité (EV+)",
                annotation_position="bottom right"
            )
            fig_clv.update_layout(
                yaxis_title="Écart moyen avec la cote finale (%)",
                xaxis_title="Volume de Paris (Échelle indépendante par marché)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            appliquer_theme_dark(fig_clv)
            st.plotly_chart(fig_clv, use_container_width=True)
        else:
            st.write("En attente des données de fermeture du marché.")
    elif not df_termines.empty:
        st.write("La colonne de CLV est introuvable dans le fichier.")
    else:
        st.write("Données insuffisantes.")

# ==========================================
# 📋 TABLEAU : PARIS EN ATTENTE
# ==========================================
st.markdown("---")
st.subheader("📡 Radar MLB : Transactions en cours")

if not df_attente.empty:
    colonnes_a_afficher = ["Date", "Match", "Pari", "Type_Pari", "Segment", "Cote", "Edge", "Mise"]
    st.dataframe(df_attente[colonnes_a_afficher], use_container_width=True)
else:
    st.success("Aucun ordre en attente sur le marché.")

# ==========================================
# 📊 ANALYSE AVANCÉE : PERFORMANCES PAR COTES
# ==========================================
st.markdown("---")
st.subheader("🔍 Analyse par Tranches de Cotes (Où est l'Edge ?)")

if not df_termines.empty:
    bins = [1.0, 1.50, 1.80, 2.20, 2.60, 5.0]
    labels = ["Très Favori (<1.50)", "Favori (1.50-1.80)", "Equilibré (1.80-2.20)", "Outsider (2.20-2.60)", "Gros Outsider (>2.60)"]

    df_termines['Tranche_Cote'] = pd.cut(df_termines['Cote'], bins=bins, labels=labels)

    pl_cotes = df_termines.groupby('Tranche_Cote', observed=False).agg(
        P_and_L=('P&L', 'sum'),
        Volume=('P&L', 'count')
    ).reset_index()
    pl_cotes = pl_cotes[pl_cotes['Volume'] > 0]

    fig_cotes = creer_graphique_pl_marche(
        pl_cotes,
        col_x='Tranche_Cote',
        titre_x="Tranches de Cotes",
        titre_y="Profit & Loss (Unités)"
    )
    st.plotly_chart(fig_cotes, use_container_width=True)

# ==========================================
# 📰 JOURNAL DE BORD : DERNIERS RÉSULTATS
# ==========================================
st.markdown("---")
st.subheader("📰 Bilan de la nuit : 10 derniers paris terminés")

if not df_termines.empty:
    df_derniers = df_termines.sort_values(by="Date", ascending=False).head(10)
    colonnes_historique = ["Date", "Match", "Pari", "Cote", "Mise", "Result", "P&L"]
    df_derniers = df_derniers[colonnes_historique].copy()
    df_derniers['P&L'] = df_derniers['P&L'].round(2)

    def colorer_resultat(val):
        if val == '✅ GAGNÉ': return 'color: #00FF00; font-weight: bold'
        elif val == '❌ PERDU': return 'color: #FF4500; font-weight: bold'
        return ''

    st.dataframe(df_derniers.style.map(colorer_resultat, subset=['Result']), use_container_width=True)
else:
    st.write("Aucun pari terminé pour le moment.")
