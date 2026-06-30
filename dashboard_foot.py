import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    afficher_alertes_chargement,
    appliquer_theme_dark,
    calculer_drawdown_serie,
    calculer_max_drawdown,
    convertir_dates,
    creer_graphique_bankroll,
    creer_graphique_pl_marche,
    filtre_temporel_sidebar,
    nettoyer_colonnes_numeriques,
    preparer_df_backtest,
    verifier_authentification,
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

URL_FOOT      = "https://chienblanc.pythonanywhere.com/data/historique_sniper.csv"
URL_BACKTEST  = "https://chienblanc.pythonanywhere.com/data/backtest_results.csv"
CAPITAL_INITIAL = 100.0

LIGUE_NOMS = {
    140: "La Liga",        78: "Bundesliga",     88: "Eredivisie",
    135: "Serie A",        94: "Primeira Liga",  203: "Süper Lig",
    113: "Allsvenskan",    71: "Série A Brésil",  61: "Ligue 1",
    141: "LaLiga 2",       39: "Premier League",  40: "Championship",
    253: "MLS",           103: "Eliteserien",    144: "Jupiler Pro",
    136: "Serie B",
}

# ==========================================
# 📥 CHARGEMENT DES DONNÉES
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
    df['Type_Marche'] = df['Ligue'].apply(
        lambda x: "Totals (Buts)" if "[totals]" in str(x) else "Handicap Asiatique"
    )
    df['Nom_Ligue'] = df['Ligue'].apply(lambda x: str(x).split(" [")[0])
    return df, "ok"


@st.cache_data(ttl=300)
def load_backtest_data():
    """
    Charge backtest_results.csv — tente d'abord l'URL PythonAnywhere,
    puis le fichier local en fallback.
    """
    df = pd.DataFrame()
    # 1) Tentative via URL distante
    try:
        df = pd.read_csv(URL_BACKTEST)
    except Exception:
        pass

    # 2) Fallback fichier local
    if df.empty:
        local_path = os.path.join(os.getcwd(), "backtest_results.csv")
        if not os.path.exists(local_path):
            return pd.DataFrame(), "missing"
        try:
            df = pd.read_csv(local_path)
        except Exception:
            return pd.DataFrame(), "error"

    if df.empty:
        return df, "empty"

    df['Nom_Ligue'] = df['ligue_id'].map(LIGUE_NOMS).fillna(df['ligue_id'].astype(str))
    df['Type_Marche'] = df['market'].map(
        {'spreads': 'Handicap Asiatique', 'totals': 'Totaux (Buts)'}
    )
    df = preparer_df_backtest(df)
    return df, "ok"


df, statut_chargement = load_football_data()

# ==========================================
# 🗂️ ONGLETS PRINCIPAUX
# ==========================================
tab_live, tab_backtest = st.tabs(["📡 Live Performance", "🔬 Back-test Historique"])


# ══════════════════════════════════════════
# TAB 1 — LIVE PERFORMANCE
# ══════════════════════════════════════════
with tab_live:
    afficher_alertes_chargement(
        statut_chargement, df,
        msg_succes="⚽ Le radar Football V25 est armé. En attente des premières transactions..."
    )

    # ── Filtres sidebar ──────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Filtres Écran Principal")
    df_live = filtre_temporel_sidebar(df)

    if not df_live.empty and "Nom_Ligue" in df_live.columns:
        ligues_dispo = sorted(df_live["Nom_Ligue"].unique().tolist())
        ligue_choisie = st.sidebar.selectbox("🏆 Compétition :", ["Toutes les Ligues"] + ligues_dispo)
        if ligue_choisie != "Toutes les Ligues":
            df_live = df_live[df_live["Nom_Ligue"] == ligue_choisie]

    if not df_live.empty and "Type_Marche" in df_live.columns:
        marches_dispo = sorted(df_live["Type_Marche"].unique().tolist())
        marche_choisi = st.sidebar.selectbox("📊 Marché ciblé :", ["Tous les Marchés"] + marches_dispo)
        if marche_choisi != "Tous les Marchés":
            df_live = df_live[df_live["Type_Marche"] == marche_choisi]

    # ── KPIs ─────────────────────────────────────────────────
    df_termines = df_live[df_live['Statut'].isin(['WON', 'HALF-WON', 'VOID', 'HALF-LOST', 'LOST'])].copy()
    df_attente  = df_live[df_live['Statut'] == 'PENDING'].copy()

    total_pl    = df_termines['Profit_Unites'].sum() if not df_termines.empty else 0.0
    capital_actuel = CAPITAL_INITIAL + total_pl
    total_mise  = df_termines['Mise'].sum() if not df_termines.empty else 0.0
    roi         = (total_pl / total_mise * 100) if total_mise > 0 else 0

    rec_gagnants  = len(df_termines[df_termines['Statut'].isin(['WON', 'HALF-WON'])])
    rec_perdants  = len(df_termines[df_termines['Statut'].isin(['LOST', 'HALF-LOST'])])
    total_tranches = rec_gagnants + rec_perdants
    winrate = (rec_gagnants / total_tranches * 100) if total_tranches > 0 else 0

    max_dd_pct = 0.0
    if not df_termines.empty:
        df_termines, max_dd_pct = calculer_max_drawdown(df_termines, 'Profit_Unites', CAPITAL_INITIAL)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Capital Actuel",       f"{capital_actuel:.2f} U", f"{total_pl:.2f} U")
    col2.metric("ROI",                  f"{roi:.2f} %")
    col3.metric("Winrate (Hors Void)",  f"{winrate:.1f} %")
    col4.metric("Max Drawdown",         f"{max_dd_pct:.1f} %")
    col5.metric("Ordres en Cours",      f"{len(df_attente)}")

    st.markdown("---")

    # ── Courbe de croissance ──────────────────────────────────
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

    # ── P&L par marché ────────────────────────────────────────
    with col_gauche:
        st.subheader("🎯 Rentabilité et Volume par Type de Pari")
        if not df_termines.empty:
            pl_detail = df_termines.groupby('Type_Marche').agg(
                P_and_L=('Profit_Unites', 'sum'),
                Volume=('Profit_Unites', 'count')
            ).reset_index()
            fig_segment = creer_graphique_pl_marche(
                pl_detail, col_x='Type_Marche', titre_x="Marché", titre_y="Profit / Perte (U)"
            )
            st.plotly_chart(fig_segment, use_container_width=True)
        else:
            st.write("Données insuffisantes.")

    # ── CLV par marché ────────────────────────────────────────
    with col_droite:
        st.subheader("⚖️ Validation Mathématique : CLV par Marché")
        if not df_termines.empty:
            df_clv = df_termines[df_termines['CLV'] != 0.0].copy()
            if not df_clv.empty:
                fig_clv = go.Figure()
                couleurs = {"Totals (Buts)": "#FF4500", "Handicap Asiatique": "#00BFFF"}
                for marche in df_clv['Type_Marche'].unique():
                    df_cat = df_clv[df_clv['Type_Marche'] == marche].sort_values("Date").reset_index(drop=True)
                    df_cat['CLV_Pct'] = df_cat['CLV'] * 100
                    df_cat['CLV_Moy_Cumulee'] = df_cat['CLV_Pct'].expanding().mean()
                    fig_clv.add_trace(go.Scatter(
                        x=df_cat.index, y=df_cat['CLV_Moy_Cumulee'],
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

    # ── Paris en attente ──────────────────────────────────────
    st.markdown("---")
    st.subheader("📡 Radar Football : Signaux Actifs / En attente de dénouement")
    if not df_attente.empty:
        df_att_display = df_attente.copy()

        # CLV temps réel : colorée si disponible
        colonnes_attente = ["Date", "Nom_Ligue", "Equipe", "Handicap", "Cote_Prise", "Mise", "Edge"]
        if "CLV" in df_att_display.columns:
            colonnes_attente.append("CLV")
            df_att_display["CLV"] = (df_att_display["CLV"] * 100).round(2)
            df_att_display = df_att_display.rename(columns={"CLV": "CLV (%)"})
            colonnes_attente[-1] = "CLV (%)"

        def style_clv(val):
            try:
                v = float(val)
                if v > 1:   return 'color: #00FF00; font-weight: bold'
                if v < -1:  return 'color: #FF4500; font-weight: bold'
                return 'color: #FFD700'
            except Exception:
                return ''

        df_sorted = df_att_display[colonnes_attente].sort_values("Date")
        if "CLV (%)" in colonnes_attente:
            st.dataframe(
                df_sorted.style.map(style_clv, subset=["CLV (%)"]),
                use_container_width=True
            )
        else:
            st.dataframe(df_sorted, use_container_width=True)
    else:
        st.success("Aucun ordre en cours sur les marchés. Le Sniper est en veille.")

    # ── 10 derniers résultats ──────────────────────────────────
    st.markdown("---")
    st.subheader("📰 Journal de bord : 10 dernières rencontres clôturées")
    if not df_termines.empty:
        df_derniers = df_termines.sort_values("Date", ascending=False).head(10)
        colonnes_hist = ["Date", "Nom_Ligue", "Equipe", "Handicap", "Cote_Prise", "Mise", "Statut", "Profit_Unites"]
        def style_statut(val):
            if val in ['WON', 'HALF-WON']:   return 'color: #00FF00; font-weight: bold'
            elif val in ['LOST', 'HALF-LOST']: return 'color: #FF4500; font-weight: bold'
            return 'color: #FFFFFF; opacity: 0.5'
        st.dataframe(
            df_derniers[colonnes_hist].style.map(style_statut, subset=['Statut']),
            use_container_width=True
        )


# ══════════════════════════════════════════
# TAB 2 — BACK-TEST HISTORIQUE
# ══════════════════════════════════════════
with tab_backtest:
    df_bt, statut_bt = load_backtest_data()

    if statut_bt == "missing":
        st.warning(
            "Aucun fichier `backtest_results.csv` trouvé. "
            "Lance `python backtest_football.py` pour générer les données."
        )
    elif statut_bt in ("error", "empty"):
        st.error("Erreur lors du chargement des données de back-test.")
    else:
        # ── Filtres sidebar back-test ─────────────────────────────
        st.sidebar.markdown("---")
        st.sidebar.header("🔬 Filtres Back-test")

        saisons_dispo = sorted(df_bt['saison'].unique().tolist())
        saisons_choisies = st.sidebar.multiselect(
            "📅 Saison(s) :", saisons_dispo, default=saisons_dispo
        )
        if saisons_choisies:
            df_bt = df_bt[df_bt['saison'].isin(saisons_choisies)]

        ligues_bt = sorted(df_bt['Nom_Ligue'].unique().tolist())
        ligue_bt = st.sidebar.selectbox("🏆 Ligue :", ["Toutes"] + ligues_bt, key="bt_ligue")
        if ligue_bt != "Toutes":
            df_bt = df_bt[df_bt['Nom_Ligue'] == ligue_bt]

        marche_bt = st.sidebar.selectbox(
            "📊 Marché :", ["Tous", "Handicap Asiatique", "Totaux (Buts)"], key="bt_marche"
        )
        if marche_bt != "Tous":
            df_bt = df_bt[df_bt['Type_Marche'] == marche_bt]

        # Séparer résultats résolus / en attente (tri chronologique conservé)
        df_bt_res = df_bt[df_bt['resultat'].notna()].copy()
        if 'Date' in df_bt_res.columns:
            df_bt_res = df_bt_res.sort_values('Date').reset_index(drop=True)
        df_bt_clv = df_bt_res[df_bt_res['clv'].notna()].copy()

        if 'date_utc' not in df_bt.columns:
            st.info(
                "CSV sans colonne `date_utc` — regénérez avec "
                "`python backtest_football.py --report` pour l'axe dates."
            )

        # ── KPIs ─────────────────────────────────────────────────
        st.subheader("📊 Vue d'ensemble — Back-test Dixon-Coles")

        total_signaux = len(df_bt_res)
        total_pnl     = df_bt_res['Profit_Unites'].sum()
        total_mise_bt = df_bt_res['mise'].sum()
        roi_bt        = (total_pnl / total_mise_bt * 100) if total_mise_bt > 0 else 0
        clv_moy       = df_bt_clv['clv'].mean() * 100 if not df_bt_clv.empty else 0
        wins_bt       = (df_bt_res['resultat'] > 0).sum()
        wr_bt         = (wins_bt / total_signaux * 100) if total_signaux > 0 else 0
        dd            = calculer_drawdown_serie(df_bt_res['Profit_Unites'], CAPITAL_INITIAL)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Signaux générés",  f"{total_signaux}")
        c2.metric("P&L total",        f"{total_pnl:+.1f} u")
        c3.metric("ROI",              f"{roi_bt:+.2f} %")
        c4.metric("CLV moyen",        f"{clv_moy:+.2f} %")
        c5.metric("Max Drawdown",     f"{dd:.1f} %")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        # ── Courbe de bankroll back-test ──────────────────────────
        with col_a:
            st.subheader("📈 Courbe de Capital (Back-test)")
            if not df_bt_res.empty:
                df_curve, _ = calculer_max_drawdown(
                    df_bt_res, 'Profit_Unites', CAPITAL_INITIAL
                )
                axe_x = df_curve['Date'] if 'Date' in df_curve.columns else df_curve.index
                lbl_x = "Date du match" if 'Date' in df_curve.columns else "Paris (index)"
                fig_bt_bank = go.Figure()
                fig_bt_bank.add_trace(go.Scatter(
                    x=axe_x, y=df_curve['Bankroll'],
                    mode='lines', fill='tozeroy',
                    line=dict(color='#00BFFF', width=2),
                    fillcolor='rgba(0,191,255,0.1)',
                    name='Bankroll simulée'
                ))
                fig_bt_bank.add_hline(y=CAPITAL_INITIAL, line_dash="dash", line_color="#FFFFFF", opacity=0.4)
                fig_bt_bank.update_layout(
                    yaxis_title="Capital (unités)",
                    xaxis_title=lbl_x,
                    showlegend=False
                )
                appliquer_theme_dark(fig_bt_bank)
                st.plotly_chart(fig_bt_bank, use_container_width=True)
            else:
                st.info("Aucune donnée à afficher.")

        # ── ROI par ligue ─────────────────────────────────────────
        with col_b:
            st.subheader("🏆 ROI & CLV par Ligue")
            if not df_bt_res.empty:
                grp = df_bt_res.groupby('Nom_Ligue').agg(
                    PnL=('Profit_Unites', 'sum'),
                    Mise=('mise', 'sum'),
                    N=('resultat', 'count')
                ).reset_index()
                grp['ROI'] = grp['PnL'] / grp['Mise'] * 100
                grp_clv = df_bt_clv.groupby('Nom_Ligue')['clv'].mean().reset_index()
                grp_clv.columns = ['Nom_Ligue', 'CLV_moy']
                grp = grp.merge(grp_clv, on='Nom_Ligue', how='left')
                grp = grp.sort_values('ROI', ascending=True)

                fig_roi = go.Figure()
                fig_roi.add_trace(go.Bar(
                    y=grp['Nom_Ligue'], x=grp['ROI'],
                    orientation='h',
                    marker_color=['#00FF00' if v >= 0 else '#FF4500' for v in grp['ROI']],
                    name='ROI (%)',
                    text=[f"{v:+.1f}%" for v in grp['ROI']],
                    textposition='outside'
                ))
                if 'CLV_moy' in grp.columns:
                    fig_roi.add_trace(go.Scatter(
                        y=grp['Nom_Ligue'], x=grp['CLV_moy'] * 100,
                        mode='markers',
                        marker=dict(color='#FFD700', size=10, symbol='diamond'),
                        name='CLV moyen (%)'
                    ))
                fig_roi.add_vline(x=0, line_dash="dash", line_color="#FFFFFF", opacity=0.5)
                fig_roi.update_layout(
                    xaxis_title="ROI / CLV (%)",
                    height=480,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                appliquer_theme_dark(fig_roi)
                st.plotly_chart(fig_roi, use_container_width=True)
            else:
                st.info("Aucune donnée à afficher.")

        st.markdown("---")
        col_c, col_d = st.columns(2)

        # ── Calibration EV ────────────────────────────────────────
        with col_c:
            st.subheader("🎯 Calibration du Modèle")
            st.caption("Fréquence de victoire réelle vs EV détecté — un modèle bien calibré doit suivre la droite.")
            if not df_bt_res.empty and 'ev_modele' in df_bt_res.columns:
                tranches = [
                    (0.05, 0.07, "5-7%"),
                    (0.07, 0.09, "7-9%"),
                    (0.09, 0.12, "9-12%"),
                    (0.12, 0.20, "12-20%"),
                ]
                calib_rows = []
                for lo, hi, label in tranches:
                    sub = df_bt_res[(df_bt_res['ev_modele'] >= lo) & (df_bt_res['ev_modele'] < hi)]
                    sub_actif = sub[sub['resultat'] != 0]
                    if sub_actif.empty:
                        continue
                    wr_reel = (sub_actif['resultat'] > 0).mean() * 100
                    ev_moy  = sub['ev_modele'].mean() * 100
                    calib_rows.append({'Tranche': label, 'Win_Rate_Reel': wr_reel, 'EV_Moyen': ev_moy, 'N': len(sub)})

                if calib_rows:
                    df_calib = pd.DataFrame(calib_rows)
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Bar(
                        x=df_calib['Tranche'], y=df_calib['Win_Rate_Reel'],
                        marker_color='#00BFFF', name='Win Rate réel (%)',
                        text=[f"{v:.1f}%<br>n={n}" for v, n in zip(df_calib['Win_Rate_Reel'], df_calib['N'])],
                        textposition='outside'
                    ))
                    fig_cal.add_hline(y=50, line_dash="dot", line_color="#FFD700",
                                      annotation_text="50% (neutre)", annotation_position="bottom right")
                    fig_cal.update_layout(
                        yaxis_title="Fréquence de victoire (%)",
                        xaxis_title="Tranche d'EV détecté",
                        yaxis_range=[0, 100]
                    )
                    appliquer_theme_dark(fig_cal)
                    st.plotly_chart(fig_cal, use_container_width=True)
                else:
                    st.info("Données insuffisantes pour la calibration.")
            else:
                st.info("Colonne `ev_modele` non trouvée dans le CSV.")

        # ── P&L par marché ────────────────────────────────────────
        with col_d:
            st.subheader("📊 P&L par Marché")
            if not df_bt_res.empty:
                grp_m = df_bt_res.groupby('Type_Marche').agg(
                    PnL=('Profit_Unites', 'sum'),
                    Mise=('mise', 'sum'),
                    N=('resultat', 'count')
                ).reset_index()
                grp_m['ROI'] = grp_m['PnL'] / grp_m['Mise'] * 100

                fig_m = go.Figure()
                couleurs_m = {'Handicap Asiatique': '#00BFFF', 'Totaux (Buts)': '#FF4500'}
                fig_m.add_trace(go.Bar(
                    x=grp_m['Type_Marche'], y=grp_m['PnL'],
                    marker_color=[couleurs_m.get(m, '#FFFFFF') for m in grp_m['Type_Marche']],
                    text=[f"P&L: {p:+.1f}u<br>ROI: {r:+.1f}%<br>N={n}"
                          for p, r, n in zip(grp_m['PnL'], grp_m['ROI'], grp_m['N'])],
                    textposition='outside',
                    name='P&L (unités)'
                ))
                fig_m.add_hline(y=0, line_dash="dash", line_color="#FFFFFF", opacity=0.5)
                fig_m.update_layout(yaxis_title="Profit / Perte (unités)", showlegend=False)
                appliquer_theme_dark(fig_m)
                st.plotly_chart(fig_m, use_container_width=True)
            else:
                st.info("Aucune donnée à afficher.")

        # ── Tableau détaillé par ligue ────────────────────────────
        st.markdown("---")
        st.subheader("📋 Tableau de Synthèse par Ligue")
        if not df_bt_res.empty:
            tbl = df_bt_res.groupby(['Nom_Ligue', 'Type_Marche']).agg(
                N=('resultat', 'count'),
                PnL=('Profit_Unites', 'sum'),
                Mise=('mise', 'sum'),
            ).reset_index()
            tbl['ROI (%)'] = (tbl['PnL'] / tbl['Mise'] * 100).round(2)
            tbl['P&L']     = tbl['PnL'].round(2)
            tbl_clv = df_bt_clv.groupby(['Nom_Ligue', 'Type_Marche'])['clv'].mean().reset_index()
            tbl_clv.columns = ['Nom_Ligue', 'Type_Marche', 'CLV moy (%)']
            tbl_clv['CLV moy (%)'] = (tbl_clv['CLV moy (%)'] * 100).round(2)
            tbl = tbl.merge(tbl_clv, on=['Nom_Ligue', 'Type_Marche'], how='left')

            def color_roi(val):
                try:
                    return 'color: #00FF00' if float(val) > 0 else 'color: #FF4500'
                except Exception:
                    return ''

            display_cols = ['Nom_Ligue', 'Type_Marche', 'N', 'P&L', 'ROI (%)', 'CLV moy (%)']
            st.dataframe(
                tbl[display_cols].sort_values('ROI (%)', ascending=False)
                                 .style.map(color_roi, subset=['ROI (%)', 'CLV moy (%)']),
                use_container_width=True
            )
