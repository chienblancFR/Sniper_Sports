import os
import sqlite3
from datetime import datetime
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from utils import (
    afficher_alertes_chargement,
    appliquer_theme_dark,
    calculer_drawdown_serie,
    calculer_max_drawdown,
    convertir_dates,
    creer_graphique_bankroll,
    creer_graphique_clv_cumule,
    creer_graphique_pl_marche,
    filtre_ligue_sidebar,
    filtre_marche_sidebar,
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
PA_DATA_DIR = "/home/chienblanc/data"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LIGUE_NOMS = {
    140: "La Liga",        78: "Bundesliga",     88: "Eredivisie",
    135: "Serie A",        94: "Primeira Liga",  203: "Süper Lig",
    113: "Allsvenskan",    71: "Série A Brésil",  61: "Ligue 1",
    141: "LaLiga 2",       39: "Premier League",  40: "Championship",
    253: "MLS",           103: "Eliteserien",    144: "Jupiler Pro",
    136: "Serie B",
}


def _candidats_fichiers(nom: str) -> list[str]:
    """Chemins possibles CSV / DB (PA data dir en priorité)."""
    paths = []
    env_key = "FOOT_HISTORIQUE_CSV" if nom.endswith(".csv") else "FOOT_SNIPER_DB"
    if os.environ.get(env_key):
        paths.append(os.environ[env_key])
    if os.path.isdir(PA_DATA_DIR):
        paths.append(os.path.join(PA_DATA_DIR, nom))
    paths.extend([
        os.path.join(_SCRIPT_DIR, nom),
        os.path.join(os.getcwd(), nom),
        os.path.join(os.path.expanduser("~"), nom),
        os.path.join(os.path.expanduser("~"), "sniper_bot_foot", nom),
    ])
    seen, out = set(), []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _score_csv(df: pd.DataFrame, mtime: float) -> tuple:
    n_pending = 0
    if "Statut" in df.columns:
        n_pending = int(
            (df["Statut"].astype(str).str.strip().str.upper() == "PENDING").sum()
        )
    return (n_pending, mtime)


def _charger_csv(url: str, fichier_local: str):
    """Fichier local le plus récent / avec le plus de PENDING → URL PA → absent."""
    erreurs = []
    best_df, best_src, best_score = None, None, (-1, -1.0)

    for path in _candidats_fichiers(fichier_local):
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                erreurs.append(f"{path}: vide")
                continue
            score = _score_csv(df, os.path.getmtime(path))
            if score > best_score:
                best_score, best_df = score, df
                mtime = datetime.fromtimestamp(score[1])
                best_src = (
                    f"fichier · {path} ({mtime:%d/%m %H:%M}, "
                    f"{score[0]} PENDING / {len(df)} lignes)"
                )
        except Exception as e:
            erreurs.append(f"{path}: {e}")

    if best_df is not None:
        return best_df, "ok", best_src

    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text))
            if not df.empty:
                n_p = _score_csv(df, 0)[0]
                return df, "ok", f"URL PA · {fichier_local} ({n_p} PENDING / {len(df)} lignes)"
            return df, "empty", f"URL PA · {fichier_local} (vide)"
        erreurs.append(f"URL HTTP {r.status_code}")
    except Exception as e:
        erreurs.append(f"URL: {e}")

    return pd.DataFrame(), "missing", " | ".join(erreurs) if erreurs else "introuvable"


def _charger_foot_depuis_db(db_path: str | None = None):
    """Fallback SQLite — essaie plusieurs emplacements sniper_data.db."""
    sql = """
        SELECT
            id_match AS ID_Match, equipe AS Equipe, handicap AS Handicap,
            cote_prise AS Cote_Prise, mise AS Mise, cote_cloture AS Cote_Cloture,
            edge_detecte AS Edge, p_modele AS Prob_Modele, clv AS CLV,
            statut AS Statut, resultat AS Profit_Unites, ligue AS Ligue,
            is_lineup_official AS Compo_Officielle,
            equipe_dom AS Equipe_Dom, equipe_ext AS Equipe_Ext, kickoff AS Kickoff,
            timestamp AS Date
        FROM paris_log ORDER BY timestamp DESC
    """
    sql_legacy = """
        SELECT
            id_match AS ID_Match, equipe AS Equipe, handicap AS Handicap,
            cote_prise AS Cote_Prise, mise AS Mise, cote_cloture AS Cote_Cloture,
            edge_detecte AS Edge, p_modele AS Prob_Modele, clv AS CLV,
            statut AS Statut, resultat AS Profit_Unites, ligue AS Ligue,
            is_lineup_official AS Compo_Officielle, timestamp AS Date
        FROM paris_log ORDER BY timestamp DESC
    """
    candidates = [db_path] if db_path else _candidats_fichiers("sniper_data.db")
    last_err = ""
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            conn = sqlite3.connect(path)
            try:
                df = pd.read_sql_query(sql, conn)
            except Exception:
                df = pd.read_sql_query(sql_legacy, conn)
            conn.close()
            if df.empty:
                return df, "empty", f"SQLite · {path} (vide)"
            n_p = int((df["Statut"].astype(str).str.strip().str.upper() == "PENDING").sum())
            return df, "ok", f"SQLite · {path} ({n_p} PENDING / {len(df)} lignes)"
        except Exception as e:
            last_err = f"{path}: {e}"
    return pd.DataFrame(), "missing", last_err or "sniper_data.db introuvable"


def _enrichir_df_live(df: pd.DataFrame) -> pd.DataFrame:
    colonnes_numeriques = ["Cote_Prise", "Mise", "Cote_Cloture", "Edge", "Prob_Modele", "CLV", "Profit_Unites"]
    df = nettoyer_colonnes_numeriques(df, colonnes_numeriques)
    df = convertir_dates(df)
    if "Statut" in df.columns:
        df["Statut"] = df["Statut"].astype(str).str.strip().str.upper()
    df['Type_Marche'] = df['Ligue'].apply(
        lambda x: "Totals (Buts)" if "[totals]" in str(x).lower() else "Handicap Asiatique"
    )
    df['Nom_Ligue'] = df['Ligue'].apply(lambda x: str(x).split(" [")[0].strip())

    def _libelle_match(row):
        dom, ext = row.get('Equipe_Dom'), row.get('Equipe_Ext')
        if pd.notna(dom) and pd.notna(ext) and str(dom).strip() and str(ext).strip():
            base = f"{dom} - {ext}"
            sel = str(row.get('Equipe', ''))
            h = row.get('Handicap')
            if '[totals]' in str(row.get('Ligue', '')).lower() and sel and pd.notna(h):
                return f"{base} | {sel} {h:g}"
            if sel and pd.notna(h):
                return f"{base} | {sel} ({h:+g})"
            return base
        return row.get('Equipe', '')

    if 'Equipe_Dom' in df.columns:
        df['Match'] = df.apply(_libelle_match, axis=1)
    else:
        df['Match'] = df['Equipe']
    return df

# ==========================================
# 📥 CHARGEMENT DES DONNÉES
# ==========================================
@st.cache_data(ttl=60)
def load_football_data():
    df_csv, stat_csv, src_csv = _charger_csv(URL_FOOT, "historique_sniper.csv")
    df_db, stat_db, src_db = _charger_foot_depuis_db()

    def _n_pending(d: pd.DataFrame) -> int:
        if d is None or d.empty or "Statut" not in d.columns:
            return 0
        return int((d["Statut"].astype(str).str.strip().str.upper() == "PENDING").sum())

    candidates = []
    if stat_csv == "ok" and not df_csv.empty:
        candidates.append((df_csv, src_csv, _n_pending(df_csv), len(df_csv)))
    if stat_db == "ok" and not df_db.empty:
        candidates.append((df_db, src_db, _n_pending(df_db), len(df_db)))

    if not candidates:
        src = src_csv if stat_csv != "ok" else src_db
        if stat_csv != "ok" and stat_db != "ok":
            src = f"{src_csv} · DB: {src_db}"
        return pd.DataFrame(), "missing", src

    # Priorité : plus de PENDING, puis plus de lignes
    df, src, _, _ = max(candidates, key=lambda x: (x[2], x[3]))
    return _enrichir_df_live(df), "ok", src

@st.cache_data(ttl=300)
def load_backtest_data():
    df, statut, source = _charger_csv(URL_BACKTEST, "backtest_results.csv")
    if statut != "ok":
        return df, statut, source

    required = {'ligue_id', 'saison', 'market', 'resultat', 'mise'}
    if not required.issubset(df.columns):
        return pd.DataFrame(), "error", source

    df['Nom_Ligue'] = df['ligue_id'].map(LIGUE_NOMS).fillna(df['ligue_id'].astype(str))
    df['Type_Marche'] = df['market'].map(
        {'spreads': 'Handicap Asiatique', 'totals': 'Totaux (Buts)'}
    )
    df = preparer_df_backtest(df)
    return df, "ok", source


# ==========================================
# 🎛️ SIDEBAR — CONTRÔLES GLOBAUX
# ==========================================
st.sidebar.header("⚙️ Contrôles")
if st.sidebar.button("🔄 Rafraîchir les données", use_container_width=True):
    load_football_data.clear()
    load_backtest_data.clear()
    st.rerun()

section = st.sidebar.radio(
    "Navigation",
    ["📡 Live Performance", "🔬 Back-test Historique"],
    key="dash_section",
)

df, statut_chargement, src_live = load_football_data()
df_bt, statut_bt, src_bt = load_backtest_data()

st.caption(
    f"📡 Live : **{src_live}** (cache 60 s) · "
    f"🔬 Back-test : **{src_bt}** (cache 300 s) · "
    f"Section : **{section.split(' ', 1)[1]}**"
)


# ══════════════════════════════════════════
# TAB 1 — LIVE PERFORMANCE
# ══════════════════════════════════════════
if section == "📡 Live Performance":
    if statut_chargement == "missing":
        st.warning(
            "⚠️ Journal live introuvable.\n\n"
            f"**Détail :** {src_live}\n\n"
            "Le dashboard lit `historique_sniper.csv` (local, URL PA, ou `sniper_data.db`). "
            "Sur PythonAnywhere : vérifiez que le bot foot tourne et que "
            "`/data/` pointe vers `/home/chienblanc/data/`."
        )
        st.stop()
    afficher_alertes_chargement(
        statut_chargement, df,
        msg_succes="⚽ Le radar Football V25 est armé. En attente des premières transactions..."
    )
    if df.empty:
        st.warning("Journal chargé mais **0 ligne** — lancez `python export_foot_csv.py` sur PA.")
        st.stop()

    n_pending_raw = int((df["Statut"] == "PENDING").sum()) if "Statut" in df.columns else 0
    with st.sidebar.expander("🔍 Diagnostic données", expanded=(n_pending_raw == 0)):
        st.write(f"**Source :** {src_live}")
        st.write(f"**Lignes totales :** {len(df)}")
        st.write(f"**PENDING (brut) :** {n_pending_raw}")
        if "Statut" in df.columns:
            st.write("**Statuts :**", df["Statut"].value_counts().to_dict())

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Filtres Live")
    df_live = filtre_temporel_sidebar(df, key_prefix="live")
    df_live = filtre_ligue_sidebar(df_live, key="live_ligue")
    df_live = filtre_marche_sidebar(df_live, key="live_marche")

    if "Statut" not in df_live.columns:
        st.error("Format CSV live invalide (colonne `Statut` manquante).")
        st.stop()
    if df_live.empty:
        if df.empty:
            st.stop()
        st.info(
            "Aucune transaction ne correspond aux filtres sidebar "
            "(période / ligue / marché). Remettez **Tout** / **Toutes les Ligues**."
        )
        st.stop()

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

    df_clv_ok = df_termines[(df_termines['CLV'].notna()) & (df_termines['CLV'] != 0)] if not df_termines.empty else pd.DataFrame()
    clv_moy_live = df_clv_ok['CLV'].mean() * 100 if not df_clv_ok.empty else None
    df_edge_ok = df_termines[df_termines['Edge'].notna()] if not df_termines.empty else pd.DataFrame()
    edge_moy = df_edge_ok['Edge'].mean() * 100 if not df_edge_ok.empty else None

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Capital Actuel",       f"{capital_actuel:.2f} U", f"{total_pl:+.2f} U")
    c2.metric("ROI",                  f"{roi:.2f} %")
    c3.metric("Winrate (Hors Void)",  f"{winrate:.1f} %")
    c4.metric("Max Drawdown",         f"{max_dd_pct:.1f} %")
    c5.metric("Ordres en Cours",      f"{len(df_attente)}")
    c6.metric(
        "CLV moyen",
        f"{clv_moy_live:+.2f} %" if clv_moy_live is not None else "N/A",
        help=f"Sur {len(df_clv_ok)} paris clôturés avec CLV",
    )
    c7.metric(
        "Edge moyen",
        f"{edge_moy:+.2f} %" if edge_moy is not None else "N/A",
        help=f"Sur {len(df_edge_ok)} paris avec edge enregistré",
    )

    st.markdown("---")

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

    with col_droite:
        st.subheader("⚖️ Validation Mathématique : CLV par Marché")
        if not df_clv_ok.empty:
            fig_clv = creer_graphique_clv_cumule(df_clv_ok)
            st.plotly_chart(fig_clv, use_container_width=True)
        elif not df_termines.empty:
            st.write("En attente de données de clôture Pinnacle.")
        else:
            st.write("Données insuffisantes.")

    st.markdown("---")
    st.subheader("📡 Radar Football : Signaux Actifs / En attente de dénouement")
    if not df_attente.empty:
        df_att_display = df_attente.copy()
        colonnes_attente = ["Date", "Nom_Ligue", "Match", "Cote_Prise", "Mise", "Edge"]
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

    st.markdown("---")
    st.subheader("📰 Journal de bord : 10 dernières rencontres clôturées")
    if not df_termines.empty:
        df_derniers = df_termines.sort_values("Date", ascending=False).head(10)
        colonnes_hist = ["Date", "Nom_Ligue", "Match", "Cote_Prise", "Mise", "Statut", "Profit_Unites"]

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
elif section == "🔬 Back-test Historique":
    if statut_bt == "missing":
        st.warning(
            "Aucun fichier `backtest_results.csv` trouvé. "
            "Lance `python backtest_football.py --report` pour générer les données."
        )
    elif statut_bt in ("error", "empty"):
        st.error("Erreur lors du chargement des données de back-test.")
    else:
        st.sidebar.markdown("---")
        st.sidebar.header("🔬 Filtres Back-test")

        saisons_dispo = sorted(df_bt['saison'].unique().tolist())
        saisons_choisies = st.sidebar.multiselect(
            "📅 Saison(s) :", saisons_dispo, default=saisons_dispo, key="bt_saisons"
        )
        if saisons_choisies:
            df_bt = df_bt[df_bt['saison'].isin(saisons_choisies)]
        else:
            st.warning("Sélectionnez au moins une saison.")
            st.stop()

        df_bt = filtre_ligue_sidebar(df_bt, key="bt_ligue", label_toutes="Toutes")
        df_bt = filtre_marche_sidebar(df_bt, key="bt_marche", label_tous="Tous")

        df_bt_res = df_bt[df_bt['resultat'].notna()].copy()
        if 'Date' in df_bt_res.columns:
            df_bt_res = df_bt_res.sort_values('Date').reset_index(drop=True)
        df_bt_clv = df_bt_res[df_bt_res['clv'].notna()].copy()

        if 'date_utc' not in df_bt.columns:
            st.info(
                "CSV sans colonne `date_utc` — regénérez avec "
                "`python backtest_football.py --report` pour l'axe dates."
            )

        st.subheader("📊 Vue d'ensemble — Back-test Dixon-Coles")

        total_signaux = len(df_bt_res)
        total_pnl     = df_bt_res['Profit_Unites'].sum()
        total_mise_bt = df_bt_res['mise'].sum()
        roi_bt        = (total_pnl / total_mise_bt * 100) if total_mise_bt > 0 else 0
        clv_moy       = df_bt_clv['clv'].mean() * 100 if not df_bt_clv.empty else 0
        wins_bt       = (df_bt_res['resultat'] > 0).sum()
        wr_bt         = (wins_bt / total_signaux * 100) if total_signaux > 0 else 0
        dd            = calculer_drawdown_serie(df_bt_res['Profit_Unites'], CAPITAL_INITIAL)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Signaux générés",  f"{total_signaux}")
        c2.metric("P&L total",        f"{total_pnl:+.1f} u")
        c3.metric("ROI",              f"{roi_bt:+.2f} %")
        c4.metric("Win rate",         f"{wr_bt:.1f} %")
        c5.metric("CLV moyen",        f"{clv_moy:+.2f} %")
        c6.metric("Max Drawdown",     f"{dd:.1f} %")

        st.markdown("---")
        col_a, col_b = st.columns(2)

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
