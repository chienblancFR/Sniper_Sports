import os
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
import streamlit as st

from utils import (
    afficher_alertes_chargement,
    calculer_max_drawdown,
    convertir_dates,
    creer_graphique_bankroll,
    creer_graphique_clv_cumule,
    creer_graphique_pl_marche,
    filtre_temporel_sidebar,
    nettoyer_colonnes_numeriques,
    verifier_authentification,
)

# ==========================================
# ⚙️ CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="MLB Quant Dashboard", page_icon="⚾", layout="wide")

verifier_authentification()

st.title("⚾ Centre de Commandement : Sniper MLB")

URL_FG = "https://chienblanc.pythonanywhere.com/data/sniper_history_SEC2026.csv"
URL_F5 = "https://chienblanc.pythonanywhere.com/data/sniper_history_f5_SEC2026.csv"
FICHIER_FG = "sniper_history_SEC2026.csv"
FICHIER_F5 = "sniper_history_f5_SEC2026.csv"
PA_DATA_DIR = "/home/chienblanc/data"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPITAL_INITIAL = float(os.environ.get("MLB_BANKROLL", "100"))


def _candidats_fichiers(nom: str) -> list[str]:
    paths = []
    env_key = "MLB_HISTORIQUE_FG" if nom == FICHIER_FG else "MLB_HISTORIQUE_F5"
    if os.environ.get(env_key):
        paths.append(os.environ[env_key])
    if os.path.isdir(PA_DATA_DIR):
        paths.append(os.path.join(PA_DATA_DIR, nom))
    paths.extend([
        os.path.join(_SCRIPT_DIR, nom),
        os.path.join(os.getcwd(), nom),
        os.path.join(os.path.expanduser("~"), nom),
    ])
    seen, out = set(), []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _score_mlb(df: pd.DataFrame, mtime: float) -> tuple:
    n_attente = 0
    if "Result" in df.columns:
        n_attente = int((df["Result"].astype(str).str.strip() == "En attente").sum())
    return (n_attente, mtime)


def _charger_csv_mlb(url: str, fichier_local: str):
    """Local le plus récent (priorité En attente) → URL PA."""
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
            score = _score_mlb(df, os.path.getmtime(path))
            if score > best_score:
                best_score, best_df = score, df
                mtime = datetime.fromtimestamp(score[1])
                best_src = (
                    f"fichier · {path} ({mtime:%d/%m %H:%M}, "
                    f"{score[0]} en attente / {len(df)} lignes)"
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
                n_a = _score_mlb(df, 0)[0]
                return df, "ok", f"URL PA · {fichier_local} ({n_a} en attente / {len(df)} lignes)"
            return df, "empty", f"URL PA · {fichier_local} (vide)"
        erreurs.append(f"URL HTTP {r.status_code}")
    except Exception as e:
        erreurs.append(f"URL: {e}")

    return pd.DataFrame(), "missing", " | ".join(erreurs) if erreurs else "introuvable"


def _parse_edge_pct(val) -> float | None:
    try:
        return float(str(val).strip().replace("%", "").replace(",", "."))
    except (TypeError, ValueError):
        return None


def _definir_type_pari(pari_str: str) -> str:
    pari_upper = str(pari_str).upper()
    if "OVER" in pari_upper or "UNDER" in pari_upper:
        return "Over/Under"
    return "Moneyline"


def _enrichir_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    colonnes_numeriques = ["Cote", "Mise", "Cote_Fermeture", "CLV_Fermeture"]
    df = nettoyer_colonnes_numeriques(df, colonnes_numeriques)
    df = convertir_dates(df)

    if "Type_Pari" not in df.columns and "Pari" in df.columns:
        df["Type_Pari"] = df["Pari"].apply(_definir_type_pari)

    df["Edge_Pct"] = df["Edge"].apply(_parse_edge_pct) if "Edge" in df.columns else None
    df["Catégorie"] = df["Segment"] + " · " + df["Type_Pari"]

    def calc_pl(row):
        if row["Result"] == "✅ GAGNÉ":
            return float(row["Mise"]) * (float(row["Cote"]) - 1)
        if row["Result"] == "❌ PERDU":
            return -float(row["Mise"])
        return 0.0

    df["P&L"] = df.apply(calc_pl, axis=1)
    return df


@st.cache_data(ttl=60)
def load_and_merge_data():
    df_fg, stat_fg, src_fg = _charger_csv_mlb(URL_FG, FICHIER_FG)
    df_f5, stat_f5, src_f5 = _charger_csv_mlb(URL_F5, FICHIER_F5)

    dfs = []
    sources = []
    if stat_fg == "ok" and not df_fg.empty:
        df_fg = df_fg.copy()
        df_fg["Segment"] = "Full Game"
        dfs.append(df_fg)
        sources.append(f"FG: {src_fg}")
    if stat_f5 == "ok" and not df_f5.empty:
        df_f5 = df_f5.copy()
        df_f5["Segment"] = "F5"
        dfs.append(df_f5)
        sources.append(f"F5: {src_f5}")

    if not dfs:
        if stat_fg == "missing" and stat_f5 == "missing":
            return pd.DataFrame(), "missing", f"{src_fg} · {src_f5}"
        if stat_fg == "error" or stat_f5 == "error":
            return pd.DataFrame(), "error", f"{src_fg} · {src_f5}"
        return pd.DataFrame(), "empty", " · ".join(sources) if sources else "aucune source"

    df = _enrichir_df(pd.concat(dfs, ignore_index=True))
    return df, "ok", " | ".join(sources)


# ==========================================
# 🎛️ SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Contrôles")
if st.sidebar.button("🔄 Rafraîchir les données", use_container_width=True):
    load_and_merge_data.clear()
    st.rerun()

df, statut_chargement, source = load_and_merge_data()
st.caption(f"📡 Live : **{source}** (cache 60 s)")

if statut_chargement == "missing":
    st.warning(
        "⚠️ Journaux MLB introuvables.\n\n"
        f"**Détail :** {source}\n\n"
        "Le dashboard lit `sniper_history_SEC2026.csv` et `sniper_history_f5_SEC2026.csv` "
        "(local, dossier PA `/data/`, ou URL PythonAnywhere). "
        "Vérifiez que le bot MLB tourne et publie les CSV."
    )
    st.stop()

afficher_alertes_chargement(
    statut_chargement, df,
    msg_succes="⚾ Le radar MLB est armé. En attente de la première transaction...",
)

n_attente_raw = int((df["Result"] == "En attente").sum()) if not df.empty else 0
with st.sidebar.expander("🔍 Diagnostic données", expanded=(n_attente_raw == 0 and not df.empty)):
    st.write(f"**Source :** {source}")
    st.write(f"**Lignes totales :** {len(df)}")
    st.write(f"**En attente (brut) :** {n_attente_raw}")
    if not df.empty and "Result" in df.columns:
        st.write("**Résultats :**", df["Result"].value_counts().to_dict())
    if not df.empty and "Segment" in df.columns:
        st.write("**Par segment :**", df["Segment"].value_counts().to_dict())

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtres Live")

df_live = filtre_temporel_sidebar(df, key_prefix="mlb")

if not df_live.empty and "Segment" in df_live.columns:
    segment_choisi = st.sidebar.selectbox(
        "⏱️ Segment :",
        ["Tous", "Full Game", "F5"],
        key="mlb_segment",
    )
    if segment_choisi != "Tous":
        df_live = df_live[df_live["Segment"] == segment_choisi]

if not df_live.empty and "Match" in df_live.columns:
    equipes = set()
    for match in df_live["Match"].dropna().unique():
        if " @ " in str(match):
            teams = str(match).split(" @ ")
            equipes.add(teams[0].strip())
            equipes.add(teams[1].strip())
    equipe_choisie = st.sidebar.selectbox(
        "⚾ Équipe :",
        ["Toutes les Équipes"] + sorted(equipes),
        key="mlb_equipe",
    )
    if equipe_choisie != "Toutes les Équipes":
        df_live = df_live[df_live["Match"].str.contains(equipe_choisie, na=False)]

if not df_live.empty and "Catégorie" in df_live.columns:
    categories = sorted(df_live["Catégorie"].unique().tolist())
    marche_choisi = st.sidebar.selectbox(
        "📊 Marché ciblé :",
        ["Tous les Marchés"] + categories,
        key="mlb_marche",
    )
    if marche_choisi != "Tous les Marchés":
        df_live = df_live[df_live["Catégorie"] == marche_choisi]

if df_live.empty:
    if df.empty:
        st.stop()
    st.info("Aucune transaction ne correspond aux filtres sélectionnés.")
    st.stop()

# ==========================================
# 📊 KPI
# ==========================================
df_termines = df_live[df_live["Result"].isin(["✅ GAGNÉ", "❌ PERDU"])].copy()
df_attente = df_live[df_live["Result"] == "En attente"].copy()

total_pl = df_termines["P&L"].sum() if not df_termines.empty else 0.0
capital_actuel = CAPITAL_INITIAL + total_pl
total_mise = df_termines["Mise"].sum() if not df_termines.empty else 0.0
roi = (total_pl / total_mise * 100) if total_mise > 0 else 0
winrate = (
    len(df_termines[df_termines["Result"] == "✅ GAGNÉ"]) / len(df_termines) * 100
    if not df_termines.empty else 0
)

max_dd_pct = 0.0
if not df_termines.empty:
    df_termines, max_dd_pct = calculer_max_drawdown(df_termines, "P&L", CAPITAL_INITIAL)

df_clv_ok = (
    df_termines[(df_termines["CLV_Fermeture"].notna()) & (df_termines["CLV_Fermeture"] != 0)]
    if not df_termines.empty else pd.DataFrame()
)
clv_moy = df_clv_ok["CLV_Fermeture"].mean() if not df_clv_ok.empty else None
df_edge_ok = df_termines[df_termines["Edge_Pct"].notna()] if not df_termines.empty else pd.DataFrame()
edge_moy = df_edge_ok["Edge_Pct"].mean() if not df_edge_ok.empty else None

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Capital Actuel", f"{capital_actuel:.2f} U", f"{total_pl:+.2f} U")
c2.metric("ROI", f"{roi:.2f} %")
c3.metric("Winrate", f"{winrate:.1f} %")
c4.metric("Max Drawdown", f"{max_dd_pct:.1f} %")
c5.metric("Paris en Attente", f"{len(df_attente)}")
c6.metric(
    "CLV moyen",
    f"{clv_moy:+.2f} %" if clv_moy is not None else "N/A",
    help=f"Sur {len(df_clv_ok)} paris clôturés avec CLV_Fermeture",
)
c7.metric(
    "Edge moyen",
    f"{edge_moy:+.2f} %" if edge_moy is not None else "N/A",
    help=f"Sur {len(df_edge_ok)} paris avec edge enregistré",
)

if not df_termines.empty and "Segment" in df_termines.columns:
    st.markdown("#### Comparaison Full Game vs F5")
    seg_cols = st.columns(2)
    for i, seg in enumerate(["Full Game", "F5"]):
        sub = df_termines[df_termines["Segment"] == seg]
        with seg_cols[i]:
            if sub.empty:
                st.caption(f"**{seg}** — aucun pari clôturé")
                continue
            pl_seg = sub["P&L"].sum()
            wr_seg = len(sub[sub["Result"] == "✅ GAGNÉ"]) / len(sub) * 100
            roi_seg = (pl_seg / sub["Mise"].sum() * 100) if sub["Mise"].sum() > 0 else 0
            st.metric(f"{seg}", f"{pl_seg:+.2f} U", f"ROI {roi_seg:.1f} % · WR {wr_seg:.0f} %")

st.markdown("---")

# ==========================================
# 📈 BANKROLL & MARCHÉS
# ==========================================
st.subheader("📈 Évolution de la Bankroll")
if not df_termines.empty:
    fig_bankroll = creer_graphique_bankroll(
        df_termines,
        hover_data=["Date", "Match", "Pari", "P&L", "Segment"],
        couleur="#00BFFF",
        unite="Unités",
    )
    st.plotly_chart(fig_bankroll, use_container_width=True)
else:
    st.info("La courbe de croissance s'affichera dès qu'un pari sera classé GAGNÉ ou PERDU.")

col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.subheader("🎯 Rentabilité et Volume par Marché")
    if not df_termines.empty:
        pl_detail = df_termines.groupby("Catégorie").agg(
            P_and_L=("P&L", "sum"),
            Volume=("P&L", "count"),
        ).reset_index()
        fig_segment = creer_graphique_pl_marche(
            pl_detail,
            col_x="Catégorie",
            titre_x="Type de Marché",
            titre_y="Profit & Loss (Unités)",
        )
        st.plotly_chart(fig_segment, use_container_width=True)
    else:
        st.write("Données insuffisantes.")

with col_droite:
    st.subheader("⚖️ Tracking CLV par Marché")
    if not df_clv_ok.empty:
        df_clv_chart = df_clv_ok.copy()
        df_clv_chart["CLV"] = df_clv_chart["CLV_Fermeture"] / 100.0
        df_clv_chart["Type_Marche"] = df_clv_chart["Catégorie"]
        fig_clv = creer_graphique_clv_cumule(df_clv_chart, col_marche="Type_Marche")
        st.plotly_chart(fig_clv, use_container_width=True)
    elif not df_termines.empty:
        st.write("En attente des données de fermeture Pinnacle.")
    else:
        st.write("Données insuffisantes.")

# ==========================================
# 📡 PARIS EN ATTENTE
# ==========================================
st.markdown("---")
st.subheader("📡 Radar MLB : Transactions en cours")

if not df_attente.empty:
    df_att = df_attente.copy()
    colonnes_att = ["Date", "Match", "Pari", "Segment", "Cote", "Edge", "Mise", "CLV_Fermeture"]
    colonnes_att = [c for c in colonnes_att if c in df_att.columns]
    df_att = df_att[colonnes_att].sort_values("Date")

    def _style_clv(val):
        try:
            v = float(val)
            if v > 1:
                return "color: #00FF00; font-weight: bold"
            if v < -1:
                return "color: #FF4500; font-weight: bold"
            return "color: #FFD700"
        except (TypeError, ValueError):
            return ""

    if "CLV_Fermeture" in colonnes_att:
        st.dataframe(
            df_att.style.map(_style_clv, subset=["CLV_Fermeture"]),
            use_container_width=True,
        )
    else:
        st.dataframe(df_att, use_container_width=True)
else:
    st.success("Aucun ordre en attente sur le marché.")

# ==========================================
# 🔍 ANALYSE PAR TRANCHES DE COTES
# ==========================================
st.markdown("---")
st.subheader("🔍 Analyse par Tranches de Cotes (Où est l'Edge ?)")

if not df_termines.empty:
    bins = [1.0, 1.50, 1.80, 2.20, 2.60, 5.0]
    labels = [
        "Très Favori (<1.50)",
        "Favori (1.50-1.80)",
        "Equilibré (1.80-2.20)",
        "Outsider (2.20-2.60)",
        "Gros Outsider (>2.60)",
    ]
    df_termines["Tranche_Cote"] = pd.cut(df_termines["Cote"], bins=bins, labels=labels)
    pl_cotes = df_termines.groupby("Tranche_Cote", observed=False).agg(
        P_and_L=("P&L", "sum"),
        Volume=("P&L", "count"),
    ).reset_index()
    pl_cotes = pl_cotes[pl_cotes["Volume"] > 0]

    fig_cotes = creer_graphique_pl_marche(
        pl_cotes,
        col_x="Tranche_Cote",
        titre_x="Tranches de Cotes",
        titre_y="Profit & Loss (Unités)",
    )
    st.plotly_chart(fig_cotes, use_container_width=True)

    if "Segment" in df_termines.columns:
        with st.expander("📊 Détail P&L par segment et tranche de cote"):
            pivot = df_termines.pivot_table(
                index="Tranche_Cote",
                columns="Segment",
                values="P&L",
                aggfunc="sum",
                observed=False,
            ).fillna(0)
            st.dataframe(pivot.style.format("{:+.2f}"), use_container_width=True)

# ==========================================
# 📰 JOURNAL DE BORD
# ==========================================
st.markdown("---")
st.subheader("📰 Bilan : 10 derniers paris terminés")

if not df_termines.empty:
    df_derniers = df_termines.sort_values(by="Date", ascending=False).head(10)
    colonnes_hist = ["Date", "Match", "Pari", "Segment", "Cote", "Mise", "Result", "P&L", "CLV_Fermeture"]
    colonnes_hist = [c for c in colonnes_hist if c in df_derniers.columns]
    df_derniers = df_derniers[colonnes_hist].copy()
    df_derniers["P&L"] = df_derniers["P&L"].round(2)

    def colorer_resultat(val):
        if val == "✅ GAGNÉ":
            return "color: #00FF00; font-weight: bold"
        if val == "❌ PERDU":
            return "color: #FF4500; font-weight: bold"
        return ""

    st.dataframe(
        df_derniers.style.map(colorer_resultat, subset=["Result"]),
        use_container_width=True,
    )
else:
    st.write("Aucun pari terminé pour le moment.")
