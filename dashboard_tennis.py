"""
Dashboard Streamlit tennis ATP — calqué sur dashboard_nhl.py.

Journal : journal_trading_tennis.csv (colonnes sniper tennis).
Normalise Statut REGLE+W/L → GAGNÉ/PERDU pour réutiliser utils (Brier, CLV, bankroll).

Lancer :
  streamlit run dashboard_tennis.py
"""
from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO

import pandas as pd
import requests
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
    verifier_authentification,
)

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="Tennis Quant Dashboard", page_icon="🎾", layout="wide")

verifier_authentification()

st.title("🎾 Centre de Commandement : Sniper Tennis ATP")

JOURNAL_NOM = os.environ.get("TENNIS_JOURNAL_FILE", "journal_trading_tennis.csv")
URL_JOURNAL = os.environ.get(
    "TENNIS_JOURNAL_URL",
    f"https://chienblanc.pythonanywhere.com/data/{JOURNAL_NOM}",
)
PA_DATA_DIR = "/home/chienblanc/data"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPITAL_INITIAL = float(os.environ.get("TENNIS_BANKROLL", "1000.0"))
TENNIS_CALIB_MIN = 5

COULEURS_CLV = {
    "Moneyline": "#00BFFF",
    "Over/Under (Jeux)": "#FF4500",
}


def _candidats_fichiers(nom: str) -> list[str]:
    paths = []
    if os.environ.get("TENNIS_JOURNAL_CSV"):
        paths.append(os.environ["TENNIS_JOURNAL_CSV"])
    if os.path.isdir(PA_DATA_DIR):
        paths.append(os.path.join(PA_DATA_DIR, nom))
    paths.extend(
        [
            os.path.join(_SCRIPT_DIR, nom),
            os.path.join(os.getcwd(), nom),
            os.path.join(os.path.expanduser("~"), nom),
        ]
    )
    seen, out = set(), []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _score_journal(df: pd.DataFrame, mtime: float) -> tuple:
    n_attente = 0
    if "Statut" in df.columns:
        n_attente = int((df["Statut"].astype(str).str.strip() == "EN ATTENTE").sum())
    return (n_attente, mtime)


def _charger_csv(url: str, fichier_local: str):
    erreurs = []
    best_df, best_src, best_score = None, None, (-1, -1.0)

    for path in _candidats_fichiers(fichier_local):
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8")
            if df.empty:
                erreurs.append(f"{path}: vide")
                continue
            score = _score_journal(df, os.path.getmtime(path))
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
            df = pd.read_csv(BytesIO(r.content), encoding="utf-8")
            if not df.empty:
                n_a = _score_journal(df, 0)[0]
                return (
                    df,
                    "ok",
                    f"URL PA · {fichier_local} ({n_a} en attente / {len(df)} lignes)",
                )
            return df, "empty", f"URL PA · {fichier_local} (vide)"
        erreurs.append(f"URL HTTP {r.status_code}")
    except Exception as e:
        erreurs.append(f"URL: {e}")

    return pd.DataFrame(), "missing", " | ".join(erreurs) if erreurs else "introuvable"


def extraire_type_marche(marche: str) -> str:
    m = str(marche or "").upper()
    if m.startswith("OU_") or "OVER" in m or "UNDER" in m:
        return "Over/Under (Jeux)"
    if m.startswith("ML") or "MONEYLINE" in m:
        return "Moneyline"
    return "Autre"


def _normaliser_statut_tennis(df: pd.DataFrame) -> pd.DataFrame:
    """REGLE + Resultat W/L → GAGNÉ/PERDU (compat utils NHL)."""
    df = df.copy()
    if "Statut" not in df.columns:
        return df
    statut = df["Statut"].astype(str).str.strip()
    resultat = (
        df["Resultat"].astype(str).str.strip().str.upper()
        if "Resultat" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    out = statut.copy()
    mask_w = (resultat == "W") | (statut.str.upper() == "GAGNÉ")
    mask_l = (resultat == "L") | (statut.str.upper() == "PERDU")
    out.loc[mask_w] = "GAGNÉ"
    out.loc[mask_l] = "PERDU"
    # REGLE sans résultat reste REGLE
    df["Statut"] = out
    return df


def enrichir_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = _normaliser_statut_tennis(df)

    # Alias colonnes tennis → schéma dashboard (comme NHL)
    if "Cote" in df.columns and "Cote_Prise" not in df.columns:
        df["Cote_Prise"] = pd.to_numeric(df["Cote"], errors="coerce")
    if "CoteClose" in df.columns and "Cote_CLV" not in df.columns:
        df["Cote_CLV"] = pd.to_numeric(df["CoteClose"], errors="coerce")
    if "FairCote" in df.columns and "Vraie_Cote_Bot" not in df.columns:
        df["Vraie_Cote_Bot"] = pd.to_numeric(df["FairCote"], errors="coerce")
    if "Edge" in df.columns and "Edge(%)" not in df.columns:
        df["Edge(%)"] = pd.to_numeric(df["Edge"], errors="coerce")
    if "Mise" in df.columns and "Mise_€" not in df.columns:
        df["Mise_€"] = pd.to_numeric(df["Mise"], errors="coerce")
    if "PnL" in df.columns and "P&L" not in df.columns:
        df["P&L"] = pd.to_numeric(df["PnL"], errors="coerce")

    if "Marche" in df.columns:
        df["Type_Marche"] = df["Marche"].apply(extraire_type_marche)
    elif "Selection" in df.columns:
        df["Type_Marche"] = df["Selection"].apply(extraire_type_marche)

    # Match label
    if "PlayerA" in df.columns and "PlayerB" in df.columns:
        df["Match"] = (
            df["PlayerA"].astype(str).str.strip()
            + " vs "
            + df["PlayerB"].astype(str).str.strip()
        )

    # CLV ratio depuis cotes (source de vérité) — comme NHL
    if "Cote_Prise" in df.columns and "Cote_CLV" in df.columns:
        cote_p = pd.to_numeric(df["Cote_Prise"], errors="coerce")
        cote_c = pd.to_numeric(df["Cote_CLV"], errors="coerce")
        mask = (cote_p > 1) & (cote_c > 1)
        df["CLV"] = pd.NA
        df.loc[mask, "CLV"] = cote_p.loc[mask] / cote_c.loc[mask] - 1
        df["CLV"] = pd.to_numeric(df["CLV"], errors="coerce")
    elif "CLV" in df.columns:
        # Ancien journal : CLV déjà en %
        clv_raw = pd.to_numeric(df["CLV"], errors="coerce")
        # Si valeurs typiques en % (|x|>1 souvent), ramener en fraction
        if clv_raw.notna().any() and clv_raw.abs().median() > 0.5:
            df["CLV"] = clv_raw / 100.0
        else:
            df["CLV"] = clv_raw

    if "DryRun" in df.columns:
        df["DryRun"] = df["DryRun"].astype(str).str.strip().str.upper()
    if "Surface" in df.columns:
        df["Surface"] = df["Surface"].astype(str).str.strip()

    return df


def preparer_calibration_tennis(df: pd.DataFrame) -> pd.DataFrame:
    """p_model = ProbaFinal (blend) sinon 1/FairCote ; issue GAGNÉ/PERDU."""
    if df.empty or "Statut" not in df.columns:
        return pd.DataFrame()
    out = df[df["Statut"].isin(["GAGNÉ", "PERDU"])].copy()
    if out.empty:
        return pd.DataFrame()
    if "ProbaFinal" in out.columns:
        out["p_model"] = pd.to_numeric(out["ProbaFinal"], errors="coerce")
    elif "Vraie_Cote_Bot" in out.columns:
        cote = pd.to_numeric(out["Vraie_Cote_Bot"], errors="coerce")
        out["p_model"] = 1.0 / cote.where(cote > 1.0)
    elif "FairCote" in out.columns:
        cote = pd.to_numeric(out["FairCote"], errors="coerce")
        out["p_model"] = 1.0 / cote.where(cote > 1.0)
    else:
        return pd.DataFrame()
    out = out[out["p_model"].between(0.05, 0.95)]
    out["outcome"] = (out["Statut"] == "GAGNÉ").astype(float)
    if "Edge(%)" in out.columns:
        out["edge_pct"] = pd.to_numeric(out["Edge(%)"], errors="coerce")
    return out


def _tableau_brier_segments(df_cal: pd.DataFrame, col_segment: str) -> pd.DataFrame:
    if df_cal.empty or col_segment not in df_cal.columns:
        return pd.DataFrame()
    rows = []
    for seg, sub in df_cal.groupby(col_segment, dropna=False):
        if len(sub) < 3:
            continue
        brier = calculer_brier_score(sub)
        rows.append(
            {
                "Segment": seg,
                "N": len(sub),
                "Brier": round(brier, 4) if brier is not None else None,
                "Win %": round(sub["outcome"].mean() * 100, 1),
                "Edge moy. %": (
                    round(sub["edge_pct"].mean(), 2)
                    if "edge_pct" in sub.columns
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def load_data():
    df, statut, source = _charger_csv(URL_JOURNAL, JOURNAL_NOM)
    if statut != "ok":
        return df, statut, source

    colonnes_numeriques = [
        "Cote",
        "Cote_Prise",
        "CoteClose",
        "Cote_CLV",
        "FairCote",
        "Vraie_Cote_Bot",
        "ProbaModele",
        "ProbaMarche",
        "ProbaFinal",
        "BlendLam",
        "Edge",
        "Edge(%)",
        "Mise",
        "Mise_€",
        "PnL",
        "P&L",
        "CLV",
    ]
    df = nettoyer_colonnes_numeriques(df, colonnes_numeriques)
    df = convertir_dates(df)
    df = enrichir_dataframe(df)
    return df, "ok", source


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Contrôles")
if st.sidebar.button("🔄 Rafraîchir les données", use_container_width=True):
    load_data.clear()
    st.rerun()

df, statut_chargement, source = load_data()
st.caption(f"📡 Live : **{source}** (cache 60 s)")

if statut_chargement == "missing":
    st.warning(
        "⚠️ Journal tennis introuvable.\n\n"
        f"**Détail :** {source}\n\n"
        f"Attendu : `{JOURNAL_NOM}` (local, dossier PA, ou `TENNIS_JOURNAL_URL`). "
        "Lance `python tennis_sniper.py --once` pour créer le journal paper."
    )
    st.stop()

afficher_alertes_chargement(
    statut_chargement,
    df,
    msg_succes="🎾 Radar tennis armé. En attente des premiers edges paper…",
)

n_attente_raw = (
    int((df["Statut"] == "EN ATTENTE").sum())
    if "Statut" in df.columns and not df.empty
    else 0
)
with st.sidebar.expander(
    "🔍 Diagnostic données", expanded=(n_attente_raw == 0 and not df.empty)
):
    st.write(f"**Source :** {source}")
    st.write(f"**Lignes totales :** {len(df)}")
    st.write(f"**EN ATTENTE (brut) :** {n_attente_raw}")
    if "Statut" in df.columns and not df.empty:
        st.write("**Statuts :**", df["Statut"].value_counts().to_dict())
    cols_attendues = [
        "CoteClose",
        "ProbaFinal",
        "BlendLam",
        "Surface",
        "CommenceTime",
    ]
    manquantes = [c for c in cols_attendues if c not in df.columns]
    if manquantes:
        st.warning(f"Colonnes absentes (journal ancien ?) : {', '.join(manquantes)}")

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtres Live")

df_live = filtre_temporel_sidebar(df, key_prefix="tennis")
df_live = filtre_marche_sidebar(
    df_live, key="tennis_marche", label_tous="Tous les Marchés"
)

if not df_live.empty and "Surface" in df_live.columns:
    surfaces = sorted(df_live["Surface"].dropna().unique().tolist())
    surf = st.sidebar.selectbox(
        "🏟️ Surface :", ["Toutes"] + surfaces, key="tennis_surface"
    )
    if surf != "Toutes":
        df_live = df_live[df_live["Surface"] == surf]

if not df_live.empty and "Tourney" in df_live.columns:
    tourneys = sorted(df_live["Tourney"].dropna().unique().tolist())
    tour = st.sidebar.selectbox(
        "🏆 Tournoi :", ["Tous"] + tourneys, key="tennis_tourney"
    )
    if tour != "Tous":
        df_live = df_live[df_live["Tourney"] == tour]

if not df_live.empty and "DryRun" in df_live.columns:
    dry_f = st.sidebar.selectbox(
        "🧪 Mode :",
        ["Tous", "Paper (DRY)", "Live argent"],
        key="tennis_dry",
    )
    if dry_f == "Paper (DRY)":
        df_live = df_live[df_live["DryRun"] == "OUI"]
    elif dry_f == "Live argent":
        df_live = df_live[df_live["DryRun"] == "NON"]

if not df_live.empty and "PlayerA" in df_live.columns and "PlayerB" in df_live.columns:
    joueurs = sorted(
        set(df_live["PlayerA"].dropna()) | set(df_live["PlayerB"].dropna())
    )
    joueur = st.sidebar.selectbox(
        "👤 Joueur :", ["Tous"] + joueurs, key="tennis_joueur"
    )
    if joueur != "Tous":
        df_live = df_live[
            (df_live["PlayerA"] == joueur) | (df_live["PlayerB"] == joueur)
        ]

if "Statut" not in df_live.columns and not df_live.empty:
    st.error("Format CSV invalide (colonne `Statut` manquante).")
    st.stop()
if df_live.empty:
    if df.empty:
        st.stop()
    st.info("Aucune transaction ne correspond aux filtres sélectionnés.")
    st.stop()

# ==========================================
# KPI
# ==========================================
df_termines = df_live[df_live["Statut"].isin(["GAGNÉ", "PERDU"])].copy()
df_attente = df_live[df_live["Statut"] == "EN ATTENTE"].copy()

total_pl = float(df_termines["P&L"].sum()) if not df_termines.empty and "P&L" in df_termines.columns else 0.0
capital_actuel = CAPITAL_INITIAL + total_pl
col_mise = "Mise_€" if "Mise_€" in df_termines.columns else None
total_mise = float(df_termines[col_mise].sum()) if col_mise and not df_termines.empty else 0.0
roi = (total_pl / total_mise * 100) if total_mise > 0 else 0.0
winrate = (
    len(df_termines[df_termines["Statut"] == "GAGNÉ"]) / len(df_termines) * 100
    if not df_termines.empty
    else 0.0
)

max_dd_pct = 0.0
if not df_termines.empty and "P&L" in df_termines.columns:
    df_termines, max_dd_pct = calculer_max_drawdown(df_termines, "P&L", CAPITAL_INITIAL)

df_clv_ok = (
    df_termines[(df_termines["CLV"].notna()) & (df_termines["CLV"] != 0)]
    if not df_termines.empty and "CLV" in df_termines.columns
    else pd.DataFrame()
)
clv_moy = df_clv_ok["CLV"].mean() * 100 if not df_clv_ok.empty else None
edge_col = "Edge(%)" if "Edge(%)" in df_termines.columns else None
df_edge_ok = (
    df_termines[df_termines[edge_col].notna()]
    if edge_col and not df_termines.empty
    else pd.DataFrame()
)
edge_moy = df_edge_ok[edge_col].mean() if not df_edge_ok.empty else None

blend_moy = None
if "BlendLam" in df_termines.columns and not df_termines.empty:
    s = df_termines["BlendLam"].dropna()
    blend_moy = s.mean() if not s.empty else None

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
c1.metric("Capital Actuel", f"{capital_actuel:.2f} €", f"{total_pl:+.2f} €")
c2.metric("ROI", f"{roi:.2f} %")
c3.metric("Winrate", f"{winrate:.1f} %")
c4.metric("Max Drawdown", f"{max_dd_pct:.1f} %")
c5.metric("Ordres en Cours", f"{len(df_attente)}")
c6.metric(
    "CLV moyen",
    f"{clv_moy:+.2f} %" if clv_moy is not None else "N/A",
    help=f"Sur {len(df_clv_ok)} paris avec close Pinnacle (pré-tip)",
)
c7.metric(
    "Edge moyen",
    f"{edge_moy:+.2f} %" if edge_moy is not None else "N/A",
)
c8.metric(
    "λ blend moy.",
    f"{blend_moy:.2f}" if blend_moy is not None else "N/A",
    help="Poids marché dans p_final = (1-λ)p_model + λ p_mkt",
)

st.markdown("---")

# ==========================================
# BANKROLL & MARCHÉS
# ==========================================
st.subheader("📈 Évolution du Capital (Sniper Tennis)")
if not df_termines.empty and "Bankroll" in df_termines.columns:
    hover = ["Date", "Match", "Selection", "P&L"]
    hover = [c for c in hover if c in df_termines.columns]
    fig_bankroll = creer_graphique_bankroll(
        df_termines,
        hover_data=hover,
        couleur="#7CFC00",
        unite="€",
    )
    st.plotly_chart(fig_bankroll, use_container_width=True)
else:
    st.info("La courbe s'affichera dès qu'un match sera réglé (`--settle-clv`).")

col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.subheader("🎯 Rentabilité par Type de Pari")
    if not df_termines.empty and "Type_Marche" in df_termines.columns and "P&L" in df_termines.columns:
        pl_detail = (
            df_termines.groupby("Type_Marche")
            .agg(P_and_L=("P&L", "sum"), Volume=("P&L", "count"))
            .reset_index()
        )
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
    st.subheader("⚖️ CLV cumulé par Marché")
    if not df_clv_ok.empty and "Type_Marche" in df_clv_ok.columns:
        fig_clv = creer_graphique_clv_cumule(
            df_clv_ok, col_marche="Type_Marche", couleurs=COULEURS_CLV
        )
        st.plotly_chart(fig_clv, use_container_width=True)
    elif not df_termines.empty:
        st.write("En attente de cotes close Pinnacle (tracking pré-tip).")
    else:
        st.write("Données insuffisantes.")

# ==========================================
# CALIBRATION
# ==========================================
st.markdown("---")
st.subheader("🎯 Calibration du modèle (Brier)")
st.caption(
    "Probabilité utilisée : `ProbaFinal` (blend modèle × marché) vs résultats. "
    "Diagnostic paper — n'influence pas les paris."
)

df_cal = preparer_calibration_tennis(df_termines)
if (
    not df_cal.empty
    and "Type_Marche" in df_termines.columns
    and "Type_Marche" not in df_cal.columns
):
    cols_meta = [c for c in ("Date", "Marche", "Type_Marche", "Surface") if c in df_termines.columns]
    if "Date" in df_cal.columns and "Marche" in df_cal.columns and "Marche" in cols_meta:
        df_cal = df_cal.merge(
            df_termines[cols_meta].drop_duplicates(subset=["Date", "Marche"]),
            on=["Date", "Marche"],
            how="left",
            suffixes=("", "_dup"),
        )

if len(df_cal) >= TENNIS_CALIB_MIN:
    brier = calculer_brier_score(df_cal)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Brier score", f"{brier:.4f}", help="0 = parfait · ~0.25 ≈ pile-ou-face")
    b2.metric("Paris analysés", len(df_cal))
    b3.metric("Prob. moyenne modèle", f"{df_cal['p_model'].mean() * 100:.1f} %")
    b4.metric("Win rate réel", f"{df_cal['outcome'].mean() * 100:.1f} %")

    col_cal_prob, col_cal_edge = st.columns(2)
    with col_cal_prob:
        st.markdown("**Fiabilité par probabilité**")
        df_bins = calibration_par_probabilite(df_cal)
        if not df_bins.empty:
            st.plotly_chart(
                creer_graphique_calibration_prob(df_bins), use_container_width=True
            )
        else:
            st.info("Pas assez de paris dans les tranches.")
    with col_cal_edge:
        st.markdown("**Win rate par tranche d'edge**")
        df_edge_cal = calibration_par_edge(df_cal)
        if not df_edge_cal.empty:
            st.plotly_chart(
                creer_graphique_calibration_edge(df_edge_cal), use_container_width=True
            )
        else:
            st.info("Edge(%) absent ou tranches vides.")

    st.markdown("**Brier par segment** (≥ 3 paris)")
    seg_cols = st.columns(2)
    with seg_cols[0]:
        st.markdown("*Par marché*")
        tb_m = _tableau_brier_segments(df_cal, "Type_Marche")
        if not tb_m.empty:
            st.dataframe(tb_m, use_container_width=True, hide_index=True)
        else:
            st.caption("Pas assez de données.")
    with seg_cols[1]:
        if "Surface" in df_cal.columns:
            st.markdown("*Par surface*")
            tb_s = _tableau_brier_segments(df_cal, "Surface")
            if not tb_s.empty:
                st.dataframe(tb_s, use_container_width=True, hide_index=True)
            else:
                st.caption("Pas assez de données.")
else:
    st.info(
        f"En attente de **{TENNIS_CALIB_MIN} paris clôturés** "
        f"({len(df_cal)} actuellement)."
    )

# ==========================================
# PARIS EN ATTENTE
# ==========================================
st.markdown("---")
st.subheader("📡 Radar Tennis : Signaux Actifs / En attente")

COLONNES_ATTENTE = [
    "Date",
    "Tourney",
    "Surface",
    "Match",
    "Selection",
    "Type_Marche",
    "Cote_Prise",
    "Cote_CLV",
    "FairCote",
    "ProbaFinal",
    "BlendLam",
    "Edge(%)",
    "Mise_€",
    "DryRun",
    "CommenceTime",
]


def _preparer_tableau(df_src: pd.DataFrame, colonnes: list) -> pd.DataFrame:
    df_out = df_src.copy()
    if "CLV" in df_out.columns:
        clv = pd.to_numeric(df_out["CLV"], errors="coerce")
        df_out["CLV (%)"] = (clv * 100).round(2)
    for c in ("ProbaFinal", "ProbaModele", "ProbaMarche", "BlendLam"):
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce").round(3)
    cols = [c for c in colonnes if c in df_out.columns]
    if "CLV (%)" in df_out.columns and "CLV (%)" not in cols:
        cols.append("CLV (%)")
    return df_out[cols]


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


if not df_attente.empty:
    df_att_display = _preparer_tableau(df_attente, COLONNES_ATTENTE)
    df_sorted = df_att_display.sort_values("Date") if "Date" in df_att_display.columns else df_att_display
    if "CLV (%)" in df_sorted.columns:
        st.dataframe(
            df_sorted.style.map(style_clv, subset=["CLV (%)"]),
            use_container_width=True,
        )
    else:
        st.dataframe(df_sorted, use_container_width=True)
else:
    st.success("Aucun ordre en cours. Le sniper est en veille.")

# ==========================================
# JOURNAL
# ==========================================
st.markdown("---")
st.subheader("📰 Journal de bord : 15 derniers paris clôturés")

COLONNES_HIST = [
    "Date",
    "Tourney",
    "Surface",
    "Match",
    "Selection",
    "Type_Marche",
    "Cote_Prise",
    "Cote_CLV",
    "ProbaFinal",
    "Edge(%)",
    "Mise_€",
    "Statut",
    "P&L",
]

if not df_termines.empty:
    df_derniers = df_termines.sort_values("Date", ascending=False).head(15)
    df_hist = _preparer_tableau(df_derniers, COLONNES_HIST)
    if "P&L" in df_hist.columns:
        df_hist["P&L"] = df_hist["P&L"].round(2)

    def style_statut(val):
        if val == "GAGNÉ":
            return "color: #00FF00; font-weight: bold"
        if val == "PERDU":
            return "color: #FF4500; font-weight: bold"
        return ""

    styler = df_hist.style
    if "Statut" in df_hist.columns:
        styler = styler.map(style_statut, subset=["Statut"])
    if "CLV (%)" in df_hist.columns:
        styler = styler.map(style_clv, subset=["CLV (%)"])
    st.dataframe(styler, use_container_width=True)
else:
    st.write("Aucun pari terminé pour le moment.")

st.caption(
    "Paper trading · CLV = Cote / CoteClose − 1 (close figée au tip, jamais de cote live)."
)
