import json
import os
from datetime import datetime
from io import StringIO

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
    preparer_calibration_journal,
    verifier_authentification,
)

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
st.set_page_config(page_title="NHL Quant Dashboard", page_icon="🏒", layout="wide")

verifier_authentification()

st.title("🏒 Centre de Commandement : Sniper NHL Oméga")

URL_JOURNAL = "https://chienblanc.pythonanywhere.com/data/journal_trading_nhl_SEC2026xOmG.csv"
JOURNAL_NOM = "journal_trading_nhl_SEC2026xOmG.csv"
PA_DATA_DIR = "/home/chienblanc/data"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPITAL_INITIAL = float(os.environ.get("NHL_BANKROLL", "1000.0"))
NHL_CALIB_MIN = 5


def _candidats_fichiers(nom: str) -> list[str]:
    paths = []
    if os.environ.get("NHL_JOURNAL_CSV"):
        paths.append(os.environ["NHL_JOURNAL_CSV"])
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


def _score_journal(df: pd.DataFrame, mtime: float) -> tuple:
    n_attente = 0
    if "Statut" in df.columns:
        n_attente = int((df["Statut"].astype(str).str.strip() == "EN ATTENTE").sum())
    return (n_attente, mtime)


def _charger_csv(url: str, fichier_local: str):
    """Local le plus récent (priorité EN ATTENTE) → URL PA."""
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
            df = pd.read_csv(StringIO(r.text))
            if not df.empty:
                n_a = _score_journal(df, 0)[0]
                return df, "ok", f"URL PA · {fichier_local} ({n_a} en attente / {len(df)} lignes)"
            return df, "empty", f"URL PA · {fichier_local} (vide)"
        erreurs.append(f"URL HTTP {r.status_code}")
    except Exception as e:
        erreurs.append(f"URL: {e}")

    return pd.DataFrame(), "missing", " | ".join(erreurs) if erreurs else "introuvable"


def _charger_json_meta(nom: str) -> dict | None:
    for path in (
        os.path.join(_SCRIPT_DIR, nom),
        os.path.join(os.getcwd(), nom),
        os.path.join(PA_DATA_DIR, nom) if os.path.isdir(PA_DATA_DIR) else None,
    ):
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return None


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
    if "Statut" in df.columns:
        df["Statut"] = df["Statut"].astype(str).str.strip()
    if "Pari" in df.columns:
        df["Type_Marche"] = df["Pari"].apply(extraire_type_pari)
    if "Cote_Prise" in df.columns and "Cote_CLV" in df.columns:
        mask = (df["Cote_Prise"] > 1) & (df["Cote_CLV"] > 1)
        df["CLV"] = None
        df.loc[mask, "CLV"] = df.loc[mask, "Cote_Prise"] / df.loc[mask, "Cote_CLV"] - 1
    if "Gardiens_Confirmes" in df.columns:
        df["Gardiens_Confirmes"] = (
            df["Gardiens_Confirmes"].astype(str).str.strip().str.upper()
        )
    for col in ("B2B_Home", "B2B_Away"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    if "B2B_Home" in df.columns and "B2B_Away" in df.columns:
        df["Contexte_B2B"] = df.apply(
            lambda r: "B2B domicile" if r.get("B2B_Home") == "OUI"
            else ("B2B visiteur" if r.get("B2B_Away") == "OUI" else "Repos"),
            axis=1,
        )
    return df


def _tableau_brier_segments(df_cal: pd.DataFrame, col_segment: str) -> pd.DataFrame:
    if df_cal.empty or col_segment not in df_cal.columns:
        return pd.DataFrame()
    rows = []
    for seg, sub in df_cal.groupby(col_segment, dropna=False):
        if len(sub) < 3:
            continue
        brier = calculer_brier_score(sub)
        rows.append({
            "Segment": seg,
            "N": len(sub),
            "Brier": round(brier, 4) if brier is not None else None,
            "Win %": round(sub["outcome"].mean() * 100, 1),
            "Edge moy. %": round(sub["edge_pct"].mean(), 2) if "edge_pct" in sub.columns else None,
        })
    return pd.DataFrame(rows)


def _afficher_etat_modele():
    rho_meta = _charger_json_meta("rho_calibrage_meta.json")
    hia_meta = _charger_json_meta("hia_equipes_meta.json")
    ligue_meta = _charger_json_meta("nhl_league_calib.json")

    if not rho_meta and not hia_meta and not ligue_meta:
        st.caption(
            "Fichiers `rho_calibrage_meta.json` / `hia_equipes_meta.json` non trouvés "
            "(copie locale ou même dossier que le journal sur PA)."
        )
        return

    cols = st.columns(4)
    if rho_meta:
        cols[0].metric("ρ Dixon-Coles", f"{float(rho_meta.get('rho', 0)):.3f}")
        cols[1].metric("HIA global", f"{float(rho_meta.get('hia', 0)):.1%}")
        cols[2].metric(
            "OT dom. calibré",
            f"{float(rho_meta.get('ot_home_adv', 0.52)):.1%}",
            help="Avantage domicile prolongation/tirs au but",
        )
        nb = int(rho_meta.get("nb_matchs", 0))
        cols[3].metric("Matchs MLE", nb, help=rho_meta.get("date", "—"))
        c1, c2, c3, c4 = st.columns(4)
        c1.caption(f"prob_tie **{float(rho_meta.get('prob_tie', 0.12)):.1%}**")
        c2.caption(f"prob_EN **{float(rho_meta.get('prob_en', 0.22)):.1%}**")
        c3.caption(f"NB O/U r **{float(rho_meta.get('nb_ou_dispersion', 25)):.1f}**")
        c4.caption(
            f"ref sens. **{float(rho_meta.get('ref_sensibilite', 0.2)):.2f}** · "
            f"FO sens. **{float(rho_meta.get('faceoff_sensibilite', 0.25)):.2f}** · "
            f"PP share **{float(rho_meta.get('pp_lam_share', 0.2)):.0%}**"
        )
        c5, c6, c7 = st.columns(3)
        c5.caption(
            f"voyage B2B **{float(rho_meta.get('travel_b2b_atk_pct', 0.04)):.0%}**/"
            f"**+{float(rho_meta.get('travel_b2b_def_pct', 0.06)):.0%}**"
        )
        c6.caption(f"GSAx→λ mult **{float(rho_meta.get('gsax_lam_mult', 1.0)):.2f}**")
        c7.caption(f"PL scale **{float(rho_meta.get('pl_scale', 1.0)):.2f}**")

    if hia_meta and hia_meta.get("teams"):
        teams = hia_meta["teams"]
        extremes = sorted(teams.items(), key=lambda kv: kv[1].get("hia", 0))
        st.caption(
            f"HIA par équipe — **{len(teams)}** indexées · "
            f"min {extremes[0][0]} {extremes[0][1]['hia']:.1%} · "
            f"max {extremes[-1][0]} {extremes[-1][1]['hia']:.1%} "
            f"({hia_meta.get('date', '')})"
        )

    if ligue_meta and ligue_meta.get("games"):
        st.caption(
            f"Historique ligue calibration — **{len(ligue_meta['games'])}** matchs "
            f"({ligue_meta.get('date', '')})"
        )


@st.cache_data(ttl=60)
def load_data():
    df, statut, source = _charger_csv(URL_JOURNAL, JOURNAL_NOM)
    if statut != "ok":
        return df, statut, source

    colonnes_numeriques = [
        "Vraie_Cote_Bot", "Cote_Prise", "Cote_CLV",
        "Edge(%)", "Risque(%)", "Mise_€", "P&L",
        "Lam_Ext", "Lam_Dom", "Rho", "Hia", "Confiance_Kelly",
    ]
    df = nettoyer_colonnes_numeriques(df, colonnes_numeriques)
    df = convertir_dates(df)
    df = enrichir_dataframe(df)
    return df, "ok", source


# ==========================================
# 🎛️ SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Contrôles")
if st.sidebar.button("🔄 Rafraîchir les données", use_container_width=True):
    load_data.clear()
    st.rerun()

df, statut_chargement, source = load_data()
st.caption(f"📡 Live : **{source}** (cache 60 s)")

if statut_chargement == "missing":
    st.warning(
        "⚠️ Journal NHL introuvable.\n\n"
        f"**Détail :** {source}\n\n"
        "Le dashboard lit `journal_trading_nhl_SEC2026xOmG.csv` (local, dossier PA, ou URL). "
        "Sur PythonAnywhere : vérifiez que le bot NHL tourne et publie le CSV dans `/data/`."
    )
    st.stop()

afficher_alertes_chargement(
    statut_chargement, df,
    msg_succes="🏒 Le radar NHL Oméga est armé. En attente des premières transactions...",
)

n_attente_raw = int((df["Statut"] == "EN ATTENTE").sum()) if "Statut" in df.columns and not df.empty else 0
with st.sidebar.expander("🔍 Diagnostic données", expanded=(n_attente_raw == 0 and not df.empty)):
    st.write(f"**Source :** {source}")
    st.write(f"**Lignes totales :** {len(df)}")
    st.write(f"**EN ATTENTE (brut) :** {n_attente_raw}")
    if "Statut" in df.columns and not df.empty:
        st.write("**Statuts :**", df["Statut"].value_counts().to_dict())
    cols_manquantes = [c for c in (
        "Gardiens_Confirmes", "Hia", "Confiance_Kelly", "Rho", "Lam_Dom", "Lam_Ext",
    ) if c not in df.columns]
    if cols_manquantes:
        st.warning(f"Colonnes absentes (journal ancien ?) : {', '.join(cols_manquantes)}")

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtres Live")

df_live = filtre_temporel_sidebar(df, key_prefix="nhl")
df_live = filtre_marche_sidebar(df_live, key="nhl_marche", label_tous="Tous les Marchés")

if not df_live.empty and "Gardiens_Confirmes" in df_live.columns:
    filtre_gard = st.sidebar.selectbox(
        "🥅 Gardiens :",
        ["Tous", "Confirmés (OUI)", "Probables (NON)"],
        key="nhl_gardiens",
    )
    if filtre_gard == "Confirmés (OUI)":
        df_live = df_live[df_live["Gardiens_Confirmes"] == "OUI"]
    elif filtre_gard == "Probables (NON)":
        df_live = df_live[df_live["Gardiens_Confirmes"] == "NON"]

if not df_live.empty and "Contexte_B2B" in df_live.columns:
    filtre_b2b = st.sidebar.selectbox(
        "🔄 Back-to-back :",
        ["Tous", "Repos", "B2B domicile", "B2B visiteur", "Tout B2B"],
        key="nhl_b2b",
    )
    if filtre_b2b == "Repos":
        df_live = df_live[df_live["Contexte_B2B"] == "Repos"]
    elif filtre_b2b == "Tout B2B":
        df_live = df_live[df_live["Contexte_B2B"].str.startswith("B2B")]
    elif filtre_b2b != "Tous":
        df_live = df_live[df_live["Contexte_B2B"] == filtre_b2b]

if not df_live.empty and "Visiteur" in df_live.columns and "Local" in df_live.columns:
    equipes = sorted(set(df_live["Visiteur"].dropna()) | set(df_live["Local"].dropna()))
    equipe_choisie = st.sidebar.selectbox(
        "🏒 Équipe :", ["Toutes les Équipes"] + equipes, key="nhl_equipe",
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
# 🧠 ÉTAT MODÈLE (hors filtres paris)
# ==========================================
with st.expander("🧠 État calibration MLE (fichiers meta locaux)", expanded=False):
    _afficher_etat_modele()

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

df_clv_ok = (
    df_termines[(df_termines["CLV"].notna()) & (df_termines["CLV"] != 0)]
    if not df_termines.empty else pd.DataFrame()
)
clv_moy = df_clv_ok["CLV"].mean() * 100 if not df_clv_ok.empty else None
edge_col = "Edge(%)" if "Edge(%)" in df_termines.columns else None
df_edge_ok = df_termines[df_termines[edge_col].notna()] if edge_col and not df_termines.empty else pd.DataFrame()
edge_moy = df_edge_ok[edge_col].mean() if not df_edge_ok.empty else None

conf_moy = None
if "Confiance_Kelly" in df_termines.columns and not df_termines.empty:
    s = df_termines["Confiance_Kelly"].dropna()
    conf_moy = s.mean() if not s.empty else None

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
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
c8.metric(
    "Conf. Kelly moy.",
    f"{conf_moy:.2f}" if conf_moy is not None else "N/A",
    help="Multiplicateur incertitude paramétrique (1 = pleine confiance)",
)

st.markdown("---")

# ==========================================
# 📈 BANKROLL & MARCHÉS
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

df_cal = preparer_calibration_journal(df_termines)
if "Type_Marche" in df_termines.columns and "Type_Marche" not in df_cal.columns:
    df_cal = df_cal.merge(
        df_termines[["Date", "Pari", "Type_Marche", "Gardiens_Confirmes", "Contexte_B2B"]],
        on=["Date", "Pari"],
        how="left",
        suffixes=("", "_dup"),
    )

if len(df_cal) >= NHL_CALIB_MIN:
    brier = calculer_brier_score(df_cal)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Brier score", f"{brier:.4f}", help="0 = parfait · ~0.25 = aléatoire sur 50/50")
    b2.metric("Paris analysés", len(df_cal))
    b3.metric(
        "Prob. moyenne modèle",
        f"{df_cal['p_model'].mean() * 100:.1f} %",
    )
    b4.metric(
        "Win rate réel",
        f"{df_cal['outcome'].mean() * 100:.1f} %",
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

    st.markdown("**Brier par segment** (≥ 3 paris)")
    seg_cols = st.columns(3)
    with seg_cols[0]:
        st.markdown("*Par marché*")
        tb_m = _tableau_brier_segments(df_cal, "Type_Marche")
        if not tb_m.empty:
            st.dataframe(tb_m, use_container_width=True, hide_index=True)
        else:
            st.caption("Pas assez de données par marché.")
    with seg_cols[1]:
        if "Gardiens_Confirmes" in df_cal.columns:
            st.markdown("*Gardiens confirmés*")
            tb_g = _tableau_brier_segments(df_cal, "Gardiens_Confirmes")
            if not tb_g.empty:
                st.dataframe(tb_g, use_container_width=True, hide_index=True)
            else:
                st.caption("Pas assez de données.")
    with seg_cols[2]:
        if "Contexte_B2B" in df_cal.columns:
            st.markdown("*Contexte B2B*")
            tb_b = _tableau_brier_segments(df_cal, "Contexte_B2B")
            if not tb_b.empty:
                st.dataframe(tb_b, use_container_width=True, hide_index=True)
            else:
                st.caption("Pas assez de données.")

    filtre_cal = st.selectbox(
        "Filtrer les graphiques calibration par marché",
        ["Tous"] + sorted(df_cal["Type_Marche"].dropna().unique().tolist())
        if "Type_Marche" in df_cal.columns else ["Tous"],
        key="nhl_cal_marche",
    )
    if filtre_cal != "Tous" and "Type_Marche" in df_cal.columns:
        df_cal_f = df_cal[df_cal["Type_Marche"] == filtre_cal]
        if len(df_cal_f) >= 3:
            st.caption(
                f"**{filtre_cal}** — Brier {calculer_brier_score(df_cal_f):.4f} "
                f"({len(df_cal_f)} paris) · CLV moy. "
                f"{df_termines[df_termines['Type_Marche'] == filtre_cal]['CLV'].mean() * 100:+.2f} %"
                if "CLV" in df_termines.columns
                and not df_termines[df_termines["Type_Marche"] == filtre_cal].empty
                else f"**{filtre_cal}** — Brier {calculer_brier_score(df_cal_f):.4f} ({len(df_cal_f)} paris)"
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

COLONNES_ATTENTE = [
    "Date", "Visiteur", "Local", "Pari", "Type_Marche",
    "Gardien_Ext", "Gardien_Dom", "Gardiens_Confirmes",
    "B2B_Home", "B2B_Away", "Cote_Prise", "Mise_€", "Edge(%)",
    "Rho", "Hia", "Confiance_Kelly", "Lam_Dom", "Lam_Ext",
]


def _preparer_tableau_affichage(df_src: pd.DataFrame, colonnes: list) -> pd.DataFrame:
    df_out = df_src.copy()
    if "CLV" in df_out.columns:
        df_out["CLV (%)"] = (df_out["CLV"] * 100).round(2)
    if "Confiance_Kelly" in df_out.columns:
        df_out["Confiance_Kelly"] = df_out["Confiance_Kelly"].round(3)
    if "Hia" in df_out.columns:
        df_out["Hia"] = (df_out["Hia"] * 100).round(2)
    cols = [c for c in colonnes if c in df_out.columns]
    if "CLV (%)" in df_out.columns and "CLV (%)" not in cols:
        cols.append("CLV (%)")
    return df_out[cols]


if not df_attente.empty:
    df_att_display = _preparer_tableau_affichage(df_attente, COLONNES_ATTENTE)

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

    df_sorted = df_att_display.sort_values("Date")
    if "CLV (%)" in df_sorted.columns:
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
st.subheader("📰 Journal de bord : 15 dernières rencontres clôturées")

COLONNES_HIST = [
    "Date", "Visiteur", "Local", "Pari", "Type_Marche",
    "Gardien_Ext", "Gardien_Dom", "Gardiens_Confirmes",
    "Cote_Prise", "Cote_CLV", "Mise_€", "Edge(%)", "Confiance_Kelly",
    "Rho", "Hia", "Statut", "P&L",
]

if not df_termines.empty:
    df_derniers = df_termines.sort_values("Date", ascending=False).head(15)
    df_hist = _preparer_tableau_affichage(df_derniers, COLONNES_HIST)
    if "P&L" in df_hist.columns:
        df_hist["P&L"] = df_hist["P&L"].round(2)

    def style_statut(val):
        if val == "GAGNÉ":
            return "color: #00FF00; font-weight: bold"
        if val == "PERDU":
            return "color: #FF4500; font-weight: bold"
        return ""

    st.dataframe(
        df_hist.style.map(style_statut, subset=["Statut"]),
        use_container_width=True,
    )
else:
    st.write("Aucun pari terminé pour le moment.")
