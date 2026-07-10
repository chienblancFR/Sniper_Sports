"""
utils_mlb_dashboard.py — Utilitaires légers pour dashboard_mlb.py (Streamlit Cloud).
Évite d'importer utils.py complet (calibration foot, numpy/scipy lourds).
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def verifier_authentification():
    import os

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


def nettoyer_colonnes_numeriques(df: pd.DataFrame, colonnes: list) -> pd.DataFrame:
    for col in colonnes:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def convertir_dates(df: pd.DataFrame) -> pd.DataFrame:
    candidates = ["Date", "date", "Kickoff", "kickoff", "datetime"]
    for col in candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if hasattr(df[col].dt, "tz") and df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
            if col != "Date":
                df.rename(columns={col: "Date"}, inplace=True)
            break
    return df


def afficher_alertes_chargement(statut: str, df: pd.DataFrame, msg_succes: str = ""):
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


def filtre_temporel_sidebar(df: pd.DataFrame, key_prefix: str = "live") -> pd.DataFrame:
    if df.empty or "Date" not in df.columns:
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
        df = df[df["Date"] >= date_min]

    return df


def calculer_max_drawdown(df: pd.DataFrame, col_profit: str, capital_initial: float):
    df = df.copy()
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    df["Bankroll"] = capital_initial + df[col_profit].cumsum()
    peak = df["Bankroll"].cummax()
    drawdown = (peak - df["Bankroll"]) / peak
    max_dd_pct = drawdown.max() * 100 if not drawdown.empty else 0.0
    return df, max_dd_pct


def _hex_to_rgb(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r},{g},{b}"


def appliquer_theme_dark(fig: go.Figure):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF", size=12),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            linecolor="rgba(255,255,255,0.2)",
            tickcolor="rgba(255,255,255,0.3)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            linecolor="rgba(255,255,255,0.2)",
            tickcolor="rgba(255,255,255,0.3)",
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )


def creer_graphique_bankroll(
    df: pd.DataFrame,
    hover_data: list = None,
    couleur: str = "#00FF00",
    unite: str = "Unités",
) -> go.Figure:
    fig = go.Figure()
    x_vals = df["Date"] if "Date" in df.columns else df.index
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=df["Bankroll"],
        mode="lines",
        fill="tozeroy",
        line=dict(color=couleur, width=2.5),
        fillcolor=f"rgba({_hex_to_rgb(couleur)},0.10)",
        name=f"Capital ({unite})",
        hovertemplate=(
            "<b>%{x|%d/%m/%Y}</b><br>"
            f"Capital : %{{y:.2f}} {unite}<extra></extra>"
        ),
    ))
    fig.update_layout(
        yaxis_title=f"Capital ({unite})",
        xaxis_title="",
        showlegend=False,
        hovermode="x unified",
    )
    appliquer_theme_dark(fig)
    return fig


def creer_graphique_pl_marche(
    df: pd.DataFrame,
    col_x: str,
    titre_x: str = "Catégorie",
    titre_y: str = "P&L",
) -> go.Figure:
    couleurs = ["#00FF00" if v >= 0 else "#FF4500" for v in df["P_and_L"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[col_x],
        y=df["P_and_L"],
        marker_color=couleurs,
        text=[f"{v:+.2f} u<br>{n} paris" for v, n in zip(df["P_and_L"], df["Volume"])],
        textposition="outside",
        name="P&L",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#FFFFFF", opacity=0.4)
    fig.update_layout(xaxis_title=titre_x, yaxis_title=titre_y, showlegend=False)
    appliquer_theme_dark(fig)
    return fig


def creer_graphique_clv_cumule(
    df_clv: pd.DataFrame,
    col_marche: str = "Type_Marche",
    couleurs: dict | None = None,
) -> go.Figure:
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
