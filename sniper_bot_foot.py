import requests, json, os, time, logging, csv, signal, shutil
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize_scalar, minimize
from thefuzz import process
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import asyncio
import aiohttp
import aiosqlite

# ==========================================
# 💾 0. CONFIGURATION LOGS & BASE DE DONNÉES
# ==========================================
logging.basicConfig(
    filename='sniper_activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S'
)

def log_info(msg):
    print(msg)
    logging.info(msg)

async def init_db():
    """Ouvre une connexion persistante et crée les tables si nécessaire."""
    global db_conn
    db_conn = await aiosqlite.connect("sniper_data.db")
    await db_conn.execute('''CREATE TABLE IF NOT EXISTS xg_cache
                          (cle TEXT PRIMARY KEY, xg_p REAL, xg_c REAL, timestamp DATETIME)''')
    await db_conn.execute('''CREATE TABLE IF NOT EXISTS paris_log
                          (id_match INTEGER, equipe TEXT, handicap REAL, cote_prise REAL,
                           mise REAL, cote_cloture REAL DEFAULT 0.0, edge_detecte REAL,
                           p_modele REAL, clv REAL DEFAULT 0.0, statut TEXT DEFAULT 'PENDING',
                           resultat REAL DEFAULT 0.0, ligue TEXT, is_lineup_official INTEGER,
                           timestamp TEXT,
                           PRIMARY KEY (id_match, equipe, handicap))''')
    # Accumulateur de scores pour l'estimation dynamique de ρ par ligue/saison
    await db_conn.execute('''CREATE TABLE IF NOT EXISTS scores_matchs
                          (id_match INTEGER PRIMARY KEY, ligue_id INTEGER,
                           saison INTEGER, buts_dom INTEGER, buts_ext INTEGER)''')
    # Table de couverture xG par ligue (auto-détectée au démarrage)
    await db_conn.execute('''CREATE TABLE IF NOT EXISTS xg_couverture_ligue
                          (ligue_id INTEGER PRIMARY KEY, a_xg INTEGER DEFAULT 0,
                           teste_le TEXT)''')
    # Migration : ajout colonne is_xg si table xg_cache existait déjà sans elle
    try:
        await db_conn.execute("ALTER TABLE xg_cache ADD COLUMN is_xg INTEGER DEFAULT 0")
    except Exception:
        pass  # Colonne déjà présente
    # Table des paramètres DC complets (α, β, γ, ρ) par équipe/ligue/saison
    await db_conn.execute('''CREATE TABLE IF NOT EXISTS dc_params
                          (ligue_id INTEGER, saison INTEGER, team_id INTEGER,
                           attack REAL, defense REAL, home_adv REAL, rho REAL,
                           computed_at TEXT,
                           PRIMARY KEY (ligue_id, saison, team_id))''')
    # Migration : IDs d'équipes + date de match dans scores_matchs
    for migration in [
        "ALTER TABLE scores_matchs ADD COLUMN team_dom_id INTEGER DEFAULT NULL",
        "ALTER TABLE scores_matchs ADD COLUMN team_ext_id INTEGER DEFAULT NULL",
        "ALTER TABLE scores_matchs ADD COLUMN match_date TEXT DEFAULT NULL",
    ]:
        try:
            await db_conn.execute(migration)
        except Exception:
            pass  # Colonne déjà présente
    # Migration : colonnes CLV enrichies (équipes Odds API + kickoff + flag notification)
    for migration in [
        "ALTER TABLE paris_log ADD COLUMN equipe_dom TEXT DEFAULT NULL",
        "ALTER TABLE paris_log ADD COLUMN equipe_ext TEXT DEFAULT NULL",
        "ALTER TABLE paris_log ADD COLUMN kickoff TEXT DEFAULT NULL",
        "ALTER TABLE paris_log ADD COLUMN clv_notifie INTEGER DEFAULT 0",
    ]:
        try:
            await db_conn.execute(migration)
        except Exception:
            pass  # Colonne déjà présente
    await db_conn.commit()

semaphore = None
db_lock = None
db_conn = None  # Connexion persistante — ouverte une seule fois dans init_db

# --- 🛡️ CACHES GLOBAUX ---
cache_standings = {}
cache_heures_matchs = {}

# ==========================================
# ⚙️ 1. SECRETS & PARAMÈTRES (LE BIG 3)
# ==========================================
load_dotenv("identifiants_différent_api.env")
load_dotenv()  # .env prioritaire si présent
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
API_ODDS_KEY = os.getenv("API_ODDS_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_METEO_KEY = os.getenv("OPENWEATHER_KEY")

KELLY_FRAC = 0.05       # Fraction Kelly de base (ajustée dynamiquement selon le drawdown)
KELLY_COURANT = 0.05   # Mise à jour chaque cycle par actualiser_kelly_adaptatif()

def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, str(default)).strip().lower()
    return v in ("1", "true", "yes", "on")

FOOT_KELLY_BRIER_ACTIF = _env_bool("FOOT_KELLY_BRIER_ACTIF", True)
FOOT_KELLY_BRIER_FENETRE = int(os.environ.get("FOOT_KELLY_BRIER_FENETRE", "40"))
FOOT_KELLY_BRIER_MIN_PARIS = int(os.environ.get("FOOT_KELLY_BRIER_MIN_PARIS", "20"))
FOOT_KELLY_BSS_SENSIBILITE = float(os.environ.get("FOOT_KELLY_BSS_SENSIBILITE", "0.30"))
FOOT_KELLY_BSS_MULT_MIN = float(os.environ.get("FOOT_KELLY_BSS_MULT_MIN", "0.4"))

# Shrinkage bayésien adaptatif : plus la ligue est petite / données xG peu fiables,
# plus on maintient longtemps les estimations proches de la moyenne de ligue.
# Grandes ligues 20 clubs / 38J → prior faible (on fait confiance aux données équipe plus vite)
# Petites ligues 16 clubs / 30J → prior élevé (on reste prudent plus longtemps)
N_PRIOR_PAR_LIGUE = {
    140: 7,   # La Liga         (20 clubs, 38J)
    39:  7,   # Premier League  (20 clubs, 38J)
    135: 7,   # Serie A         (20 clubs, 38J)
    40:  7,   # Championship    (24 clubs, 46J — beaucoup de matchs)
    78:  8,   # Bundesliga      (18 clubs, 34J)
    61:  8,   # Ligue 1         (18 clubs, 34J)
    141: 8,   # LaLiga 2        (22 clubs, 42J)
    88:  9,   # Eredivisie      (18 clubs, 34J — moins de couverture médiatique)
    94:  9,   # Primeira Liga   (18 clubs, 34J)
    203: 9,   # Süper Lig       (19 clubs, 36J)
    71:  9,   # Série A Brésil  (20 clubs, 38J — xG parfois incomplets)
    136: 10,  # Serie B         (20 clubs, 38J — xG souvent absents → fallback buts)
    253: 10,  # MLS             (29 clubs, calendrier irrégulier, hétérogénéité élevée)
    144: 11,  # Jupiler Pro     (16 clubs + championship round)
    113: 12,  # Allsvenskan     (16 clubs, 30J — petite ligue nordique)
    103: 12,  # Eliteserien     (16 clubs, 30J — petite ligue nordique)
}
N_PRIOR_DEFAULT = 8

# Ligues confirmées sans xG via auto-détection (voir detecter_ligues_sans_xg).
# Ce set est peuplé dynamiquement au démarrage — les valeurs ci-dessous sont
# des valeurs initiales conservatrices, remplacées dès le premier scan.
LIGUES_SANS_XG = {
    113,  # Allsvenskan  → testé : expected_goals null
    136,  # Serie B      → à confirmer via auto-détection
    203,  # Süper Lig    → à confirmer via auto-détection
    # 71  Brésil         → CONFIRMÉ xG ✅ (29/06/2026)
    # 103 Eliteserien    → CONFIRMÉ xG ✅ (29/06/2026)
    # 253 MLS            → CONFIRMÉ xG ✅ (29/06/2026)
}

async def detecter_ligues_sans_xg(session):
    """
    Détecte automatiquement quelles ligues ont des données xG réelles dans API-Football.
    Teste chaque ligue en récupérant un match récent et vérifiant la présence de
    'expected_goals' dans /fixtures/statistics.

    Résultats mis en cache dans la table xg_couverture_ligue (retesté tous les 30 jours).
    Met à jour LIGUES_SANS_XG en mémoire.
    """
    global LIGUES_SANS_XG
    log_info("🔬 Détection automatique de la couverture xG par ligue...")
    maintenant = datetime.now().isoformat()
    seuil_retest = (datetime.now() - timedelta(days=30)).isoformat()
    nouvelles_sans_xg = set()

    for ligue in CHAMPIONNATS:
        ligue_id = ligue['id']

        # Vérifier si un test récent existe en DB (< 30 jours)
        async with db_lock:
            async with db_conn.execute(
                "SELECT a_xg, teste_le FROM xg_couverture_ligue WHERE ligue_id=?",
                (ligue_id,)
            ) as cursor:
                row = await cursor.fetchone()

        if row and row[1] and row[1] > seuil_retest:
            # Résultat encore frais — utiliser le cache
            if row[0] == 0:
                nouvelles_sans_xg.add(ligue_id)
            continue

        # Trouver un match récent terminé pour cette ligue
        saison = obtenir_saison_api(ligue['nom'])
        url = f"{URL_FOOTBALL}/fixtures?league={ligue_id}&season={saison}&last=3&status=FT"
        data = await fetch_async(session, url, HEADERS_FB)
        fixtures = data.get('response', []) if data else []

        if not fixtures:
            # Essayer saison précédente
            url = f"{URL_FOOTBALL}/fixtures?league={ligue_id}&season={saison-1}&last=3&status=FT"
            data = await fetch_async(session, url, HEADERS_FB)
            fixtures = data.get('response', []) if data else []

        if not fixtures:
            log_info(f"  ⚠️ {ligue['nom']} : aucun match récent trouvé, xG supposé absent.")
            nouvelles_sans_xg.add(ligue_id)
            async with db_lock:
                await db_conn.execute(
                    "INSERT OR REPLACE INTO xg_couverture_ligue VALUES (?, ?, ?)",
                    (ligue_id, 0, maintenant)
                )
                await db_conn.commit()
            continue

        # Tester le premier match disponible
        fixture_id = fixtures[0]['fixture']['id']
        url_stats = f"{URL_FOOTBALL}/fixtures/statistics?fixture={fixture_id}"
        stats = await fetch_async(session, url_stats, HEADERS_FB)

        a_xg = False
        if stats and stats.get('response'):
            for team_stat in stats['response']:
                xg_raw = next(
                    (s['value'] for s in team_stat['statistics'] if s['type'] == 'expected_goals'),
                    None
                )
                try:
                    if xg_raw not in (None, 'null', '', 'None') and float(xg_raw) >= 0:
                        a_xg = True
                        break
                except (TypeError, ValueError):
                    pass

        statut = "✅ xG disponibles" if a_xg else "❌ pas de xG"
        log_info(f"  {ligue['nom']} (fixture {fixture_id}) → {statut}")

        if not a_xg:
            nouvelles_sans_xg.add(ligue_id)

        async with db_lock:
            await db_conn.execute(
                "INSERT OR REPLACE INTO xg_couverture_ligue VALUES (?, ?, ?)",
                (ligue_id, int(a_xg), maintenant)
            )
            await db_conn.commit()

    LIGUES_SANS_XG = nouvelles_sans_xg
    log_info(f"🔬 Couverture xG détectée. Ligues sans xG : {LIGUES_SANS_XG}")

# ρ de Dixon-Coles calibré par ligue (issu de la littérature académique).
# Les ligues à fort volume de buts (Bundesliga, Eredivisie) ont un ρ plus négatif
# car la correction 0-0/1-1 est plus marquée relativement.
RHO_PAR_LIGUE = {
    78:  -0.16,  # Bundesliga (scores élevés)
    88:  -0.15,  # Eredivisie
    39:  -0.13,  # Premier League
    40:  -0.12,  # Championship
    61:  -0.12,  # Ligue 1
    141: -0.12,  # LaLiga 2
    136: -0.11,  # Serie B
    140: -0.10,  # La Liga
    94:  -0.10,  # Primeira Liga
    135: -0.09,  # Serie A (faible scoring)
    203: -0.08,  # Süper Lig
    71:  -0.08,  # Brésil Série A
    113: -0.11,  # Allsvenskan
    103: -0.10,  # Eliteserien
    144: -0.12,  # Jupiler Pro League
    253: -0.09,  # MLS
}
RHO_DEFAULT = -0.12  # Fallback pour toute ligue non listée

# Dict mis à jour dynamiquement chaque saison par estimer_rho_saison()
# Prioritaire sur RHO_PAR_LIGUE quand on dispose d'assez de données.
RHO_DYNAMIQUE: dict[tuple, float] = {}  # clé = (ligue_id, saison)

# Paramètres Dixon-Coles complets par équipe — estimés conjointement par MLE
# Clé = (ligue_id, saison) → {'gamma': float, 'rho': float,
#                              'teams': {team_id: {'attack': float, 'defense': float}}}
DC_PARAMS: dict[tuple, dict] = {}

async def estimer_rho_saison(ligue_id: int, saison: int, mu_h: float, mu_a: float) -> float | None:
    """
    Estimation MLE de ρ (Dixon-Coles) à partir des scores accumulés dans scores_matchs.

    La correction τ n'affecte que les 4 cellules {0,1}×{0,1}.
    Amélioration vs approche naïve : on utilise les λ réels par match (xG de xg_cache)
    au lieu d'un unique (μ_h, μ_a) ligue. Fallback sur μ_h / μ_a si xG absent.

    Algorithme :
      1. Charger tous les scores + IDs équipes
      2. Batch-fetch des vrais xG depuis xg_cache (1 seule requête IN)
      3. MLE avec τ(d, e, λ_h, λ_a) individuel par match
    """
    async with db_lock:
        async with db_conn.execute(
            "SELECT id_match, buts_dom, buts_ext, team_dom_id, team_ext_id "
            "FROM scores_matchs WHERE ligue_id=? AND saison=?",
            (ligue_id, saison)
        ) as cursor:
            rows = await cursor.fetchall()

    if len(rows) < 30:
        return None

    # Batch-fetch des xG réels pour les matchs à faible score (seuls utiles pour τ)
    # Clé xg_cache = "xg_{fixture_id}_{team_id}" ; xg_p = λ produit, xg_c = λ concédé
    low_rows = [(fid, d, e, did, eid) for fid, d, e, did, eid in rows if d <= 1 and e <= 1]
    xg_lookup: dict[str, tuple[float, float]] = {}

    cles_dom = {f"xg_{r[0]}_{r[3]}" for r in low_rows if r[3]}
    if cles_dom:
        placeholders = ','.join('?' * len(cles_dom))
        async with db_lock:
            async with db_conn.execute(
                f"SELECT cle, xg_p, xg_c FROM xg_cache WHERE cle IN ({placeholders}) AND is_xg=1",
                list(cles_dom)
            ) as cursor:
                for cle, xg_p, xg_c in await cursor.fetchall():
                    # xg_p = λ_home, xg_c = λ_away (perspective de l'équipe domicile)
                    xg_lookup[cle] = (max(0.3, xg_p or mu_h), max(0.3, xg_c or mu_a))

    # Construire les paires (λ_h, λ_a) par match — fallback sur μ ligue si absent
    match_lambdas: list[tuple[int, int, float, float]] = []
    for fid, d, e, dom_id, ext_id in low_rows:
        cle_dom = f"xg_{fid}_{dom_id}" if dom_id else None
        if cle_dom and cle_dom in xg_lookup:
            lh, la = xg_lookup[cle_dom]
        else:
            lh, la = mu_h, mu_a
        match_lambdas.append((d, e, lh, la))

    if not match_lambdas:
        return None

    # MLE : maximise Σ log τ(d, e, λ_h, λ_a, ρ) sur les 4 cellules correctives
    def neg_ll(rho: float) -> float:
        ll = 0.0
        for d, e, lh, la in match_lambdas:
            if   d == 0 and e == 0: tau = max(1e-9, 1.0 - lh * la * rho)
            elif d == 1 and e == 0: tau = max(1e-9, 1.0 + la * rho)
            elif d == 0 and e == 1: tau = max(1e-9, 1.0 + lh * rho)
            else:                   tau = max(1e-9, 1.0 - rho)          # 1-1
            ll += np.log(tau)
        return -ll

    res = minimize_scalar(neg_ll, bounds=(-0.30, -0.01), method='bounded')
    if res.success:
        rho_est = round(res.x, 4)
        n_xg = sum(1 for r in low_rows if f"xg_{r[0]}_{r[3]}" in xg_lookup)
        logging.info(f"[ρ MLE] ligue={ligue_id} saison={saison} "
                     f"n={len(rows)} n_low={len(match_lambdas)} "
                     f"n_xg={n_xg} → ρ={rho_est}")
        return rho_est
    return None


async def estimer_parametres_dc_complet(ligue_id: int, saison: int,
                                         mu_h: float, mu_a: float) -> dict | None:
    """
    Estimation MLE jointe des paramètres Dixon-Coles complets par équipe.

    Modèle multiplicatif (log-espace) :
        λ_home = exp(a_home + d_away + γ)   a = log-attaque, d = log-défense
        λ_away = exp(a_away + d_home)
    Contrainte d'identification (soft penalty) : Σ a_i = 0

    Pondération temporelle (demi-vie 90 jours) + saison précédente à poids 0.5.
    Actif dès 40 matchs avec IDs d'équipes ET ≥ 4 matchs par équipe.
    Résultat stocké dans DC_PARAMS et en base dc_params.
    """
    async with db_lock:
        async with db_conn.execute(
            "SELECT id_match, buts_dom, buts_ext, team_dom_id, team_ext_id, match_date "
            "FROM scores_matchs WHERE ligue_id=? AND saison=?",
            (ligue_id, saison)
        ) as cursor:
            rows_curr = await cursor.fetchall()
        async with db_conn.execute(
            "SELECT id_match, buts_dom, buts_ext, team_dom_id, team_ext_id, match_date "
            "FROM scores_matchs WHERE ligue_id=? AND saison=?",
            (ligue_id, saison - 1)
        ) as cursor:
            rows_prev = await cursor.fetchall()

    valid_curr = [(d, e, h, a, md) for _, d, e, h, a, md in rows_curr if h and a]
    valid_prev = [(d, e, h, a, md) for _, d, e, h, a, md in rows_prev if h and a]

    if len(valid_curr) < 40:
        return None

    # Construire l'index des équipes avec assez de données
    team_counts: dict[int, int] = {}
    for _, _, h, a, _ in valid_curr:
        team_counts[h] = team_counts.get(h, 0) + 1
        team_counts[a] = team_counts.get(a, 0) + 1
    eligible = {t for t, c in team_counts.items() if c >= 4}
    if len(eligible) < 8:
        return None

    teams = sorted(eligible)
    N = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # Pondération temporelle : demi-vie 90 jours
    now_ts = datetime.now(timezone.utc)
    HALF_LIFE_DAYS = 90.0
    decay = np.log(2) / HALF_LIFE_DAYS

    def weight(match_date_str: str | None, base_w: float) -> float:
        if not match_date_str:
            return base_w
        try:
            dt = datetime.fromisoformat(match_date_str.replace('Z', '+00:00'))
            days_ago = max(0, (now_ts - dt).total_seconds() / 86400)
            return base_w * np.exp(-decay * days_ago)
        except Exception:
            return base_w

    all_matches = []
    for d, e, h, a, md in valid_curr:
        if h in idx and a in idx:
            all_matches.append((int(d), int(e), idx[h], idx[a], weight(md, 1.0)))
    for d, e, h, a, md in valid_prev:
        if h in idx and a in idx:
            all_matches.append((int(d), int(e), idx[h], idx[a], weight(md, 0.5)))

    if len(all_matches) < 40:
        return None

    # Cache factorielles
    max_g = max(max(d, e) for d, e, _, _, _ in all_matches)
    log_fact = np.zeros(max_g + 2)
    for k in range(1, max_g + 2):
        log_fact[k] = log_fact[k - 1] + np.log(k)

    def neg_ll(x: np.ndarray) -> float:
        log_a = x[:N]
        log_d = x[N:2 * N]
        log_g = x[2 * N]        # log(home_advantage)
        rho   = float(np.clip(x[2 * N + 1], -0.40, -0.001))

        # Soft identification penalty : Σ log_a = 0
        pen = (float(np.sum(log_a))) ** 2 * 15.0

        ll = 0.0
        for gh, ga, hi, ai, w in all_matches:
            lh = float(np.clip(np.exp(log_a[hi] + log_d[ai] + log_g), 0.1, 12.0))
            la = float(np.clip(np.exp(log_a[ai] + log_d[hi]),          0.1, 12.0))

            ll += w * (gh * np.log(lh) - lh - float(log_fact[gh]))
            ll += w * (ga * np.log(la) - la - float(log_fact[ga]))

            if gh == 0 and ga == 0: tau = max(1e-9, 1.0 - lh * la * rho)
            elif gh == 1 and ga == 0: tau = max(1e-9, 1.0 + la * rho)
            elif gh == 0 and ga == 1: tau = max(1e-9, 1.0 + lh * rho)
            elif gh == 1 and ga == 1: tau = max(1e-9, 1.0 - rho)
            else: tau = 1.0
            ll += w * np.log(tau)

        return -(ll - pen)

    x0 = np.zeros(2 * N + 2)
    x0[2 * N]     = np.log(max(0.5, mu_h))   # home advantage ≈ mu_h
    x0[2 * N + 1] = -0.13                     # rho initial typique

    bounds = ([(-2.5, 2.5)] * N +   # log_attack
              [(-2.5, 2.5)] * N +   # log_defense
              [(-0.5, 0.5)]  +      # log_home_adv
              [(-0.40, -0.001)])     # rho

    try:
        res = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 3000, 'ftol': 1e-9})
    except Exception as e:
        logging.warning(f"[DC MLE] ligue={ligue_id} saison={saison} erreur scipy : {e}")
        return None

    x = res.x
    log_a = x[:N]
    log_d = x[N:2 * N]
    gamma = float(np.exp(x[2 * N]))
    rho   = float(np.clip(x[2 * N + 1], -0.40, -0.001))

    # Normalisation : centrer log_d pour que mean(defense) = 1 (log = 0)
    # et compenser dans gamma → préserve λ_home_avg = gamma
    mean_log_d = float(np.mean(log_d))
    log_d -= mean_log_d
    gamma *= np.exp(mean_log_d)

    # Calibration absolue : ajuster gamma pour que λ_home_moyen ≈ mu_h
    # Pour une équipe moyenne : λ_home = exp(0 + 0 + log_gamma) = gamma → gamma ← mu_h
    if gamma > 0:
        scale = mu_h / gamma
        log_a += np.log(max(scale, 1e-6))
        gamma *= scale

    params: dict = {
        'gamma': gamma,
        'rho': rho,
        'teams': {
            team_id: {
                'attack':  float(np.exp(log_a[i])),
                'defense': float(np.exp(log_d[i]))
            }
            for i, team_id in enumerate(teams)
        }
    }

    # Persistance en base
    computed_at = datetime.now().isoformat()
    async with db_lock:
        await db_conn.executemany(
            "INSERT OR REPLACE INTO dc_params "
            "(ligue_id, saison, team_id, attack, defense, home_adv, rho, computed_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [(ligue_id, saison, tid,
              params['teams'][tid]['attack'], params['teams'][tid]['defense'],
              gamma, rho, computed_at)
             for tid in params['teams']]
        )
        await db_conn.commit()

    logging.info(f"[DC MLE] ligue={ligue_id} saison={saison} N={N} "
                 f"matchs={len([m for m in all_matches if m[4] > 0.4])} "
                 f"γ={gamma:.3f} ρ={rho:.4f}")
    return params


NAME_MAPPING = {
    # ============================================================
    # 🇫🇷 LIGUE 1 — API-Football utilise les noms officiels complets
    # ============================================================
    "Paris Saint-Germain":      "Paris Saint Germain",
    "Olympique Lyonnais":       "Lyon",
    "Olympique de Marseille":   "Marseille",
    "Stade Rennais FC":         "Rennes",
    "Stade Rennais":            "Rennes",
    "Stade de Reims":           "Reims",
    "AS Monaco":                "Monaco",
    "OGC Nice":                 "Nice",
    "RC Lens":                  "Lens",
    "Lille OSC":                "Lille",
    "FC Nantes":                "Nantes",
    "RC Strasbourg Alsace":     "Strasbourg",
    "RC Strasbourg":            "Strasbourg",
    "Montpellier HSC":          "Montpellier",
    "Stade Brestois 29":        "Brest",
    "FC Lorient":               "Lorient",
    "AJ Auxerre":               "Auxerre",
    "AC Ajaccio":               "Ajaccio",
    "Toulouse FC":              "Toulouse",
    "Le Havre AC":              "Le Havre",
    "Girondins de Bordeaux":    "Bordeaux",
    "Angers SCO":               "Angers",
    "AS Saint-Étienne":         "Saint-Etienne",
    "AS Saint-Etienne":         "Saint-Etienne",

    # ============================================================
    # 🇪🇸 LA LIGA — Noms officiels espagnols vs noms anglais Odds API
    # ============================================================
    "Athletic Club":            "Athletic Bilbao",
    "Atlético Madrid":          "Atletico Madrid",
    "Atletico Madrid":          "Atletico Madrid",
    "Deportivo Alavés":         "Alaves",
    "Cádiz CF":                 "Cadiz",
    "Granada CF":               "Granada",
    "UD Almería":               "Almeria",
    "RC Celta":                 "Celta Vigo",
    "RCD Espanyol":             "Espanyol",
    "RCD Mallorca":             "Mallorca",
    "Getafe CF":                "Getafe",
    "CA Osasuna":               "Osasuna",
    "Real Betis Balompié":      "Real Betis",
    "Sevilla FC":               "Sevilla",
    "Valencia CF":              "Valencia",
    "Real Valladolid":          "Valladolid",
    "Girona FC":                "Girona",
    "UD Las Palmas":            "Las Palmas",
    "CD Leganés":               "Leganes",
    "Real Sociedad":            "Real Sociedad",
    "Real Sociedad B":          "Real Sociedad II",
    "Villarreal CF":            "Villarreal",
    "Athletic Club de Bilbao":  "Athletic Bilbao",

    # ============================================================
    # 🇩🇪 BUNDESLIGA — Préfixes numérotés et umlauts
    # ============================================================
    "FC Bayern München":        "Bayern Munich",
    "Bayern München":           "Bayern Munich",
    "1. FC Köln":               "FC Koeln",
    "Borussia Mönchengladbach": "Borussia Monchengladbach",
    "TSG Hoffenheim":           "Hoffenheim",
    "TSG 1899 Hoffenheim":      "Hoffenheim",
    "SC Freiburg":              "Freiburg",
    "Sport-Club Freiburg":      "Freiburg",
    "VfB Stuttgart":            "Stuttgart",
    "1. FSV Mainz 05":          "Mainz",
    "FSV Mainz 05":             "Mainz",
    "FC Augsburg":              "Augsburg",
    "SV Werder Bremen":         "Werder Bremen",
    "VfL Wolfsburg":            "Wolfsburg",
    "VfL Bochum 1848":          "Bochum",
    "Hertha BSC":               "Hertha Berlin",
    "1. FC Union Berlin":       "Union Berlin",
    "FC Union Berlin":          "Union Berlin",
    "1. FC Heidenheim 1846":    "Heidenheim",
    "1. FC Heidenheim":         "Heidenheim",
    "FC St. Pauli":             "St. Pauli",
    "SV Darmstadt 98":          "Darmstadt 98",
    "Holstein Kiel":            "Holstein Kiel",

    # ============================================================
    # 🇮🇹 SERIE A & SERIE B — Noms officiels italiens
    # ============================================================
    "Inter":                    "Inter Milan",
    "AC Milan":                 "AC Milan",
    "AS Roma":                  "Roma",
    "SS Lazio":                 "Lazio",
    "ACF Fiorentina":           "Fiorentina",
    "Atalanta BC":              "Atalanta",
    "Hellas Verona":            "Verona",
    "Torino FC":                "Torino",
    "Bologna FC 1909":          "Bologna",
    "Genoa CFC":                "Genoa",
    "US Sassuolo Calcio":       "Sassuolo",
    "Udinese Calcio":           "Udinese",
    "Cagliari Calcio":          "Cagliari",
    "Empoli FC":                "Empoli",
    "US Lecce":                 "Lecce",
    "Parma Calcio 1913":        "Parma",
    "Como 1907":                "Como",
    "Venezia FC":               "Venezia",
    "US Salernitana 1919":      "Salernitana",
    "US Salernitana":           "Salernitana",
    "Frosinone Calcio":         "Frosinone",
    "US Cremonese":             "Cremonese",
    "AC Pisa 1909":             "Pisa",
    "Brescia Calcio":           "Brescia",
    "Spezia Calcio":            "Spezia",
    "SSC Bari":                 "Bari",
    "UC Sampdoria":             "Sampdoria",
    "Modena FC 2018":           "Modena",
    "Modena FC":                "Modena",

    # ============================================================
    # 🏴󠁧󠁢󠁥󠁮󠁧󠁿 PREMIER LEAGUE & CHAMPIONSHIP
    # ============================================================
    "Manchester United":        "Manchester United",
    "Manchester City":          "Manchester City",
    "Tottenham Hotspur":        "Tottenham Hotspur",
    "Wolverhampton Wanderers":  "Wolverhampton",
    "Nottingham Forest":        "Nottingham Forest",
    "Nottm Forest":             "Nottingham Forest",
    "Brighton & Hove Albion":   "Brighton",
    "West Ham United":          "West Ham",
    "Newcastle United":         "Newcastle",
    "Sheffield United":         "Sheffield United",
    "Sheff Utd":                "Sheffield United",
    "Sheffield Wednesday":      "Sheffield Wednesday",
    "Sheff Wed":                "Sheffield Wednesday",
    "Queens Park Rangers":      "QPR",
    "Leeds United":             "Leeds",
    "West Bromwich Albion":     "West Brom",
    "Swansea City":             "Swansea",
    "Luton Town":               "Luton",
    "Hull City":                "Hull",
    "Middlesbrough":            "Middlesbrough",
    "Coventry City":            "Coventry",
    "Sunderland":               "Sunderland",
    "Plymouth Argyle":          "Plymouth",
    "Bristol City":             "Bristol City",
    "Watford":                  "Watford",
    "Norwich City":             "Norwich",
    "Stoke City":               "Stoke",
    "Blackburn Rovers":         "Blackburn",
    "Preston North End":        "Preston",
    "Burnley FC":               "Burnley",

    # ============================================================
    # 🇳🇱 EREDIVISIE
    # ============================================================
    "AFC Ajax":                 "Ajax",
    "PSV Eindhoven":            "PSV",
    "AZ Alkmaar":               "AZ",
    "FC Utrecht":               "Utrecht",
    "FC Twente":                "Twente",
    "SC Heerenveen":            "Heerenveen",
    "PEC Zwolle":               "PEC Zwolle",
    "Go Ahead Eagles":          "Go Ahead Eagles",
    "FC Emmen":                 "Emmen",
    "Almere City FC":           "Almere City",
    "FC Groningen":             "Groningen",
    "Sparta Rotterdam":         "Sparta Rotterdam",
    "RKC Waalwijk":             "RKC Waalwijk",
    "NEC Nijmegen":             "NEC",
    "NEC":                      "NEC Nijmegen",

    # ============================================================
    # 🇵🇹 PRIMEIRA LIGA
    # ============================================================
    "SL Benfica":               "Benfica",
    "FC Porto":                 "Porto",
    "Sporting CP":              "Sporting CP",
    "SC Braga":                 "Braga",
    "Vitória SC":               "Vitoria Guimaraes",
    "Vitoria SC":               "Vitoria Guimaraes",
    "Gil Vicente FC":           "Gil Vicente",
    "Boavista FC":              "Boavista",
    "CD Santa Clara":           "Santa Clara",
    "Moreirense FC":            "Moreirense",
    "GD Estoril Praia":         "Estoril",
    "GD Chaves":                "Chaves",
    "Casa Pia AC":              "Casa Pia",
    "FC Famalicao":             "Famalicao",
    "CD Famalicão":             "Famalicao",
    "Rio Ave FC":               "Rio Ave",
    "SC Farense":               "Farense",
    "CF Arouca":                "Arouca",
    "CD Nacional":              "Nacional",

    # ============================================================
    # 🇹🇷 SÜPER LIG
    # ============================================================
    "Galatasaray SK":           "Galatasaray",
    "Fenerbahçe SK":            "Fenerbahce",
    "Fenerbahce SK":            "Fenerbahce",
    "Beşiktaş JK":              "Besiktas",
    "Besiktas JK":              "Besiktas",
    "Kasımpaşa SK":             "Kasimpasa",
    "Kasimpasa SK":             "Kasimpasa",
    "İstanbul Başakşehir FK":   "Basaksehir",
    "Istanbul Basaksehir FK":   "Basaksehir",
    "Göztepe SK":               "Goztepe",
    "Goztepe SK":               "Goztepe",
    "Yılport Samsunspor":       "Samsunspor",
    "Fatih Karagümrük SK":      "Karagumruk",
    "Fatih Karagumruk":         "Karagumruk",
    "Sivasspor":                "Sivasspor",
    "Alanyaspor":               "Alanyaspor",
    "Konyaspor":                "Konyaspor",
    "Antalyaspor":              "Antalyaspor",
    "Kayserispor":              "Kayserispor",
    "Gaziantep FK":             "Gaziantep",
    "Rizespor":                 "Rizespor",
    "Adana Demirspor":          "Adana Demirspor",
    "MKE Ankaragücü":           "Ankaragucu",

    # ============================================================
    # 🇧🇷 SÉRIE A BRÉSIL
    # ============================================================
    "Athletico Paranaense":     "Athletico-PR",
    "Atlético Paranaense":      "Athletico-PR",
    "Atlético-MG":              "Atletico Mineiro",
    "Atlético Mineiro":         "Atletico Mineiro",
    "Bragantino":               "Red Bull Bragantino",
    "Red Bull Bragantino":      "Red Bull Bragantino",
    "Grêmio":                   "Gremio",
    "Gremio":                   "Gremio",
    "São Paulo FC":             "Sao Paulo",
    "Sao Paulo":                "Sao Paulo",
    "Sport Club Corinthians Paulista": "Corinthians",
    "Sociedade Esportiva Palmeiras": "Palmeiras",
    "Club de Regatas do Flamengo": "Flamengo",
    "Fluminense FC":            "Fluminense",
    "Botafogo FR":              "Botafogo",
    "CR Vasco da Gama":         "Vasco da Gama",
    "EC Bahia":                 "Bahia",
    "Fortaleza EC":             "Fortaleza",
    "Sport Club Internacional": "Internacional",
    "Cruzeiro EC":              "Cruzeiro",
    "América Mineiro":          "America Mineiro",
    "Cuiabá EC":                "Cuiaba",
    "Goiás EC":                 "Goias",
    "Atlético Goianiense":      "Atletico Goianiense",
    "EC Juventude":             "Juventude",
    "Criciúma EC":              "Criciuma",
    "Sport Recife":             "Sport Recife",
    "Coritiba FC":              "Coritiba",
    "Ceará SC":                 "Ceara",
    "Santos FC":                "Santos",

    # ============================================================
    # 🇸🇪 ALLSVENSKAN
    # ============================================================
    "AIK Fotboll":              "AIK",
    "Malmö FF":                 "Malmo FF",
    "Djurgårdens IF":           "Djurgarden",
    "IFK Göteborg":             "IFK Goteborg",
    "BK Häcken":                "BK Hacken",
    "IFK Norrköping":           "IFK Norrkoping",
    "Mjällby AIF":              "Mjallby",
    "Halmstads BK":             "Halmstad",
    "IK Sirius FK":             "Sirius",
    "Varbergs BoIS FC":         "Varbergs BoIS",
    "GIF Sundsvall":            "Sundsvall",
    "Degerfors IF":             "Degerfors",
    "GAIS":                     "GAIS Goteborg",
    "Brommapojkarna":           "Brommapojkarna",
    "Kalmar FF":                "Kalmar",

    # ============================================================
    # 🇳🇴 ELITESERIEN
    # ============================================================
    "FK Bodø/Glimt":            "Bodo/Glimt",
    "Bodø/Glimt":               "Bodo/Glimt",
    "Bodo/Glimt":               "Bodø/Glimt",
    "Molde FK":                 "Molde",
    "Rosenborg BK":             "Rosenborg",
    "SK Brann":                 "Brann",
    "Viking FK":                "Viking",
    "Tromsø IL":                "Tromso",
    "IL Tromso":                "Tromso",
    "Stabæk Fotball":           "Stabek",
    "Stabek IF":                "Stabek",
    "Strømsgodset IF":          "Stromsgodset",
    "FK Haugesund":             "Haugesund",
    "Odd BK":                   "Odd",
    "Sandefjord Fotball":       "Sandefjord",
    "Aalesunds FK":             "Aalesund",
    "FK Jerv":                  "Jerv",
    "Lillestrøm SK":            "Lillestrom",

    # ============================================================
    # 🇧🇪 JUPILER PRO LEAGUE
    # ============================================================
    "RSC Anderlecht":           "Anderlecht",
    "Club Brugge KV":           "Club Brugge",
    "KAA Gent":                 "Gent",
    "Standard Liège":           "Standard Liege",
    "Standard de Liège":        "Standard Liege",
    "KRC Genk":                 "Genk",
    "Royal Antwerp FC":         "Antwerp",
    "Royale Union Saint-Gilloise": "Union Saint Gilloise",
    "R. Charleroi SC":          "Charleroi",
    "Sporting Charleroi":       "Charleroi",
    "Cercle Brugge KSV":        "Cercle Brugge",
    "Sint-Truidense VV":        "Sint-Truiden",
    "KV Mechelen":              "Mechelen",
    "KV Kortrijk":              "Kortrijk",
    "K. Beerschot VA":          "Beerschot",
    "Beerschot VA":             "Beerschot",
    "KAS Eupen":                "Eupen",
    "OH Leuven":                "OHL Leuven",
    "Westerlo":                 "Westerlo",
    "RWDM Brussels FC":         "RWDM",

    # ============================================================
    # 🇪🇸 LALIGA 2
    # ============================================================
    "SD Huesca":                "Huesca",
    "Real Oviedo":              "Oviedo",
    "Sporting Gijón":           "Sporting Gijon",
    "Sporting de Gijón":        "Sporting Gijon",
    "Real Zaragoza":            "Zaragoza",
    "SD Eibar":                 "Eibar",
    "Málaga CF":                "Malaga",
    "Racing Club de Santander": "Racing Santander",
    "Burgos CF":                "Burgos",
    "Elche CF":                 "Elche",
    "Levante UD":               "Levante",
    "UD Almería":               "Almeria",
    "Albacete Balompié":        "Albacete",
    "FC Cartagena":             "Cartagena",
    "CD Eldense":               "Eldense",
    "Córdoba CF":               "Cordoba",
    "CD Tenerife":              "Tenerife",

    # ============================================================
    # 🇺🇸 MLS
    # ============================================================
    "Inter Miami CF":           "Inter Miami",
    "LA Galaxy":                "Los Angeles Galaxy",
    "LAFC":                     "Los Angeles FC",
    "New York Red Bulls":       "NY Red Bulls",
    "New York City FC":         "New York City FC",
    "NYCFC":                    "New York City FC",
    "Seattle Sounders FC":      "Seattle Sounders",
    "Portland Timbers":         "Portland Timbers",
    "Atlanta United FC":        "Atlanta United",
    "Columbus Crew":            "Columbus Crew",
    "FC Cincinnati":            "FC Cincinnati",
    "Orlando City SC":          "Orlando City",
    "FC Dallas":                "FC Dallas",
    "Toronto FC":               "Toronto FC",
    "CF Montréal":              "CF Montreal",
    "New England Revolution":   "New England Revolution",
    "Philadelphia Union":       "Philadelphia Union",
    "D.C. United":              "DC United",
    "Colorado Rapids":          "Colorado Rapids",
    "Houston Dynamo FC":        "Houston Dynamo",
    "Minnesota United FC":      "Minnesota United",
    "Austin FC":                "Austin FC",
    "Nashville SC":             "Nashville SC",
    "Charlotte FC":             "Charlotte FC",
    "St. Louis City SC":        "St. Louis City",
    "San Jose Earthquakes":     "San Jose Earthquakes",
    "Vancouver Whitecaps FC":   "Vancouver Whitecaps",
    "Real Salt Lake":           "Real Salt Lake",
    "Sporting Kansas City":     "Sporting Kansas City",
    "Chicago Fire FC":          "Chicago Fire",
    "Los Angeles FC":           "Los Angeles FC",
}

STADES_GPS = {
    # --- 🇫🇷 LIGUE 1 (ID: 61) ---
    85: (48.8414, 2.2530),   # PSG (Parc des Princes)
    81: (43.2698, 5.3959),   # Marseille (Vélodrome)
    80: (45.7653, 4.9819),   # Lyon (Groupama Stadium)
    79: (48.1147, -1.7034),  # Rennes (Roazhon Park)
    82: (43.7057, 7.2458),   # Nice (Allianz Riviera)
    84: (44.8285, -0.5616),  # Bordeaux (Matmut Atlantique)
    91: (43.7276, 7.4155),   # Monaco (Stade Louis II)
    94: (47.7485, -3.3695),  # Lorient (Stade du Moustoir)
    106: (48.4103, -4.4925), # Brest (Stade Francis-Le Blé)
    97: (50.6119, 3.1305),   # Lille (Stade Pierre-Mauroy)
    108: (50.4328, 2.8149),  # Lens (Stade Bollaert-Delelis)
    98: (49.2467, 4.0251),   # Reims (Stade Auguste-Delaune)
    110: (43.5833, 1.4342),  # Toulouse (Stadium de Toulouse)
    96: (47.2556, -1.5251),  # Nantes (Stade de la Beaujoire)
    116: (48.5844, 7.7486),  # Strasbourg (Stade de la Meinau)
    105: (47.7924, 3.5852),  # Auxerre (Stade de l'Abbé-Deschamps)
    77: (47.4725, -0.5512),  # Angers (Stade Raymond-Kopa)
    112: (45.4607, 4.3901),  # St Etienne (Stade Geoffroy-Guichard)
    111: (49.4938, 0.1077),  # Le Havre (Stade Océane)

    # --- 🇪🇸 LA LIGA (ID: 140) ---
    541: (40.4531, -3.6883), # Real Madrid (Santiago Bernabéu)
    529: (41.3809, 2.1228),  # FC Barcelona (Camp Nou)
    530: (40.4362, -3.5995), # Atletico Madrid (Metropolitano)
    546: (37.3840, -5.9705), # Real Betis (Benito Villamarín)
    536: (37.3841, -5.9902), # Sevilla (Sánchez-Pizjuán)
    532: (39.4746, -0.3582), # Valencia (Mestalla)
    531: (43.2635, -2.9483), # Athletic Bilbao (San Mamés)
    543: (43.3014, -1.9736), # Real Sociedad (Anoeta)
    533: (42.2318, -8.7126), # Celta Vigo (Balaídos)
    538: (40.4017, -3.7167), # Getafe (Coliseum Alfonso Pérez)
    540: (41.3478, 2.0755),  # Espanyol (RCDE Stadium)
    537: (42.8222, -1.6372), # Osasuna (El Sadar)
    728: (36.5026, -6.2731), # Cadiz (Nuevo Mirandilla)
    727: (39.5471, 2.6301),  # Mallorca (Visit Mallorca Estadi)
    534: (39.9441, -0.1032), # Villarreal (Estadio de la Cerámica)
    548: (41.6445, -4.7612), # Valladolid (José Zorrilla)
    547: (41.9614, 2.8286),  # Girona (Montilivi)
    730: (42.8371, -2.6882), # Alaves (Mendizorroza)
    554: (36.7341, -4.4265), # Malaga (La Rosaleda)
    551: (38.3572, -0.4912), # Hercules / Elche (Martínez Valero)

    # --- 🇮🇹 SERIE A (ID: 135) ---
    489: (45.4781, 9.1240),  # AC Milan (San Siro)
    497: (45.4781, 9.1240),  # Inter Milan (San Siro)
    496: (45.0901, 7.6413),  # Juventus (Allianz Stadium)
    492: (40.8279, 14.2495), # Napoli (Diego Armando Maradona)
    495: (41.9339, 12.4547), # AS Roma (Stadio Olimpico)
    487: (41.9339, 12.4547), # Lazio (Stadio Olimpico)
    502: (43.7766, 11.2822), # Fiorentina (Artemio Franchi)
    503: (45.6807, 9.6301),  # Atalanta (Gewiss Stadium)
    499: (45.4354, 10.9686), # Hellas Verona (Marc'Antonio Bentegodi)
    494: (45.0519, 7.6994),  # Torino (Stadio Olimpico Grande Torino)
    488: (44.4921, 11.3090), # Bologna (Renato Dall'Ara)
    505: (44.4423, 8.9281),  # Genoa (Luigi Ferraris)
    500: (44.6675, 10.6501), # Sassuolo (Mapei Stadium)
    504: (46.0815, 13.2007), # Udinese (Dacia Arena)
    498: (43.7198, 10.9421), # Empoli (Carlo Castellani)
    493: (40.3541, 18.2014), # Lecce (Via del Mare)
    491: (39.2001, 9.1264),  # Cagliari (Unipol Domus)
    514: (44.8015, 10.3385), # Parma (Ennio Tardini)
    515: (45.8114, 9.0717),  # Como (Stadio Giuseppe Sinigaglia)
    481: (45.4517, 12.3271), # Venezia (Pier Luigi Penzo)

    # --- 🇩🇪 BUNDESLIGA (ID: 78) ---
    157: (48.2188, 11.6247), # Bayern Munich (Allianz Arena)
    165: (51.4926, 7.4519),  # Dortmund (Signal Iduna Park)
    168: (51.0382, 7.0022),  # Leverkusen (BayArena)
    161: (52.4319, 10.8039), # Wolfsburg (Volkswagen Arena)
    169: (50.0686, 8.6455),  # Eintracht Frankfurt (Deutsche Bank Park)
    173: (51.3458, 12.3483), # RB Leipzig (Red Bull Arena)
    160: (48.7922, 9.2320),  # Stuttgart (MHPArena)
    163: (51.2179, 6.7328),  # M'gladbach (Borussia-Park)
    162: (53.0664, 8.8376),  # Werder Bremen (Weserstadion)
    167: (49.2394, 8.8875),  # Hoffenheim (PreZero Arena)
    170: (47.9889, 7.8932),  # Freiburg (Europa-Park Stadion)
    174: (49.9838, 8.2244),  # Mainz (Mewa Arena)
    172: (48.3328, 10.8861), # Augsburg (WWK Arena)
    176: (51.4889, 7.2361),  # Bochum (Vonovia Ruhrstadion)
    191: (52.4572, 13.5681), # Union Berlin (An der Alten Försterei)
    188: (48.6896, 10.1394), # Heidenheim (Voith-Arena)
    182: (53.5545, 9.9678),  # St. Pauli (Millerntor-Stadion)
    179: (54.3486, 10.1239), # Holstein Kiel (Holstein-Stadion)

    # --- 🇳🇱 EREDIVISIE (ID: 88) ---
    194: (52.3144, 4.9427),  # Ajax (Johan Cruyff Arena)
    197: (51.4417, 5.4674),  # PSV Eindhoven (Philips Stadion)
    209: (51.8939, 4.5231),  # Feyenoord (De Kuip)
    201: (52.6124, 4.7423),  # AZ Alkmaar (AFAS Stadion)
    198: (52.2366, 6.8375),  # FC Twente (De Grolsch Veste)
    202: (51.1928, 5.9812),  # Vitesse Arnhem (GelreDome)
    415: (52.5133, 6.0747),  # PEC Zwolle (MAC³PARK Stadion)
    424: (52.2037, 5.9208),  # Go Ahead Eagles (De Adelaarshorst)
    210: (52.9585, 5.9361),  # Heerenveen (Abe Lenstra Stadion)
    204: (51.6914, 5.2444),  # Willem II (Koning Willem II Stadion)
    670: (51.4172, 5.4411),  # NAC Breda (Rat Verlegh Stadion)

    # --- 🇵🇹 PRIMEIRA LIGA (ID: 94) ---
    211: (38.7527, -9.1847), # Benfica (Estádio da Luz)
    212: (41.1717, -8.5839), # FC Porto (Estádio do Dragão)
    228: (38.7612, -9.1617), # Sporting CP (Alvalade)
    214: (41.5614, -8.4312), # SC Braga (Estádio Municipal de Braga)
    217: (41.4531, -8.3175), # Vitoria Guimaraes (Dom Afonso Henriques)
    218: (41.3414, -8.4117), # Gil Vicente (Estádio Cidade de Barcelos)
    224: (41.1619, -8.6425), # Boavista (Estádio do Bessa)
    712: (39.2435, -8.6874), # Santa Clara (Estádio de São Miguel)
    221: (41.6939, -8.8471), # Vitoria Setubal / Moreirense

    # --- 🇹🇷 SÜPER LIG (ID: 203) ---
    610: (41.1034, 28.9910), # Galatasaray (Rams Park)
    611: (40.9877, 29.0370), # Fenerbahce (Şükrü Saracoğlu)
    607: (41.0394, 29.0020), # Besiktas (Tüpraş Stadyumu)
    549: (40.9515, 39.6300), # Trabzonspor (Papara Park)
    605: (39.9531, 32.8101), # Ankaragucu (Eryaman Stadyumu)
    596: (41.0116, 28.8184), # Kasimpasa (Recep Tayyip Erdoğan)
    618: (37.0583, 37.3800), # Gaziantep FK (Gaziantep Stadyumu)
    1001: (38.3375, 27.1350),# Goztepe (Gürsel Aksel Stadyumu)
    600: (41.2291, 36.4628), # Samsunspor (Samsun 19 Mayıs)

    # --- 🇧🇷 SÉRIE A (ID: 71) ---
    127: (-22.9121, -43.2302), # Flamengo (Maracanã)
    126: (-23.5489, -46.6361), # São Paulo (Morumbi)
    121: (-23.5275, -46.6783), # Palmeiras (Allianz Parque)
    131: (-22.8932, -43.2922), # Botafogo (Nilton Santos)
    118: (-23.5453, -46.4742), # Corinthians (Neo Química Arena)
    135: (-30.0655, -51.2359), # Internacional (Beira-Rio)
    130: (-30.0277, -51.1939), # Gremio (Arena do Grêmio)
    120: (-19.9231, -43.9458), # Atletico Mineiro (Arena MRV)
    124: (-22.9121, -43.2302), # Fluminense (Maracanã)
    128: (-25.4167, -49.2667), # Athletico-PR (Ligga Arena)
    134: (-12.9711, -38.5108), # Bahia (Arena Fonte Nova)
    122: (-15.8201, -47.8981), # Bragantino / Cruzeiro
    119: (-19.8658, -43.9711), # Cruzeiro (Mineirão)
    133: (-3.8067, -38.5217), # Fortaleza (Castelão)
    147: (-16.6971, -49.2562), # Atletico Goianiense

    # --- 🇸🇪 ALLSVENSKAN (ID: 113) ---
    342: (59.3728, 18.0000),  # AIK Solna (Strawberry Arena)
    343: (55.5849, 12.9884),  # Malmö FF (Eleda Stadion)
    344: (57.7351, 12.9348),  # IF Elfsborg (Borås Arena)
    345: (59.2911, 18.0839),  # Hammarby IF (Tele2 Arena)
    346: (59.2911, 18.0839),  # Djurgårdens IF (Tele2 Arena)
    347: (57.7061, 11.9803),  # IFK Göteborg (Gamla Ullevi)
    348: (59.6247, 16.5401),  # Västerås SK (Hitachi Energy Arena)
    349: (59.2274, 14.4447),  # Degerfors IF (Stora Valla)
    350: (58.5848, 16.1706),  # IFK Norrköping (PlatinumCars Arena)
    351: (57.7212, 11.9345),  # BK Häcken (Bravida Arena)
    352: (56.6784, 16.3475),  # Kalmar FF (Guldfågeln Arena)
    353: (59.8492, 17.6456),  # IK Sirius (Studenternas IP)
    354: (59.3622, 17.8681),  # Brommapojkarna (Grimsta IP)
    355: (57.7061, 11.9803),  # GAIS (Gamla Ullevi)
    356: (56.6853, 12.8664),  # Halmstads BK (Örjans Vall)
    357: (56.0223, 14.6974),  # Mjällby AIF (Strandvallen)

    # --- 🇪🇸 LALIGA 2 (ID: 141) ---
    535: (43.4628, -3.8047), # Racing Santander (El Sardinero)
    542: (43.5361, -5.6372), # Sporting Gijon (El Molinón)
    545: (43.3653, -5.8612), # Real Oviedo (Carlos Tartiere)
    544: (41.6366, -0.9018), # Zaragoza (La Romareda)
    539: (38.3572, -0.4912), # Elche (Martínez Valero)
    720: (39.4444, -0.3501), # Levante (Ciutat de València)

    # ==========================================
    # 🏴󠁧󠁢󠁥󠁮󠁧󠁿 PREMIER LEAGUE (Saison Complète)
    # ==========================================
    42: (51.5549, -0.1084),   # Arsenal (Emirates Stadium)
    66: (52.4827, -1.8847),   # Aston Villa (Villa Park)
    35: (50.7352, -1.8383),   # Bournemouth (Vitality Stadium)
    55: (51.4906, -0.2890),   # Brentford (Gtech Community)
    51: (50.8616, -0.0837),   # Brighton (Amex Stadium)
    49: (51.4817, -0.1910),   # Chelsea (Stamford Bridge)
    52: (51.3983, -0.0856),   # Crystal Palace (Selhurst Park)
    45: (53.4389, -2.9664),   # Everton (Goodison Park)
    43: (51.4750, -0.2217),   # Fulham (Craven Cottage)
    62: (52.0530, 1.1446),    # Ipswich Town (Portman Road)
    46: (52.6204, -1.1422),   # Leicester City (King Power)
    40: (53.4308, -2.9608),   # Liverpool (Anfield)
    50: (53.4831, -2.2004),   # Manchester City (Etihad)
    33: (53.4631, -2.2913),   # Manchester United (Old Trafford)
    34: (54.9756, -1.6217),   # Newcastle United (St. James' Park)
    65: (52.9400, -1.1328),   # Nottingham Forest (City Ground)
    41: (50.9058, -1.3911),   # Southampton (St Mary's)
    47: (51.6044, -0.0664),   # Tottenham Hotspur (Spurs Stadium)
    48: (51.5386, -0.0166),   # West Ham United (London Stadium)
    39: (52.5902, -2.1304),   # Wolverhampton (Molineux)

    # ==========================================
    # 🏴󠁧󠁢󠁥󠁮󠁧󠁿 CHAMPIONSHIP (Les principaux cadors)
    # ==========================================
    63: (53.7778, -1.5722),   # Leeds United (Elland Road)
    44: (53.7890, -2.2302),   # Burnley (Turf Moor)
    70: (53.3703, -1.4708),   # Sheffield United (Bramall Lane)
    1359: (51.8842, -0.4317), # Luton Town (Kenilworth Road)
    60: (52.5090, -1.9639),   # West Bromwich (The Hawthorns)
    71: (52.6221, 1.3086),    # Norwich City (Carrow Road)
    67: (53.7461, -0.3678),   # Hull City (MKM Stadium)
    746: (54.9144, -1.3883),  # Sunderland (Stadium of Light)
    68: (54.5783, -1.2169),   # Middlesbrough (Riverside Stadium)
    74: (52.4481, -1.4956),   # Coventry City (CBS Arena)
    58: (51.5093, -0.2322),   # QPR (Loftus Road)
    72: (51.6428, -3.9351),   # Swansea City (Swansea.com)
    64: (51.4816, -0.0507),   # Millwall (The Den)

    # ==========================================
    # 🇺🇸 MAJOR LEAGUE SOCCER (Franchises Majeures)
    # ==========================================
    9568: (26.1931, -80.1611), # Inter Miami (Chase Stadium)
    1602: (33.8644, -118.2611),# LA Galaxy (Dignity Health)
    1603: (34.0128, -118.2847),# LAFC (BMO Stadium)
    1615: (40.7367, -74.1503), # NY Red Bulls (Red Bull Arena)
    1614: (40.8296, -73.9262), # NYCFC (Yankee Stadium)
    1595: (47.5952, -122.3316),# Seattle Sounders (Lumen Field)
    1611: (45.5214, -122.6917),# Portland Timbers (Providence Park)
    1604: (33.7550, -84.4008), # Atlanta United (Mercedes-Benz)
    1608: (39.9680, -83.0173), # Columbus Crew (Lower.com)
    9569: (39.1114, -84.5228), # FC Cincinnati (TQL Stadium)
    1616: (28.5411, -81.3893), # Orlando City (Exploria Stadium)
    1598: (33.1544, -96.8353), # FC Dallas (Toyota Stadium)

    # ==========================================
    # 🇳🇴 ELITESERIEN (Les clubs du Grand Nord)
    # ==========================================
    268: (67.2778, 14.3917),  # Bodø/Glimt (Aspmyra - Cercle Polaire !)
    273: (62.7333, 7.1236),   # Molde (Aker Stadion)
    278: (63.4128, 10.4103),  # Rosenborg (Lerkendal Stadion)
    269: (60.3669, 5.3300),   # Brann Bergen (Brann Stadion)
    282: (58.9514, 5.7306),   # Viking (SR-Bank Arena)
    281: (69.6416, 18.9400),  # Tromso (Alfheim Stadion)
    274: (59.1969, 9.6108),   # Odd (Skagerak Arena)
    279: (59.1364, 10.1878),  # Sandefjord (Release Arena)

    # ==========================================
    # 🇧🇪 JUPILER PRO LEAGUE (Les clubs belges)
    # ==========================================
    569: (50.8344, 4.3006),   # Anderlecht (Lotto Park)
    561: (51.2117, 3.1931),   # Club Brugge (Jan Breydel)
    573: (51.0064, 3.7375),   # KAA Gent (Ghelamco Arena)
    564: (50.6094, 5.5444),   # Standard Liège (Sclessin)
    565: (51.0044, 5.5333),   # KRC Genk (Cegeka Arena)
    3393: (50.8178, 4.3267),  # Union SG (Stade Joseph Marien)
    558: (50.4161, 4.4533),   # Charleroi (Stade du Pays de Charleroi)

    # ==========================================
    # 🇮🇹 SERIE B (La bataille pour la montée)
    # ==========================================
    524: (38.1528, 13.3400),  # Palermo (Stadio Renzo Barbera)
    508: (44.4281, 8.9525),   # Sampdoria (Luigi Ferraris)
    512: (40.6711, 14.7872),  # Salernitana (Stadio Arechi)
    511: (41.6339, 13.3331),  # Frosinone (Stadio Benito Stirpe)
    520: (45.1328, 10.0356),  # Cremonese (Stadio Giovanni Zini)
    518: (43.7198, 10.4067),  # Pisa (Arena Garibaldi)
    517: (45.5683, 10.2311),  # Brescia (Stadio Mario Rigamonti)
    513: (44.1039, 9.8081),   # Spezia (Stadio Alberto Picco)
    509: (41.0847, 16.8400),  # Bari (Stadio San Nicola)
}

# 💎 LE CATALOGUE OPTIMISÉ PAR GRID SEARCH 💎
CHAMPIONNATS = [
    # Les 3 ligues "Or" (Volume sain + Efficience battue)
    {"nom": "🇪🇸 La Liga", "id": 140, "key": "soccer_spain_la_liga", "ev_min": 0.05, "ev_max": 0.15, "c1": 4, "euro": 6, "rel": 18},
    {"nom": "🇩🇪 Bundesliga", "id": 78, "key": "soccer_germany_bundesliga", "ev_min": 0.05, "ev_max": 0.15, "c1": 4, "euro": 6, "rel": 16},
    {"nom": "🇳🇱 Eredivisie", "id": 88, "key": "soccer_netherlands_eredivisie", "ev_min": 0.05, "ev_max": 0.15, "c1": 2, "euro": 5, "rel": 16},
    {"nom": "🇮🇹 Serie A", "id": 135, "key": "soccer_italy_serie_a", "ev_min": 0.05, "ev_max": 0.15, "c1": 4, "euro": 6, "rel": 18},
    {"nom": "🇵🇹 Primeira Liga", "id": 94, "key": "soccer_portugal_primeira_liga", "ev_min": 0.05, "ev_max": 0.15, "c1": 2, "euro": 5, "rel": 16},
    {"nom": "🇹🇷 Süper Lig", "id": 203, "key": "soccer_turkey_super_league", "ev_min": 0.05, "ev_max": 0.15, "c1": 2, "euro": 4, "rel": 17},
    {"nom": "🇸🇪 Allsvenskan", "id": 113, "key": "soccer_sweden_allsvenskan", "ev_min": 0.05, "ev_max": 0.15, "c1": 3, "euro": 3, "rel": 14},
    {"nom": "🇧🇷 Série A Brésil", "id": 71, "key": "soccer_brazil_campeonato", "ev_min": 0.05, "ev_max": 0.15, "c1": 6, "euro": 6, "rel": 17},
    {"nom": "🇫🇷 Ligue 1", "id": 61, "key": "soccer_france_ligue_one", "ev_min": 0.05, "ev_max": 0.15, "c1": 4, "euro": 6, "rel": 16},
    {"nom": "🇪🇸 LaLiga 2", "id": 141, "key": "soccer_spain_segunda_division", "ev_min": 0.05, "ev_max": 0.15, "c1": 2, "euro": 6, "rel": 19},
    {"nom": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "id": 39, "key": "soccer_epl", "ev_min": 0.05, "ev_max": 0.15, "c1": 4, "euro": 6, "rel": 18},
    {"nom": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship", "id": 40, "key": "soccer_efl_champ", "ev_min": 0.05, "ev_max": 0.15, "c1": 2, "euro": 6, "rel": 22},
    {"nom": "🇺🇸 MLS", "id": 253, "key": "soccer_usa_mls", "ev_min": 0.05, "ev_max": 0.15, "c1": 7, "euro": 9, "rel": 99},
    {"nom": "🇳🇴 Eliteserien", "id": 103, "key": "soccer_norway_eliteserien", "ev_min": 0.05, "ev_max": 0.15, "c1": 2, "euro": 4, "rel": 14},
    {"nom": "🇧🇪 Jupiler Pro League", "id": 144, "key": "soccer_belgium_first_div", "ev_min": 0.05, "ev_max": 0.15, "c1": 6, "euro": 12, "rel": 13},
    {"nom": "🇮🇹 Serie B", "id": 136, "key": "soccer_italy_serie_b", "ev_min": 0.05, "ev_max": 0.15, "c1": 2, "euro": 8, "rel": 16},
]

URL_FOOTBALL = "https://v3.football.api-sports.io"
HEADERS_FB = {"x-apisports-key": API_FOOTBALL_KEY, "v": "3"}

# ==========================================
# 🛠️ 2. OUTILS SYSTÈME
# ==========================================
async def fetch_async(session, url, headers=None, retries=3):
    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.get(url, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        remaining = response.headers.get('x-ratelimit-requests-remaining')
                        if remaining is not None and int(remaining) < 10:
                            log_info(f"⚠️ QUOTA API-Football bas : {remaining} requêtes restantes !")
                        return await response.json()
                    if response.status == 429:
                        wait = 60 * (attempt + 1)
                        log_info(f"🚫 Rate limit 429 (tentative {attempt+1}/{retries}) — pause {wait}s : {url[:80]}")
                        await asyncio.sleep(wait)
                        continue
                    log_info(f"⚠️ HTTP {response.status} (tentative {attempt+1}/{retries}) : {url[:80]}")
                    return None
            except Exception as e:
                wait = 2 ** attempt
                log_info(f"⚠️ fetch_async erreur (tentative {attempt+1}/{retries}, pause {wait}s) : {e}")
                await asyncio.sleep(wait)
        return None

async def envoyer_telegram_async(session, msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200: log_info("📱 Signal Telegram envoyé.")
    except Exception as e: log_info(f"⚠️ Erreur Telegram : {e}")

async def exporter_historique_csv():
    """
    Exporte paris_log vers le CSV lu par le dashboard Streamlit.
    Colonnes alignées exactement sur ce qu'attend dashboard_foot.py.
    Chemin : /home/chienblanc/data/ sur PythonAnywhere, sinon répertoire courant.
    """
    async with db_lock:
        async with db_conn.execute("""
            SELECT
                id_match          AS ID_Match,
                equipe            AS Equipe,
                handicap          AS Handicap,
                cote_prise        AS Cote_Prise,
                mise              AS Mise,
                cote_cloture      AS Cote_Cloture,
                edge_detecte      AS Edge,
                p_modele          AS Prob_Modele,
                clv               AS CLV,
                statut            AS Statut,
                resultat          AS Profit_Unites,
                ligue             AS Ligue,
                is_lineup_official AS Compo_Officielle,
                timestamp         AS Date
            FROM paris_log
            ORDER BY timestamp DESC
        """) as cursor:
            rows = await cursor.fetchall()
            colonnes = [desc[0] for desc in cursor.description]

    # Chemin adaptatif : PythonAnywhere → /home/chienblanc/data/, sinon répertoire courant
    pa_path = "/home/chienblanc/data/historique_sniper.csv"
    csv_path = pa_path if os.path.isdir("/home/chienblanc/data") else "historique_sniper.csv"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(colonnes)
        writer.writerows(rows)

    log_info(f"📄 CSV exporté ({len(rows)} paris) → {csv_path}")

async def actualiser_kelly_adaptatif():
    """
    Ajuste KELLY_COURANT selon le drawdown actuel de la bankroll,
    puis applique un multiplicateur Brier (BSS récent) si activé.

    Paliers drawdown (basés sur KELLY_FRAC = 0.05) :
      • Drawdown < 8%   → Kelly normal (0.05)   — performance nominale
      • Drawdown 8-15%  → Kelly réduit (0.035)  — phase de récupération
      • Drawdown > 15%  → Kelly réduit (0.025)  — protection capital critique
    """
    global KELLY_COURANT
    try:
        async with db_lock:
            async with db_conn.execute(
                "SELECT resultat FROM paris_log WHERE statut != 'PENDING' ORDER BY timestamp ASC"
            ) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            KELLY_COURANT = KELLY_FRAC
            return

        bankroll = 100.0
        peak = bankroll
        for (pnl,) in rows:
            bankroll += (pnl or 0)
            if bankroll > peak:
                peak = bankroll

        drawdown = (peak - bankroll) / peak if peak > 0 else 0.0

        if drawdown > 0.15:
            kelly_base = 0.025
            log_info(f"⚠️ Kelly réduit à 0.025 — Drawdown critique : {drawdown:.1%}")
        elif drawdown > 0.08:
            kelly_base = 0.035
            log_info(f"⚠️ Kelly réduit à 0.035 — Drawdown modéré : {drawdown:.1%}")
        else:
            kelly_base = KELLY_FRAC

        mult_brier = await _multiplicateur_kelly_brier_foot()
        KELLY_COURANT = round(kelly_base * mult_brier, 4)
    except Exception as e:
        log_info(f"⚠️ actualiser_kelly_adaptatif : {e}")
        KELLY_COURANT = KELLY_FRAC


async def _lire_brier_recent_foot(fenetre: int | None = None):
    """
    Brier score sur les N derniers paris clôturés + baseline p*(1-p).
    Utilise p_modele (probabilité fair implicite 1/cote_fair, pas l'EV).
    Exclut les anciennes lignes où p_modele stockait par erreur ev_modele (~0.05-0.15).
    """
    fenetre = fenetre or FOOT_KELLY_BRIER_FENETRE
    async with db_lock:
        async with db_conn.execute(
            """SELECT p_modele, statut FROM paris_log
               WHERE statut IN ('WON','HALF-WON','LOST','HALF-LOST')
               ORDER BY timestamp ASC"""
        ) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        return None, None, 0

    # p_modele valide : probabilité implicite (0.15–0.95), pas un EV historique
    valid = [
        (p, 1.0 if s in ('WON', 'HALF-WON') else 0.0)
        for p, s in rows
        if p and 0.15 < p < 0.95
    ]
    if len(valid) < FOOT_KELLY_BRIER_MIN_PARIS:
        return None, None, len(valid)

    recent = valid[-fenetre:]
    if len(recent) < FOOT_KELLY_BRIER_MIN_PARIS:
        return None, None, len(recent)

    ps = np.array([p for p, _ in recent])
    outcomes = np.array([o for _, o in recent])
    brier = float(((ps - outcomes) ** 2).mean())
    brier_baseline = float((ps * (1 - ps)).mean())
    return brier, brier_baseline, len(recent)


async def _multiplicateur_kelly_brier_foot() -> float:
    """Réduit Kelly si BSS récent négatif (modèle mal calibré vs confiance affichée)."""
    if not FOOT_KELLY_BRIER_ACTIF:
        return 1.0
    brier, brier_baseline, n = await _lire_brier_recent_foot()
    if brier is None or not brier_baseline or FOOT_KELLY_BSS_SENSIBILITE <= 0:
        return 1.0
    bss = 1.0 - (brier / brier_baseline)
    if bss >= 0:
        return 1.0
    mult = round(min(max(1.0 + bss / FOOT_KELLY_BSS_SENSIBILITE, FOOT_KELLY_BSS_MULT_MIN), 1.0), 3)
    if mult < 1.0:
        log_info(
            f"⚠️ Kelly Brier : BSS récent {bss:+.2f} sur {n} paris "
            f"(Brier {brier:.3f} vs baseline {brier_baseline:.3f}) → mises x{mult:.2f}"
        )
    return mult


def obtenir_saison_api(nom_ligue):
    now = datetime.now()
    annee_actuelle = now.year
    mois_actuel = now.month

    # Ligues estivales (calendrier Printemps→Automne, saison = année civile)
    ligues_estivales = ["Série A Brésil", "Allsvenskan", "MLS", "Eliteserien"]
    if any(mot in nom_ligue for mot in ligues_estivales):
        return annee_actuelle

    # Ligues Européennes (calendrier Hivernal : Août → Mai)
    # En juin-juillet : saison précédente terminée, nouvelle pas encore commencée
    if mois_actuel < 7:
        return annee_actuelle - 1
    else:
        return annee_actuelle

async def obtenir_meteo(session, team_home_id, kickoff_dt=None):
    """
    Récupère la météo prévue à l'heure du coup d'envoi (forecast).
    Si kickoff_dt est None ou H < 4, fallback sur la météo actuelle.
    Utilise forecast 3h si disponible (OpenWeather gratuit).
    """
    coords = STADES_GPS.get(team_home_id)
    if not coords:
        return 0.0, 0.0

    lat, lon = coords
    now_utc = datetime.now(timezone.utc)

    try:
        # Forecast disponible si le match est dans les 5 jours et H > 1h
        if kickoff_dt and (kickoff_dt - now_utc).total_seconds() > 3600:
            url_fc = (f"https://api.openweathermap.org/data/2.5/forecast"
                      f"?lat={lat}&lon={lon}&appid={API_METEO_KEY}&units=metric&cnt=16")
            res_fc = await fetch_async(session, url_fc)
            if res_fc and res_fc.get('list'):
                # Trouver l'entrée forecast la plus proche de l'heure du KO
                ko_ts = kickoff_dt.timestamp()
                best = min(res_fc['list'], key=lambda e: abs(e['dt'] - ko_ts))
                vent_kmh = best.get('wind', {}).get('speed', 0) * 3.6
                pluie_mm = best.get('rain', {}).get('3h', 0)
                return vent_kmh, pluie_mm

        # Fallback : météo actuelle (si match < 1h ou forecast indisponible)
        url_cur = (f"https://api.openweathermap.org/data/2.5/weather"
                   f"?lat={lat}&lon={lon}&appid={API_METEO_KEY}&units=metric")
        res = await fetch_async(session, url_cur)
        if res:
            vent_kmh = res.get('wind', {}).get('speed', 0) * 3.6
            pluie_mm = res.get('rain', {}).get('1h', 0)
            return vent_kmh, pluie_mm
    except Exception as e:
        log_info(f"⚠️ Erreur Météo : {e}")

    return 0.0, 0.0

def appliquer_penalite_meteo(xg_base, vent_kmh, pluie_mm):
    """Baisse les xG si les conditions sont dantesques."""
    multiplicateur = 1.0

    if vent_kmh > 35:
        multiplicateur -= 0.10
    elif vent_kmh > 25:
        multiplicateur -= 0.05

    if pluie_mm > 5:
        multiplicateur -= 0.05

    multiplicateur = max(0.80, multiplicateur)
    return xg_base * multiplicateur

# ==========================================
# 🗄️ COLLECTE HISTORIQUE — scores_matchs
# ==========================================
async def collecter_scores_historiques(session, ligue, saison):
    """
    Récupère tous les matchs FT d'une ligue/saison et les insère dans scores_matchs.
    Appelé une fois par jour pour alimenter l'estimation MLE de ρ.
    Fonctionne par lots de 20 fixtures pour respecter les limites API-Football.
    """
    url = f"{URL_FOOTBALL}/fixtures?league={ligue['id']}&season={saison}&status=FT"
    data = await fetch_async(session, url, HEADERS_FB)
    if not data or not data.get('response'):
        return 0

    fixtures = data['response']
    inserts = 0
    for i in range(0, len(fixtures), 20):
        lot = fixtures[i:i+20]
        async with db_lock:
            await db_conn.executemany(
                "INSERT OR IGNORE INTO scores_matchs "
                "(id_match, ligue_id, saison, buts_dom, buts_ext, team_dom_id, team_ext_id, match_date) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [
                    (f['fixture']['id'],
                     f['league']['id'],
                     f['league']['season'],
                     f['goals']['home'],
                     f['goals']['away'],
                     f['teams']['home']['id'],
                     f['teams']['away']['id'],
                     f['fixture']['date'])
                    for f in lot
                    if f['goals']['home'] is not None and f['goals']['away'] is not None
                ]
            )
            await db_conn.commit()
        inserts += sum(
            1 for f in lot
            if f['goals']['home'] is not None and f['goals']['away'] is not None
        )

    log_info(f"📊 scores_matchs : {inserts} matchs FT chargés pour {ligue['nom']} {saison}.")
    return inserts


# ==========================================
# 🧮 3. RÈGLEMENTS & STATS
# ==========================================
async def verifier_resultats_matchs(session):
    async with db_lock:
        async with db_conn.execute(
            "SELECT id_match, equipe, handicap, cote_prise, mise, ligue, equipe_dom, equipe_ext "
            "FROM paris_log WHERE statut='PENDING'"
        ) as cursor:
            paris = await cursor.fetchall()

    if not paris: return

    ids_uniques = list(set(p[0] for p in paris))
    ids_a_verifier = []

    for id_m in ids_uniques:
        heure_match = cache_heures_matchs.get(id_m)
        if heure_match and datetime.now(timezone.utc) < heure_match + timedelta(hours=2):
            continue
        ids_a_verifier.append(id_m)

    if not ids_a_verifier: return

    for i in range(0, len(ids_a_verifier), 20):
        lot_ids = ids_a_verifier[i:i+20]
        ids_str = "-".join(map(str, lot_ids))
        data = await fetch_async(session, f"{URL_FOOTBALL}/fixtures?ids={ids_str}", HEADERS_FB)

        if not data or not data.get('response'): continue

        resultats_dict = {f['fixture']['id']: f for f in data['response']}

        for p in paris:
            id_m, equipe_pari, h_val, cote, mise_initiale, ligue_tag, eq_dom_log, eq_ext_log = p
            f = resultats_dict.get(id_m)
            if not f: continue

            status_short = f['fixture']['status']['short']

            # Statuts "en cours / pas encore joué" : rien à régler, on garde le pari PENDING.
            # PST (reporté) et TBD sont inclus ici — le fixture ID reste stable chez API-Football,
            # le match finira par repasser à FT une fois rejoué (ou CANC s'il est définitivement annulé).
            if status_short in ['NS', 'TBD', '1H', 'HT', '2H', 'ET', 'BT', 'P', 'SUSP', 'INT', 'PST']:
                try:
                    dt_obj = datetime.fromisoformat(f['fixture']['date'].replace('Z', '+00:00'))
                    cache_heures_matchs[id_m] = dt_obj
                except Exception:
                    pass
                continue

            # Statuts terminaux sans score exploitable : annulé, abandonné, ou résultat technique
            # (walkover / défaite administrative). On rembourse (VOID) plutôt que de laisser le pari
            # PENDING indéfiniment — on ne peut pas déterminer un règlement fiable sur handicap/total.
            if status_short in ['CANC', 'ABD', 'AWD', 'WO']:
                async with db_lock:
                    await db_conn.execute(
                        "UPDATE paris_log SET statut='VOID', resultat=0.0 "
                        "WHERE id_match=? AND equipe=? AND handicap=?",
                        (id_m, equipe_pari, h_val)
                    )
                    await db_conn.commit()
                log_info(f"🚫 Match {status_short} : {equipe_pari} ({h_val}) -> VOID (remboursé, non réglable)")
                continue

            if status_short in ['FT', 'AET', 'PEN']:
                gh, ga = f['goals']['home'], f['goals']['away']
                if gh is None or ga is None: continue

                # Enregistrer le score pour l'estimation dynamique de ρ
                ligue_id_match = f['league']['id']
                saison_match   = f['league']['season']
                dom_id_match   = f['teams']['home']['id']
                ext_id_match   = f['teams']['away']['id']
                async with db_lock:
                    await db_conn.execute(
                        "INSERT OR IGNORE INTO scores_matchs "
                        "(id_match, ligue_id, saison, buts_dom, buts_ext, team_dom_id, team_ext_id, match_date) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (id_m, ligue_id_match, saison_match, gh, ga, dom_id_match, ext_id_match,
                         f['fixture']['date'])
                    )
                    await db_conn.commit()

                if "[totals]" in ligue_tag:
                    total_buts = gh + ga
                    diff = (total_buts - h_val) if "Over" in equipe_pari else (h_val - total_buts)
                else:
                    nom_home_api = f['teams']['home']['name']
                    nom_away_api = f['teams']['away']['name']
                    # Priorité : équipe_dom/ext stockées en DB (noms Odds API exacts)
                    # Fallback : fuzzy match si anciennes lignes sans ces colonnes
                    if eq_dom_log:
                        is_h = (equipe_pari == eq_dom_log)
                    else:
                        score_home = process.extractOne(equipe_pari, [nom_home_api])[1]
                        score_away = process.extractOne(equipe_pari, [nom_away_api])[1]
                        is_h = score_home > score_away
                    diff = (gh + h_val - ga) if is_h else (ga + h_val - gh)

                if diff > 0.25 + _EPS_AH:               mult, stat = cote - 1,         "WON"
                elif abs(diff - 0.25) < _EPS_AH:        mult, stat = (cote - 1) / 2,  "HALF-WON"
                elif abs(diff) < _EPS_AH:               mult, stat = 0,                "VOID"
                elif abs(diff + 0.25) < _EPS_AH:        mult, stat = -0.5,             "HALF-LOST"
                else:                                    mult, stat = -1.0,             "LOST"

                profit_reel = round(mult * mise_initiale, 2)

                async with db_lock:
                    await db_conn.execute(
                        "UPDATE paris_log SET statut=?, resultat=? WHERE id_match=? AND equipe=? AND handicap=?",
                        (stat, profit_reel, id_m, equipe_pari, h_val)
                    )
                    await db_conn.commit()
                log_info(f"🏁 Règlement : {equipe_pari} ({h_val}) -> {stat} ({profit_reel:+.2f}u)")

async def evaluer_impact_blessures(session, fixture_id, team_id, default_power=1.0):
    """Filtre H-24 : Vérifie le nombre de joueurs à l'infirmerie."""
    try:
        url = f"{URL_FOOTBALL}/injuries?fixture={fixture_id}&team={team_id}"
        res = await fetch_async(session, url, HEADERS_FB)

        if not res or not res.get('response'):
            return default_power

        nb_absents = len(res['response'])

        if nb_absents <= 2:
            return default_power
        else:
            malus = min((nb_absents - 2) * 0.05, 0.20)
            return default_power * (1.0 - malus)
    except Exception:
        return default_power

async def evaluer_force_lineup(session, fixture_id, team_id, saison_correcte, default_power=1.0):
    """Bouclier H-1 : Vérifie si les titulaires habituels sont bien sur le terrain."""
    try:
        url_lineup = f"{URL_FOOTBALL}/fixtures/lineups?fixture={fixture_id}&team={team_id}"
        res_lineup = await fetch_async(session, url_lineup, HEADERS_FB)

        if not res_lineup or not res_lineup.get('response'):
            return default_power

        current_starters = [str(player['player']['id']) for player in res_lineup['response'][0]['startXI']]

        # Récupération paginée de l'effectif complet pour un XI type fiable
        all_players_res = []
        page = 1
        while True:
            url_players = f"{URL_FOOTBALL}/players?team={team_id}&season={saison_correcte}&page={page}"
            res_players = await fetch_async(session, url_players, HEADERS_FB)
            if not res_players or not res_players.get('response'):
                break
            all_players_res.extend(res_players['response'])
            total_pages = res_players.get('paging', {}).get('total', 1)
            if page >= total_pages:
                break
            page += 1

        if not all_players_res:
            return default_power

        player_minutes = []
        for p in all_players_res:
            p_id = str(p['player']['id'])
            stats_league = next((s for s in p['statistics'] if s['league']['id'] in [c['id'] for c in CHAMPIONNATS]), None)
            mins = stats_league['games']['minutes'] if stats_league and stats_league['games']['minutes'] else 0
            player_minutes.append((p_id, mins))

        player_minutes.sort(key=lambda x: x[1], reverse=True)
        xi_type = [p[0] for p in player_minutes[:11]]

        stars_missing = sum(1 for p_id in xi_type if p_id not in current_starters)

        if stars_missing <= 2:
            return default_power
        else:
            malus = min((stars_missing - 2) * 0.05, 0.25)
            return default_power * (1.0 - malus)
    except Exception:
        return default_power

# ==========================================
# 🎯 4. ANALYSEUR HYBRIDE
# ==========================================
async def analyser_un_match(session, m, ligue, saison_correcte, sos_map, sos_attack_map, mot_map, luck_map, m_dom_l, m_ext_l, cotes_data):
    n_d, n_e = m['teams']['home']['name'], m['teams']['away']['name']
    id_d, id_e, id_m = m['teams']['home']['id'], m['teams']['away']['id'], m['fixture']['id']

    dt_obj = datetime.fromisoformat(m['fixture']['date'].replace('Z', '+00:00'))
    hr = (dt_obj - datetime.now(dt_obj.tzinfo)).total_seconds() / 3600

    # Alimenter le cache d'heures dès le scan pour que calculer_pause() soit réactif
    cache_heures_matchs[id_m] = dt_obj

    # 🎯 LE FILTRE TEMPOREL : Le Sniper ne s'active qu'à partir de H-36
    if not (0.9 <= hr <= 36.0): return

    # poids_dyn calibré sur la fenêtre réelle [0.9h, 36h] du bot.
    # L'ancienne formule (hr/168) produisait une variation de 15.1% à 18.2% — trop étroite.
    # Nouvelle plage : 10% (H-0.9, marché très sharp) → 30% (H-36, signal modèle plus libre).
    poids_dyn = 0.10 + 0.20 * min(1.0, (hr - 0.9) / (36.0 - 0.9))

    # 🕐 EV MINIMUM GRADUÉ SELON L'HEURE AU KO
    # Plus on se rapproche du coup d'envoi, plus Pinnacle a intégré l'information
    # (compositions, blessures de dernière minute, mouvements sharp).
    # On compense en exigeant un edge plus élevé pour valider le pari.
    #   H > 6  → seuil normal   (modèle pertinent, peu d'info de dernière minute)
    #   H 3-6  → +1.5%          (premiers mouvements pré-KO, compositions probables)
    #   H < 3  → +3%            (compositions diffusées, marché au plus serré)
    if hr < 3.0:
        ev_min_effectif = ligue['ev_min'] + 0.03
    elif hr < 6.0:
        ev_min_effectif = ligue['ev_min'] + 0.015
    else:
        ev_min_effectif = ligue['ev_min']

    # Label H-KO pour transparence dans les alertes Telegram
    if hr < 3.0:
        hko_label = f"🔴 H-{hr:.1f} (marché serré +3%)"
    elif hr < 6.0:
        hko_label = f"🟡 H-{hr:.1f} (pré-KO +1.5%)"
    else:
        hko_label = f"🟢 H-{hr:.1f}"

    n_d_m = NAME_MAPPING.get(n_d, n_d)
    n_e_m = NAME_MAPPING.get(n_e, n_e)
    match_o = next((c for c in cotes_data if (process.extractOne(n_d_m, [c['home_team']])[1] + process.extractOne(n_e_m, [c['away_team']])[1])/2 > 85), None)
    if not match_o: return
    pinnacle = next((b for b in match_o['bookmakers'] if b['key'] == 'pinnacle'), None)

    if not pinnacle or not pinnacle.get('markets') or len(pinnacle['markets']) == 0: return
    outcomes = pinnacle['markets'][0].get('outcomes')
    if not outcomes or len(outcomes) < 2: return


    res_lineups = await fetch_async(session, f"{URL_FOOTBALL}/fixtures/lineups?fixture={id_m}", HEADERS_FB)
    lineup_ok = True if res_lineups and res_lineups.get('response') else False

    # ligue_avg_def : moyenne des buts encaissés par match selon le venue
    # Home team concède en moyenne m_ext_l (buts marqués par les visiteurs)
    # Away team concède en moyenne m_dom_l (buts marqués par les équipes à domicile)
    xg_d_home = await obtenir_xg_moyenne_async(session, id_d, ligue['id'], saison_correcte, sos_map, m_dom_l, venue='home', ligue_avg_def=m_ext_l, sos_attack_map=sos_attack_map)
    xg_e_away = await obtenir_xg_moyenne_async(session, id_e, ligue['id'], saison_correcte, sos_map, m_ext_l, venue='away', ligue_avg_def=m_dom_l, sos_attack_map=sos_attack_map)
    ligue_avg_all = (m_dom_l + m_ext_l) / 2
    xg_d_all = await obtenir_xg_moyenne_async(session, id_d, ligue['id'], saison_correcte, sos_map, ligue_avg_all, venue='all', ligue_avg_def=ligue_avg_all, sos_attack_map=sos_attack_map)
    xg_e_all = await obtenir_xg_moyenne_async(session, id_e, ligue['id'], saison_correcte, sos_map, ligue_avg_all, venue='all', ligue_avg_def=ligue_avg_all, sos_attack_map=sos_attack_map)

    # Pondération venue adaptative : la confiance dans les stats spécifiques (dom/ext)
    # croît avec la taille d'échantillon. Moins de 5 matchs → on se fie surtout au global.
    # n_spec[2] = nombre de matchs utilisés, retourné par obtenir_xg_moyenne_async
    def w_venue(n_spec, max_spec=0.80):
        w = min(max_spec, (n_spec / 10.0) * max_spec)
        return w, 1.0 - w

    w_d_spec, w_d_glob = w_venue(xg_d_home[2])
    w_e_spec, w_e_glob = w_venue(xg_e_away[2])

    xg_d = (
        (xg_d_home[0] * w_d_spec) + (xg_d_all[0] * w_d_glob),
        (xg_d_home[1] * w_d_spec) + (xg_d_all[1] * w_d_glob)
    )
    xg_e = (
        (xg_e_away[0] * w_e_spec) + (xg_e_all[0] * w_e_glob),
        (xg_e_away[1] * w_e_spec) + (xg_e_all[1] * w_e_glob)
    )

    # L'avantage domicile/extérieur est capturé par venue='home'/'away' dans obtenir_xg_moyenne_async.
    d_d = (mot_map.get(id_d, 1.0)-1) + (luck_map.get(id_d, 1.0)-1)
    d_e = (mot_map.get(id_e, 1.0)-1) + (luck_map.get(id_e, 1.0)-1)
    m_d, m_e = 1.0 + max(-0.25, min(0.25, d_d)), 1.0 + max(-0.25, min(0.25, d_e))

    if not m_dom_l or not m_ext_l:
        return  # Début de saison : aucun match joué, moyennes ligue à 0

    L_A_base = (xg_d[0] * m_d / m_dom_l) * (xg_e[1] / m_dom_l) * m_dom_l
    L_B_base = (xg_e[0] * m_e / m_ext_l) * (xg_d[1] / m_ext_l) * m_ext_l

    # --- 🧮 ENRICHISSEMENT DIXON-COLES COMPLET ---
    # Si les paramètres α/β/γ/ρ par équipe sont disponibles (MLE quotidien),
    # on blende les λ xG (forme récente) avec les λ DC (structure saison entière).
    # Blend 50/50 : DC apporte la structure, xG apporte la forme récente.
    dc = DC_PARAMS.get((ligue['id'], saison_correcte))
    if dc and id_d in dc['teams'] and id_e in dc['teams']:
        td, te = dc['teams'][id_d], dc['teams'][id_e]
        L_A_dc = td['attack'] * te['defense'] * dc['gamma']
        L_B_dc = te['attack'] * td['defense']
        L_A_base = 0.50 * L_A_dc + 0.50 * L_A_base
        L_B_base = 0.50 * L_B_dc + 0.50 * L_B_base
        # Propager ρ DC (estimé conjointement, plus fiable)
        RHO_DYNAMIQUE[(ligue['id'], saison_correcte)] = dc['rho']

    # --- ⛈️ INTÉGRATION DU WEATHER EDGE ---
    # Météo activée seulement à H < 6 (prévision fiable + marché peu anticipé)
    if hr < 6.0:
        vent, pluie = await obtenir_meteo(session, id_d, kickoff_dt=dt_obj)
        if vent > 25 or pluie > 5:
            L_A_base = appliquer_penalite_meteo(L_A_base, vent, pluie)
            L_B_base = appliquer_penalite_meteo(L_B_base, vent, pluie)
            log_info(f"⛈️ Alerte Météo ({n_d}) : Vent {vent:.1f}km/h | Pluie {pluie}mm. xG réduits.")

    # --- 🛡️ FILTRE PRÉLIMINAIRE (ÉCONOMISEUR D'API) ---
    # Noms des équipes côté odds API pour comparaison exacte dans la boucle
    home_team_odds = match_o['home_team']
    away_team_odds = match_o['away_team']

    mat_preliminaire = generer_matrice_dixon(max(0.4, L_A_base), max(0.4, L_B_base), ligue['id'], saison_correcte)
    match_potentiel = False

    for market in pinnacle['markets']:
        market_key_prelim = market['key']
        if market_key_prelim not in ('spreads', 'totals'):
            continue
        for out in market.get('outcomes', []):
            try:
                h_prelim = float(out.get('point', 0))
                cote_prelim = float(out['price'])

                if market_key_prelim == 'spreads':
                    is_h_odds_prelim = (out['name'] == home_team_odds)
                    ev_prelim = calculer_ev_ah(mat_preliminaire, h_prelim, is_h_odds_prelim, cote_prelim)
                else:
                    is_over_prelim = out['name'].lower() == 'over'
                    ev_prelim = calculer_ev_total_asiatique(mat_preliminaire, h_prelim, is_over_prelim, cote_prelim)

                if ev_prelim > -0.05:
                    match_potentiel = True
                    break
            except Exception:
                continue

        if match_potentiel:
            break

    if not match_potentiel:
        return

    # 2. FILTRE H-24 : Impact des Blessures (Uniquement sur les matchs prometteurs)
    force_blessure_d = await evaluer_impact_blessures(session, id_m, id_d)
    force_blessure_e = await evaluer_impact_blessures(session, id_m, id_e)

    # Réutiliser la matrice préliminaire si aucune blessure ne modifie les xG
    if force_blessure_d == 1.0 and force_blessure_e == 1.0:
        mat = mat_preliminaire
    else:
        L_A = L_A_base * force_blessure_d
        L_B = L_B_base * force_blessure_e
        mat = generer_matrice_dixon(max(0.4, L_A), max(0.4, L_B), ligue['id'], saison_correcte)

    # 3. BOUCLIER H-1 : Impact de la Composition Officielle
    mat_lineup_drop = None
    alerte_lineup_text = ""
    if lineup_ok and hr <= 1.5:
        force_lineup_d = await evaluer_force_lineup(session, id_m, id_d, saison_correcte)
        force_lineup_e = await evaluer_force_lineup(session, id_m, id_e, saison_correcte)

        if force_lineup_d <= 0.90 or force_lineup_e <= 0.90:
            L_A_drop = L_A_base * force_blessure_d * force_lineup_d
            L_B_drop = L_B_base * force_blessure_e * force_lineup_e
            mat_lineup_drop = generer_matrice_dixon(max(0.4, L_A_drop), max(0.4, L_B_drop), ligue['id'], saison_correcte)
            alerte_lineup_text = f"🔄 ROTATION MASSIVE DÉTECTÉE (Dom: {force_lineup_d:.2f}x | Ext: {force_lineup_e:.2f}x)"

    # 🔒 1 seul pari actif par match — spreads + totaux confondus
    async with db_lock:
        async with db_conn.execute(
            "SELECT id_match FROM paris_log WHERE id_match=? AND statut='PENDING'",
            (id_m,)
        ) as cursor:
            deja_parie_ce_match = await cursor.fetchone()

    if deja_parie_ce_match:
        return

    # Matrice active (rotation ou normale) + seuil EV adapté
    if mat_lineup_drop is not None:
        mat_actif = mat_lineup_drop
        ev_min_rotation = ev_min_effectif + 0.02
    else:
        mat_actif = mat
        ev_min_rotation = ev_min_effectif

    # Collecte de tous les candidats (AH + Totaux) → meilleur EV global
    candidats = []
    for market in pinnacle['markets']:
        market_key = market['key']
        if market_key not in ('spreads', 'totals'):
            continue
        outcomes = market['outcomes']

        for out in outcomes:
            h, cote, nom = float(out['point']), float(out['price']), out['name']

            if cote < 1.70:
                continue  # Cotes courtes : ratio edge/bruit défavorable (aligné backtest)

            async with db_lock:
                async with db_conn.execute(
                    "SELECT cote_prise FROM paris_log WHERE id_match=? AND equipe=? AND handicap=?",
                    (id_m, nom, h)
                ) as cursor:
                    if await cursor.fetchone():
                        continue

            if market_key == 'spreads':
                is_h_odds = (nom == home_team_odds)
                cote_partenaire = next(
                    (float(o['price']) for o in outcomes
                     if o['name'] != nom and abs(float(o.get('point', 0)) + h) < 0.01),
                    None
                )
                if cote_partenaire and cote_partenaire > 1.0:
                    ovr_ligne = (1.0 / cote) + (1.0 / cote_partenaire)
                    cote_novig = cote * ovr_ligne
                else:
                    cote_novig = cote
                ev_modele = calculer_ev_ah(mat_actif, h, is_h_odds, cote)
                ev_pinnacle = calculer_ev_ah(mat_actif, h, is_h_odds, cote_novig)
                kelly_theorique = calculer_kelly_ah(mat_actif, h, is_h_odds, cote)
                p_modele = calculer_prob_modele_pari(mat_actif, 'spreads', h, is_h_odds=is_h_odds)
                market_label = "🏆 HANDICAP"
                pari_display = f"{nom} ({h:+g})"
            else:
                is_over = nom.lower() == 'over'
                cote_partenaire = next(
                    (float(o['price']) for o in outcomes
                     if o['name'] != nom and abs(float(o.get('point', 0)) - h) < 0.01),
                    None
                )
                if cote_partenaire and cote_partenaire > 1.0:
                    ovr_ligne = (1.0 / cote) + (1.0 / cote_partenaire)
                    cote_novig = cote * ovr_ligne
                else:
                    cote_novig = cote
                ev_modele = calculer_ev_total_asiatique(mat_actif, h, is_over, cote)
                ev_pinnacle = calculer_ev_total_asiatique(mat_actif, h, is_over, cote_novig)
                kelly_theorique = calculer_kelly_total(mat_actif, h, is_over, cote)
                p_modele = calculer_prob_modele_pari(mat_actif, 'totals', h, is_over=is_over)
                market_label = "⚽ TOTAL"
                pari_display = f"{nom} {h:g}"

            ev_final = (ev_modele * poids_dyn) + (ev_pinnacle * (1 - poids_dyn))

            if ev_min_rotation <= ev_final <= ligue['ev_max']:
                mise_u = round((kelly_theorique * 100) * KELLY_COURANT, 2)
                mise_u = min(mise_u, 5.0)
                if mise_u >= 0.1:
                    candidats.append({
                        'ev_final': ev_final, 'ev_modele': ev_modele, 'ev_pinnacle': ev_pinnacle,
                        'h': h, 'cote': cote, 'nom': nom, 'mise_u': mise_u,
                        'p_modele': p_modele,
                        'market_key': market_key, 'market_label': market_label,
                        'pari_display': pari_display,
                    })

    if not candidats:
        return

    best = max(candidats, key=lambda x: x['ev_final'])
    h, cote, nom = best['h'], best['cote'], best['nom']
    ev_final, ev_modele, ev_pinnacle, mise_u, p_modele = (
        best['ev_final'], best['ev_modele'], best['ev_pinnacle'], best['mise_u'], best['p_modele']
    )

    badge = "✅ *XI CONFIRMÉ*" if lineup_ok else "⏳ *Compo Probable*"
    rotation_note = f"\n⚠️ {alerte_lineup_text}" if alerte_lineup_text else ""

    msg = (f"🎯 *SIGNAL [{ligue['nom']}] {best['market_label']}*\n"
           f"🏟️ {n_d} - {n_e}\n{badge} | {hko_label}{rotation_note}\n"
           f"💎 Pari : *{best['pari_display']}* @ {cote:.2f}\n"
           f"🔥 Value : *+{ev_final:.1%}* (Mod: {ev_modele:+.1%} | Pin: {ev_pinnacle:+.1%})\n"
           f"📊 Prob. modèle : *{p_modele:.1%}* (fair @ {1/p_modele:.2f})\n"
           f"📏 Mise Kelly : *{mise_u} u*")
    await envoyer_telegram_async(session, msg)

    async with db_lock:
        ligue_tag = f"{ligue['nom']} [{best['market_key']}]"
        await db_conn.execute("""INSERT OR IGNORE INTO paris_log
            (id_match, equipe, handicap, cote_prise, mise, edge_detecte, p_modele, ligue,
             is_lineup_official, timestamp, equipe_dom, equipe_ext, kickoff)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (id_m, nom, h, cote, mise_u, round(ev_final, 4), round(p_modele, 4), ligue_tag,
             int(lineup_ok), datetime.now().isoformat(),
             home_team_odds, away_team_odds, m['fixture']['date']))
        await db_conn.commit()

# ==========================================
# 🚀 5. FONCTIONS MATHS & xG
# ==========================================
async def obtenir_xg_moyenne_async(session, team_id, l_id, saison_actuelle, sos_map, ligue_avg, venue='all', ligue_avg_def=None, sos_attack_map=None):
    """
    Moteur Hybride : Gère le passage de témoin entre la saison passée et actuelle,
    applique un shrinkage bayésien et une décroissance temporelle basée sur les jours réels.
    ligue_avg      : moyenne buts MARQUÉS par match (cible shrinkage offensif)
    ligue_avg_def  : moyenne buts ENCAISSÉS par match (cible shrinkage défensif)
                     Si None → utilise ligue_avg (compatibilité ascendante).
    sos_attack_map : buts marqués par équipe (force offensive adverse) pour normaliser xG défensif.
                     Si None → fallback sur sos_map (compatibilité ascendante, sous-optimal).
    Retourne (xg_off, xg_def, n_matchs_utilises).
    """
    if ligue_avg_def is None:
        ligue_avg_def = ligue_avg
    url_n = f"{URL_FOOTBALL}/fixtures?team={team_id}&league={l_id}&season={saison_actuelle}&status=FT"
    data_n = await fetch_async(session, url_n, HEADERS_FB)
    matchs_n = data_n['response'] if data_n else []

    matchs_totaux = matchs_n
    if len(matchs_n) < 10:
        saison_prev = saison_actuelle - 1
        url_prev = f"{URL_FOOTBALL}/fixtures?team={team_id}&league={l_id}&season={saison_prev}&status=FT"
        data_prev = await fetch_async(session, url_prev, HEADERS_FB)
        if data_prev and data_prev['response']:
            matchs_totaux = data_prev['response'] + matchs_n

    matchs_valides = matchs_totaux
    if venue == 'home':
        matchs_valides = [m for m in matchs_totaux if m['teams']['home']['id'] == team_id]
    elif venue == 'away':
        matchs_valides = [m for m in matchs_totaux if m['teams']['away']['id'] == team_id]

    matchs_a_analyser = matchs_valides[-15:]

    # 🚨 LE FACTEUR PROMU : moins de 5 matchs en D1 sur 2 saisons = promu
    if len(matchs_a_analyser) < 5:
        xg_off_promu = ligue_avg * 0.85
        xg_def_promu = (ligue_avg_def or ligue_avg) * 1.25  # cible défensive, pas offensive
        return (xg_off_promu, xg_def_promu, len(matchs_a_analyser))

    tp, tc, tw = 0.0, 0.0, 0.0
    ALPHA_DEGRADATION = 0.80
    maintenant_utc = datetime.now(timezone.utc)

    for i, m in enumerate(matchs_a_analyser):
        f_id = m['fixture']['id']
        p, c = None, None

        async with db_lock:
            async with db_conn.execute("SELECT xg_p, xg_c, is_xg FROM xg_cache WHERE cle=?", (f"xg_{f_id}_{team_id}",)) as cursor:
                db_res = await cursor.fetchone()

        # Utiliser le cache seulement si les données sont des vrais xG (is_xg=1).
        # Si is_xg=0 (buts comme proxy) et que la ligue fournit maintenant des xG,
        # on invalide pour re-fetch les statistiques réelles.
        cache_valide = db_res and (db_res[2] == 1 or l_id in LIGUES_SANS_XG)
        if cache_valide:
            p, c = db_res[0], db_res[1]
        else:
            # Par défaut : buts réels comme proxy (fallback garanti)
            is_h_team = (m['teams']['home']['id'] == team_id)
            p = float(m['goals']['home'] or 0) if is_h_team else float(m['goals']['away'] or 0)
            c = float(m['goals']['away'] or 0) if is_h_team else float(m['goals']['home'] or 0)
            used_real_xg = False

            # Tentative d'obtenir les xG réels (Opta via API-Football)
            # Skip pour les ligues connues sans xG : évite des appels API inutiles
            if l_id not in LIGUES_SANS_XG:
                stats = await fetch_async(session, f"{URL_FOOTBALL}/fixtures/statistics?fixture={f_id}", HEADERS_FB)
                if stats and stats.get('response'):
                    for team_stat in stats['response']:
                        xg_raw = next(
                            (s['value'] for s in team_stat['statistics'] if s['type'] == 'expected_goals'),
                            None
                        )
                        # L'API renvoie parfois "null" (str), None, "" ou un float en string
                        try:
                            xg_val = float(xg_raw) if xg_raw not in (None, 'null', '', 'None') else None
                        except (TypeError, ValueError):
                            xg_val = None

                        if xg_val is not None:
                            # Remplacement individuel : on ne jette pas le xG valide
                            # si l'autre équipe manque de donnée
                            if team_stat['team']['id'] == team_id:
                                p = xg_val
                                used_real_xg = True
                            else:
                                c = xg_val
                                used_real_xg = True

            async with db_lock:
                await db_conn.execute(
                    "INSERT OR REPLACE INTO xg_cache (cle, xg_p, xg_c, is_xg, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (f"xg_{f_id}_{team_id}", p, c, int(used_real_xg), datetime.now().isoformat())
                )
                await db_conn.commit()

        # Décroissance temporelle réelle (demi-vie ≈ 46 jours)
        # Un match joué il y a 1 semaine pèse ~0.90, il y a 3 mois ~0.15
        try:
            match_date = datetime.fromisoformat(m['fixture']['date'].replace('Z', '+00:00'))
            days_ago = max(0, (maintenant_utc - match_date).days)
        except Exception:
            days_ago = (len(matchs_a_analyser) - 1 - i) * 7  # fallback : 1 match/semaine
        weight = np.exp(-0.015 * days_ago)

        if m['league']['season'] < saison_actuelle:
            weight *= ALPHA_DEGRADATION

        opp_id = m['teams']['away']['id'] if m['teams']['home']['id'] == team_id else m['teams']['home']['id']
        # ratio_def : force défensive de l'adversaire (buts concédés / moy.) → normalise notre attaque
        # ratio_att : force offensive de l'adversaire (buts marqués / moy.)  → normalise notre défense
        ratio_def = max(0.6, min(1.6, sos_map.get(opp_id, ligue_avg) / (ligue_avg or 1)))
        _att_map  = sos_attack_map if sos_attack_map else sos_map
        ratio_att = max(0.6, min(1.6, _att_map.get(opp_id, ligue_avg) / (ligue_avg or 1)))

        tp += (p / ratio_def) * weight   # si adversaire défense faible → on a marqué facile → normalise DOWN
        tc += (c / ratio_att) * weight   # si adversaire attaque forte  → on a encaissé normal → normalise DOWN
        tw += weight

    xg_off_brut = tp / tw
    xg_def_brut = tc / tw

    # --- 🎯 SHRINKAGE BAYÉSIEN ADAPTATIF (James-Stein) ---
    # Régresse les estimations vers la moyenne de ligue pour éviter la sur-réaction
    # aux petits échantillons (ex: 5-8 matchs en début de saison).
    # N_PRIOR varie selon la ligue : plus elle est petite / données peu fiables → prior élevé.
    n_prior = N_PRIOR_PAR_LIGUE.get(l_id, N_PRIOR_DEFAULT)
    n = len(matchs_a_analyser)
    w_equipe = n / (n + n_prior)
    w_ligue  = 1.0 - w_equipe

    xg_off = w_equipe * xg_off_brut + w_ligue * ligue_avg
    xg_def = w_equipe * xg_def_brut + w_ligue * ligue_avg_def

    return (xg_off, xg_def, n)

async def actualiser_stats_ligue(session, ligue_cfg, season):
    cle = f"{ligue_cfg['id']}_{season}"

    # 🛡️ Cache de 12 heures : économise ~4500 requêtes API par jour !
    if cle in cache_standings and datetime.now(timezone.utc) < cache_standings[cle]['expire']:
        return cache_standings[cle]['data']

    # Ligues européennes en off-season (juin-juillet) : inutile d'interroger l'API
    mois = datetime.now().month
    is_euro = not any(mot in ligue_cfg['nom'] for mot in ["Brésil", "Allsvenskan", "MLS", "Eliteserien"])
    if is_euro and mois in (6, 7):
        log_info(f"💤 {ligue_cfg['nom']} : off-season (Coupe du Monde / intersaison). Ignorée jusqu'en août.")
        return None

    async def _fetch_standings(s):
        """Interroge /standings pour la saison s. Retourne la liste standings ou None."""
        url = f"{URL_FOOTBALL}/standings?league={ligue_cfg['id']}&season={s}"
        response = await fetch_async(session, url, HEADERS_FB)
        if not response or not response.get('response') or len(response['response']) == 0:
            return None
        data = response['response']
        league_data = data[0].get('league', {})
        if not league_data.get('standings') or len(league_data['standings']) == 0:
            return None
        return league_data['standings'][0]

    try:
        standings = await _fetch_standings(season)
        saison_utilisee = season

        if standings is None:
            # Fallback : essayer la saison précédente (utile en début de saison ou World Cup break)
            standings = await _fetch_standings(season - 1)
            saison_utilisee = season - 1
            if standings is not None:
                log_info(f"⚠️ {ligue_cfg['nom']} : saison {season} vide — utilisation de la saison {season-1} (pause/début de saison).")
            else:
                log_info(f"❌ {ligue_cfg['nom']} : aucune donnée standings pour {season} ni {season-1}.")
                return None

        if len(standings) < 10:
            avg = ligue_cfg.get('avg_goals', 1.3)
            return ({t['team']['id']: avg for t in standings},
                    {t['team']['id']: avg for t in standings},
                    {t['team']['id']: 1.0 for t in standings}, {}, avg, avg * 0.85)

        pts_c1 = standings[min(ligue_cfg['c1']-1, len(standings)-1)]['points']
        pts_rel = standings[min(ligue_cfg['rel']-1, len(standings)-1)]['points']
        pts_euro = standings[min(ligue_cfg['euro']-1, len(standings)-1)]['points'] if ligue_cfg.get('euro') else None

        m_dom_l = sum(t['home']['goals']['for'] for t in standings) / (sum(t['home']['played'] for t in standings) or 1)
        m_ext_l = sum(t['away']['goals']['for'] for t in standings) / (sum(t['away']['played'] for t in standings) or 1)

        # Moyenne de buts encaissés par match dans la ligue (pour le PDO défensif)
        league_avg_gf = (m_dom_l + m_ext_l) / 2
        league_avg_ga = sum(t['all']['goals']['against'] for t in standings) / (sum(t['all']['played'] for t in standings) or 1)

        sos, sos_attack, mot, luck = {}, {}, {}, {}
        for team in standings:
            t_id = team['team']['id']
            j = team['all']['played'] or 1
            # Distance minimale à un enjeu de classement : titre/promo (c1), place européenne (euro) ou relégation (rel).
            # Auparavant seuls c1/rel étaient pris en compte : les équipes en course pour l'Europe
            # (souvent en milieu de tableau, loin du podium et de la zone rouge) restaient à tort neutres.
            enjeux = [pts_c1, pts_rel] + ([pts_euro] if pts_euro is not None else [])
            d = min(abs(team['points'] - p_enjeu) for p_enjeu in enjeux)
            mot[t_id] = 1.0 + (0.10 * (1/(d+1))) if d <= 4 else (0.95 if d > 12 else 1.0)
            sos[t_id]        = team['all']['goals']['against'] / j  # buts concédés : force défensive adverse (normalise notre attaque)
            sos_attack[t_id] = team['all']['goals']['for']     / j  # buts marqués  : force offensive adverse (normalise notre défense)

            # PDO-proxy : combine sur-performance offensive ET défensive
            # gf_ratio > 1 = équipe qui marque plus que la moyenne (peut être chanceuse)
            # ga_ratio > 1 = équipe qui encaisse plus que la moyenne (peut être malchanceuse)
            # pdo > 1 = équipe qui sur-performe → correction vers le bas (régression à la moyenne)
            gf_ratio = (team['all']['goals']['for'] / j) / (league_avg_gf or 1)
            ga_ratio = (team['all']['goals']['against'] / j) / (league_avg_ga or 1)
            pdo = (gf_ratio + (2.0 - ga_ratio)) / 2.0
            luck[t_id] = 1.0 - (pdo - 1.0) * 0.30

        resultat = (sos, sos_attack, mot, luck, m_dom_l, m_ext_l)

        # --- ρ DYNAMIQUE : estimation MLE saison courante ---
        # Lance en tâche de fond pour ne pas bloquer le scan si < 30 matchs.
        rho_mle = await estimer_rho_saison(ligue_cfg['id'], saison_utilisee, m_dom_l, m_ext_l)
        if rho_mle is not None:
            RHO_DYNAMIQUE[(ligue_cfg['id'], saison_utilisee)] = rho_mle

        cache_standings[cle] = {
            'data': resultat,
            'expire': datetime.now(timezone.utc) + timedelta(hours=12)  # UTC-aware, cohérent
        }
        return resultat
    except Exception as e:
        log_info(f"⚠️ {ligue_cfg['nom']} : exception dans actualiser_stats_ligue — {e}")
        return None

def generer_matrice_dixon(l_dom, l_ext, ligue_id=None, saison=None):
    """
    Génère la matrice Dixon-Coles.
    Priorité : ρ estimé dynamiquement (MLE saison courante) > ρ statique par ligue > défaut.
    Taille dynamique : couvre 99.8% de la masse Poisson (min 10, max 15) pour éviter
    la troncature sur les ligues à fort volume de buts (Bundesliga, Eredivisie, λ ≥ 3.5).
    """
    rho = (RHO_DYNAMIQUE.get((ligue_id, saison))
           or RHO_PAR_LIGUE.get(ligue_id, RHO_DEFAULT) if ligue_id
           else RHO_DEFAULT)
    max_goals = max(10, min(int(np.ceil(poisson.ppf(0.998, max(l_dom, l_ext)))) + 1, 15))
    p_d = [poisson.pmf(i, l_dom) for i in range(max_goals)]
    p_e = [poisson.pmf(i, l_ext) for i in range(max_goals)]
    m = np.outer(p_d, p_e).astype(float)
    m[0, 0] *= max(0, 1 - (l_dom * l_ext * rho))
    m[1, 0] *= max(0, 1 + (l_ext * rho))
    m[0, 1] *= max(0, 1 + (l_dom * rho))
    m[1, 1] *= max(0, 1 - rho)
    return m / np.sum(m)

_EPS_AH = 1e-6  # tolérance float pour quarts de handicap

def calculer_ev_ah(matrice, h, is_h, cote):
    esperance = 0.0
    n = matrice.shape[0]
    for i in range(n):
        for j in range(n):
            prob = matrice[i, j]
            if prob < 0.0001: continue
            score_diff = (i - j) if is_h else (j - i)
            res_net = score_diff + h

            if res_net > 0.25 + _EPS_AH:            payout = cote
            elif abs(res_net - 0.25) < _EPS_AH:     payout = 1.0 + (cote - 1.0) / 2
            elif abs(res_net) < _EPS_AH:             payout = 1.0
            elif abs(res_net + 0.25) < _EPS_AH:     payout = 0.5
            else:                                    payout = 0.0
            esperance += prob * payout
    return esperance - 1.0

def calculer_ev_total_asiatique(matrice, ligne, is_over, cote):
    """Calcule l'EV pour les Totaux Asiatiques (0.0, 0.25, 0.5, 0.75)."""
    esperance = 0.0
    n = matrice.shape[0]
    for i in range(n):
        for j in range(n):
            prob = matrice[i, j]
            if prob < 0.0001: continue

            total_match = i + j
            res_net = (total_match - ligne) if is_over else (ligne - total_match)

            if res_net > 0.25 + _EPS_AH:            payout = cote
            elif abs(res_net - 0.25) < _EPS_AH:     payout = 1.0 + (cote - 1.0) / 2
            elif abs(res_net) < _EPS_AH:             payout = 1.0
            elif abs(res_net + 0.25) < _EPS_AH:     payout = 0.5
            else:                                    payout = 0.0

            esperance += prob * payout
    return esperance - 1.0

def _calculer_cote_fair(ev_fn, cote_max=500.0):
    """
    Résout EV(cote)=0 par bisection.
    EV croît avec la cote : on cherche lo (EV<=0) et hi (EV>=0).
    """
    def ev_at(c):
        return ev_fn(max(c, 1.001))

    lo = 1.01
    if ev_at(lo) >= 0:
        return lo

    hi = 2.0
    while ev_at(hi) < 0:
        hi = min(hi * 1.5, cote_max)
        if hi >= cote_max and ev_at(hi) < 0:
            return cote_max

    for _ in range(80):
        mid = (lo + hi) / 2
        if ev_at(mid) <= 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def calculer_cote_fair_ah(matrice, h, is_h):
    """Cote décimale fair (EV=0) pour un pari handicap asiatique."""
    return _calculer_cote_fair(lambda c: calculer_ev_ah(matrice, h, is_h, c))


def calculer_cote_fair_total(matrice, ligne, is_over):
    """Cote décimale fair (EV=0) pour un total asiatique."""
    return _calculer_cote_fair(lambda c: calculer_ev_total_asiatique(matrice, ligne, is_over, c))


def calculer_prob_modele_pari(matrice, market_key, h, is_h_odds=None, is_over=None):
    """
    Probabilité implicite du modèle pour le pari (1 / cote_fair).
    Aligné sur le NHL bot (Vraie_Cote_Bot → p_model = 1/cote_fair).
    """
    if market_key == 'spreads':
        c_fair = calculer_cote_fair_ah(matrice, h, is_h_odds)
    else:
        c_fair = calculer_cote_fair_total(matrice, h, is_over)
    return min(max(1.0 / c_fair, 0.001), 0.999)

def calculer_kelly_ah(matrice, h, is_h, cote):
    """
    Kelly exact pour Asian Handicap via l'approximation mean-variance.
    Utilise les 5 issues réelles (gagné, demi-gagné, push, demi-perdu, perdu)
    au lieu de la formule binaire p=(EV+1)/cote qui sous-estime la taille.
    f* = E[X] / E[X²]
    """
    e_x, e_x2 = 0.0, 0.0
    n = matrice.shape[0]
    for i in range(n):
        for j in range(n):
            prob = matrice[i, j]
            if prob < 0.0001: continue
            score_diff = (i - j) if is_h else (j - i)
            res_net = score_diff + h
            if res_net > 0.25 + _EPS_AH:        x = cote - 1.0
            elif abs(res_net - 0.25) < _EPS_AH: x = (cote - 1.0) / 2.0
            elif abs(res_net) < _EPS_AH:         x = 0.0
            elif abs(res_net + 0.25) < _EPS_AH: x = -0.5
            else:                                x = -1.0
            e_x  += prob * x
            e_x2 += prob * x * x
    return e_x / e_x2 if e_x2 > 1e-9 else 0.0

def calculer_kelly_total(matrice, ligne, is_over, cote):
    """
    Kelly exact pour Total Asiatique via l'approximation mean-variance.
    f* = E[X] / E[X²]
    """
    e_x, e_x2 = 0.0, 0.0
    n = matrice.shape[0]
    for i in range(n):
        for j in range(n):
            prob = matrice[i, j]
            if prob < 0.0001: continue
            total_match = i + j
            res_net = (total_match - ligne) if is_over else (ligne - total_match)
            if res_net > 0.25 + _EPS_AH:        x = cote - 1.0
            elif abs(res_net - 0.25) < _EPS_AH: x = (cote - 1.0) / 2.0
            elif abs(res_net) < _EPS_AH:         x = 0.0
            elif abs(res_net + 0.25) < _EPS_AH: x = -0.5
            else:                                x = -1.0
            e_x  += prob * x
            e_x2 += prob * x * x
    return e_x / e_x2 if e_x2 > 1e-9 else 0.0

# ==========================================
# 🔄 6. BOUCLE PRINCIPALE
# ==========================================

def nettoyer_caches_memoire():
    """Vide la RAM en supprimant les vieilles données."""
    maintenant = datetime.now(timezone.utc)

    cles_matchs_a_supprimer = [k for k, v in cache_heures_matchs.items() if v < maintenant - timedelta(days=2)]
    for k in cles_matchs_a_supprimer:
        del cache_heures_matchs[k]

    # Expire est maintenant toujours UTC-aware (cohérent avec maintenant)
    cles_standings_a_supprimer = [k for k, v in cache_standings.items() if v['expire'] < maintenant]
    for k in cles_standings_a_supprimer:
        del cache_standings[k]

    # Purge RHO_DYNAMIQUE / DC_PARAMS : garder saison courante + N-1 uniquement
    annee = maintenant.year
    saison_courante = annee if maintenant.month >= 7 else annee - 1
    min_saison = saison_courante - 1
    cles_rho = [k for k in RHO_DYNAMIQUE if k[1] < min_saison]
    cles_dc  = [k for k in DC_PARAMS if k[1] < min_saison]
    for k in cles_rho:
        del RHO_DYNAMIQUE[k]
    for k in cles_dc:
        del DC_PARAMS[k]

    log_info(f"🧹 Nettoyage RAM : {len(cles_matchs_a_supprimer)} matchs, "
             f"{len(cles_standings_a_supprimer)} ligues, "
             f"{len(cles_rho)} ρ et {len(cles_dc)} DC params purgés.")

async def traiter_une_ligue(session, ligue) -> int:
    """Traite une ligue : récupère stats, cotes et matchs, lance les analyses.
    Retourne le nombre de matchs trouvés dans la fenêtre 7 jours."""
    saison_correcte = obtenir_saison_api(ligue['nom'])
    log_info(f"⌛ Analyse : {ligue['nom']} (Saison API: {saison_correcte})...")

    stats = await actualiser_stats_ligue(session, ligue, saison_correcte)
    if not stats:
        return 0

    url_cotes = f"https://api.the-odds-api.com/v4/sports/{ligue['key']}/odds/?apiKey={API_ODDS_KEY}&regions=eu&markets=spreads,totals"
    cotes = await fetch_async(session, url_cotes)
    date_deb = datetime.now().strftime('%Y-%m-%d')
    date_fin = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    matchs = await fetch_async(session, f"{URL_FOOTBALL}/fixtures?league={ligue['id']}&season={saison_correcte}&from={date_deb}&to={date_fin}&timezone=Europe/Paris", HEADERS_FB)

    n_matchs = len(matchs.get('response', [])) if matchs else 0
    if n_matchs:
        log_info(f"🔎 {ligue['nom']} : {n_matchs} match(s) dans les 7 prochains jours.")

    if isinstance(cotes, list) and matchs and 'response' in matchs:
        # Traitement par lots de 5 matchs pour limiter les pics de requêtes API
        # et préserver le quota journalier (100 req/j sur plan gratuit).
        MATCH_BATCH = 5
        all_matchs = matchs['response']
        for i in range(0, len(all_matchs), MATCH_BATCH):
            chunk = all_matchs[i:i + MATCH_BATCH]
            chunk_tasks = [analyser_un_match(session, m, ligue, saison_correcte, *stats, cotes) for m in chunk]
            results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    log_info(f"⚠️ Match ignoré suite à une erreur de donnée : {res}")
            if i + MATCH_BATCH < len(all_matchs):
                await asyncio.sleep(2)  # Respiration entre lots

    return n_matchs

async def tracker_clv_async(session):
    """
    Tracker CLV dédié — tourne à chaque cycle du scan.

    Pour chaque pari PENDING :
      1. Re-fetch les cotes Pinnacle actuelles (groupées par ligue pour économiser les appels API)
      2. Met à jour cote_cloture et clv en continu
      3. Envoie une alerte Telegram UNIQUE quand le match entre dans la fenêtre H-60
         (clv_notifie passe de 0 à 1 pour éviter le spam)

    CLV = (cote_prise / cote_clôture) - 1
      > 0 : on a pris des cotes meilleures que le marché final → edge confirmé
      < 0 : le marché a remonté → signal d'erreur de modèle
    """
    async with db_lock:
        async with db_conn.execute(
            """SELECT id_match, equipe, handicap, cote_prise, ligue,
                      equipe_dom, equipe_ext, kickoff, clv_notifie
               FROM paris_log WHERE statut='PENDING'"""
        ) as cursor:
            pending = await cursor.fetchall()

    if not pending:
        return

    now = datetime.now(timezone.utc)

    # Groupe par (sport_key, market_key) → 1 appel Odds API par ligue au lieu de N
    par_ligue: dict = {}
    for row in pending:
        id_m, equipe, handicap, cote_prise, ligue_tag, eq_dom, eq_ext, kickoff, clv_notifie = row
        if not eq_dom or not eq_ext:
            continue  # Paris anciens sans métadonnées — ignorés

        # Vérifier que le match n'a pas déjà commencé (on ne track plus après KO)
        if kickoff:
            try:
                ko_dt = datetime.fromisoformat(kickoff.replace('Z', '+00:00'))
                if now > ko_dt + timedelta(minutes=10):
                    continue
            except Exception:
                pass

        market_key = 'spreads' if '[spreads]' in ligue_tag else 'totals'
        ligue_nom = ligue_tag.split(' [')[0]
        ligue_cfg = next((l for l in CHAMPIONNATS if l['nom'] == ligue_nom), None)
        if not ligue_cfg:
            continue

        cle = (ligue_cfg['key'], market_key)
        if cle not in par_ligue:
            par_ligue[cle] = {'cfg': ligue_cfg, 'market_key': market_key, 'bets': []}
        par_ligue[cle]['bets'].append(
            (id_m, equipe, handicap, cote_prise, eq_dom, eq_ext, kickoff, clv_notifie)
        )

    mises_a_jour = 0
    for (sport_key, market_key), info in par_ligue.items():
        ligue_cfg = info['cfg']
        url_odds = (f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
                    f"?apiKey={API_ODDS_KEY}&bookmakers=pinnacle"
                    f"&markets={market_key}&oddsFormat=decimal")
        data = await fetch_async(session, url_odds, {})
        if not data or not isinstance(data, list):
            continue

        for (id_m, equipe, handicap, cote_prise, eq_dom, eq_ext, kickoff, clv_notifie) in info['bets']:
            # Trouver l'événement Pinnacle par fuzzy-match equipe_dom / equipe_ext
            event = next(
                (e for e in data
                 if ((process.extractOne(eq_dom, [e['home_team']])[1] +
                      process.extractOne(eq_ext, [e['away_team']])[1]) / 2) > 80),
                None
            )
            if not event:
                continue

            pinnacle = next((b for b in event.get('bookmakers', []) if b['key'] == 'pinnacle'), None)
            if not pinnacle:
                continue

            market = next((mkt for mkt in pinnacle['markets'] if mkt['key'] == market_key), None)
            if not market:
                continue

            # Outcome exact : même nom (équipe ou Over/Under) et même handicap
            outcome = next(
                (o for o in market['outcomes']
                 if o['name'] == equipe and abs(float(o.get('point', 0)) - handicap) < 0.01),
                None
            )
            if not outcome:
                continue

            cote_actuelle = float(outcome['price'])
            clv_val = round((cote_prise / cote_actuelle) - 1, 4)

            # Déterminer si une alerte CLV doit être envoyée
            # Deux fenêtres :
            #   • H-60 : pré-clôture (XI officiel connu, Pinnacle commence à serrer)
            #   • H-5  : vraie closing line de référence (marché au plus sharp)
            envoyer_alerte = False
            label_alerte = ""
            mins_avant = None

            if kickoff:
                try:
                    ko_dt = datetime.fromisoformat(kickoff.replace('Z', '+00:00'))
                    mins_avant = (ko_dt - now).total_seconds() / 60

                    if clv_notifie == 0 and 55 <= mins_avant <= 65:
                        envoyer_alerte = True
                        label_alerte = "⏱️ Pré-clôture H-60 (XI confirmé)"
                    elif clv_notifie == 1 and 3 <= mins_avant <= 8:
                        # Deuxième alerte : vraie closing line
                        envoyer_alerte = True
                        label_alerte = "🔒 Closing line H-5 (référence sharp)"
                except Exception:
                    pass

            async with db_lock:
                await db_conn.execute(
                    """UPDATE paris_log SET cote_cloture=?, clv=?, clv_notifie=?
                       WHERE id_match=? AND equipe=? AND handicap=?""",
                    (cote_actuelle, clv_val,
                     2 if (envoyer_alerte and clv_notifie == 1) else (1 if envoyer_alerte else clv_notifie),
                     id_m, equipe, handicap)
                )
                await db_conn.commit()
            mises_a_jour += 1

            if envoyer_alerte:
                clv_emoji = "🟢" if clv_val > 0 else "🔴"
                pct = f"{clv_val:+.1%}"
                await envoyer_telegram_async(session,
                    f"📊 *CLV — {ligue_cfg['nom']}*\n"
                    f"⚽ {eq_dom} vs {eq_ext}\n"
                    f"💎 *{equipe} ({handicap:+.1f})* pris @ {cote_prise:.2f}\n"
                    f"📉 Cote actuelle Pinnacle : *{cote_actuelle:.2f}*\n"
                    f"{clv_emoji} CLV : *{pct}*\n"
                    f"_{label_alerte}_"
                )

    if mises_a_jour:
        log_info(f"[CLV] {mises_a_jour} paris mis à jour.")


async def lancer_scan_global_async() -> int:
    """Retourne le nombre de matchs détectés dans la fenêtre 0-24h."""
    log_info(f"--- DÉBUT CYCLE ({datetime.now().strftime('%H:%M')}) ---")
    nettoyer_caches_memoire()

    matchs_detectes = 0
    async with aiohttp.ClientSession() as session:
        await actualiser_kelly_adaptatif()        # 📐 Kelly dynamique selon drawdown
        await verifier_resultats_matchs(session)
        await tracker_clv_async(session)          # CLV proactif avant le scan des ligues

        # Traitement par lots de 4 ligues en parallèle (réduit le dead-wait de 160s à ~40s)
        BATCH_SIZE = 4
        for i in range(0, len(CHAMPIONNATS), BATCH_SIZE):
            batch = CHAMPIONNATS[i:i+BATCH_SIZE]
            resultats = await asyncio.gather(*[traiter_une_ligue(session, l) for l in batch])
            matchs_detectes += sum(r or 0 for r in resultats)
            await asyncio.sleep(10)  # Rate-limit entre les batches

        await exporter_historique_csv()

    return matchs_detectes


def calculer_pause(matchs_detectes: int) -> int:
    """
    Durée de pause adaptative selon l'activité détectée.

    Logique :
      • Matchs dans les 2h     →  5 min  (mode actif, suivi rapproché)
      • Matchs dans 2-24h      → 15 min  (mode veille, scan toutes les 15 min)
      • Aucun match aujourd'hui → 60 min  (mode économie, limite API préservée)
    """
    now_utc = datetime.now(timezone.utc)
    prochains = []
    for dt in cache_heures_matchs.values():
        # Normaliser en UTC-aware pour la comparaison
        dt_aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        if dt_aware > now_utc:
            prochains.append(dt_aware)

    if prochains:
        mins_prochain = (min(prochains) - now_utc).total_seconds() / 60
        if mins_prochain <= 120:
            return 300      # 5 min
        if mins_prochain <= 1440:
            return 900      # 15 min
    if matchs_detectes > 0:
        return 900          # 15 min
    return 3600             # 60 min


async def main_loop():
    global semaphore, db_lock
    semaphore = asyncio.Semaphore(5)
    db_lock = asyncio.Lock()

    await init_db()

    log_info("🚀 Démarrage du Superviseur Sniper Football V25 Gold (aiosqlite)...")

    # Détection initiale de la couverture xG + collecte historique des scores
    dernier_retest_xg = None
    dernier_collecte_scores = None
    dernier_dc = None
    dernier_backup = None
    async with aiohttp.ClientSession() as session:
        await detecter_ligues_sans_xg(session)
        for lg in CHAMPIONNATS:
            await collecter_scores_historiques(session, lg, obtenir_saison_api(lg['nom']))
    dernier_retest_xg = datetime.now(timezone.utc)
    dernier_collecte_scores = datetime.now(timezone.utc)

    while True:
        try:
            maintenant = datetime.now(timezone.utc)

            # Backup quotidien de la DB (rotation sur 7 jours)
            if dernier_backup is None or (maintenant - dernier_backup).total_seconds() > 86400:
                try:
                    backup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backups")
                    os.makedirs(backup_dir, exist_ok=True)
                    date_str = maintenant.strftime("%Y-%m-%d")
                    dest = os.path.join(backup_dir, f"sniper_data_{date_str}.db")
                    # Flush aiosqlite avant la copie
                    async with db_lock:
                        await db_conn.commit()
                    shutil.copy2("sniper_data.db", dest)
                    # Rotation : garder seulement les 7 derniers backups
                    backups = sorted(
                        [f for f in os.listdir(backup_dir) if f.startswith("sniper_data_") and f.endswith(".db")]
                    )
                    for old in backups[:-7]:
                        os.remove(os.path.join(backup_dir, old))
                    log_info(f"💾 Backup DB : {dest} ({len(backups)} fichiers, rotation 7j).")
                except Exception as e:
                    log_info(f"⚠️ Backup DB échoué : {e}")
                dernier_backup = maintenant

            # Retest quotidien xG : si une ligue commence à fournir des xG en cours de saison,
            # le bot le détecte sans redémarrage.
            if dernier_retest_xg is None or (maintenant - dernier_retest_xg).total_seconds() > 86400:
                async with aiohttp.ClientSession() as session:
                    await detecter_ligues_sans_xg(session)
                dernier_retest_xg = maintenant
                log_info("🔄 Retest couverture xG par ligue effectué.")

            # Collecte quotidienne des scores FT pour alimenter l'estimation MLE de ρ
            if dernier_collecte_scores is None or (maintenant - dernier_collecte_scores).total_seconds() > 86400:
                async with aiohttp.ClientSession() as session:
                    for lg in CHAMPIONNATS:
                        await collecter_scores_historiques(session, lg, obtenir_saison_api(lg['nom']))
                dernier_collecte_scores = maintenant
                log_info("📊 Collecte quotidienne des scores FT terminée.")

            # Estimation quotidienne des paramètres Dixon-Coles complets (α, β, γ, ρ par équipe)
            if dernier_dc is None or (maintenant - dernier_dc).total_seconds() > 86400:
                async with aiohttp.ClientSession() as session:
                    for lg in CHAMPIONNATS:
                        saison_lg = obtenir_saison_api(lg['nom'])
                        stats_lg  = await actualiser_stats_ligue(session, lg, saison_lg)
                        if stats_lg:
                            _, _, _, _, mu_h_lg, mu_a_lg = stats_lg
                            dc = await estimer_parametres_dc_complet(
                                lg['id'], saison_lg, mu_h_lg, mu_a_lg)
                            if dc:
                                DC_PARAMS[(lg['id'], saison_lg)] = dc
                                RHO_DYNAMIQUE[(lg['id'], saison_lg)] = dc['rho']
                dernier_dc = maintenant
                log_info("🧮 Paramètres DC complets recalculés pour toutes les ligues.")

            matchs = await lancer_scan_global_async()
            pause = calculer_pause(matchs)
            pause_label = f"{pause//60} min" if pause >= 60 else f"{pause}s"
            log_info(f"😴 Pause adaptative : {pause_label} "
                     f"({'mode actif' if pause == 300 else 'mode veille' if pause == 900 else 'mode économie'}).")
            await asyncio.sleep(pause)
        except Exception as e:
            log_info(f"⚠️ Erreur Critique : {e}")
            await asyncio.sleep(60)

async def rapport_calibration_foot(min_paris: int = 5):
    """Diagnostic Brier + calibration sur paris_log (sans impact live)."""
    from utils import formater_rapport_calibration_texte, preparer_calibration_foot
    import pandas as pd

    global db_lock, db_conn
    if db_conn is None:
        db_lock = asyncio.Lock()
        await init_db()

    async with db_lock:
        async with db_conn.execute(
            """SELECT p_modele AS Prob_Modele, statut AS Statut, edge_detecte AS Edge
               FROM paris_log
               WHERE statut IN ('WON','HALF-WON','LOST','HALF-LOST')
               ORDER BY timestamp ASC"""
        ) as cursor:
            rows = await cursor.fetchall()
            cols = [d[0] for d in cursor.description]

    df_cal = preparer_calibration_foot(pd.DataFrame(rows, columns=cols))
    print(formater_rapport_calibration_texte(
        df_cal, min_paris=min_paris, titre="CALIBRATION FOOT"
    ))


async def _graceful_shutdown(sig_name: str):
    """Ferme la connexion DB proprement avant de quitter."""
    log_info(f"🛑 Signal {sig_name} reçu — fermeture propre en cours...")
    if db_conn:
        await db_conn.close()
        log_info("💾 Connexion DB fermée.")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    import sys
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if len(sys.argv) > 1 and sys.argv[1] in ("--calibration", "--calib"):
        try:
            loop.run_until_complete(rapport_calibration_foot())
        finally:
            loop.close()
        raise SystemExit(0)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: asyncio.ensure_future(_graceful_shutdown(s.name)))
        except NotImplementedError:
            pass  # Windows ne supporte pas add_signal_handler — shutdown propre ignoré
    try:
        loop.run_until_complete(main_loop())
    except KeyboardInterrupt:
        # Sur Windows, add_signal_handler n'est pas supporté → fallback ici
        loop.run_until_complete(_graceful_shutdown("SIGINT"))
    finally:
        loop.close()
