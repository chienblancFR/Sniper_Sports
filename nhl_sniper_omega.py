import copy
import requests
import csv
import math
import time
import os
from datetime import datetime, timedelta
from io import StringIO
import traceback
from scipy.optimize import minimize
from dotenv import load_dotenv

load_dotenv("identifiants_différent_api.env")
load_dotenv()

# ==========================================
# ⚙️ CONFIGURATION GLOBALE
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
FICHIER_MEMOIRE = "alertes_nhl_envoyees.txt"
FICHIER_JOURNAL = "journal_trading_nhl_SEC2026xOmG.csv"
BANKROLL_INITIALE = float(os.environ.get("NHL_BANKROLL", "1000.0"))
NHL_SEASON = int(os.environ.get("NHL_SEASON", "2026"))
EDGE_MINIMUM = float(os.environ.get("NHL_EDGE_MIN", "0.02"))
KELLY_FRACTION = float(os.environ.get("NHL_KELLY_FRACTION", "0.25"))
KELLY_FRACTION_GARDIEN_INCERTAIN = float(os.environ.get("NHL_KELLY_GARDIEN", "0.125"))
FLOAT_TOL = 0.01
# Part du temps de jeu gardien (~54 min) pour convertir GSAx/60 → impact/match
GSAX_MINUTES_PAR_MATCH = float(os.environ.get("NHL_GSAX_MINUTES", "54.0"))

# Mappage des abréviations (NHL API) vers noms complets (The-Odds-API)
NHL_TEAMS_MAPPING = {
    "ANA": "Anaheim Ducks", "UTA": "Utah Hockey Club",
    "BOS": "Boston Bruins", "BUF": "Buffalo Sabres", "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars", "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers", "FLA": "Florida Panthers", "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens", "NSH": "Nashville Predators",
    "NJD": "New Jersey Devils", "NYI": "New York Islanders", "NYR": "New York Rangers",
    "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins",
    "SJS": "San Jose Sharks", "SEA": "Seattle Kraken", "STL": "St Louis Blues",
    "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs", "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights", "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}

# Alias The-Odds-API (noms alternatifs Pinnacle → clé interne)
ODDS_API_ALIASES = {
    "Montreal Canadiens": ["Montréal Canadiens"],
    "Utah Hockey Club": ["Utah Mammoth"],
}
# Index inverse : alias → nom primaire
_ODDS_NOM_PRIMAIRE = {}
for _primaire, _alias_liste in ODDS_API_ALIASES.items():
    for _alias in _alias_liste:
        _ODDS_NOM_PRIMAIRE[_alias] = _primaire

# Carte des fuseaux horaires (0 = EST, 1 = CENTRAL, 2 = MOUNTAIN, 3 = PACIFIC)
NHL_TIMEZONES = {
    "BOS": 0, "BUF": 0, "CAR": 0, "CBJ": 0, "DET": 0, "FLA": 0, "MTL": 0, "NJD": 0,
    "NYI": 0, "NYR": 0, "OTT": 0, "PHI": 0, "PIT": 0, "TBL": 0, "TOR": 0, "WSH": 0,
    "CHI": 1, "DAL": 1, "MIN": 1, "NSH": 1, "STL": 1, "WPG": 1,
    "COL": 2, "UTA": 2, "CGY": 2, "EDM": 2,
    "ANA": 3, "LAK": 3, "SJS": 3, "SEA": 3, "VAN": 3, "VGK": 3
}

# ==========================================
# 1. ASPIRATEURS DE DONNÉES (MoneyPuck)
# ==========================================
def _float_proche(a, b, tol=FLOAT_TOL):
    return abs(float(a) - float(b)) < tol


def _arrondir_cut(point):
    """Arrondit une ligne O/U au demi-point le plus proche (5.5, 6.0…)."""
    return round(float(point) * 2) / 2


def _trouver_cle_float(dictionnaire, cible):
    """Retourne (cle, valeur) en tolérant les imprécisions float."""
    for cle, val in dictionnaire.items():
        if _float_proche(cle, cible):
            return cle, val
    return None, None


def _fetch_moneypuck_csv(kind, season):
    """Télécharge un CSV MoneyPuck (saison N, puis N-1 si échec)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    for s in (season, season - 1):
        url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{s}/regular/{kind}.csv"
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200 and response.text.strip():
                return response.text, s
            print(f"⚠️ MoneyPuck {kind} saison {s} : HTTP {response.status_code}")
        except Exception as e:
            print(f"⚠️ MoneyPuck {kind} saison {s} : {e}")
    return None, None


def get_team_stats(season=None):
    if season is None:
        season = NHL_SEASON
    """Aspire les xG à 5 contre 5 et les Unités Spéciales (PP/PK)."""
    try:
        texte, saison_utilisee = _fetch_moneypuck_csv("teams", season)
        if not texte:
            return []
        if saison_utilisee != season:
            print(f"ℹ️ MoneyPuck équipes : repli sur la saison {saison_utilisee}")
        csv_reader = csv.DictReader(StringIO(texte))
        teams_dict = {}
        for row in csv_reader:
            team, sit = row["team"], row.get("situation")
            if team not in teams_dict:
                teams_dict[team] = {
                    "team": team, "xGF_per_game": 0.0, "xGA_per_game": 0.0,
                    "xGF_PP": 0.0, "xGA_PK": 0.0,
                }
            gp = max(float(row.get("games_played", 1)), 1)
            if sit == "5on5":
                teams_dict[team]["xGF_per_game"] = round(float(row.get("xGoalsFor", 0)) / gp, 3)
                teams_dict[team]["xGA_per_game"] = round(float(row.get("xGoalsAgainst", 0)) / gp, 3)
            elif sit == "5on4":
                teams_dict[team]["xGF_PP"] = round(float(row.get("xGoalsFor", 0)) / gp, 3)
            elif sit == "4on5":
                teams_dict[team]["xGA_PK"] = round(float(row.get("xGoalsAgainst", 0)) / gp, 3)
        return list(teams_dict.values())
    except Exception as e:
        print(f"⚠️ Erreur extracteur équipes: {e}")
        return []


def get_goalie_stats(season=None):
    if season is None:
        season = NHL_SEASON
    """Aspire les performances avancées des gardiens (GSAx)."""
    try:
        texte, saison_utilisee = _fetch_moneypuck_csv("goalies", season)
        if not texte:
            return []
        if saison_utilisee != season:
            print(f"ℹ️ MoneyPuck gardiens : repli sur la saison {saison_utilisee}")
        csv_reader = csv.DictReader(StringIO(texte))
        goalies = []
        for row in csv_reader:
            if row.get("situation") == "all":
                if float(row.get("games_played", 0)) < 5:
                    continue
                games_eq = max(float(row.get("icetime", 1)) / 3600, 1)
                gsax_per_60 = round(
                    (float(row.get("xGoals", 0)) - float(row.get("goals", 0))) / games_eq, 3
                )
                goalies.append({"name": row["name"], "gsax_per_60": gsax_per_60})
        return goalies
    except Exception as e:
        print(f"⚠️ Erreur extracteur gardiens: {e}")
        return []


def get_stars_impact(season=None):
    if season is None:
        season = NHL_SEASON
    """Génère le Top 30 des patineurs les plus cruciaux de la ligue."""
    try:
        texte, saison_utilisee = _fetch_moneypuck_csv("skaters", season)
        if not texte:
            return {}
        if saison_utilisee != season:
            print(f"ℹ️ MoneyPuck patineurs : repli sur la saison {saison_utilisee}")
        csv_reader = csv.DictReader(StringIO(texte))
        skaters = []
        for row in csv_reader:
            if row.get("situation") == "5on5":
                gp = float(row.get("games_played", 0))
                if gp < 10:
                    continue
                skaters.append({
                    "name": row["name"], "team": row["team"],
                    "game_score_per_game": float(row.get("gameScore", 0)) / gp,
                })
        skaters_tries = sorted(skaters, key=lambda x: x["game_score_per_game"], reverse=True)[:30]
        stars_impact = {}
        for rank, player in enumerate(skaters_tries):
            if rank < 5:
                xgf_pen, xga_pen = 0.12, 0.05
            elif rank < 15:
                xgf_pen, xga_pen = 0.08, 0.03
            else:
                xgf_pen, xga_pen = 0.05, 0.02
            stars_impact[player["name"]] = {
                "team": player["team"], "xgf_penalty": xgf_pen, "xga_penalty": xga_pen,
            }
        return stars_impact
    except Exception as e:
        print(f"⚠️ Erreur extracteur stars: {e}")
        return {}

# ==========================================
# 2. RADAR API OFFICIEL NHL & MOMENTUM
# ==========================================
def get_nhl_games_today():
    date_str = datetime.now().strftime("%Y-%m-%d")
    url = f"https://api-web.nhle.com/v1/score/{date_str}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return []
        data = response.json()
        if "games" not in data: return []
        matchs = []
        for game in data["games"]:
            if game["gameState"] in ["FUT", "PRE"]:
                matchs.append({'game_id': game["id"], 'away_team': game["awayTeam"]["abbrev"], 'home_team': game["homeTeam"]["abbrev"]})
        return matchs
    except: return []

def get_active_rosters(game_id):
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return None, None, [], []
        data = response.json()
        away_goalies = data.get("playerByGameStats", {}).get("awayTeam", {}).get("goalies", [])
        home_goalies = data.get("playerByGameStats", {}).get("homeTeam", {}).get("goalies", [])
        away_starter = away_goalies[0]["name"]["default"] if away_goalies else None
        home_starter = home_goalies[0]["name"]["default"] if home_goalies else None
        away_skaters, home_skaters = [], []
        for pos in ["forwards", "defense"]:
            for player in data.get("playerByGameStats", {}).get("awayTeam", {}).get(pos, []): away_skaters.append(player["name"]["default"])
            for player in data.get("playerByGameStats", {}).get("homeTeam", {}).get(pos, []): home_skaters.append(player["name"]["default"])
        return away_starter, home_starter, away_skaters, home_skaters
    except: return None, None, [], []

def get_nhl_momentum():
    """Calcule le différentiel de forme L10 (Momentum)."""
    url = "https://api-web.nhle.com/v1/standings/now"
    headers = {'User-Agent': 'Mozilla/5.0'}
    momentum_dict = {}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return {}
        data = response.json()
        for team in data.get("standings", []):
            abbrev = team.get("teamAbbrev", {}).get("default")
            gp = team.get("gamesPlayed", 1)
            if gp < 10:
                momentum_dict[abbrev] = 0.0
                continue
            season_pts_pct = team.get("points", 0) / (gp * 2.0)
            l10_wins = team.get("l10Wins", 0)
            l10_ot_losses = team.get("l10OtLosses", 0)
            l10_pts_pct = ((l10_wins * 2) + l10_ot_losses) / 20.0
            delta_form = l10_pts_pct - season_pts_pct
            momentum_dict[abbrev] = round(delta_form * 0.15, 3) # Modérateur conservateur
        return momentum_dict
    except: return {}

def get_goalie_confirmation_status(game_id):
    """
    Explore le Landing Page du match pour détecter si les gardiens
    sont validés ou simplement pressentis.
    Retourner un dictionnaire avec le statut de confirmation.
    """
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/landing"
    headers = {'User-Agent': 'Mozilla/5.0'}
    status = {"away_confirmed": False, "home_team_confirmed": False}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return status
        data = response.json()

        # On fouille l'aperçu du match (Match Preview / Summary)
        game_info = data.get("gameInfo", {})

        # L'API NHL utilise le drapeau 'isStarter' ou place le gardien dans
        # la section 'confirmedStarters' lorsqu'un journaliste officiel valide l'info.
        away_goalis_list = data.get("awayTeam", {}).get("probableGoalies", [])
        home_goalis_list = data.get("homeTeam", {}).get("probableGoalies", [])

        if away_goalis_list and away_goalis_list[0].get("status") == "CONFIRMED":
            status["away_confirmed"] = True
        if home_goalis_list and home_goalis_list[0].get("status") == "CONFIRMED":
            status["home_team_confirmed"] = True

        return status
    except:
        return status

def get_teams_played_yesterday():
    """Récupère la liste des abréviations des équipes qui ont joué hier."""
    hier = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    url = f"https://api-web.nhle.com/v1/score/{hier}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    teams_played = set()

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return teams_played
        data = response.json()
        if "games" not in data: return teams_played

        for game in data["games"]:
            teams_played.add(game["awayTeam"]["abbrev"])
            teams_played.add(game["homeTeam"]["abbrev"])
        return teams_played
    except:
        return teams_played

# ==========================================
# 3. UTILITAIRES D'IDENTIFICATION
# ==========================================
def formater_nom(nom_complet):
    parties = nom_complet.split(" ", 1)
    if len(parties) == 2:
        return f"{parties[0][0]}. {parties[1]}"
    return nom_complet


def normaliser_nom_joueur(nom):
    """Clé de comparaison MoneyPuck (nom complet) <-> NHL API (F. Nom)."""
    nom = str(nom).strip()
    parties = nom.split(" ", 1)
    if len(parties) == 2 and parties[0].endswith(".") and len(parties[0]) <= 3:
        return f"{parties[0][0].lower()}.{parties[1].lower()}"
    if len(parties) == 2:
        return f"{parties[0][0].lower()}.{parties[1].lower()}"
    return nom.lower()


def joueur_present(nom_reference, roster_nhl):
    cle = normaliser_nom_joueur(nom_reference)
    return any(normaliser_nom_joueur(n) == cle for n in roster_nhl)


def detecter_stars_absentes(team_abbr, roster_nhl, stars_vip):
    return [
        nom for nom, info in stars_vip.items()
        if info["team"] == team_abbr and not joueur_present(nom, roster_nhl)
    ]


def trouver_gsax(nom_nhl, goalies_data):
    for g in goalies_data:
        if formater_nom(g['name']) == nom_nhl or joueur_present(g['name'], [nom_nhl]):
            return g['gsax_per_60']
    return 0.0


def gsax_per_60_vers_lambda(gsax_per_60):
    """
    Convertit GSAx/60 MoneyPuck en ajustement lambda buts/match.
    GSAx > 0 = gardien performant → réduit les buts encaissés par l'équipe adverse.
    """
    return gsax_per_60 * (GSAX_MINUTES_PAR_MATCH / 60.0)

def apply_star_absence_penalty(team_name, base_xgf, base_xga, missing_players_list, stars_dict):
    adjusted_xgf, adjusted_xga = base_xgf, base_xga
    for player in missing_players_list:
        if player in stars_dict and stars_dict[player]["team"] == team_name:
            print(f"🚨 ABSENCE DE MARQUE : {player} est forfait pour {team_name} !")
            adjusted_xgf *= (1.0 - stars_dict[player]["xgf_penalty"])
            adjusted_xga *= (1.0 + stars_dict[player]["xga_penalty"])
    return round(adjusted_xgf, 3), round(adjusted_xga, 3)

def calculate_circadian_fatigue(team_abbrev, opponent_abbrev, is_b2b, is_home):
    """
    Calcule le modificateur de fatigue basé sur le décalage horaire et le B2B.
    Retourne (modificateur_attaque, modificateur_defense).
    1.0 = Forme normale. < 1.0 = Baisse d'attaque. > 1.0 = Hausse des erreurs défensives.
    """
    # Pénalité de base pour un Back-to-Back classique (ce qu'on avait avant)
    mod_atk, mod_def = (0.95, 1.10) if is_b2b else (1.0, 1.0)

    # Si l'équipe joue à domicile, on considère qu'elle est "acclimatée" à son fuseau
    if is_home:
        return mod_atk, mod_def

    tz_team = NHL_TIMEZONES.get(team_abbrev, 0)
    tz_opp = NHL_TIMEZONES.get(opponent_abbrev, 0)

    # Calcul du décalage horaire absolu traversé
    decalage = abs(tz_team - tz_opp)

    if decalage >= 2:
        # Jet Lag modéré à sévère (ex: Est vers Montagne ou Pacifique)
        if is_b2b:
            # Le pire scénario possible : B2B + Jet Lag (Schedule Loss)
            mod_atk *= 0.93  # L'attaque s'effondre
            mod_def *= 1.15  # La défense craque complètement (+15% de xGA)
        else:
            # Juste le Jet Lag sans B2B
            mod_atk *= 0.98
            mod_def *= 1.04

    elif decalage == 1 and is_b2b:
        # Petit décalage mais en B2B (ex: Chicago joue à New York le lendemain)
        mod_atk *= 0.97
        mod_def *= 1.05

    return round(mod_atk, 3), round(mod_def, 3)

# ==========================================
# 4. MOTEUR MATHÉMATIQUE & INTELLIGENCE
# ==========================================
def poisson(lam, k):
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def tau_dixon_coles(lam_home, lam_away, h, a, rho=-0.12):
    if h == 0 and a == 0: return max(0, 1 - (lam_home * lam_away * rho))
    elif h == 1 and a == 0: return max(0, 1 + (lam_home * rho))
    elif h == 0 and a == 1: return max(0, 1 + (lam_away * rho))
    elif h == 1 and a == 1: return max(0, 1 - rho)
    return 1.0

def log_likelihood_rho(rho_array, historique_matchs):
    rho = rho_array[0]
    ll = 0.0
    for match in historique_matchs:
        h_goals, a_goals = match['vrai_score_domicile'], match['vrai_score_exterieur']
        lam_h, lam_a = match['lambda_domicile_calcule'], match['lambda_exterieur_calcule']
        p_h = (math.exp(-lam_h) * (lam_h ** h_goals)) / math.factorial(h_goals)
        p_a = (math.exp(-lam_a) * (lam_a ** a_goals)) / math.factorial(a_goals)
        tau = tau_dixon_coles(lam_h, lam_a, h_goals, a_goals, rho)
        ll += math.log(max(p_h * p_a * tau, 1e-10))
    return -ll

def optimiser_rho_saison(historique_matchs):
    """Calibre le Rho par Maximum de Vraisemblance (MLE)."""
    print(f"🔬 Début du calibrage MLE sur {len(historique_matchs)} matchs...")
    resultat = minimize(log_likelihood_rho, [-0.12], args=(historique_matchs,), bounds=[(-0.30, 0.05)], method='L-BFGS-B')
    meilleur_rho = round(resultat.x[0], 4)
    print(f"✅ Calibrage terminé. Le nouveau Rho optimal est : {meilleur_rho}")
    return meilleur_rho

def calculate_master_odds_v4(teams_data, home_team, away_team, home_gsax, away_gsax, mom_home=0.0, mom_away=0.0, home_is_b2b=False, away_is_b2b=False, rho=-0.12):
    league_avg_5v5 = sum(t['xGF_per_game'] for t in teams_data) / len(teams_data)
    league_avg_pp = sum(t['xGF_PP'] for t in teams_data) / len(teams_data)
    safe_league_pp = max(league_avg_pp, 0.01)

    home = next((t for t in teams_data if t['team'] == home_team), None)
    away = next((t for t in teams_data if t['team'] == away_team), None)
    if not home or not away: return None

    # HIA + Momentum
    home_xgf = home['xGF_per_game'] * 1.05 * (1.0 + mom_home)
    home_xga = home['xGA_per_game'] * 0.95 * (1.0 - mom_home)
    away_xgf = away['xGF_per_game'] * (1.0 + mom_away)
    away_xga = away['xGA_per_game'] * (1.0 - mom_away)

    home_pp_att = home['xGF_PP'] * (1.0 + mom_home)
    home_pk_def = home['xGA_PK'] * (1.0 - mom_home)
    away_pp_att = away['xGF_PP'] * (1.0 + mom_away)
    away_pk_def = away['xGA_PK'] * (1.0 - mom_away)

    # Matrice de Fatigue Circadienne et Kilométrique
    home_fatigue_atk, home_fatigue_def = calculate_circadian_fatigue(home_team, away_team, home_is_b2b, is_home=True)
    away_fatigue_atk, away_fatigue_def = calculate_circadian_fatigue(away_team, home_team, away_is_b2b, is_home=False)

    home_xgf *= home_fatigue_atk
    home_xga *= home_fatigue_def
    home_pp_att *= home_fatigue_atk
    home_pk_def *= home_fatigue_def

    away_xgf *= away_fatigue_atk
    away_xga *= away_fatigue_def
    away_pp_att *= away_fatigue_atk
    away_pk_def *= away_fatigue_def

    lam_home_5v5 = (home_xgf / league_avg_5v5) * (away_xga / league_avg_5v5) * league_avg_5v5
    lam_away_5v5 = (away_xgf / league_avg_5v5) * (home_xga / league_avg_5v5) * league_avg_5v5
    lam_home_pp = (home_pp_att / safe_league_pp) * (away_pk_def / safe_league_pp) * safe_league_pp
    lam_away_pp = (away_pp_att / safe_league_pp) * (home_pk_def / safe_league_pp) * safe_league_pp

    adj_gsax_away = gsax_per_60_vers_lambda(away_gsax)
    adj_gsax_home = gsax_per_60_vers_lambda(home_gsax)
    final_lam_home = max(lam_home_5v5 + lam_home_pp - adj_gsax_away, 0.1)
    final_lam_away = max(lam_away_5v5 + lam_away_pp - adj_gsax_home, 0.1)

    # Matrice initiale Poisson + Dixon-Coles
    matrice_brute = [[0.0]*12 for _ in range(12)]
    for h in range(12):
        for a in range(12):
            matrice_brute[h][a] = poisson(final_lam_home, h) * poisson(final_lam_away, a) * tau_dixon_coles(final_lam_home, final_lam_away, h, a, rho)

    # Correction Surdispersion (Empty Net)
    matrice_finale = [[0.0]*12 for _ in range(12)]
    prob_tie, prob_en = 0.12, 0.22

    for h in range(12):
        for a in range(12):
            p = matrice_brute[h][a]
            if p == 0: continue
            if h - a == 1 and h < 11 and a < 11:
                matrice_finale[h][a] += p - (p*prob_tie + p*prob_en)
                matrice_finale[h][a+1] += p*prob_tie
                matrice_finale[h+1][a] += p*prob_en
            elif a - h == 1 and h < 11 and a < 11:
                matrice_finale[h][a] += p - (p*prob_tie + p*prob_en)
                matrice_finale[h+1][a] += p*prob_tie
                matrice_finale[h][a+1] += p*prob_en
            else:
                matrice_finale[h][a] += p

    # Renormalisation : garantit des probas cohérentes (PL, ML, O/U)
    masse_totale = sum(matrice_finale[h][a] for h in range(12) for a in range(12))
    if masse_totale > 0:
        for h in range(12):
            for a in range(12):
                matrice_finale[h][a] /= masse_totale

    # Lecture des probabilités
    prob_1, prob_X, prob_2 = 0, 0, 0
    prob_pl_home, prob_pl_away = 0, 0
    cuts_cibles = [4.5, 5.5, 6.5, 7.5]
    prob_over_cuts = {cut: 0.0 for cut in cuts_cibles}
    prob_under_cuts = {cut: 0.0 for cut in cuts_cibles}

    for h in range(12):
        for a in range(12):
            p_final = matrice_finale[h][a]
            if h > a: prob_1 += p_final
            elif a > h: prob_2 += p_final
            else: prob_X += p_final

            if (h - a) >= 2: prob_pl_home += p_final
            if (a - h) >= 2: prob_pl_away += p_final

            total_buts = h + a
            for cut in cuts_cibles:
                if total_buts > cut: prob_over_cuts[cut] += p_final
                else: prob_under_cuts[cut] += p_final

    s_1x2 = prob_1 + prob_X + prob_2
    if s_1x2 <= 0:
        return None

    # ML Pinnacle = 2-way incluant prolongation : répartir prob_X (nul régul.) au prorata
    denom_ml = prob_1 + prob_2
    if denom_ml > 0:
        prob_ml_home = prob_1 + prob_X * (prob_1 / denom_ml)
        prob_ml_away = prob_2 + prob_X * (prob_2 / denom_ml)
    else:
        prob_ml_home, prob_ml_away = 0.5, 0.5

    prob_ml_home = min(max(prob_ml_home, 0.001), 0.999)
    prob_ml_away = min(max(prob_ml_away, 0.001), 0.999)
    prob_pl_home = min(max(prob_pl_home, 0.001), 0.999)
    prob_pl_away = min(max(prob_pl_away, 0.001), 0.999)

    for cut in cuts_cibles:
        s_ou = prob_over_cuts[cut] + prob_under_cuts[cut]
        if s_ou > 0:
            prob_over_cuts[cut] = min(max(prob_over_cuts[cut] / s_ou, 0.001), 0.999)
            prob_under_cuts[cut] = min(max(prob_under_cuts[cut] / s_ou, 0.001), 0.999)

    return {
        'prob_1': prob_ml_home, 'prob_2': prob_ml_away,
        'prob_pl_home': prob_pl_home, 'prob_pl_away': prob_pl_away,
        'prob_over_cuts': prob_over_cuts, 'prob_under_cuts': prob_under_cuts,
        'cote_1': round(1 / prob_ml_home, 2), 'cote_2': round(1 / prob_ml_away, 2),
        'cote_pl_home': round(1 / prob_pl_home, 2), 'cote_pl_away': round(1 / prob_pl_away, 2),
        'lam_home': round(final_lam_home, 3), 'lam_away': round(final_lam_away, 3),
    }

# ==========================================
# 5. INTEGRATION THE-ODDS-API & FINANCIAL
# ==========================================
def _noms_odds_pour_equipe(abbrev):
    """Liste des noms possibles pour une équipe (primaire + alias Pinnacle)."""
    primaire = NHL_TEAMS_MAPPING.get(abbrev)
    if not primaire:
        return []
    return [primaire] + ODDS_API_ALIASES.get(primaire, [])


def _nom_odds_vers_primaire(nom_api):
    """Mappe un nom Odds API vers notre nom primaire interne."""
    if nom_api in NHL_TEAMS_MAPPING.values():
        return nom_api
    return _ODDS_NOM_PRIMAIRE.get(nom_api, nom_api)


def _noms_equipe_equivalents(nom):
    """Ensemble de noms API interchangeables pour une même franchise."""
    if not nom:
        return set()
    primaire = _nom_odds_vers_primaire(nom)
    return {nom, primaire} | set(ODDS_API_ALIASES.get(primaire, []))


def _outcome_est_equipe(outcome_name, team_name):
    """True si le libellé Pinnacle correspond à l'équipe (alias inclus)."""
    if outcome_name == team_name:
        return True
    equiv_outcome = _noms_equipe_equivalents(outcome_name)
    equiv_team = _noms_equipe_equivalents(team_name)
    return bool(equiv_outcome & equiv_team)


def _est_puck_line_moins_15(point):
    return _float_proche(point, -1.5)


def _parse_pinnacle_game(game):
    """Extrait h2h, totals et puck line (-1.5) d'un match Pinnacle."""
    if not game.get("bookmakers"):
        return None
    home_full = game["home_team"]
    away_full = game["away_team"]
    bookmaker = game["bookmakers"][0]
    cotes = {"home_full": home_full, "away_full": away_full}
    for market in bookmaker["markets"]:
        if market["key"] == "h2h":
            for o in market["outcomes"]:
                if _outcome_est_equipe(o["name"], home_full):
                    cotes["cote_1"] = o["price"]
                elif _outcome_est_equipe(o["name"], away_full):
                    cotes["cote_2"] = o["price"]
        elif market["key"] == "totals":
            if "totals" not in cotes:
                cotes["totals"] = {}
            for o in market["outcomes"]:
                point = _arrondir_cut(o["point"])
                side = o["name"].capitalize()
                if side not in ("Over", "Under"):
                    continue
                if point not in cotes["totals"]:
                    cotes["totals"][point] = {}
                cotes["totals"][point][side] = o["price"]
        elif market["key"] == "spreads":
            for o in market["outcomes"]:
                if _est_puck_line_moins_15(o["point"]):
                    if _outcome_est_equipe(o["name"], home_full):
                        cotes["cote_pl_home"] = o["price"]
                    elif _outcome_est_equipe(o["name"], away_full):
                        cotes["cote_pl_away"] = o["price"]
    if "cote_1" not in cotes or "cote_2" not in cotes:
        return None
    return cotes


def _indexer_cotes_cache(cache, parsed):
    """Indexe un match sous toutes les combinaisons de noms connus."""
    home_api = parsed["home_full"]
    away_api = parsed["away_full"]
    home_primaire = _nom_odds_vers_primaire(home_api)
    away_primaire = _nom_odds_vers_primaire(away_api)
    cles_home = {home_api, home_primaire} | set(ODDS_API_ALIASES.get(home_primaire, []))
    cles_away = {away_api, away_primaire} | set(ODDS_API_ALIASES.get(away_primaire, []))
    for h in cles_home:
        for a in cles_away:
            cache[(h, a)] = parsed


def fetch_all_pinnacle_odds():
    """Une seule requête Odds API pour tous les matchs NHL (quota économisé)."""
    if not ODDS_API_KEY:
        return {}
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals,spreads",
        "bookmakers": "pinnacle",
        "oddsFormat": "decimal",
    }
    cache = {}
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            print(f"⚠️ Odds API : HTTP {response.status_code}")
            return {}
        for game in response.json():
            parsed = _parse_pinnacle_game(game)
            if parsed:
                _indexer_cotes_cache(cache, parsed)
        return cache
    except Exception as e:
        print(f"⚠️ Erreur Odds API globale : {e}")
        return {}


def get_odds_for_match(home_team_abbrev, away_team_abbrev, odds_cache=None, log_si_absent=False):
    """Retourne les cotes Pinnacle pour un match (depuis le cache ou une requête dédiée)."""
    noms_home = _noms_odds_pour_equipe(home_team_abbrev)
    noms_away = _noms_odds_pour_equipe(away_team_abbrev)
    if not noms_home or not noms_away:
        if log_si_absent:
            print(f"⚠️ Abréviation inconnue : {away_team_abbrev} @ {home_team_abbrev}")
        return None

    if odds_cache is not None:
        for home in noms_home:
            for away in noms_away:
                hit = odds_cache.get((home, away))
                if hit:
                    return hit
        if log_si_absent:
            print(
                f"⚠️ Pas de cotes Pinnacle : {away_team_abbrev} @ {home_team_abbrev} "
                f"(testé : {noms_away[0]} @ {noms_home[0]})"
            )
        return None

    single = fetch_all_pinnacle_odds()
    return get_odds_for_match(home_team_abbrev, away_team_abbrev, single, log_si_absent)


def get_real_live_odds(home_team_abbrev, away_team_abbrev, odds_cache=None):
    """Aspire Moneyline et tous les Cuts Over/Under."""
    cotes = get_odds_for_match(home_team_abbrev, away_team_abbrev, odds_cache)
    if not cotes:
        return None
    return {k: v for k, v in cotes.items() if k not in ("home_full", "away_full")}


def get_real_live_odds_puckline(home_team_abbrev, away_team_abbrev, odds_cache=None):
    """Aspire les spreads pour le Puck Line."""
    cotes = get_odds_for_match(home_team_abbrev, away_team_abbrev, odds_cache)
    if not cotes:
        return None
    pl = {}
    if "cote_pl_home" in cotes:
        pl["cote_pl_home"] = cotes["cote_pl_home"]
    if "cote_pl_away" in cotes:
        pl["cote_pl_away"] = cotes["cote_pl_away"]
    return pl if pl else None

def calculate_kelly(true_prob, book_odds, bankroll, gardiens_confirmes=True):
    if book_odds <= 1.0 or true_prob <= 0.01 or true_prob >= 0.99:
        return None
    edge = true_prob - (1 / book_odds)
    if edge <= EDGE_MINIMUM:
        return None
    b = book_odds - 1.0
    fraction_kelly = KELLY_FRACTION if gardiens_confirmes else KELLY_FRACTION_GARDIEN_INCERTAIN
    if not gardiens_confirmes:
        print("🛡️ SÉCURITÉ GARDIEN : Alignement non confirmé à 100%. Mise divisée par 2.")
    safe_kelly = ((b * true_prob - (1 - true_prob)) / b) * fraction_kelly
    return {
        'edge': round(edge * 100, 2),
        'pct_bankroll': round(safe_kelly * 100, 2),
        'mise': round(bankroll * safe_kelly, 2),
        'statut_gardiens': "CONFIRMÉ" if gardiens_confirmes else "PROBABLE",
    }

# ==========================================
# 6. JOURNAL DE TRADING & NOTIFICATIONS
# ==========================================
def match_deja_notifie(id_match):
    if not os.path.exists(FICHIER_MEMOIRE): return False
    with open(FICHIER_MEMOIRE, "r") as f: return id_match in f.read()

def enregistrer_notification(id_match):
    with open(FICHIER_MEMOIRE, "a") as f: f.write(id_match + "\n")

def enregistrer_transaction(id_match, ext, dom, type_pari, vraie_cote_pari, cotes_vraies_dict, investissement, cote_bookmaker):
    fichier_existe = os.path.isfile(FICHIER_JOURNAL)
    with open(FICHIER_JOURNAL, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not fichier_existe:
            writer.writerow([
                "Date", "ID_Match", "Visiteur", "Local", "Pari",
                "Vraie_Cote_Bot", "Cote_Prise", "Cote_CLV",
                "Lam_Ext", "Lam_Dom", "Score_Ext", "Score_Dom",
                "Edge(%)", "Risque(%)", "Mise_€", "Statut", "P&L"
            ])
        date_jour = datetime.now().strftime("%Y-%m-%d %H:%M")
        writer.writerow([
            date_jour, id_match, ext, dom, type_pari,
            vraie_cote_pari, cote_bookmaker, cote_bookmaker,
            cotes_vraies_dict['lam_away'], cotes_vraies_dict['lam_home'], "-", "-",
            investissement['edge'], investissement['pct_bankroll'],
            investissement['mise'], "EN ATTENTE", "0.00"
        ])

def envoyer_alerte(ext, g_ext, dom, g_dom, vraie_cote_pari, investissement, type_pari):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram non configuré — alerte non envoyée.")
        return
    statut = investissement.get('statut_gardiens', 'CONFIRMÉ')
    alerte_gardien = "✅ Gardiens Confirmés" if statut == "CONFIRMÉ" else "🛡️ GARDIENS PROBABLES (Mise / 2)"
    msg = f"🚨 **SNIPER NHL DÉCLENCHÉ** 🚨\n\nLoc: 🏟️ **{dom}** ({g_dom})\nVis: ✈️ **{ext}** ({g_ext})\n"
    msg += f"ℹ️ Statut : {alerte_gardien}\n──────────────\n"
    msg += f"🎯 **ORDRE : PARIER {type_pari}**\n🔥 Edge : **+{investissement['edge']}%**\n⚖️ Kelly : **{investissement['pct_bankroll']}%**\n"
    msg += f"💵 **MISE : {investissement['mise']} €**\n──────────────\n📊 True Odds: {vraie_cote_pari}"
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
        timeout=10,
    )


def _extraire_cote_clv(home_abbr, away_abbr, type_pari, cote_actuelle, odds_cache=None):
    """Détermine la cote Pinnacle actuelle pour un pari en attente."""
    cotes = get_odds_for_match(home_abbr, away_abbr, odds_cache)
    if not cotes:
        return cote_actuelle
    if "OVER" in type_pari or "UNDER" in type_pari:
        parts = type_pari.split(" ")
        side = parts[0].capitalize()
        cut = _arrondir_cut(parts[1])
        totals = cotes.get("totals", {})
        _, ligne = _trouver_cle_float(totals, cut)
        if ligne and side in ligne:
            return str(ligne[side])
    elif "Puck Line" in type_pari:
        if home_abbr in type_pari and "cote_pl_home" in cotes:
            return str(cotes["cote_pl_home"])
        if away_abbr in type_pari and "cote_pl_away" in cotes:
            return str(cotes["cote_pl_away"])
    else:
        if home_abbr in type_pari and "cote_1" in cotes:
            return str(cotes["cote_1"])
        if away_abbr in type_pari and "cote_2" in cotes:
            return str(cotes["cote_2"])
    return cote_actuelle


def traquer_et_actualiser_clv():
    if not os.path.exists(FICHIER_JOURNAL):
        return
    odds_cache = fetch_all_pinnacle_odds()
    lignes, mise_a_jour_effectuee = [], False
    with open(FICHIER_JOURNAL, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lignes.append(next(reader))
        for row in reader:
            if len(row) > 15 and row[15] == "EN ATTENTE":
                ext, dom, type_pari = row[2], row[3], row[4]
                nv_cote = _extraire_cote_clv(dom, ext, type_pari, row[7], odds_cache)
                if nv_cote != row[7]:
                    row[7] = nv_cote
                    mise_a_jour_effectuee = True
            lignes.append(row)
    if mise_a_jour_effectuee:
        with open(FICHIER_JOURNAL, 'w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerows(lignes)

def calculer_bankroll_dynamique(capital_de_base=1000.0):
    """
    Lit le fichier CSV, additionne les profits et pertes (P&L),
    et retourne le capital actuel.
    """
    if not os.path.exists(FICHIER_JOURNAL):
        return capital_de_base

    profit_total = 0.0
    try:
        with open(FICHIER_JOURNAL, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # On ne compte que les paris qui sont terminés et balayés
                if row['Statut'] in ["GAGNÉ", "PERDU"]:
                    profit_total += float(row['P&L'])

        # On calcule le solde actuel
        bankroll_actuelle = capital_de_base + profit_total

        # Sécurité Anti-Ruine : Si la bankroll tombe sous 10€,
        # on la bloque à 10€ pour éviter que le bot ne plante avec des mises négatives ou à 0.
        return round(max(bankroll_actuelle, 10.0), 2)

    except Exception as e:
        print(f"⚠️ Erreur de calcul de la Bankroll : {e}")
        return capital_de_base

# ==========================================
# 7. BOUCLE DE SCANNAGE PRINCIPALE
# ==========================================
def run_sniper():
    print("🤖 Lancement de l'arme absolue NHL...")

    while True:
        try:
            lancer_la_balayeuse()
            entrainer_ia_dixon_coles()

            heure_actuelle = datetime.now().hour
            if 18 <= heure_actuelle or heure_actuelle <= 1:
                print("🕵️ Tracking CLV en cours...")
                traquer_et_actualiser_clv()

            # ---> NOUVEAU : MISE À JOUR DU CAPITAL
            bankroll_actuelle = calculer_bankroll_dynamique(BANKROLL_INITIALE)
            print(f"💰 Capital Dynamique Disponible : {bankroll_actuelle} €")
            # ------------------------------------

            rho_actuel = lire_rho_dynamique()
            print(f"🧠 Configuration Mathématique : Rho = {rho_actuel}")

            print("\n📡 Synchronisation bases de données...")
            teams, goalies, stars_vip, momentum_data = (
                get_team_stats(), get_goalie_stats(), get_stars_impact(), get_nhl_momentum()
            )
            equipes_en_b2b_hier = get_teams_played_yesterday()
            odds_cache = fetch_all_pinnacle_odds()

            if not teams or not goalies:
                print(f"⚠️ Données MoneyPuck indisponibles (saison {NHL_SEASON}). Nouvelle tentative dans 5 min...")
                time.sleep(300)
                continue

            matchs = get_nhl_games_today()
            if not matchs:
                print("🏒 Aucun match NHL programmé aujourd'hui — veille active.")
            for m in matchs:
                id_match = f"{m['game_id']}_notified"
                if match_deja_notifie(id_match): continue

                g_ext, g_dom, skaters_ext, skaters_dom = get_active_rosters(m['game_id'])
                if g_ext and g_dom:
                    statut_confirmation = get_goalie_confirmation_status(m['game_id'])
                    gardiens_verrouilles = statut_confirmation["away_confirmed"] and statut_confirmation["home_team_confirmed"]

                    # ---> NOUVEAU : Détection automatique du drapeau B2B
                    home_b2b = m['home_team'] in equipes_en_b2b_hier
                    away_b2b = m['away_team'] in equipes_en_b2b_hier

                    if home_b2b: print(f"🔄 {m['home_team']} détecté en Back-to-Back !")
                    if away_b2b: print(f"🔄 {m['away_team']} détecté en Back-to-Back !")

                    gsax_ext, gsax_dom = trouver_gsax(g_ext, goalies), trouver_gsax(g_dom, goalies)
                    absents_ext = detecter_stars_absentes(m['away_team'], skaters_ext, stars_vip)
                    absents_dom = detecter_stars_absentes(m['home_team'], skaters_dom, stars_vip)

                    teams_match = copy.deepcopy(teams)
                    home_base = next((t for t in teams_match if t['team'] == m['home_team']), None)
                    away_base = next((t for t in teams_match if t['team'] == m['away_team']), None)
                    if not home_base or not away_base:
                        continue

                    adj_xgf_ext, adj_xga_ext = apply_star_absence_penalty(
                        m['away_team'], away_base['xGF_per_game'], away_base['xGA_per_game'],
                        absents_ext, stars_vip,
                    )
                    adj_xgf_dom, adj_xga_dom = apply_star_absence_penalty(
                        m['home_team'], home_base['xGF_per_game'], home_base['xGA_per_game'],
                        absents_dom, stars_vip,
                    )
                    away_base['xGF_per_game'], away_base['xGA_per_game'] = adj_xgf_ext, adj_xga_ext
                    home_base['xGF_per_game'], home_base['xGA_per_game'] = adj_xgf_dom, adj_xga_dom

                    mom_dom, mom_ext = momentum_data.get(m['home_team'], 0.0), momentum_data.get(m['away_team'], 0.0)

                    cotes_vraies = calculate_master_odds_v4(
                        teams_match, m['home_team'], m['away_team'], gsax_dom, gsax_ext,
                        mom_home=mom_dom, mom_away=mom_ext,
                        home_is_b2b=home_b2b, away_is_b2b=away_b2b, # <--- ICI
                        rho=rho_actuel
                    )
                    cotes_bookmaker = get_real_live_odds(
                        m['home_team'], m['away_team'], odds_cache, log_si_absent=True,
                    )
                    cotes_puckline = get_real_live_odds_puckline(m['home_team'], m['away_team'], odds_cache)

                    if cotes_vraies and cotes_bookmaker:
                        pari_trouve, best_pari, max_edge = False, None, 0.0

                        # --- Domicile ---
                        inv_ml_home = calculate_kelly(cotes_vraies['prob_1'], cotes_bookmaker['cote_1'], bankroll_actuelle, gardiens_confirmes=gardiens_verrouilles)
                        if inv_ml_home and inv_ml_home['edge'] > max_edge:
                            max_edge = inv_ml_home['edge']
                            best_pari = {"type": f"Victoire {m['home_team']}", "inv": inv_ml_home, "cote_book": cotes_bookmaker['cote_1'], "cote_vraie": cotes_vraies['cote_1']}

                        if cotes_puckline and 'cote_pl_home' in cotes_puckline:
                            inv_pl_home = calculate_kelly(cotes_vraies['prob_pl_home'], cotes_puckline['cote_pl_home'], bankroll_actuelle, gardiens_confirmes=gardiens_verrouilles)
                            if inv_pl_home and inv_pl_home['edge'] > max_edge:
                                max_edge = inv_pl_home['edge']
                                best_pari = {"type": f"Puck Line {m['home_team']} -1.5", "inv": inv_pl_home, "cote_book": cotes_puckline['cote_pl_home'], "cote_vraie": cotes_vraies['cote_pl_home']}

                        # --- Extérieur ---
                        inv_ml_away = calculate_kelly(cotes_vraies['prob_2'], cotes_bookmaker['cote_2'], bankroll_actuelle, gardiens_confirmes=gardiens_verrouilles)
                        if inv_ml_away and inv_ml_away['edge'] > max_edge:
                            max_edge = inv_ml_away['edge']
                            best_pari = {"type": f"Victoire {m['away_team']}", "inv": inv_ml_away, "cote_book": cotes_bookmaker['cote_2'], "cote_vraie": cotes_vraies['cote_2']}

                        if cotes_puckline and 'cote_pl_away' in cotes_puckline:
                            inv_pl_away = calculate_kelly(cotes_vraies['prob_pl_away'], cotes_puckline['cote_pl_away'], bankroll_actuelle, gardiens_confirmes=gardiens_verrouilles)
                            if inv_pl_away and inv_pl_away['edge'] > max_edge:
                                max_edge = inv_pl_away['edge']
                                best_pari = {"type": f"Puck Line {m['away_team']} -1.5", "inv": inv_pl_away, "cote_book": cotes_puckline['cote_pl_away'], "cote_vraie": cotes_vraies['cote_pl_away']}

                        # --- Cible Universelle OVER/UNDER ---
                        if 'totals' in cotes_bookmaker:
                            for cut, prices in cotes_bookmaker['totals'].items():
                                cut_arrondi = _arrondir_cut(cut)
                                _, prob_over = _trouver_cle_float(cotes_vraies['prob_over_cuts'], cut_arrondi)
                                _, prob_under = _trouver_cle_float(cotes_vraies['prob_under_cuts'], cut_arrondi)
                                if prob_over is None:
                                    continue
                                if 'Over' in prices:
                                    inv_over = calculate_kelly(prob_over, prices['Over'], bankroll_actuelle, gardiens_confirmes=gardiens_verrouilles)
                                    if inv_over and inv_over['edge'] > max_edge:
                                        max_edge = inv_over['edge']
                                        vraie_cote_over = round(1 / max(prob_over, 0.001), 2)
                                        best_pari = {"type": f"OVER {cut_arrondi}", "inv": inv_over, "cote_book": prices['Over'], "cote_vraie": vraie_cote_over}

                                if 'Under' in prices and prob_under is not None:
                                    inv_under = calculate_kelly(prob_under, prices['Under'], bankroll_actuelle, gardiens_confirmes=gardiens_verrouilles)
                                    if inv_under and inv_under['edge'] > max_edge:
                                        max_edge = inv_under['edge']
                                        vraie_cote_under = round(1 / max(prob_under, 0.001), 2)
                                        best_pari = {"type": f"UNDER {cut_arrondi}", "inv": inv_under, "cote_book": prices['Under'], "cote_vraie": vraie_cote_under}

                        # --- Envoi de la transaction finale ---
                        if best_pari:
                            envoyer_alerte(m['away_team'], g_ext, m['home_team'], g_dom, best_pari['cote_vraie'], best_pari['inv'], best_pari['type'])
                            enregistrer_transaction(id_match, m['away_team'], m['home_team'], best_pari['type'], best_pari['cote_vraie'], cotes_vraies, best_pari['inv'], best_pari['cote_book'])
                            pari_trouve = True

                        if pari_trouve: enregistrer_notification(id_match)
            time.sleep(900)
        except Exception as e:
            print(f"⚠️ Erreur système : {e}")
            traceback.print_exc()
            time.sleep(60)

# ==========================================
# 8. LA BALAYEUSE & INTELLIGENCE
# ==========================================
def get_match_result(game_id):
    try:
        response = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore", timeout=10)
        data = response.json()
        if response.status_code == 200 and data["gameState"] == "FINAL": return data["awayScore"], data["homeScore"]
        return None
    except: return None

def lancer_la_balayeuse():
    if not os.path.exists(FICHIER_JOURNAL): return
    lignes = []
    with open(FICHIER_JOURNAL, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lignes.append(next(reader))
        for row in reader:
            if row[15] == "EN ATTENTE":
                res = get_match_result(row[1].split('_')[0])
                if res:
                    score_v, score_d = res
                    row[10], row[11] = str(score_v), str(score_d)
                    ext, dom, pari = row[2], row[3], row[4]
                    mise, cote_book = float(row[14]), float(row[6])

                    gagne = False
                    if "Victoire" in pari:
                        if (score_d > score_v and dom in pari) or (score_v > score_d and ext in pari): gagne = True
                    elif "Puck Line" in pari:
                        if (score_d - score_v >= 2 and dom in pari) or (score_v - score_d >= 2 and ext in pari): gagne = True
                    elif "OVER" in pari or "UNDER" in pari:
                        parts = pari.split(" ")
                        cut = float(parts[1])
                        total_buts = score_d + score_v
                        if "OVER" in pari and total_buts > cut: gagne = True
                        elif "UNDER" in pari and total_buts < cut: gagne = True

                    row[15] = "GAGNÉ" if gagne else "PERDU"
                    row[16] = f"{round(mise * (cote_book - 1), 2) if gagne else -mise}"
            lignes.append(row)
    with open(FICHIER_JOURNAL, 'w', encoding='utf-8', newline='') as f: csv.writer(f).writerows(lignes)

def lire_rho_dynamique():
    if os.path.exists("rho_optimal.txt"):
        with open("rho_optimal.txt", "r") as f: return float(f.read().strip())
    return -0.12

def entrainer_ia_dixon_coles():
    if not os.path.exists(FICHIER_JOURNAL): return
    historique_unique = {}
    with open(FICHIER_JOURNAL, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['Statut'] in ["GAGNÉ", "PERDU"]:
                if row['ID_Match'] not in historique_unique:
                    historique_unique[row['ID_Match']] = {'vrai_score_domicile': int(row['Score_Dom']), 'vrai_score_exterieur': int(row['Score_Ext']), 'lambda_domicile_calcule': float(row['Lam_Dom']), 'lambda_exterieur_calcule': float(row['Lam_Ext'])}
    dataset = list(historique_unique.values())
    if len(dataset) >= 30:
        with open("rho_optimal.txt", "w") as f: f.write(str(optimiser_rho_saison(dataset)))

if __name__ == "__main__":
    manquants = []
    if not ODDS_API_KEY:
        manquants.append("ODDS_API_KEY")
    if not TELEGRAM_TOKEN:
        manquants.append("TELEGRAM_TOKEN")
    if manquants:
        print(f"⚠️ Variables manquantes dans identifiants_différent_api.env : {', '.join(manquants)}")
        print("   Le bot peut tourner en mode veille, mais ne pourra pas parier sans clé Odds API.")
    print(f"🏒 Sniper NHL Oméga — saison MoneyPuck {NHL_SEASON} | bankroll initiale {BANKROLL_INITIALE} €")
    run_sniper()