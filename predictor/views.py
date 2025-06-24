import joblib
import pandas as pd
from django.shortcuts import render
from django.utils import timezone

# Load once at import time
saved = joblib.load('predictor/ml/full_model.pkl')
pipeline = saved['pipeline']
label_encoder = saved['label_encoder']

# Betting columns defaults (you can adjust as desired)
BETTING_COLS = [
    'B365H','B365D','B365A','BWH','BWD','BWA','IWH','IWD','IWA',
    'LBH','LBD','LBA','WHH','WHD','WHA','VCH','VCD','VCA'
]
DEFAULT_ODDS = {col: 0 for col in BETTING_COLS}

# define dropdown lists
LEAGUES = sorted([
    'Belgium Jupiler League', 'England Premier League', 'France Ligue 1',
    'Germany 1. Bundesliga', 'Italy Serie A', 'Netherlands Eredivisie',
    'Poland Ekstraklasa', 'Portugal Liga ZON Sagres', 'Scotland Premier League',
    'Spain LIGA BBVA', 'Switzerland Super League'
])

TEAMS = sorted(['KRC Genk', 'Beerschot AC', 'SV Zulte-Waregem', 'Sporting Lokeren', 'KSV Cercle Brugge', 'RSC Anderlecht', 
                'KAA Gent', 'RAEC Mons', 'FCV Dender EH', 'Standard de Liège', 'KV Mechelen', 'Club Brugge KV', 'KSV Roeselare', 
                'KV Kortrijk', 'Tubize', 'Royal Excel Mouscron', 'KVC Westerlo', 'Sporting Charleroi', 'Sint-Truidense VV', 
                'Lierse SK', 'KAS Eupen', 'Oud-Heverlee Leuven', 'Waasland-Beveren', 'KV Oostende', 'Manchester United', 
                'Newcastle United', 'Arsenal', 'West Bromwich Albion', 'Sunderland', 'Liverpool', 'West Ham United', 'Wigan Athletic', 
                'Aston Villa', 'Manchester City', 'Everton', 'Blackburn Rovers', 'Middlesbrough', 'Tottenham Hotspur', 'Bolton Wanderers', 
                'Stoke City', 'Hull City', 'Fulham', 'Chelsea', 'Portsmouth', 'Birmingham City', 'Wolverhampton Wanderers', 'Burnley', 
                'Blackpool', 'Swansea City', 'Queens Park Rangers', 'Norwich City', 'Southampton', 'Reading', 'Crystal Palace', 
                'Cardiff City', 'Leicester City', 'Bournemouth', 'Watford', 'AJ Auxerre', 'FC Nantes', 'Girondins de Bordeaux', 'SM Caen', 
                'Le Havre AC', 'OGC Nice', 'Le Mans FC', 'FC Lorient', 'Olympique Lyonnais', 'Toulouse FC', 'AS Monaco', 'Paris Saint-Germain', 
                'AS Nancy-Lorraine', 'LOSC Lille', 'Stade Rennais FC', 'Olympique de Marseille', 'FC Sochaux-Montbéliard', 'Grenoble Foot 38', 
                'Valenciennes FC', 'AS Saint-Étienne', 'RC Lens', 'Montpellier Hérault SC', "US Boulogne Cote D'Opale", 'AC Arles-Avignon', 
                'Stade Brestois 29', 'AC Ajaccio', 'Évian Thonon Gaillard FC', 'Dijon FCO', 'Stade de Reims', 'SC Bastia', 'ES Troyes AC', 
                'En Avant de Guingamp', 'FC Metz', 'Angers SCO', 'GFC Ajaccio', 'FC Bayern Munich', 'Hamburger SV', 'Bayer 04 Leverkusen', 
                'Borussia Dortmund', 'FC Schalke 04', 'Hannover 96', 'VfL Wolfsburg', '1. FC Köln', 'Eintracht Frankfurt', 'Hertha BSC Berlin', 
                'DSC Arminia Bielefeld', 'SV Werder Bremen', 'FC Energie Cottbus', 'TSG 1899 Hoffenheim', 'Borussia Mönchengladbach', 
                'VfB Stuttgart', 'Karlsruher SC', 'VfL Bochum', 'SC Freiburg', '1. FC Nürnberg', '1. FSV Mainz 05', '1. FC Kaiserslautern', 
                'FC St. Pauli', 'FC Augsburg', 'Fortuna Düsseldorf', 'SpVgg Greuther Fürth', 'Eintracht Braunschweig', 'SC Paderborn 07', 
                'FC Ingolstadt 04', 'SV Darmstadt 98', 'Atalanta', 'Siena', 'Cagliari', 'Lazio', 'Catania', 'Genoa', 'Chievo Verona', 
                'Reggio Calabria', 'Fiorentina', 'Juventus', 'Milan', 'Bologna', 'Roma', 'Napoli', 'Sampdoria', 'Inter', 'Torino', 'Lecce', 
                'Udinese', 'Palermo', 'Bari', 'Livorno', 'Parma', 'Cesena', 'Brescia', 'Novara', 'Pescara', 'Hellas Verona', 'Sassuolo', 
                'Empoli', 'Frosinone', 'Carpi', 'Vitesse', 'FC Groningen', 'Roda JC Kerkrade', 'FC Twente', 'Willem II', 'Ajax', 'N.E.C.', 
                'De Graafschap', 'FC Utrecht', 'PSV', 'Heracles Almelo', 'Feyenoord', 'Sparta Rotterdam', 'ADO Den Haag', 'FC Volendam', 
                'SC Heerenveen', 'AZ', 'NAC Breda', 'RKC Waalwijk', 'VVV-Venlo', 'Excelsior', 'PEC Zwolle', 'SC Cambuur', 'Go Ahead Eagles', 
                'FC Dordrecht', 'Wisła Kraków', 'Polonia Bytom', 'Ruch Chorzów', 'Legia Warszawa', 'P. Warszawa', 'Śląsk Wrocław', 
                'Lechia Gdańsk', 'Widzew Łódź', 'Odra Wodzisław', 'Lech Poznań', 'GKS Bełchatów', 'Arka Gdynia', 'Jagiellonia Białystok', 
                'Piast Gliwice', 'Cracovia', 'Korona Kielce', 'Zagłębie Lubin', 'Podbeskidzie Bielsko-Biała', 'Pogoń Szczecin', 
                'Zawisza Bydgoszcz', 'Górnik Łęczna', 'Termalica Bruk-Bet Nieciecza', 'FC Porto', 'CF Os Belenenses', 'Sporting CP', 
                'Trofense', 'Vitória Guimarães', 'Vitória Setúbal', 'FC Paços de Ferreira', 'SC Braga', 'Amadora', 'Académica de Coimbra', 
                'Rio Ave FC', 'SL Benfica', 'Leixões SC', 'CD Nacional', 'Naval 1° de Maio', 'CS Marítimo', 'União de Leiria, SAD', 
                'S.C. Olhanense', 'Portimonense', 'SC Beira Mar', 'Feirense', 'Gil Vicente FC', 'Moreirense FC', 'Estoril Praia', 'FC Arouca', 
                'FC Penafiel', 'Boavista FC', 'Uniao da Madeira', 'Tondela', 'Falkirk', 'Rangers', 'Heart of Midlothian', 'Motherwell', 
                'Kilmarnock', 'Hibernian', 'Aberdeen', 'Inverness Caledonian Thistle', 'Celtic', 'St. Mirren', 'Hamilton Academical FC', 
                'Dundee United', 'St. Johnstone FC', 'Dunfermline Athletic', 'Dundee FC', 'Ross County FC', 'Partick Thistle F.C.', 
                'Valencia CF', 'RCD Mallorca', 'CA Osasuna', 'Villarreal CF', 'RC Deportivo de La Coruña', 'Real Madrid CF', 'CD Numancia', 
                'FC Barcelona', 'Racing Santander', 'Sevilla FC', 'Real Sporting de Gijón', 'Getafe CF', 'Real Betis Balompié', 
                'RC Recreativo', 'RCD Espanyol', 'Real Valladolid', 'Athletic Club de Bilbao', 'UD Almería', 'Atlético Madrid', 'Málaga CF', 
                'Xerez Club Deportivo', 'Real Zaragoza', 'CD Tenerife', 'Hércules Club de Fútbol', 'Levante UD', 'Real Sociedad', 'Granada CF', 
                'Rayo Vallecano', 'RC Celta de Vigo', 'Elche CF', 'SD Eibar', 'Córdoba CF', 'UD Las Palmas', 'Grasshopper Club Zürich', 
                'AC Bellinzona', 'BSC Young Boys', 'FC Basel', 'FC Aarau', 'FC Sion', 'FC Luzern', 'FC Vaduz', 'Neuchâtel Xamax', 
                'FC Zürich', 'FC St. Gallen', 'FC Thun', 'Servette FC', 'FC Lausanne-Sports', 'Lugano']
)

def index(request):
    return render(request, 'predictor/index.html', {
        'leagues': LEAGUES,
        'teams': TEAMS,
    })

def predict(request):
    if request.method != 'POST':
        return render(request, 'predictor/index.html', {
            'leagues': LEAGUES,
            'teams': TEAMS,
        })

    # 1. Gather inputs
    stage      = int(request.POST['stage'])
    home_team  = request.POST['home_team']
    away_team  = request.POST['away_team']
    league     = request.POST['league']
    year       = timezone.now().year

    # 2. Build DataFrame
    row = {
        'home_team_long_name': home_team,
        'away_team_long_name': away_team,
        'league_name'        : league,
        'stage'              : stage,
        'year'               : year,
        **DEFAULT_ODDS
    }
    X = pd.DataFrame([row])

    # 3. Predict
    pred_enc   = pipeline.predict(X)[0]
    outcome_raw = label_encoder.inverse_transform([pred_enc])[0]

    # 4. Turn into a message
    if outcome_raw == 'Home Win':
        result_message = f"{home_team} Win!"
    elif outcome_raw == 'Away Win':
        result_message = f"{away_team} Win!"
    else:
        result_message = "Draw"

    # 5. Render
    return render(request, 'predictor/result.html', {
        'home_team': home_team,
        'away_team': away_team,
        'league':    league,
        'stage':     stage,
        'prediction': result_message,    # now contains "Real Madrid CF Win!" etc.
    })
