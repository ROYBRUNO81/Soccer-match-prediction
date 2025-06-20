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

def index(request):
    return render(request, 'predictor/index.html')

def predict(request):
    if request.method != 'POST':
        return render(request, 'predictor/index.html')

    # 1. Gather inputs
    stage      = int(request.POST['stage'])
    home_team  = request.POST['home_team']
    away_team  = request.POST['away_team']
    league     = request.POST['league']
    year       = timezone.now().year

    # 2. Build a single-row DataFrame
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
    pred_enc = pipeline.predict(X)[0]
    outcome  = label_encoder.inverse_transform([pred_enc])[0]

    return render(request, 'predictor/result.html', {
        'home_team': home_team,
        'away_team': away_team,
        'league': league,
        'stage': stage,
        'prediction': outcome,
    })
