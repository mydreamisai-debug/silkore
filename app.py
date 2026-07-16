import os
import json
import datetime
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the trained model and metadata
MODEL_PATH = 'cocoon_price_model.joblib'
METADATA_PATH = 'model_metadata.json'
DATASET_PATH = 'synthetic_karnataka_cocoon_dataset_10000_records.xlsx'

model = None
metadata = None
trends_cache = {}

def init_app():
    global model, metadata, trends_cache
    
    # 1. Load Model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print("WARNING: Model file not found. Run train_model.py first.")
        
    # 2. Load Metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print("Metadata loaded successfully.")
    else:
        print("WARNING: Metadata file not found. Run train_model.py first.")
        
    # 3. Load Dataset & Precompute Trends for speed
    if os.path.exists(DATASET_PATH):
        print("Loading dataset for trends...")
        df = pd.read_excel(DATASET_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        # Sort by date
        df = df.sort_values('Date')
        
        # Monthly averages for trends
        df['YearMonth'] = df['Date'].dt.strftime('%Y-%m')
        grouped = df.groupby(['Market', 'Variety', 'YearMonth']).agg({
            'CocoonPrice_RsKg': 'mean',
            'RawSilkPrice_RsKg': 'mean'
        }).reset_index()
        
        # Structure the cache: trends_cache[(market, variety)] = list of monthly price/raw_silk_price
        for _, row in grouped.iterrows():
            key = (row['Market'], row['Variety'])
            if key not in trends_cache:
                trends_cache[key] = []
            trends_cache[key].append({
                'date': row['YearMonth'],
                'price': round(float(row['CocoonPrice_RsKg']), 2),
                'raw_silk_price': round(float(row['RawSilkPrice_RsKg']), 2)
            })
        print("Trends cache prepared.")
    else:
        print("WARNING: Dataset xlsx file not found for trend visualization.")

init_app()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    if not metadata:
        return jsonify({'error': 'Metadata not loaded'}), 500
    
    return jsonify({
        'markets': metadata['markets'],
        'varieties': metadata['varieties'],
        'grade_presets': metadata['grade_presets'],
        'metrics': metadata['metrics']
    })

@app.route('/api/trends', methods=['GET'])
def get_trends():
    market = request.args.get('market')
    variety = request.args.get('variety')
    
    if not market or not variety:
        return jsonify({'error': 'Missing market or variety parameter'}), 400
        
    key = (market, variety)
    trends = trends_cache.get(key, [])
    
    # If no exact match, try matching just the variety or just return first available
    if not trends:
        # Fallback
        keys = list(trends_cache.keys())
        for k in keys:
            if k[1] == variety:
                trends = trends_cache[k]
                break
        if not trends and keys:
            trends = trends_cache[keys[0]]
            
    return jsonify(trends)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model or not metadata:
        return jsonify({'error': 'Model or metadata not initialized on backend'}), 500
        
    data = request.json
    try:
        # Extract inputs
        date_str = data.get('date', datetime.date.today().isoformat())
        market = data.get('market', metadata['markets'][0])
        variety = data.get('variety', metadata['varieties'][0])
        
        # Get defaults from metadata if not provided
        var_defaults = metadata['defaults_by_variety'].get(variety, {})
        
        def get_val(key, default_pct=0.5):
            val = data.get(key)
            if val is not None:
                return float(val)
            # fallback to mean
            return float(var_defaults.get(key, {}).get('mean', 0.0))

        avg_weight = get_val('AvgWeight_g')
        shell_weight = get_val('ShellWeight_g')
        shell_ratio = get_val('ShellRatio_pct')
        filament_length = get_val('FilamentLength_m')
        reelability = get_val('Reelability_pct')
        moisture = get_val('Moisture_pct')
        defects = get_val('Defects_pct')
        temp = get_val('Temp_C')
        humidity = get_val('Humidity_pct')
        rainfall = get_val('Rainfall_mm')
        raw_silk_price = get_val('RawSilkPrice_RsKg')
        
        # Parse date and engineer date features
        dt = pd.to_datetime(date_str)
        year = dt.year
        month = dt.month
        day = dt.day
        day_of_week = dt.dayofweek
        day_of_year = dt.dayofyear
        
        start_date = pd.to_datetime('2021-01-01')
        date_ordinal = (dt - start_date).days
        
        # Categorical Encoding
        market_encoded = metadata['market_mapping'].get(market, 0)
        variety_encoded = metadata['variety_mapping'].get(variety, 0)
        
        # Build features vector in correct order
        # features list order:
        # ['Market_encoded', 'Variety_encoded', 'Year', 'Month', 'Day', 'DayOfWeek', 
        #  'DayOfYear', 'DateOrdinal', 'AvgWeight_g', 'ShellWeight_g', 'ShellRatio_pct', 
        #  'FilamentLength_m', 'Reelability_pct', 'Moisture_pct', 'Defects_pct', 
        #  'Temp_C', 'Humidity_pct', 'Rainfall_mm', 'RawSilkPrice_RsKg']
        features_vector = [
            market_encoded,
            variety_encoded,
            year,
            month,
            day,
            day_of_week,
            day_of_year,
            date_ordinal,
            avg_weight,
            shell_weight,
            shell_ratio,
            filament_length,
            reelability,
            moisture,
            defects,
            temp,
            humidity,
            rainfall,
            raw_silk_price
        ]
        
        # Model inference
        prediction = model.predict([features_vector])[0]
        
        # Calculate a quality score (0 to 100) based on features
        # Higher ShellRatio, Reelability, FilamentLength, RawSilkPrice and lower defects/moisture is good
        # We normalize features based on their dataset ranges
        # Ranges: 
        # ShellRatio: 15% to 25% (mean ~ 20.7%)
        # Reelability: 60% to 85% (mean ~ 73%)
        # FilamentLength: 600m to 1400m (mean ~ 1000m)
        # Defects: 2% to 10% (mean ~ 5.8%) -> lower is better
        
        shell_ratio_score = np.clip((shell_ratio - 15) / (25 - 15), 0, 1) * 30
        reelability_score = np.clip((reelability - 60) / (85 - 60), 0, 1) * 25
        filament_score = np.clip((filament_length - 600) / (1400 - 600), 0, 1) * 25
        defect_score = np.clip(1 - (defects - 2) / (10 - 2), 0, 1) * 20
        
        quality_score = shell_ratio_score + reelability_score + filament_score + defect_score
        quality_score = round(float(quality_score), 1)
        
        # Determine visual grade label
        if quality_score >= 80:
            grade_label = "Premium Grade (A+)"
            grade_color = "#10B981" # Green
        elif quality_score >= 50:
            grade_label = "Standard Grade (B)"
            grade_color = "#F59E0B" # Amber
        else:
            grade_label = "Economy Grade (C)"
            grade_color = "#EF4444" # Red

        # Add price range estimation (MAE is ~18.2)
        lower_bound = round(float(prediction - 1.96 * 18.2), 2)
        upper_bound = round(float(prediction + 1.96 * 18.2), 2)
        
        return jsonify({
            'success': True,
            'prediction': round(float(prediction), 2),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'quality_score': quality_score,
            'grade_label': grade_label,
            'grade_color': grade_color,
            'inputs': {
                'date': date_str,
                'market': market,
                'variety': variety,
                'avg_weight': avg_weight,
                'shell_weight': shell_weight,
                'shell_ratio': shell_ratio,
                'filament_length': filament_length,
                'reelability': reelability,
                'moisture': moisture,
                'defects': defects,
                'raw_silk_price': raw_silk_price
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Start Flask Server
    app.run(debug=True, host='0.0.0.0', port=5000)
