from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import random
import os
import datetime
import csv
import sys

app = Flask(__name__)

# Адрес MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
MODEL_NAME = "WineQualityModel"
LOG_FILE = "/app/logs/traffic_log.csv"
STAGING_RATIO = 0.5 

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'user_id', 'inputs', 'target_stage', 'actual_stage', 'prediction', 'ratio_at_time'])

# Загружает модель из MLflow
def get_model(stage):
    model_uri = f"models:/{MODEL_NAME}/{stage}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model, None
    except Exception as e:
        return None, str(e)

# Меняет долю трафика на Staging
@app.route('/set_traffic', methods=['POST'])
def set_traffic():
    global STAGING_RATIO
    try:
        ratio = float(request.json.get('ratio', 0.5))
        if not (0 <= ratio <= 1.0):
            return jsonify({'error': 'Ratio must be between 0.0 and 1.0'}), 400
        
        STAGING_RATIO = ratio
        return jsonify({
            'message': 'Traffic split updated', 
            'staging_ratio': STAGING_RATIO,
            'production_ratio': 1.0 - STAGING_RATIO
        })
    except ValueError:
        return jsonify({'error': 'Invalid ratio format'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data.get('features')
    user_id = data.get('user_id', random.randint(1, 10000))
    
    if not features:
        return jsonify({'error': 'No features provided'}), 400

    if random.random() < STAGING_RATIO:
        target_stage = "Staging"
    else:
        target_stage = "Production"

    model, error = get_model(target_stage)
    
    actual_stage = target_stage

    if model is None:
        print(f"Warning: Failed to load {target_stage}. Falling back...", file=sys.stderr)
        if target_stage == "Production":
            fallback_stage = "Staging"
        else:
            fallback_stage = "Production"
            
        model, error_fallback = get_model(fallback_stage)
        actual_stage = fallback_stage
        
        if model is None:
            return jsonify({
                'error': 'No models available', 
                'details': f"Main: {error}, Fallback: {error_fallback}"
            }), 500

    # Предсказание
    try:
        cols = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 
                'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 
                'hue', 'od280/od315_of_diluted_wines', 'proline']
        
        df = pd.DataFrame([features], columns=cols)
        prediction = int(model.predict(df)[0])
        
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now(),
                user_id,
                str(features),
                target_stage,
                actual_stage,
                prediction,
                STAGING_RATIO
            ])

        return jsonify({
            'prediction': prediction,
            'model_stage': actual_stage,
            'traffic_split_target': target_stage
        })
        
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)