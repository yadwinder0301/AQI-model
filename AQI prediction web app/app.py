import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

import gzip

# Load Model and Scaler
with gzip.open('aqi_model.pkl.gz', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Extract column names mapping from the scaler exactly as the model expects
features = list(scaler.feature_names_in_)

# Get list of all cities to feed into the Frontend Dropdown
# We can find all encoded city names from the feature list
city_cols = [col for col in features if col.startswith('City_')]
cities = [col.replace('City_', '') for col in city_cols]

# Note: The first city alphabetically was dropped during one-hot encoding (drop_first=True).
# Let's extract all cities directly from the raw dataset once to get the full list for the dropdown.
try:
    raw_df = pd.read_csv('city_day.csv')
    all_cities = sorted(raw_df['City'].dropna().unique().tolist())
except Exception:
    all_cities = cities # Fallback if csv is missing


@app.route('/')
def home():
    return render_template('index.html', cities=all_cities)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Initialize a dictionary mapping every model feature to 0.0 by default
        input_dict = {col: 0.0 for col in features}
        
        # 1. Fill Numerical Pollutant values
        input_dict['PM2.5'] = float(data.get('PM2.5', 0))
        input_dict['PM10'] = float(data.get('PM10', 0))
        input_dict['NO2'] = float(data.get('NO2', 0))
        input_dict['CO'] = float(data.get('CO', 0))
        input_dict['SO2'] = float(data.get('SO2', 0))
        input_dict['O3'] = float(data.get('O3', 0))
        
        # 2. Extract and assign Date parts
        date_str = data.get('Date')
        if not date_str: # fallback to today
            dt = datetime.now()
        else:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            
        input_dict['Year'] = dt.year
        input_dict['Month'] = dt.month
        input_dict['Day'] = dt.day
        
        # 3. One-hot encode the selected City
        selected_city = data.get('City')
        city_col_name = f"City_{selected_city}"
        
        # If the city was dropped (e.g. Ahmedabad), it naturally remains all 0s, which is correct!
        if city_col_name in input_dict:
            input_dict[city_col_name] = 1.0 
            
        # 4. Predict
        # Convert dictionary directly into a dataframe row holding exact column order
        df_input = pd.DataFrame([input_dict], columns=features)
        
        # Transform data using pre-fitted scaler
        scaled_input = scaler.transform(df_input)
        
        # Predict the AQI
        prediction = model.predict(scaled_input)[0]
        prediction_val = round(float(prediction), 2)
        
        # Determine AQI Category and UI Color
        if prediction_val <= 50:
            category, color = "Good 😊", "#4ade80" # Green
        elif prediction_val <= 100:
            category, color = "Moderate 😐", "#facc15" # Yellow
        elif prediction_val <= 200:
            category, color = "Unhealthy ☹️", "#f97316" # Orange
        elif prediction_val <= 300:
            category, color = "Very Unhealthy 😷", "#ef4444" # Red
        else:
            category, color = "Hazardous ☠️", "#7f1d1d" # Dark Red
            
        return jsonify({
            "success": True,
            "aqi": prediction_val,
            "category": category,
            "color": color
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
