import pickle, gzip, os
from flask import Flask, request, jsonify, render_template
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# ── Load Model & Scaler ──────────────────────────────────────────────────────
with gzip.open('aqi_model.pkl.gz', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

features   = list(scaler.feature_names_in_)
city_cols  = [c for c in features if c.startswith('City_')]
cities_enc = [c.replace('City_', '') for c in city_cols]

# ── Load Raw Dataset ─────────────────────────────────────────────────────────
try:
    raw_df     = pd.read_csv('city_day.csv')
    all_cities = sorted(raw_df['City'].dropna().unique().tolist())
except Exception:
    all_cities = cities_enc

# ── Load AQI station data for dashboard charts ───────────────────────────────
try:
    station_df = pd.read_csv('AQI.csv')
    station_df.columns = station_df.columns.str.strip()
    # Normalize column names (handle both cases)
    col_map = {c: c.lower() for c in station_df.columns}
    station_df.rename(columns=col_map, inplace=True)
    HAS_STATION_DATA = True
except Exception:
    HAS_STATION_DATA = False
    station_df = pd.DataFrame()


# ── Helper: get top cities by PM2.5 ──────────────────────────────────────────
def get_top_cities(n=10):
    if not HAS_STATION_DATA:
        return []
    pm = station_df[station_df['pollutant_id'] == 'PM2.5']
    top = pm.groupby('city')['pollutant_avg'].mean().nlargest(n).reset_index()
    return top.rename(columns={'city': 'City', 'pollutant_avg': 'PM2_5'}).to_dict('records')


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', cities=all_cities)


@app.route('/api/stats')
def api_stats():
    if not HAS_STATION_DATA:
        return jsonify({})

    num_cities    = int(station_df['city'].nunique())
    num_stations  = int(station_df['station'].nunique())
    num_states    = int(station_df['state'].nunique())

    pm25 = station_df[station_df['pollutant_id'] == 'PM2.5']
    if not pm25.empty:
        city_pm25     = pm25.groupby('city')['pollutant_avg'].mean()
        most_polluted = str(city_pm25.idxmax())
        cleanest      = str(city_pm25.idxmin())
        avg_pm25      = round(float(city_pm25.mean()), 1)
    else:
        most_polluted = 'N/A'
        cleanest      = 'N/A'
        avg_pm25      = 0

    return jsonify({
        'cities':        num_cities,
        'stations':      num_stations,
        'states':        num_states,
        'avg_pm25':      avg_pm25,
        'most_polluted': most_polluted,
        'cleanest':      cleanest,
    })


@app.route('/api/top-cities')
def api_top_cities():
    n = request.args.get('n', 12, type=int)
    pollutant = request.args.get('pollutant', 'PM2.5')
    if not HAS_STATION_DATA:
        return jsonify([])
    filtered = station_df[station_df['pollutant_id'] == pollutant]
    top = filtered.groupby('city')['pollutant_avg'].mean().nlargest(n).reset_index()
    return jsonify(top.rename(columns={'city': 'City', 'pollutant_avg': 'Value'}).to_dict('records'))


@app.route('/api/pollutant-summary')
def api_pollutant_summary():
    if not HAS_STATION_DATA:
        return jsonify([])
    summary = station_df.groupby('pollutant_id').agg(
        avg=('pollutant_avg', 'mean'),
        min_val=('pollutant_min', 'mean'),
        max_val=('pollutant_max', 'mean'),
    ).round(2).reset_index()
    return jsonify(summary.rename(columns={'pollutant_id': 'Pollutant'}).to_dict('records'))


@app.route('/api/state-summary')
def api_state_summary():
    if not HAS_STATION_DATA:
        return jsonify([])
    pm = station_df[station_df['pollutant_id'].isin(['PM2.5', 'PM10'])]
    state_avg = pm.groupby('state')['pollutant_avg'].mean().nlargest(15).reset_index()
    return jsonify(state_avg.rename(columns={'state': 'State', 'pollutant_avg': 'Value'}).to_dict('records'))


@app.route('/api/cities-list')
def api_cities_list():
    return jsonify(all_cities)


# ── Predict ───────────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_dict = {col: 0.0 for col in features}

        for key in ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']:
            input_dict[key] = float(data.get(key, 0))

        date_str = data.get('Date')
        dt = datetime.strptime(date_str, '%Y-%m-%d') if date_str else datetime.now()
        input_dict['Year']  = dt.year
        input_dict['Month'] = dt.month
        input_dict['Day']   = dt.day

        city_col = f"City_{data.get('City', '')}"
        if city_col in input_dict:
            input_dict[city_col] = 1.0

        df_input     = pd.DataFrame([input_dict], columns=features)
        scaled_input = scaler.transform(df_input)
        prediction   = round(float(model.predict(scaled_input)[0]), 2)

        if   prediction <= 50:  category, color = "Good",           "#10b981"
        elif prediction <= 100: category, color = "Moderate",       "#f59e0b"
        elif prediction <= 200: category, color = "Unhealthy",      "#f97316"
        elif prediction <= 300: category, color = "Very Unhealthy", "#ef4444"
        else:                   category, color = "Hazardous",      "#991b1b"

        return jsonify(success=True, aqi=prediction, category=category, color=color)

    except Exception as e:
        return jsonify(success=False, error=str(e)), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
