# AQI Dashboard — Setup Guide

## File Structure
```
your-project/
├── app.py
├── aqi_model.pkl.gz       ← your existing file
├── scaler.pkl             ← your existing file
├── city_day.csv           ← your existing training data
├── AQI.csv                ← station data for dashboard charts
├── requirements.txt
├── Procfile
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Run Locally
```bash
pip install -r requirements.txt
python app.py
# Visit: http://localhost:5000
```

## Deploy to Render / Railway / Heroku
Same as before — just push the new files. The Procfile stays unchanged.
