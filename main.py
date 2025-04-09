from flask import Flask, render_template,request,jsonify
import pandas as pd
import requests
import joblib 
from datetime import datetime

app = Flask(__name__)
model = joblib.load("weather_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    data = requests.get("http://api.weatherapi.com/v1/current.json?key=4b950b61e4944d4aaf091607250904&q=mumbai&aqi=no")
    data = data.json()

    # Extract values
    precipitation = data["current"]["precip_mm"]
    temp_min = data["current"]["temp_c"]  # Approximation
    wind = data["current"]["wind_kph"]

    # Parse date
    localtime_str = data["location"]["localtime"]
    dt = datetime.strptime(localtime_str, "%Y-%m-%d %H:%M")
    month = dt.month
    day = dt.day
    dayofweek = dt.weekday()  # Monday=0, Sunday=6

    X_input = pd.DataFrame([[precipitation, temp_min, wind, month, day, dayofweek]], columns=["precipitation", "temp_avg", "wind", "month", "day", "dayofweek"])
    # Make prediction
    predicted_class = model.predict(X_input)[0]

    # Decode it
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    match predicted_label:
        case "sun":
            predicted_label = "☀️ Sunny"
        case "rain":
            predicted_label = "🌧️ Rainy"
        case "fog":
            predicted_label = "🌫️ Foggy"
        case "drizzle":
            predicted_label = "🌦️ Drizzly"
        case "snow":
            predicted_label = "❄️ Snowy"

    return render_template('home.html', data=data, prediction=predicted_label)


# API endpoint to predict weather
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Parse input
        precipitation = float(data['precipitation'])
        temp_avg = float(data['temp_avg'])
        wind = float(data['wind'])
        date_str = data['date']  # format: YYYY-MM-DD

        dt = datetime.strptime(date_str, "%Y-%m-%d")
        month = dt.month
        day = dt.day
        dayofweek = dt.weekday()

        X_input = pd.DataFrame([[precipitation, temp_avg, wind, month, day, dayofweek]],
                               columns=["precipitation", "temp_avg", "wind", "month", "day", "dayofweek"])
        
        # Predict
        predicted_class = model.predict(X_input)[0]
        label = label_encoder.inverse_transform([predicted_class])[0]

        # Add emojis
        match label:
            case "sun":
                label = "☀️ Sunny"
            case "rain":
                label = "🌧️ Rainy"
            case "fog":
                label = "🌫️ Foggy"
            case "drizzle":
                label = "🌦️ Drizzly"
            case "snow":
                label = "❄️ Snowy"

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
app.run(debug=True)