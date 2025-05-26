import requests
import pandas as pd
from datetime import datetime
import os

def fetch_weather(api_key, lat = 55.7522, lon = 37.6156, city="Moscow"):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    
    return {
        'datetime': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S'),
        'city': city,
        'weather_main': data['weather'][0]['main'],
        'weather_description': data['weather'][0]['description'],
        'temp': data['main']['temp'],
        'feels_like': data['main']['feels_like'],
        'pressure': data['main']['pressure'],
        'wind_speed': data['wind']['speed']
    }

def save_weather_to_csv(api_key):  # Теперь явно принимает api_key
    weather_data = fetch_weather(api_key)
    df = pd.DataFrame([weather_data])
    
    file_path = '/opt/airflow/data/weather.csv'
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)