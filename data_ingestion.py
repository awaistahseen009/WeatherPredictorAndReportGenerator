import requests
import pandas as pd
from datetime import datetime

def fetch_weather_data(city, start_date="2025-01-01"):
    """
    Fetch historical weather data for a city from Open-Meteo API and return a DataFrame.
    
    Args:
        city (str): Name of the city (e.g., 'London').
        start_date (str): Start date for historical data in 'YYYY-MM-DD' format (default: '2025-01-01').
    
    Returns:
        pd.DataFrame: DataFrame with timestamp, temperature, humidity, and precipitation.
    """
    # Configuration
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    END_DATE = datetime.now().strftime("%Y-%m-%d")  # Today’s date
    
    # Fetch city coordinates
    try:
        geo_params = {"name": city, "count": 1}
        geo_response = requests.get(GEOCODING_URL, params=geo_params)
        if geo_response.status_code != 200 or not geo_response.json().get("results"):
            print(f"Error: Could not find coordinates for {city}")
            return pd.DataFrame()
        
        coords = geo_response.json()["results"][0]
        lat, lon = coords["latitude"], coords["longitude"]
    except Exception as e:
        print(f"Error fetching coordinates: {e}")
        return pd.DataFrame()
    
    # Fetch historical weather data
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation"
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            hourly = data.get("hourly", {})
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(hourly.get("time", [])),
                "temperature": hourly.get("temperature_2m", []),
                "humidity": hourly.get("relative_humidity_2m", []),
                "precipitation": hourly.get("precipitation", [])
            })
            if not df.empty:
                df = df.sort_values(by="timestamp").reset_index(drop=True)
            return df
        else:
            print(f"Error fetching weather data: {response.status_code} - {response.text}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return pd.DataFrame()