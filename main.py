import yaml
from prophet.diagnostics import cross_validation
from workflows.worflows import run_langgraph_report
from prophet import Prophet
import pandas as pd
from data_ingestion import fetch_weather_data
from load_config import load_config
config = load_config()
FORECAST_H = config['forecast']['horizon']


def pipeline(data, city , country):
        hourly = data.get("hourly", {})
        df = pd.DataFrame({
                "timestamp": pd.to_datetime(hourly.get("time", [])),
                "temperature": hourly.get("temperature_2m", []),
                "humidity": hourly.get("relative_humidity_2m", []),
                "precipitation": hourly.get("precipitation", [])
            })
        if not df.empty:
                model = Prophet()
                df = df.sort_values(by="timestamp").reset_index(drop=True)
                data = data[['timestamp' , 'temperature']].copy()
                data.columns = ['ds', 'y']
                model.fit(data)
                future = model.make_future_dataframe(periods = 12, freq = 'H')
                forecast = model.predict(future)
                df_cv = cross_validation(model , initial = config['model']['initial'] , period = config['model']['period'] , horizon = config['model']['horizon'])
                
                forecast_values = forecast[forecast['ds'] >= future['ds'].iloc[-FORECAST_H]]

                return run_langgraph_report(forecast_data = forecast_values, city = city , country = country)['report']
        else:
                print("No dataframe loaded")

if __name__=="__main__":
        model = Prophet()
        data = fetch_weather_data(config['location']['city'])
        if not data.empty:
                print(data.head(), "Successfully retrieved the data")
        else:
                print("Failed to retrieve data.")
        data = data[['timestamp' , 'temperature']].copy()
        data.columns = ['ds', 'y']
        model.fit(data)
        future = model.make_future_dataframe(periods = 12, freq = 'H')
        forecast = model.predict(future)
        df_cv = cross_validation(model , initial = config['model']['initial'] , period = config['model']['period'] , horizon = config['model']['horizon'])

        forecast_values = forecast[forecast['ds'] >= future['ds'].iloc[-FORECAST_H]]

        print(run_langgraph_report(forecast_data = forecast_values)['report'])


