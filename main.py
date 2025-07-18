import yaml
from prophet.diagnostics import cross_validation
import requests
import sys
from datetime import datetime
import os
import json
from schemas.schema import PipelineData
from workflows.worflows import run_langgraph_report
from prophet import Prophet
import pandas as pd
from data_ingestion import fetch_weather_data
from load_config import load_config
config = load_config()
FORECAST_H = config['forecast']['horizon']


def pipeline(request:PipelineData):
        data_params = request.data_params
        params = data_params['params']
        url = data_params['url']
        geo_response = requests.get(url, params= params)
        data = geo_response.json()
        city = request.city
        country = request.country
        hourly = data['hourly']
        # print(hourly)
        # print(f"Data: {data}")
        df = pd.DataFrame({
                "timestamp": pd.to_datetime(hourly["time"]),
                "temperature": hourly["temperature_2m"],
                "humidity": hourly["relative_humidity_2m"],
                "precipitation": hourly["precipitation"]
            })
        # print(f"Head values: {df.head()}")
        if not df.empty:
                df.head()
                model = Prophet()
                df = df.sort_values(by="timestamp").reset_index(drop=True)
                df = df[['timestamp' , 'temperature']].copy()
                df.columns = ['ds', 'y']
                model.fit(df)
                future = model.make_future_dataframe(periods = 12, freq = 'H')
                forecast = model.predict(future)
                df_cv = cross_validation(model , initial = config['model']['initial'] , period = config['model']['period'] , horizon = config['model']['horizon'])
                
                forecast_values = forecast[forecast['ds'] >= future['ds'].iloc[-FORECAST_H]]

                return run_langgraph_report(forecast_data = forecast_values, city = city , country = country)['report']
        else:
                print("No dataframe loaded")
def main(input_json):
        request = PipelineData(
        city=input_json['city'],
        country=input_json['country'],
        data_params=input_json['data_params']
    )
        result = pipeline(request)
        return json.dumps(result)

if __name__=="__main__":
    input_json = json.loads(sys.argv[1])
    result = main(input_json)
#     print("***PRINTING THE RESULTS***")
    print(json.dumps(result))

