import yaml
from prophet.diagnostics import cross_validation
from workflows.worflows import run_langgraph_report
from fastapi import FastAPI
import requests
import json
from prophet import Prophet
from fastapi.middleware.cors import CORSMiddleware
from schemas.schema import CityRequest, PipelineData
from fastapi.responses import JSONResponse
import pandas as pd
from data_ingestion import fetch_weather_data
from load_config import load_config
config = load_config()
FORECAST_H = config['forecast']['horizon']

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins = ["*"], 
    allow_credentials = True, 
    allow_methods = ["*"], 
    allow_headers = ["*"]
)

@app.post("/predict")
async def predict(request:CityRequest):
    if not request.city:
        return JSONResponse(content = "Please enter a valid city", status_code = 400)
    city = request.city
    country = request.country
    data = fetch_weather_data(city)
    if not data.empty:
        model = Prophet()
        print(data.head(), "Successfully retrieved the data")
        data = data[['timestamp' , 'temperature']].copy()
        data.columns = ['ds', 'y']
        model.fit(data)
        future = model.make_future_dataframe(periods = 12, freq = 'H')
        forecast = model.predict(future)
        df_cv = cross_validation(model , initial = config['model']['initial'] , period = config['model']['period'] , horizon = config['model']['horizon'])

        forecast_values = forecast[forecast['ds'] >= future['ds'].iloc[-FORECAST_H]]

        result = run_langgraph_report(forecast_data = forecast_values, city = city , country = country )
        print(result['report'])
        print(type(result['report']))
        return JSONResponse(content = result['report'] , status_code = 200)

    else:
        return JSONResponse(content = "Failed to retrieve data.", status_code = 400)       

@app.post("/pipeline")
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
        print(f"Head values: {df.head()}")
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