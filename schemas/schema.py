import json
from typing import TypedDict, Optional
from pydantic import BaseModel, Field
from typing import TypedDict, Optional, List, Dict

import pandas as pd
class RestaurantInfo(BaseModel):
    name: str = Field(description="Restaurant name")
    location: str = Field(description="Restaurant address or area")
    description: Optional[str] = Field(default=None, description="Brief description")
    weather_suitability: Optional[str] = Field(default=None, description="Why it's suitable for the weather")

# Pydantic model for structured report output
class WeatherReport(BaseModel):
    header: str = Field(description="Report title with date and location")
    temperature_overview: str = Field(description="Summary of temperature range and key times")
    general_trend: str = Field(description="Overall temperature trend for the day")
    key_patterns: str = Field(description="Notable daily patterns, e.g., diurnal cycles")
    notable_changes: str = Field(description="Significant temperature shifts")
    clothing_recommendations: str = Field(description="Clothing advice based on temperatures")
    activity_suggestions: str = Field(description="Suggested activities based on weather")
    weather_context: Optional[str] = Field(default=None, description="Comparison to historical norms")
    additional_tips: Optional[str] = Field(default=None, description="Tips based on humidity or other variables")
    local_events: Optional[str] = Field(default=None, description="Event-based recommendations")
    restaurant_recommendations: Optional[List[Dict[str, str]]] = Field(default=None, description="List of restaurants with locations")

# State object for LangGraph
class WeatherState(TypedDict):
    forecast_data: pd.DataFrame
    historical_avg: Optional[float]
    analysis: Optional[dict]
    plot_data: Optional[str]
    event_info: Optional[str]
    restaurant_info: Optional[List[Dict[str, str]]]
    event_image: Optional[str]
    restaurant_image: Optional[str]
    city_image: Optional[str]
    report: Optional[dict]
    html_path: Optional[str]
    report_path: Optional[str]
    city: str
    country:str

class CityRequest(BaseModel):
    city: str
    country:str

class PipelineData(BaseModel):
    data_params: Dict
    city: str
    country: str