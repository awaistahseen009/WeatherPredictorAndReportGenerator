import pandas as pd
import matplotlib.pyplot as plt
from typing import TypedDict, Optional, List, Dict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from schemas.schema import WeatherReport , WeatherState , RestaurantInfo , CityRequest
from datetime import datetime
from langchain_core.runnables import RunnableLambda
import json
import os
from minio import Minio
from utils.utils import upload_to_minio
from tavily import TavilyClient
import base64
from io import BytesIO
import logging
import requests
from pydantic import BaseModel, Field, ValidationError

# Suppress prophet and cmdstanpy logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
from load_config import load_config
load_dotenv()

config = load_config()

# Analyzing forecast data to extract key metrics and patterns
def analyze_forecast(state: WeatherState) -> WeatherState:
    print(state)
    df = state["forecast_data"]
    city = state["city"]  # Access from state
    country = state["country"]  # Access from state
    
    max_temp = df["yhat"].max()
    print(f"Max temp: {max_temp}")
    min_temp = df["yhat"].min()
    max_time = df.loc[df["yhat"].idxmax(), "ds"].strftime("%I %p")
    min_time = df.loc[df["yhat"].idxmin(), "ds"].strftime("%I %p")
    
    temp_diff = df["yhat"].diff().abs()
    print(f"Temp diff: {temp_diff.max()}")
    notable_change = temp_diff.max()
    print(f"Notable change: {notable_change}")
    change_time = df.loc[temp_diff.idxmax(), "ds"].strftime("%I %p") if notable_change > 2 else None
    print(f"Change time: {change_time}")
    
    trend = "warming" if df["yhat"].iloc[-1] > df["yhat"].iloc[0] else "cooling" if df["yhat"].iloc[-1] < df["yhat"].iloc[0] else "stable"
    
    afternoon_temps = df[(df["ds"].dt.hour >= 12) & (df["ds"].dt.hour < 17)]["yhat"].mean()
    night_temps = df[(df["ds"].dt.hour >= 0) & (df["ds"].dt.hour < 6)]["yhat"].mean()
    pattern = "warmest in the afternoon, coolest at night" if afternoon_temps > night_temps else "stable throughout"

    analysis = {
        "max_temp": max_temp,
        "min_temp": min_temp,
        "max_time": max_time,
        "min_time": min_time,
        "notable_change": notable_change if notable_change > 2 else None,
        "change_time": change_time,
        "trend": trend,
        "pattern": pattern,
        "afternoon_temps": afternoon_temps,
        "night_temps": night_temps
    }
    
    # Return the updated state
    new_state = dict(state)
    new_state["analysis"] = analysis
    return new_state

# Generating Matplotlib forecast plot and saving as base64
def generate_plot(state: WeatherState) -> WeatherState:
    city = state['city']
    forecast_values = state["forecast_data"].copy()
    forecast_values['ds'] = pd.to_datetime(forecast_values['ds'])
    forecast_values.set_index('ds', inplace=True)
    forecast_values['date_hour'] = forecast_values.index.strftime('%Y-%m-%d %Hh')

    plt.figure(figsize=(12, 8))
    plt.plot(forecast_values['date_hour'], forecast_values['yhat'], 
             color='blue', label='Forecast', linewidth=2, marker='o', markersize=4)
    
    # Add confidence intervals if available
    if 'yhat_lower' in forecast_values.columns and 'yhat_upper' in forecast_values.columns:
        plt.fill_between(
            forecast_values['date_hour'],
            forecast_values['yhat_lower'],
            forecast_values['yhat_upper'],
            color='blue',
            alpha=0.2,
            label='Confidence Interval'
        )
    
    plt.title(f'Hourly Temperature Forecast for {city}', fontsize=16, fontweight='bold')
    plt.xlabel('Date and Hour', fontsize=12)
    plt.ylabel('Predicted Temperature (°C)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot as base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    print(f"Generated plot data length: {len(plot_data)}")
    
    # Return the updated state
    new_state = dict(state)
    new_state["plot_data"] = plot_data
    return new_state

# Helper function to determine weather suitability for restaurants
def get_weather_suitability(restaurant_name: str, restaurant_location: str, analysis: dict) -> str:
    """Determine why a restaurant is suitable based on weather conditions"""
    max_temp = analysis["max_temp"]
    min_temp = analysis["min_temp"]
    trend = analysis["trend"]
    
    # Check for outdoor dining keywords
    outdoor_keywords = ["terrace", "garden", "patio", "outdoor", "rooftop", "balcony"]
    indoor_keywords = ["cozy", "warm", "indoor", "heated", "cafe", "bistro"]
    
    restaurant_text = f"{restaurant_name} {restaurant_location}".lower()
    
    if any(keyword in restaurant_text for keyword in outdoor_keywords):
        if max_temp > 18 and min_temp > 12:
            return "Perfect for outdoor dining with pleasant temperatures"
        elif max_temp > 15:
            return "Good for outdoor seating during warmer hours"
        else:
            return "Cozy indoor seating recommended due to cooler weather"
    
    elif any(keyword in restaurant_text for keyword in indoor_keywords):
        if min_temp < 15:
            return "Warm and cozy atmosphere perfect for cooler weather"
        else:
            return "Comfortable indoor dining with great ambiance"
    
    # General recommendations based on temperature
    if max_temp > 20:
        return "Great weather for dining - indoor or outdoor options available"
    elif max_temp > 15:
        return "Pleasant temperatures for a comfortable dining experience"
    else:
        return "Perfect for warm, cozy indoor dining"

# Searching for local events, restaurants, and city image using Tavily
def search_events_restaurants(state: WeatherState) -> WeatherState:
    city = state['city']
    country = state['country']
    date_str = datetime.now().strftime("%B %d, %Y")
    analysis = state["analysis"]
    
    # Default values
    event_info = f"No specific festivals found for {date_str}"
    restaurant_info = []
    event_image = None
    restaurant_image = None
    city_image = None
    
    # Try to use Tavily if API key is available
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        try:
            tavily = TavilyClient(api_key=tavily_api_key)
            
            # Search for events
            event_query = f"current festivals events {city} {country} {date_str}"
            event_results = tavily.search(query=event_query, max_results=5, include_images=True)
            
            festivals = []
            for result in event_results.get('results', []):
                if any(keyword in result['title'].lower() for keyword in ['festival', 'event', 'concert', 'exhibition']):
                    festivals.append(result['title'])
            
            if festivals:
                event_info = ", ".join(festivals[:3])
            
            # Search for restaurants with locations
            restaurant_query = f"best restaurants {city} {country} with addresses locations"
            restaurant_results = tavily.search(query=restaurant_query, max_results=8, include_images=True)
            
            restaurants_found = []
            for result in restaurant_results.get('results', []):
                title = result['title']
                content = result['content']
                
                # Extract restaurant information
                if any(keyword in title.lower() for keyword in ['restaurant', 'cafe', 'bistro', 'dining', 'eatery']):
                    # Try to extract location information from content
                    location = "Location not specified"
                    
                    # Common patterns for location extraction
                    location_patterns = [
                        f"{city}",
                        "address:", "located at", "situated in", "found at", "street", "avenue", "road"
                    ]
                    
                    # Simple location extraction
                    content_lower = content.lower()
                    for pattern in location_patterns:
                        if pattern in content_lower:
                            # Extract a reasonable portion that might contain location
                            start_idx = content_lower.find(pattern)
                            location_snippet = content[start_idx:start_idx+100]
                            if len(location_snippet) > 10:
                                location = location_snippet.split('.')[0].strip()
                                break
                    
                    # Clean up the restaurant name
                    restaurant_name = title.split(' - ')[0].split(' | ')[0].strip()
                    
                    # Get weather suitability
                    weather_suitability = get_weather_suitability(restaurant_name, location, analysis)
                    
                    restaurant_info_dict = {
                        "name": restaurant_name,
                        "location": location,
                        "weather_suitability": weather_suitability
                    }
                    
                    restaurants_found.append(restaurant_info_dict)
                    
                    if len(restaurants_found) >= 4:  # Limit to 4 restaurants
                        break
            
            restaurant_info = restaurants_found
            
            # Search for images
            image_query = f"{city} restaurants food dining"
            image_results = tavily.search(query=image_query, max_results=5, include_images=True)
            
            # Fetch event and restaurant images
            all_images = event_results.get('images', []) + restaurant_results.get('images', []) + image_results.get('images', [])
            
            for image_url in all_images:
                try:
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        image_data = base64.b64encode(response.content).decode('utf-8')
                        if not event_image and any(keyword in image_url.lower() for keyword in ['festival', 'event', 'concert']):
                            event_image = image_data
                        elif not restaurant_image and any(keyword in image_url.lower() for keyword in ['restaurant', 'food', 'dining', 'cafe']):
                            restaurant_image = image_data
                        
                        if event_image and restaurant_image:
                            break
                except Exception as e:
                    logger.warning(f"Failed to fetch image {image_url}: {e}")
            
            # Search for city image
            try:
                city_results = tavily.search(query=f"{city} {country} city skyline landmarks", max_results=3, include_images=True)
                for image_url in city_results.get('images', []):
                    try:
                        response = requests.get(image_url, timeout=10)
                        if response.status_code == 200 and not city_image:
                            city_image = base64.b64encode(response.content).decode('utf-8')
                            break
                    except Exception as e:
                        logger.warning(f"Failed to fetch city image {image_url}: {e}")
            except Exception as e:
                logger.warning(f"Tavily search for city image failed: {e}")
                
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
    else:
        logger.warning("TAVILY_API_KEY not found. Using default values.")
        # Create some default restaurant recommendations
        restaurant_info = [
            {
                "name": f"Local Restaurant in {city}",
                "location": f"City Center, {city}",
                "weather_suitability": get_weather_suitability("Local Restaurant", "City Center", analysis)
            },
            {
                "name": f"Cozy Cafe {city}",
                "location": f"Main Street, {city}",
                "weather_suitability": get_weather_suitability("Cozy Cafe", "Main Street", analysis)
            }
        ]
    
    # Return the updated state
    new_state = dict(state)
    new_state.update({
        "event_info": event_info,
        "restaurant_info": restaurant_info,
        "event_image": event_image,
        "restaurant_image": restaurant_image,
        "city_image": city_image
    })
    return new_state

# Generating detailed weather report using LLM
def generate_report(state: WeatherState) -> WeatherState:
    print("Generating the report")
    analysis = state["analysis"]
    historical_avg = state.get("historical_avg")
    event_info = state.get("event_info")
    restaurant_info = state.get("restaurant_info", [])
    city = state['city']
    country = state['country']
    date_str = datetime.now().strftime("%B %d, %Y")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Format restaurant information for the prompt
    restaurant_text = ""
    if restaurant_info:
        restaurant_text = "Restaurants: " + "; ".join([
            f"{r['name']} at {r['location']}" for r in restaurant_info
        ])
    
    prompt = ChatPromptTemplate.from_template(
        """Based on this weather forecast analysis for {city}, {country} on {date_str}:
        - Max temperature: {max_temp:.1f}°C at {max_time}
        - Min temperature: {min_temp:.1f}°C at {min_time}
        - Trend: {trend}
        - Pattern: {pattern}
        - Notable change: {notable_change}
        - Historical average: {historical_avg}
        - Local events: {event_info}
        - {restaurant_text}
        
        Generate a detailed, friendly weather report in JSON format. Include:
        - "header": Title (e.g., "Weather Forecast for {city}, {country}, {date_str}")
        - "temperature_overview": Detailed temperature range and times as a STRING (e.g., "Low of 24.0°C at 12 AM, high of 33.1°C at 3 PM")
        - "general_trend": Descriptive trend explanation as a STRING
        - "key_patterns": Detailed temperature patterns as a STRING
        - "notable_changes": Significant temperature shifts as a STRING (if any)
        - "clothing_recommendations": Detailed clothing suggestions as a STRING
        - "activity_suggestions": Specific activity recommendations as a STRING
        - "weather_context": Historical comparison as a STRING (null if not provided)
        - "additional_tips": Practical tips as a STRING (null if not applicable)
        - "local_events": Event recommendations tied to weather as a STRING
        - "restaurant_recommendations": List of restaurants with their locations and weather suitability
        
        For restaurant_recommendations, create a list of objects with "name", "location", and "weather_suitability" fields.
        Use vivid, non-technical language. Keep under 250 words total. Return ONLY valid JSON without any markdown formatting."""
    )
    
    notable_change_str = f"Quick {analysis['notable_change']:.1f}°C change at {analysis['change_time']}" if analysis["notable_change"] else "No significant changes"
    historical_avg_str = f"{historical_avg:.1f}°C" if historical_avg else "not available"
    
    max_retries = 2
    retry_count = 0
    report_dict = None
    
    while retry_count <= max_retries:
        try:
            report_json_str = llm.invoke(prompt.format(
                max_temp=analysis["max_temp"],
                min_time=analysis["min_time"],
                min_temp=analysis["min_temp"],
                max_time=analysis["max_time"],
                trend=analysis["trend"],
                pattern=analysis["pattern"],
                notable_change=notable_change_str,
                historical_avg=historical_avg_str,
                city=city,
                country=country,
                date_str=date_str,
                event_info=event_info,
                restaurant_text=restaurant_text
            )).content
            
            # Clean the response to extract JSON
            report_json_str = report_json_str.strip()
            if report_json_str.startswith('```json'):
                report_json_str = report_json_str[7:-3]
            elif report_json_str.startswith('```'):
                report_json_str = report_json_str[3:-3]
            
            report_dict = json.loads(report_json_str)
            
            # Ensure restaurant_recommendations is properly formatted
            if 'restaurant_recommendations' not in report_dict and restaurant_info:
                report_dict['restaurant_recommendations'] = [
                    {
                        "name": r["name"],
                        "location": r["location"],
                        "weather_suitability": r["weather_suitability"]
                    }
                    for r in restaurant_info
                ]
            
            # Validate with Pydantic
            WeatherReport(**report_dict)
            break  # Exit loop if validation succeeds
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Report generation attempt {retry_count + 1} failed: {e}")
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Failed to generate valid report after {max_retries} retries")
                # Fallback report
                report_dict = {
                    "header": f"Weather Forecast for {city}, {country}, {date_str}",
                    "temperature_overview": f"Low of {analysis['min_temp']:.1f}°C at {analysis['min_time']}, high of {analysis['max_temp']:.1f}°C at {analysis['max_time']}",
                    "general_trend": f"The day will be {analysis['trend']}",
                    "key_patterns": analysis["pattern"],
                    "notable_changes": notable_change_str,
                    "clothing_recommendations": "Wear appropriate clothing for the temperature range",
                    "activity_suggestions": "Plan activities based on the weather conditions",
                    "weather_context": f"Historical average: {historical_avg_str}" if historical_avg else None,
                    "additional_tips": "Stay informed about weather changes",
                    "local_events": event_info,
                    "restaurant_recommendations": [
                        {
                            "name": r["name"],
                            "location": r["location"],
                            "weather_suitability": r["weather_suitability"]
                        }
                        for r in restaurant_info
                    ] if restaurant_info else None
                }
                try:
                    WeatherReport(**report_dict)  # Validate fallback
                except ValidationError as e:
                    logger.error(f"Fallback report validation failed: {e}")
    
    print("Finished with the report")
    print(report_dict)
    
    # Return the updated state
    new_state = dict(state)
    new_state["report"] = report_dict
    return new_state
def generate_html(state: WeatherState):
    report = state["report"]
    plot_data = state.get("plot_data")
    event_image = state.get("event_image")
    restaurant_image = state.get("restaurant_image")
    city_image = state.get("city_image")
    event_info = state.get("event_info")
    restaurant_info = state.get("restaurant_info", [])
    city = state['city']
    country = state['country']
    date_str = datetime.now().strftime("%B %d, %Y")
    print(f"Generating HTML with plot_data length: {len(plot_data) if plot_data else 0}")
    print(f"City image available: {city_image is not None}")
    print(f"Event image available: {event_image is not None}")
    print(f"Restaurant image available: {restaurant_image is not None}")
    
    # Generate restaurant recommendations HTML
    restaurant_html = ""
    if report.get('restaurant_recommendations'):
        restaurant_html = """
        <div class="bg-white rounded-2xl shadow-xl p-8 mb-8 fade-in">
            <h2 class="text-3xl font-bold text-gray-800 mb-6 text-center">Restaurant Recommendations</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        """
        
        for restaurant in report['restaurant_recommendations']:
            restaurant_html += f"""
                <div class="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-6 border border-gray-200">
                    <h3 class="text-xl font-bold text-gray-800 mb-2">{restaurant['name']}</h3>
                    <p class="text-gray-600 mb-2">
                        <span class="font-semibold">📍 Location:</span> {restaurant['location']}
                    </p>
                    <p class="text-gray-700 text-sm">
                        <span class="font-semibold">🌤️ Weather Suitability:</span> {restaurant['weather_suitability']}
                    </p>
                </div>
            """
        
        restaurant_html += "</div></div>"
    
    # Prepare event information with specific details
    event_display = event_info if event_info and event_info != f"No specific festivals found for {date_str}" else "No specific events found for today. Check local listings for other activities."
    
    # Prepare dining tips
    dining_tips = report.get('additional_tips', 'Explore a variety of dining experiences tailored to the weather!')
    if restaurant_info:
        dining_tips = f"Enjoy dining at top spots like {', '.join([r['name'] for r in restaurant_info[:2]])} or explore more local cuisines."

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weather Forecast for {city}, {country}, {date_str}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes fadeIn {{ 
            from {{ opacity: 0; transform: translateY(20px); }} 
            to {{ opacity: 1; transform: translateY(0); }} 
        }}
        .fade-in {{ animation: fadeIn 0.8s ease-out; }}
        .card {{ 
            transition: transform 0.3s, box-shadow 0.3s; 
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        }}
        .card:hover {{ 
            transform: translateY(-5px); 
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2); 
        }}
        .plot-container {{
            max-width: 100%;
            overflow: hidden;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        .plot-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}
    </style>
</head>
<body class="bg-gradient-to-br from-blue-100 via-purple-50 to-pink-100 min-h-screen font-sans">
    <div class="container mx-auto p-6 max-w-6xl">
        <header class="text-center mb-8">
            <h1 class="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600 mb-4 fade-in">
                {report['header']}
            </h1>
            <p class="text-xl text-gray-600 fade-in">Complete Weather Analysis & Local Recommendations</p>
        </header>
        
        {f'''<div class="mb-8 fade-in">
            <div class="bg-white rounded-2xl shadow-xl overflow-hidden">
                <img src="data:image/jpeg;base64,{city_image}" alt="{state['city']} Cityscape" class="w-full h-80 object-cover">
                <div class="p-4 bg-gradient-to-r from-blue-500 to-purple-500 text-white">
                    <h2 class="text-2xl font-bold">Beautiful {state['city']}, {country}</h2>
                    <p class="text-blue-100">Perfect backdrop for today's weather</p>
                </div>
            </div>
        </div>''' if city_image else f'<div class="text-center text-gray-500 mb-8 fade-in bg-white rounded-2xl p-8 shadow-lg"><p class="text-xl">Weather Report for {city}, {country}</p></div>'}
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <div class="bg-white rounded-2xl shadow-xl p-8 card fade-in">
                <div class="flex items-center mb-4">
                    <div class="w-4 h-4 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full mr-3"></div>
                    <h2 class="text-2xl font-bold text-gray-800">Temperature Overview</h2>
                </div>
                <p class="text-gray-700 text-lg leading-relaxed">{report['temperature_overview']}</p>
            </div>
            
            <div class="bg-white rounded-2xl shadow-xl p-8 card fade-in">
                <div class="flex items-center mb-4">
                    <div class="w-4 h-4 bg-gradient-to-r from-green-500 to-blue-500 rounded-full mr-3"></div>
                    <h2 class="text-2xl font-bold text-gray-800">General Trend</h2>
                </div>
                <p class="text-gray-700 text-lg leading-relaxed">{report['general_trend']}</p>
            </div>
        </div>
        
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            <div class="bg-white rounded-2xl shadow-xl p-8 card fade-in">
                <h2 class="text-xl font-bold text-gray-800 mb-4">Key Patterns</h2>
                <p class="text-gray-700 leading-relaxed">{report['key_patterns']}</p>
            </div>
            
            <div class="bg-white rounded-2xl shadow-xl p-8 card fade-in">
                <h2 class="text-xl font-bold text-gray-800 mb-4">Notable Changes</h2>
                <p class="text-gray-700 leading-relaxed">{report['notable_changes']}</p>
            </div>
            
            <div class="bg-white rounded-2xl shadow-xl p-8 card fade-in">
                <h2 class="text-xl font-bold text-gray-800 mb-4">Clothing Recommendations</h2>
                <p class="text-gray-700 leading-relaxed">{report['clothing_recommendations']}</p>
            </div>
        </div>
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <div class="bg-white rounded-2xl shadow-xl p-8 card fade-in">
                <h2 class="text-2xl font-bold text-gray-800 mb-4">Activity Suggestions</h2>
                <p class="text-gray-700 text-lg leading-relaxed">{report['activity_suggestions']}</p>
            </div>
            
            {f'''<div class="bg-white rounded-2xl shadow-xl p-8 card fade-in">
                <h2 class="text-2xl font-bold text-gray-800 mb-4">Weather Context</h2>
                <p class="text-gray-700 text-lg leading-relaxed">{report['weather_context']}</p>
            </div>''' if report.get('weather_context') else ''}
        </div>
        
        {f'''<div class="mb-8 fade-in">
            <h2 class="text-3xl font-bold text-center text-gray-800 mb-6">24-Hour Temperature Forecast</h2>
            <div class="plot-container bg-white p-6 rounded-2xl shadow-xl">
                <img src="data:image/png;base64,{plot_data}" alt="Hourly Temperature Forecast" class="w-full">
            </div>
        </div>''' if plot_data else '<div class="text-center text-red-500 text-xl mb-8 fade-in bg-white rounded-2xl p-8 shadow-lg">Forecast plot unavailable</div>'}
        
        {restaurant_html}
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            {f'''<div class="bg-white rounded-2xl shadow-xl overflow-hidden fade-in">
                <img src="data:image/jpeg;base64,{event_image}" alt="Local Events" class="w-full h-64 object-cover">
                <div class="p-6">
                    <h2 class="text-2xl font-bold text-gray-800 mb-3">Local Events</h2>
                    <p class="text-gray-700 text-lg">{event_display}</p>
                </div>''' if event_image else f'''<div class="bg-white rounded-2xl shadow-xl p-8 fade-in">
                <h2 class="text-2xl font-bold text-gray-800 mb-3">Local Events</h2>
                <p class="text-gray-700 text-lg">{event_display}</p>'''}
            </div>
            
            {f'''<div class="bg-white rounded-2xl shadow-xl overflow-hidden fade-in">
                <img src="data:image/jpeg;base64,{restaurant_image}" alt="More Dining Options" class="w-full h-64 object-cover">
                <div class="p-6">
                    <h2 class="text-2xl font-bold text-gray-800 mb-3">More Dining Options</h2>
                    <p class="text-gray-700 text-lg">{dining_tips}</p>
                </div>
            </div>''' if restaurant_image else f'''<div class="bg-white rounded-2xl shadow-xl p-8 fade-in">
                <h2 class="text-2xl font-bold text-gray-800 mb-3">More Dining Options</h2>
                <p class="text-gray-700 text-lg">{dining_tips}</p>
            </div>'''}
        </div>
        
        <footer class="text-center text-gray-500 text-sm mt-12 fade-in">
            <p>© {datetime.now().year} Weather Report Generator. All rights reserved.</p>
            <p>Powered by LangGraph, OpenAI, and Tavily</p>
        </footer>
    </div>
</body>
</html>"""
    
    # Save the HTML content to a file
    output_dir = "weather_reports"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"weather_report_{city.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    # html_path = os.path.join(output_dir, file_name)
    
    # with open(html_path, "w", encoding="utf-8") as f:
    #     f.write(html_content)
    html_path = upload_to_minio(html_content , "weather-reports-html",file_name) 
    print(f"Generated HTML report at: {html_path}")
    
    # Save the report JSON to a file as well
    report_file_name = f"weather_report_data_{city.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # report_path = os.path.join(output_dir, report_file_name)
    # with open(report_path, "w", encoding="utf-8") as f:
    #     json.dump(report, f, indent=4)
    report_path = upload_to_minio(report , "weather-reports-json",report_file_name)   
    print(f"Successfully upload the JSON report at bucket")
    
    # Return the updated state
    new_state = dict(state)
    new_state["html_path"] = html_path
    new_state["report_path"] = report_path
    return new_state

def create_langgraph_workflow():
    workflow = StateGraph(WeatherState)

    # Add nodes to the workflow
    workflow.add_node("analyze_forecast", analyze_forecast)  # Use the function directly
    workflow.add_node("generate_plot", generate_plot)
    workflow.add_node("search_events_restaurants", search_events_restaurants)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("generate_html", generate_html)

    # Set up the graph
    workflow.set_entry_point("analyze_forecast")
    workflow.add_edge("analyze_forecast", "generate_plot")
    workflow.add_edge("generate_plot", "search_events_restaurants")
    workflow.add_edge("search_events_restaurants", "generate_report")
    workflow.add_edge("generate_report", "generate_html")
    workflow.add_edge("generate_html", END)

    # Compile the graph
    return workflow.compile()

# Ensure run_langgraph_report initializes the state correctly
def run_langgraph_report(forecast_data: pd.DataFrame, city: str, country: str, historical_avg: Optional[float] = None) -> str:
    workflow = create_langgraph_workflow()
    result = workflow.invoke({
        "forecast_data": forecast_data,
        "historical_avg": historical_avg,
        "city": city,
        "country": country
    })
    return result

# Example usage
if __name__ == "__main__":
    # Create dummy forecast data for demonstration
    today = datetime.now()
    dates = pd.to_datetime([today + pd.Timedelta(hours=i) for i in range(24)])
    temperatures = [20 + (i % 5) * 0.5 - (i % 7) * 0.2 for i in range(24)]  # Example temperatures
    forecast_df = pd.DataFrame({'ds': dates, 'yhat': temperatures})

    # Run the report
    html_path = run_langgraph_report(forecast_df, historical_avg=18.5)
    print(f"\nFinal HTML report available at: {html_path}")