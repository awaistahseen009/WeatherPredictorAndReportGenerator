from datetime import datetime
import json
import os
from schemas.schema import WeatherState
from minio import Minio
from minio.error import S3Error
import io
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

# Generating creative HTML report with forecast plot and images
def generate_html(state: WeatherState, city , country ):
    report = state["report"]
    plot_data = state.get("plot_data")
    event_image = state.get("event_image")
    restaurant_image = state.get("restaurant_image")
    city_image = state.get("city_image")
    event_info = state.get("event_info")
    restaurant_info = state.get("restaurant_info", [])
    city = city
    country = country
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
                <img src="data:image/jpeg;base64,{city_image}" alt="{city} Cityscape" class="w-full h-80 object-cover">
                <div class="p-4 bg-gradient-to-r from-blue-500 to-purple-500 text-white">
                    <h2 class="text-2xl font-bold">Beautiful {city}, {country}</h2>
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
    html_path = os.path.join(output_dir, file_name)
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Generated HTML report at: {html_path}")
    
    # Save the report JSON to a file as well
    report_file_name = f"weather_report_data_{city.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.join(output_dir, report_file_name)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print(f"Generated JSON report at: {report_path}")
    
    # Return the updated state
    new_state = dict(state)
    new_state["html_path"] = html_path
    new_state["report_path"] = report_path
    return new_state

def upload_to_minio(data, bucket_name, object_name):
    # Initialize MinIO client
    minio_client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),  # MinIO service hostname and port
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False  # Set to True if using HTTPS
    )

    try:
        # Ensure bucket exists, create if it doesn't
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Convert data to JSON string and upload to MinIO
        data_bytes = json.dumps(data).encode('utf-8')
        minio_client.put_object(
            bucket_name,
            object_name,
            data=io.BytesIO(data_bytes),
            length=len(data_bytes),
            content_type='application/json'
        )
        print(f"Successfully uploaded {object_name} to MinIO bucket {bucket_name}")
        # Return the full S3 path
        return f"s3://{bucket_name}/{object_name}"
    except S3Error as e:
        print(f"Error uploading to MinIO: {e}")
        raise