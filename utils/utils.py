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
