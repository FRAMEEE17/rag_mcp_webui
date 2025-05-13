from app.core.registry import tool
import httpx
import json
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

@tool(name="get_weather", description="Get current weather information for a location")
async def get_weather(location: str, units: str = "metric") -> dict:

    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return {"error": "API key not found."}
    url = f"https://api.openweathermap.org/data/2.5/weather"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            params={
                "q": location,
                "units": units,
                "appid": api_key
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            result = {
                "location": f"{data['name']}, {data.get('sys', {}).get('country', '')}",
                "temperature": {
                    "current": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "min": data["main"]["temp_min"],
                    "max": data["main"]["temp_max"]
                },
                "conditions": {
                    "main": data["weather"][0]["main"],
                    "description": data["weather"][0]["description"],
                },
                "wind": {
                    "speed": data["wind"]["speed"],
                    "direction": data["wind"]["deg"]
                },
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "sunrise": data.get("sys", {}).get("sunrise"),
                "sunset": data.get("sys", {}).get("sunset")
            }
            return result
        else:
            return {
                "error": f"Failed to get weather: {response.status_code}",
                "message": response.text
            }