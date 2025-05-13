from app.core.registry import tool
import httpx
import json
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv
load_dotenv()
@tool(name="get_directions", description="Get directions between two locations")
async def get_directions(
    origin: str,
    destination: str,
    mode: str = "driving",
    alternatives: bool = False
) -> dict:
    """
    Get directions between two locations using Google Maps.
    
    Args:
        origin: Starting location (address or coordinates)
        destination: End location (address or coordinates)
        mode: Travel mode (driving, walking, bicycling, transit)
        alternatives: Whether to provide alternative routes
        
    Returns:
        Directions information including route, distance, and duration
    """
    
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            params={
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "alternatives": "true" if alternatives else "false",
                "key": api_key
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("status") == "OK":
                routes = []
                
                for route in data["routes"]:
                    route_info = {
                        "summary": route.get("summary", ""),
                        "distance": route["legs"][0]["distance"]["text"],
                        "duration": route["legs"][0]["duration"]["text"],
                        "steps": []
                    }
                    
                    for step in route["legs"][0]["steps"]:
                        route_info["steps"].append({
                            "instruction": step["html_instructions"].replace("<b>", "").replace("</b>", "").replace("<div>", " ").replace("</div>", ""),
                            "distance": step["distance"]["text"],
                            "duration": step["duration"]["text"]
                        })
                    
                    routes.append(route_info)
                
                return {
                    "origin": origin,
                    "destination": destination,
                    "mode": mode,
                    "routes": routes
                }
            else:
                return {
                    "error": f"Google Maps API error: {data.get('status')}",
                    "message": data.get("error_message", "Unknown error")
                }
        else:
            return {
                "error": f"Failed to get directions: {response.status_code}",
                "message": response.text
            }

@tool(name="search_places", description="Search for places nearby or by query")
async def search_places(
    query: str,
    location: Optional[str] = None,
    radius: int = 1000,
    type: Optional[str] = None
) -> dict:
    """
    Search for places using Google Maps Places API.
    
    Args:
        query: Search query
        location: Center point for nearby search (lat,lng format)
        radius: Search radius in meters (max 50000)
        type: Type of place (restaurant, cafe, park, etc.)
        
    Returns:
        List of places matching the search criteria
    """
    
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return {"error": "API key not found."}  
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    
    params = {
        "query": query,
        "key": api_key
    }
    
    if location:
        params["location"] = location
        params["radius"] = min(radius, 50000)
    
    if type:
        params["type"] = type
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("status") == "OK":
                places = []
                
                for place in data["results"]:
                    places.append({
                        "name": place.get("name", ""),
                        "address": place.get("formatted_address", ""),
                        "rating": place.get("rating", 0),
                        "total_ratings": place.get("user_ratings_total", 0),
                        "location": place.get("geometry", {}).get("location", {}),
                        "types": place.get("types", []),
                        "place_id": place.get("place_id", "")
                    })
                
                return {
                    "query": query,
                    "location": location,
                    "places": places,
                    "next_page_token": data.get("next_page_token")
                }
            else:
                return {
                    "error": f"Google Maps API error: {data.get('status')}",
                    "message": data.get("error_message", "Unknown error")
                }
        else:
            return {
                "error": f"Failed to search places: {response.status_code}",
                "message": response.text
            }