import os
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Resolve project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed_astram_events.csv")

@st.cache_data
def load_address_database():
    """
    Loads unique addresses from processed training data for instant lookup.
    Returns:
        dict: address -> (lat, lon, corridor, police_station, zone)
    """
    if not os.path.exists(CSV_PATH):
        return {}
    try:
        # Load columns of interest to save memory and speed up
        df = pd.read_csv(CSV_PATH, usecols=['address', 'latitude', 'longitude', 'corridor', 'police_station', 'zone'])
        # Drop rows with null address or coords
        df = df.dropna(subset=['address', 'latitude', 'longitude'])
        # Drop duplicates based on address, keeping first
        df_unique = df.drop_duplicates(subset=['address'])
        
        # Build dictionary
        address_dict = {}
        for _, row in df_unique.iterrows():
            corr = str(row['corridor']) if pd.notna(row['corridor']) and str(row['corridor']) != 'nan' else 'Non-corridor'
            ps = str(row['police_station']) if pd.notna(row['police_station']) and str(row['police_station']) != 'nan' else 'unknown'
            zone = str(row['zone']) if pd.notna(row['zone']) and str(row['zone']) != 'nan' else 'unknown'
            address_dict[str(row['address'])] = (
                float(row['latitude']),
                float(row['longitude']),
                corr,
                ps,
                zone
            )
        return address_dict
    except Exception as e:
        st.error(f"Error loading address database: {e}")
        return {}

def resolve_address(address: str) -> dict:
    """
    Resolves address to latitude and longitude.
    First tries database lookup. Falls back to Nominatim.
    Returns:
        dict: {
            "lat": float or None,
            "lon": float or None,
            "corridor": str or None,
            "police_station": str or None,
            "zone": str or None,
            "source": str
        }
    """
    if not address or not address.strip():
        return {"lat": None, "lon": None, "corridor": None, "police_station": None, "zone": None, "source": "Empty Input"}

    address_str = address.strip()
    
    # 1. Database Lookup
    db = load_address_database()
    if address_str in db:
        lat, lon, corridor, police_station, zone = db[address_str]
        return {
            "lat": lat,
            "lon": lon,
            "corridor": corridor,
            "police_station": police_station,
            "zone": zone,
            "source": "Local Database Match"
        }
    
    # 2. Nominatim Geocoding
    try:
        # Bounded viewbox for Bengaluru
        # Bounding box coordinates:
        # Min Lat: 12.80, Max Lat: 13.25
        # Min Lon: 77.35, Max Lon: 77.85
        geolocator = Nominatim(user_agent="astram_traffic_intel")
        # Query with Bengaluru suffix to guide Nominatim
        query = f"{address_str}, Bengaluru, Karnataka, India"
        location = geolocator.geocode(
            query,
            viewbox=[(12.80, 77.35), (13.25, 77.85)],
            bounded=True,
            timeout=5
        )
        if location:
            # Check if resolved lat/lon falls within Bengaluru bounds
            lat, lon = location.latitude, location.longitude
            if 12.80 <= lat <= 13.25 and 77.35 <= lon <= 77.85:
                return {
                    "lat": lat,
                    "lon": lon,
                    "corridor": "Non-corridor",
                    "police_station": "unknown",
                    "zone": "unknown",
                    "source": "OSM Nominatim Geocoder"
                }
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        # Fail silently and let the user know through source field or log
        pass
    
    return {
        "lat": None,
        "lon": None,
        "corridor": "Non-corridor",
        "police_station": "unknown",
        "zone": "unknown",
        "source": "Failed (Not found or Rate-limited)"
    }

def find_nearest_spatial_context(lat: float, lon: float) -> tuple:
    """
    Finds the nearest historical coordinates in the database to inherit corridor, zone, and police station.
    Skips missing/unknown values to inherit the nearest valid labeled boundaries in the neighborhood.
    Returns:
        (corridor, police_station, zone)
    """
    db = load_address_database()
    if not db:
        return "Non-corridor", "unknown", "unknown"
        
    min_dist_corr = float('inf')
    best_corr = "Non-corridor"
    
    min_dist_ps = float('inf')
    best_ps = "unknown"
    
    min_dist_zone = float('inf')
    best_zone = "unknown"
    
    for addr, (db_lat, db_lon, corr, ps, zone) in db.items():
        # Euclidean distance squared
        dist = (lat - db_lat)**2 + (lon - db_lon)**2
        
        # Match nearest valid corridor
        if corr != 'Non-corridor' and dist < min_dist_corr:
            min_dist_corr = dist
            best_corr = corr
            
        # Match nearest valid police station
        if ps != 'unknown' and dist < min_dist_ps:
            min_dist_ps = dist
            best_ps = ps
            
        # Match nearest valid zone
        if zone != 'unknown' and dist < min_dist_zone:
            min_dist_zone = dist
            best_zone = zone
            
    # Apply distance threshold filter (~5 km or 0.0025 degrees squared)
    # If the closest labeled record is too far away, default to unknown
    if min_dist_corr > 0.0025:
        best_corr = "Non-corridor"
    if min_dist_ps > 0.0025:
        best_ps = "unknown"
    if min_dist_zone > 0.0025:
        best_zone = "unknown"
        
    return best_corr, best_ps, best_zone

