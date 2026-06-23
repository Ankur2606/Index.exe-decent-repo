"""
Standalone geocoder for the voice_agent package.
Stripped of Streamlit dependencies for use in FastAPI/Docker contexts.
"""
import os
import pandas as pd
from functools import lru_cache

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

# Resolve project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed_astram_events.csv")


@lru_cache(maxsize=1)
def load_address_database() -> dict:
    """
    Loads unique addresses from processed training data for instant lookup.
    Returns:
        dict: address -> (lat, lon, corridor, police_station, zone)
    """
    if not os.path.exists(CSV_PATH):
        print(f"[geocoder] CSV not found at {CSV_PATH}, returning empty database.")
        return {}
    try:
        df = pd.read_csv(CSV_PATH, usecols=['address', 'latitude', 'longitude', 'corridor', 'police_station', 'zone'])
        df = df.dropna(subset=['address', 'latitude', 'longitude'])
        df_unique = df.drop_duplicates(subset=['address'])

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
        print(f"[geocoder] Loaded {len(address_dict)} addresses from local database.")
        return address_dict
    except Exception as e:
        print(f"[geocoder] Error loading address database: {e}")
        return {}


def resolve_address(address: str) -> dict:
    """
    Resolves address to latitude and longitude.
    First tries database lookup. Falls back to Nominatim.
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

    # 2. Nominatim Geocoding (if geopy is installed)
    if HAS_GEOPY:
        try:
            geolocator = Nominatim(user_agent="astram_traffic_intel")
            query = f"{address_str}, Bengaluru, Karnataka, India"
            location = geolocator.geocode(
                query,
                viewbox=[(12.80, 77.35), (13.25, 77.85)],
                bounded=True,
                timeout=5
            )
            if location:
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
        except Exception:
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
        dist = (lat - db_lat)**2 + (lon - db_lon)**2

        if corr != 'Non-corridor' and dist < min_dist_corr:
            min_dist_corr = dist
            best_corr = corr

        if ps != 'unknown' and dist < min_dist_ps:
            min_dist_ps = dist
            best_ps = ps

        if zone != 'unknown' and dist < min_dist_zone:
            min_dist_zone = dist
            best_zone = zone

    # ~5 km threshold
    if min_dist_corr > 0.0025:
        best_corr = "Non-corridor"
    if min_dist_ps > 0.0025:
        best_ps = "unknown"
    if min_dist_zone > 0.0025:
        best_zone = "unknown"

    return best_corr, best_ps, best_zone
