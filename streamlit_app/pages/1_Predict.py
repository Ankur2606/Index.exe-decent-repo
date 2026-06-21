import streamlit as st
import os
import sys
import json
from datetime import datetime

# Resolve project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "streamlit_app"))

from utils.inference import InferenceEngine
from utils.geocoder import load_address_database
import components.prediction_card as card
import components.map_component as mc
from streamlit_folium import st_folium

# Inject styles from main app (streamlit caches set_page_config only for the first run, page files don't need it)
st.markdown("""
<style>
    /* Styling for Predict page */
    .stNumberInput input {
        background-color: #1A1D26 !important;
        color: #FAFAFA !important;
        border: 1px solid #2C303E !important;
    }
    .stTextArea textarea {
        background-color: #1A1D26 !important;
        color: #FAFAFA !important;
        border: 1px solid #2C303E !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1A1D26 !important;
        color: #FAFAFA !important;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        font-weight: bold;
        border-radius: 4px;
        margin-right: 0.5rem;
        margin-top: 0.25rem;
    }
    .badge-info {
        background-color: #3498DB22;
        color: #3498DB;
        border: 1px solid #3498DB44;
    }
</style>
""", unsafe_allow_html=True)

# 1. Initialize Inference Engine and Address Database
@st.cache_resource
def get_inference_engine():
    return InferenceEngine()

engine = get_inference_engine()
db = load_address_database()

# 2. Page Title
st.title("Traffic Incident Dispatch Panel")
st.caption("Input incident parameters to forecast Event Impact Score (EIS) and dispatch resources.")

# 3. Initialize Session States for reactive coordinates and dropdowns
if 'lat' not in st.session_state:
    st.session_state.lat = 12.9716  # Bangalore center lat
if 'lon' not in st.session_state:
    st.session_state.lon = 77.5946  # Bangalore center lon
if 'address_select_idx' not in st.session_state:
    st.session_state.address_select_idx = 0
if 'custom_address' not in st.session_state:
    st.session_state.custom_address = "Manual Coordinate Entry"
if 'corridor_val' not in st.session_state:
    st.session_state.corridor_val = "Non-corridor"
if 'police_station_val' not in st.session_state:
    st.session_state.police_station_val = "unknown"
if 'zone_val' not in st.session_state:
    st.session_state.zone_val = "unknown"
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Vocabularies
vocabularies = engine.vocabularies

# 5. Create Columns Layout: Left = Inputs, Right = Results
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Incident Log Entry")
    
    # Address selection
    address_options = ["Other / Manual GPS Input"] + list(db.keys())
    selected_address = st.selectbox(
        "Search Incident Address (Autocomplete)",
        options=address_options,
        index=st.session_state.address_select_idx
    )
    st.caption("Autocomplete details load raw historical logs. You can manually adjust the Corridor, Zone, or Police Station fields below if these entries are outdated or missing.")
    
    # Check if address changed
    if 'prev_selected_address' not in st.session_state or st.session_state.prev_selected_address != selected_address:
        st.session_state.prev_selected_address = selected_address
        st.session_state.address_select_idx = address_options.index(selected_address)
        if selected_address != "Other / Manual GPS Input" and selected_address in db:
            lat, lon, corridor, police_station, zone = db[selected_address]
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.session_state.corridor_val = corridor
            st.session_state.police_station_val = police_station
            st.session_state.zone_val = zone
            st.rerun()
            
    # Display custom address text input if manual coordinate entry is active
    if selected_address == "Other / Manual GPS Input":
        custom_address_val = st.text_input(
            "Custom Address / Location Details (Reverse Geocoded from Map or Typed)",
            value=st.session_state.custom_address
        )
        st.session_state.custom_address = custom_address_val
        
        # Display Spatial Resolution Heritage metadata card
        st.markdown(f"""<div style="background-color: #1A1D26; border: 1px solid #3498DB44; border-radius: 6px; padding: 0.8rem; margin-top: -0.5rem; margin-bottom: 1rem;">
<span style="font-size: 0.75rem; color: #3498DB; text-transform: uppercase; font-weight: bold; display: block; margin-bottom: 0.3rem;">Spatial Intelligence Resolved</span>
<span style="font-size: 0.85rem; color: #CFD2D6; display: block; line-height: 1.4;">Address resolved. Environmental boundaries inherited from the nearest historical log:</span>
<div style="margin-top: 0.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.4rem;">
<span class="badge badge-info">Station: {st.session_state.police_station_val}</span>
<span class="badge badge-info">Corridor: {st.session_state.corridor_val}</span>
<span class="badge badge-info">Zone: {st.session_state.zone_val}</span>
</div>
<span style="font-size: 0.75rem; color: #8A8F98; display: block;">Note: You can manually override these inputs in the configurations panel below if needed.</span>
</div>""", unsafe_allow_html=True)
    
    # Manual coordinates overrides
    coord_col1, coord_col2 = st.columns(2)
    with coord_col1:
        lat_input = st.number_input(
            "Latitude (Bengaluru: 12.80 - 13.25)",
            min_value=12.80,
            max_value=13.25,
            value=st.session_state.lat,
            step=0.0001,
            format="%.5f"
        )
    with coord_col2:
        lon_input = st.number_input(
            "Longitude (Bengaluru: 77.35 - 77.85)",
            min_value=77.35,
            max_value=77.85,
            value=st.session_state.lon,
            step=0.0001,
            format="%.5f"
        )
        
    # Synchronize manual edits back to session state
    st.session_state.lat = lat_input
    st.session_state.lon = lon_input
        
    # Map for visual validation / coordinates click capture
    st.caption("Preview Location Pin (Click on map to capture coordinates manually):")
    mini_map = mc.create_mini_predict_map(st.session_state.lat, st.session_state.lon)
    map_data = st_folium(mini_map, key="predict_location_map", height=220, width=None)
    
    # Capture map clicks to update lat/lon session state
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lng = map_data["last_clicked"]["lng"]
        if 12.80 <= clicked_lat <= 13.25 and 77.35 <= clicked_lng <= 77.85:
            # Only update if coordinates changed noticeably to prevent infinite loop reruns
            if abs(st.session_state.lat - clicked_lat) > 0.0001 or abs(st.session_state.lon - clicked_lng) > 0.0001:
                st.session_state.lat = clicked_lat
                st.session_state.lon = clicked_lng
                st.session_state.address_select_idx = 0  # reset autocomplete selection to "Other / Manual GPS Input"
                
                # Fetch closest corridor, police station, and zone from training database
                from utils.geocoder import find_nearest_spatial_context
                corr, ps, zone_val = find_nearest_spatial_context(clicked_lat, clicked_lng)
                st.session_state.corridor_val = corr
                st.session_state.police_station_val = ps
                st.session_state.zone_val = zone_val
                
                # Reverse geocode the location using Nominatim
                try:
                    from geopy.geocoders import Nominatim
                    geolocator = Nominatim(user_agent="astram_traffic_intel")
                    location = geolocator.reverse((clicked_lat, clicked_lng), timeout=5)
                    if location:
                        st.session_state.custom_address = location.address
                    else:
                        st.session_state.custom_address = f"Manual Location ({clicked_lat:.5f}, {clicked_lng:.5f})"
                except Exception:
                    st.session_state.custom_address = f"Manual Location ({clicked_lat:.5f}, {clicked_lng:.5f})"
                    
                st.rerun()

    # Event configurations
    conf_col1, conf_col2 = st.columns(2)
    with conf_col1:
        event_type = st.selectbox(
            "Event Type",
            options=vocabularies.get('event_type', ['unplanned', 'planned'])
        )
        event_cause = st.selectbox(
            "Event Cause",
            options=vocabularies.get('event_cause', ['others'])
        )
        priority = st.selectbox(
            "Priority",
            options=vocabularies.get('priority', ['High', 'Low'])
        )
        veh_type = st.selectbox(
            "Vehicle Type",
            options=vocabularies.get('veh_type', ['unknown'])
        )
    with conf_col2:
        # Corridor Select
        corr_options = vocabularies.get('corridor', ['Non-corridor'])
        try:
            corr_idx = corr_options.index(st.session_state.corridor_val)
        except ValueError:
            corr_idx = 0
        corridor = st.selectbox(
            "Corridor",
            options=corr_options,
            index=corr_idx
        )
        st.session_state.corridor_val = corridor
        
        # Police Station Select
        ps_options = vocabularies.get('police_station', ['unknown'])
        try:
            ps_idx = ps_options.index(st.session_state.police_station_val)
        except ValueError:
            ps_idx = 0
        police_station = st.selectbox(
            "Police Station",
            options=ps_options,
            index=ps_idx
        )
        st.session_state.police_station_val = police_station
        
        # Zone Select
        zone_options = vocabularies.get('zone', ['unknown'])
        try:
            zone_idx = zone_options.index(st.session_state.zone_val)
        except ValueError:
            zone_idx = 0
        zone = st.selectbox(
            "Zone",
            options=zone_options,
            index=zone_idx
        )
        st.session_state.zone_val = zone
        
        # Datetime configuration
        date_input = st.date_input("Incident Date", datetime.now().date())
        time_input = st.time_input("Incident Time", datetime.now().time())
        
    # Text Description input (max 500 chars, max 3 rows)
    description = st.text_area(
        "Incident Description (English / Kannada)",
        placeholder="Type description here... e.g. Traffic jam near Silk Board due to waterlogging.",
        max_chars=500,
        height=100
    )
    
    # 6. Predict Button
    predict_btn = st.button("PREDICT IMPACT & RESOURCES", width="stretch")

with right_col:
    st.subheader("Dispatch Recommendations")
    
    if predict_btn:
        with st.spinner("Processing ensembled models..."):
            # Prepare dictionary for inference
            raw_input = {
                "latitude": st.session_state.lat,
                "longitude": st.session_state.lon,
                "event_type": event_type,
                "event_cause": event_cause,
                "priority": priority,
                "veh_type": veh_type,
                "corridor": corridor,
                "police_station": police_station,
                "zone": zone,
                "date": date_input,
                "time": time_input,
                "description": description
            }
            
            # Predict
            res = engine.predict(raw_input)
            st.session_state.prediction_result = (res, raw_input)
            
    # Show predictions if they exist in session state
    if st.session_state.prediction_result is not None:
        res, raw_input = st.session_state.prediction_result
        
        # Render the 4 custom HTML metric cards
        # 1. Event Impact Score
        st.markdown(card.render_eis_card(res['eis'], res['eis_severity']), unsafe_allow_html=True)
        
        # 2. Manpower
        st.markdown(card.render_manpower_card(res['manpower'], raw_input['time'].hour in [6,7,8,9,18,19,20,21], raw_input['time'].hour), unsafe_allow_html=True)
        
        # 3. Barricades
        st.markdown(card.render_barricades_card(res['barricades']), unsafe_allow_html=True)
        
        # 4. Diversion
        st.markdown(card.render_diversion_card(res['diversion']), unsafe_allow_html=True)
        
        # Show mini map colored by predicted severity
        st.markdown("<p style='font-size:0.9rem; color:#8A8F98; text-transform:uppercase; font-weight:bold;'>Incident Impact Coordinates</p>", unsafe_allow_html=True)
        pred_map = mc.create_mini_predict_map(raw_input['latitude'], raw_input['longitude'], raw_input['event_cause'].upper(), res['eis'])
        st_folium(pred_map, key="predict_result_map", height=200, width=None)
        
        # Export options
        st.write("")
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "incident_details": {
                "address": st.session_state.custom_address if selected_address == "Other / Manual GPS Input" else selected_address,
                "latitude": raw_input['latitude'],
                "longitude": raw_input['longitude'],
                "event_type": raw_input['event_type'],
                "event_cause": raw_input['event_cause'],
                "priority": raw_input['priority'],
                "veh_type": raw_input['veh_type'],
                "corridor": raw_input['corridor'],
                "police_station": raw_input['police_station'],
                "zone": raw_input['zone'],
                "datetime": f"{raw_input['date'].isoformat()}T{raw_input['time'].isoformat()}",
                "description": raw_input['description']
            },
            "predictions": {
                "event_impact_score": res['eis'],
                "severity_band": res['eis_severity'],
                "recommended_officers": res['manpower'],
                "recommended_barricades": res['barricades'],
                "diversion_required": res['diversion'],
                "ensemble_confidence": f"{res['confidence']:.2f}%"
            }
        }
        
        json_str = json.dumps(report_data, indent=4)
        st.download_button(
            label="EXPORT INCIDENT DISPATCH REPORT (JSON)",
            data=json_str,
            file_name=f"astram_dispatch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            width="stretch"
        )
    else:
        st.info("Enter incident details in the left panel and click Predict to generate recommendations.")
