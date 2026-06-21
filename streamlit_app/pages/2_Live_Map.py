import streamlit as st
import os
import sys
import pandas as pd

# Resolve paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "streamlit_app"))

import components.map_component as mc
from streamlit_folium import st_folium

st.title("Live Operations Map")
st.caption("Interactive spatial view of historical logs and operational recommendations across Bengaluru.")

# 1. Load data
@st.cache_data
def get_historical_events():
    csv_path = os.path.join(PROJECT_ROOT, "data", "processed_astram_events.csv")
    if not os.path.exists(csv_path):
        st.error(f"Processed events file not found at {csv_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        # Type cast to prevent mapping errors
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        df['target_eis'] = df['target_eis'].astype(float)
        df['target_manpower'] = df['target_manpower'].astype(int)
        df['target_barricades'] = df['target_barricades'].astype(int)
        df['target_diversion'] = df['target_diversion'].astype(int)
        df['hour'] = df['hour'].astype(int)
        df['is_peak_hour'] = df['is_peak_hour'].astype(int)
        df['zone'] = df['zone'].fillna('unknown').astype(str).replace('nan', 'unknown')
        df['event_cause'] = df['event_cause'].fillna('others').astype(str).replace('nan', 'others')
        df['corridor'] = df['corridor'].fillna('Non-corridor').astype(str).replace('nan', 'Non-corridor')
        df['address'] = df['address'].fillna('unknown').astype(str).replace('nan', 'unknown')
        return df
    except Exception as e:
        st.error(f"Error loading events for map: {e}")
        return pd.DataFrame()

df_raw = get_historical_events()

if df_raw.empty:
    st.warning("No data loaded. Please make sure data/processed_astram_events.csv exists.")
else:
    # 2. Sidebar Filters
    st.sidebar.header("Map Controls")
    
    # Event Cause multiselect
    all_causes = sorted(list(df_raw['event_cause'].unique()))
    selected_causes = st.sidebar.multiselect(
        "Filter by Event Cause",
        options=all_causes,
        default=[]
    )
    
    # Zone multiselect
    all_zones = sorted(list(df_raw['zone'].unique()))
    selected_zones = st.sidebar.multiselect(
        "Filter by Police Zone",
        options=all_zones,
        default=[]
    )
    
    # EIS Slider
    eis_range = st.sidebar.slider(
        "Event Impact Score (EIS) Range",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=5.0
    )
    
    # Hour Slider
    hour_range = st.sidebar.slider(
        "Incident Hour Range",
        min_value=0,
        max_value=23,
        value=(0, 23),
        step=1
    )
    
    # Diversion Toggle
    diversion_only = st.sidebar.checkbox("Show Only Diversion Required", value=False)
    
    # Map Type Selector
    map_type = st.sidebar.radio(
        "Visualization Mode",
        options=["Marker View (Resource Proportional)", "Demand Heatmap"],
        index=0
    )
    show_heatmap = (map_type == "Demand Heatmap")

    # 3. Filter DataFrame for stats calculation
    df_filtered = df_raw.copy()
    if selected_causes:
        df_filtered = df_filtered[df_filtered['event_cause'].isin(selected_causes)]
    if selected_zones:
        df_filtered = df_filtered[df_filtered['zone'].isin(selected_zones)]
    df_filtered = df_filtered[(df_filtered['target_eis'] >= eis_range[0]) & (df_filtered['target_eis'] <= eis_range[1])]
    df_filtered = df_filtered[(df_filtered['hour'] >= hour_range[0]) & (df_filtered['hour'] <= hour_range[1])]
    if diversion_only:
        df_filtered = df_filtered[df_filtered['target_diversion'] == 1]

    # 4. Operations Stats Row
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    total_incidents = len(df_filtered)
    avg_eis = df_filtered['target_eis'].mean() if total_incidents > 0 else 0.0
    
    # Highest impact corridor calculation
    if total_incidents > 0:
        corridor_groups = df_filtered.groupby('corridor')['target_eis'].mean()
        # Filter out Non-corridor to make it meaningful, unless only Non-corridor exists
        meaningful_corridors = corridor_groups.drop('Non-corridor', errors='ignore')
        if not meaningful_corridors.empty:
            highest_corridor = meaningful_corridors.idxmax()
        else:
            highest_corridor = corridor_groups.idxmax()
    else:
        highest_corridor = "N/A"
        
    peak_pct = (df_filtered['is_peak_hour'].mean() * 100.0) if total_incidents > 0 else 0.0
    
    with stat_col1:
        st.metric("Total Incidents Shown", f"{total_incidents:,}")
    with stat_col2:
        st.metric("Average EIS Severity", f"{avg_eis:.1f} / 100")
    with stat_col3:
        st.metric("Highest Impact Corridor", highest_corridor)
    with stat_col4:
        st.metric("Peak Hour Incidents", f"{peak_pct:.1f}%")
        
    st.write("")
    
    # 5. Render Interactive Folium Map
    with st.spinner("Generating map canvas..."):
        m, limit_applied = mc.create_live_events_map(
            df_events=df_raw,
            show_heatmap=show_heatmap,
            filter_cause=selected_causes,
            filter_zone=selected_zones,
            eis_range=eis_range,
            hour_range=hour_range,
            diversion_only=diversion_only
        )
        
        if limit_applied:
            st.warning("**Rendering Safeguard Active**: Showing the top 400 highest impact events on the map to ensure fluid performance. Use the sidebar controls to filter by zone, hour, or cause.")
            
        # Display the map full-width
        st_folium(m, key="live_folium_map", height=600, width=1400)
        
    st.info("Pro Tip: In Marker View, the circle radius is proportional to the manpower officers deployment. Click on any circle to view active popup dispatch parameters.")
