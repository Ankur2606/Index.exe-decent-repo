import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import streamlit as st

def get_severity_color(eis):
    """
    Returns hex color code based on EIS score.
    """
    if eis <= 30.0:
        return "#2ECC71"  # green
    elif eis <= 60.0:
        return "#F1C40F"  # yellow
    elif eis <= 80.0:
        return "#E67E22"  # orange
    else:
        return "#E74C3C"  # red

def create_mini_predict_map(lat: float, lon: float, address_name: str = "Selected Location", eis: float = None):
    """
    Creates a mini Folium map pinned at (lat, lon) with a colored marker.
    If eis is provided, color matches severity. Otherwise, defaults to red.
    """
    m = folium.Map(
        location=[lat, lon],
        zoom_start=14,
        control_scale=True,
        tiles="CartoDB dark_matter"
    )
    
    # Choose color based on EIS
    color = get_severity_color(eis) if eis is not None else "#FF4B4B"
    popup_text = f"<b>{address_name}</b><br>Coordinates: {lat:.5f}, {lon:.5f}"
    if eis is not None:
        popup_text += f"<br>Predicted EIS: {eis:.1f}"
        
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color="red" if color == "#E74C3C" else ("orange" if color == "#E67E22" else ("lightgray" if color == "#F1C40F" else "green")), icon="info-sign")
    ).add_to(m)
    
    return m

def create_live_events_map(df_events, show_heatmap=False, filter_cause=None, filter_zone=None, eis_range=(0.0, 100.0), hour_range=(0, 23), diversion_only=False):
    """
    Creates an interactive map filtering and showing historical incidents.
    """
    # Apply filtering
    df = df_events.copy()
    
    if filter_cause:
        df = df[df['event_cause'].isin(filter_cause)]
    if filter_zone:
        df = df[df['zone'].isin(filter_zone)]
        
    df = df[(df['target_eis'] >= eis_range[0]) & (df['target_eis'] <= eis_range[1])]
    df = df[(df['hour'] >= hour_range[0]) & (df['hour'] <= hour_range[1])]
    
    if diversion_only:
        df = df[df['target_diversion'] == 1]
        
    # Map center defaults to Bangalore center
    center_lat = 12.9716
    center_lon = 77.5946
    
    if len(df) > 0:
        center_lat = float(df['latitude'].mean())
        center_lon = float(df['longitude'].mean())
        
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        control_scale=True,
        tiles="CartoDB dark_matter"
    )
    
    limit_applied = False
    if show_heatmap:
        # Generate heat weights from EIS
        heat_data = df[['latitude', 'longitude', 'target_eis']].dropna().values.tolist()
        HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
    else:
        # Circle markers
        # If there are too many events, sort by EIS and show the top 400 to keep the UI responsive
        if len(df) > 400:
            df = df.sort_values(by='target_eis', ascending=False).head(400)
            limit_applied = True
            
        for idx, row in df.iterrows():
            eis = float(row['target_eis'])
            manpower = int(row['target_manpower'])
            barricades = int(row['target_barricades'])
            diversion_str = "YES" if row['target_diversion'] == 1 else "NO"
            color = get_severity_color(eis)
            
            # Map circle marker size proportional to manpower requirement (radius at least 6px, max 30px)
            radius = max(5, min(25, manpower * 2))
            
            popup_html = f"""
            <div style="font-family: sans-serif; color: #1E293B; font-size: 0.85rem; line-height: 1.4; min-width: 200px;">
                <h4 style="margin: 0 0 5px 0; color: #E74C3C; font-size: 0.95rem;">Incident Details</h4>
                <b>Address:</b> {row['address']}<br>
                <b>Cause:</b> {row['event_cause']}<br>
                <b>Priority:</b> {row['priority']}<br>
                <b>EIS Congestion:</b> <span style="color: {color}; font-weight: bold;">{eis:.1f}</span>/100<br>
                <b>Manpower Deploy:</b> {manpower} Officers<br>
                <b>Barricades:</b> {barricades} Units<br>
                <b>Diversion Required:</b> {diversion_str}
            </div>
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
            
    return m, limit_applied
