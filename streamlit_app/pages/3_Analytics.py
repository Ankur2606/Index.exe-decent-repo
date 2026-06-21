import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Resolve paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "streamlit_app"))

st.title("Traffic Event Intelligence Analytics")
st.caption("Deep-dive visual analysis of temporal patterns, corridor hotspots, and deployment resource correlations.")

# 1. Load data
@st.cache_data
def get_processed_data():
    csv_path = os.path.join(PROJECT_ROOT, "data", "processed_astram_events.csv")
    if not os.path.exists(csv_path):
        st.error(f"Processed events file not found at {csv_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        # Parse dates
        df['start_dt'] = pd.to_datetime(df['start_dt'])
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        df['target_eis'] = df['target_eis'].astype(float)
        df['target_manpower'] = df['target_manpower'].astype(int)
        df['target_barricades'] = df['target_barricades'].astype(int)
        df['target_diversion'] = df['target_diversion'].astype(int)
        df['hour'] = df['hour'].astype(int)
        df['day_of_week'] = df['day_of_week'].astype(int)
        df['zone'] = df['zone'].fillna('unknown').astype(str).replace('nan', 'unknown')
        df['event_cause'] = df['event_cause'].fillna('others').astype(str).replace('nan', 'others')
        df['corridor'] = df['corridor'].fillna('Non-corridor').astype(str).replace('nan', 'Non-corridor')
        return df
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")
        return pd.DataFrame()

df_raw = get_processed_data()

if df_raw.empty:
    st.warning("No data found. Please run the data pipeline first.")
else:
    # 2. Sidebar Filters
    st.sidebar.header("Filter Analytics")
    
    # Date Range Slider
    min_date = df_raw['start_dt'].min().date()
    max_date = df_raw['start_dt'].max().date()
    selected_dates = st.sidebar.slider(
        "Incident Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
    
    # Cause multiselect
    all_causes = sorted(list(df_raw['event_cause'].unique()))
    selected_causes = st.sidebar.multiselect(
        "Event Cause",
        options=all_causes,
        default=[]
    )
    
    # Zone multiselect
    all_zones = sorted(list(df_raw['zone'].unique()))
    selected_zones = st.sidebar.multiselect(
        "Police Zone",
        options=all_zones,
        default=[]
    )
    
    # 3. Filter DataFrame
    df_filtered = df_raw.copy()
    
    # Filter Date
    df_filtered = df_filtered[
        (df_filtered['start_dt'].dt.date >= selected_dates[0]) & 
        (df_filtered['start_dt'].dt.date <= selected_dates[1])
    ]
    
    # Filter Cause
    if selected_causes:
        df_filtered = df_filtered[df_filtered['event_cause'].isin(selected_causes)]
        
    # Filter Zone
    if selected_zones:
        df_filtered = df_filtered[df_filtered['zone'].isin(selected_zones)]
        
    total_records = len(df_filtered)
    
    # Show warning if empty
    if total_records == 0:
        st.warning("No data matching current filters. Please adjust the sidebar controls.")
    else:
        st.markdown(f"Displaying analytics computed over **{total_records:,}** active incidents.")
        
        # 4. Group charts in tabs for premium clean layout
        tab1, tab2, tab3 = st.tabs(["Temporal & Weekly Patterns", "Severity & Resource Allocations", "Corridor & Hotspots Analysis"])
        
        # Plotly dark theme layout styling
        plotly_layout_args = {
            'paper_bgcolor': '#1A1D26',
            'plot_bgcolor': '#1A1D26',
            'font_color': '#FAFAFA',
            'title_font': dict(size=16, family='Inter, sans-serif'),
            'xaxis': dict(gridcolor='#2C303E'),
            'yaxis': dict(gridcolor='#2C303E')
        }

        # TAB 1: Temporal Patterns
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # 1. Incidents by Hour of Day
                df_hour = df_filtered.groupby(['hour', 'event_cause'])['id'].count().reset_index(name='count')
                fig_hour = px.bar(
                    df_hour,
                    x='hour',
                    y='count',
                    color='event_cause',
                    title='Incidents by Hour of Day (Stacked by Cause)',
                    labels={'hour': 'Hour of Day (0-23)', 'count': 'Incident Count', 'event_cause': 'Cause'},
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_hour.update_layout(**plotly_layout_args)
                
                # Add peak markers (Morning Peak 6-9, Evening Peak 18-21)
                fig_hour.add_vrect(x0=6.0, x1=9.0, fillcolor="#2980B9", opacity=0.15, line_width=0, annotation_text="AM Peak", annotation_position="top left")
                fig_hour.add_vrect(x0=18.0, x1=21.0, fillcolor="#C0392B", opacity=0.15, line_width=0, annotation_text="PM Peak", annotation_position="top left")
                st.plotly_chart(fig_hour, width="stretch")
                
            with col2:
                # 2. Weekly Pattern Heatmap (Hour x Day of Week)
                df_week = df_filtered.groupby(['hour', 'day_of_week'])['id'].count().reset_index(name='count')
                pivot_week = df_week.pivot_table(index='hour', columns='day_of_week', values='count', fill_value=0)
                
                # Re-index to ensure all hours 0-23 are shown
                pivot_week = pivot_week.reindex(index=range(24), columns=range(7), fill_value=0)
                
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                fig_heatmap = px.imshow(
                    pivot_week.values,
                    labels=dict(x="Day of Week", y="Hour of Day", color="Incident Count"),
                    x=day_names,
                    y=list(range(24)),
                    color_continuous_scale='Viridis',
                    title='Weekly Bottleneck Matrix (Hour of Day × Day of Week)'
                )
                fig_heatmap.update_layout(**plotly_layout_args)
                st.plotly_chart(fig_heatmap, width="stretch")
                
        # TAB 2: Severity and Resources
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # 3. EIS Distribution (Histogram with lines)
                mean_eis = float(df_filtered['target_eis'].mean())
                median_eis = float(df_filtered['target_eis'].median())
                
                fig_dist = px.histogram(
                    df_filtered,
                    x='target_eis',
                    nbins=40,
                    title='Event Impact Score (EIS) Distribution',
                    labels={'target_eis': 'EIS Congestion Score', 'count': 'Frequency'},
                    color_discrete_sequence=['#FF4B4B']
                )
                fig_dist.update_layout(**plotly_layout_args)
                
                # Add Mean/Median vertical lines
                fig_dist.add_vline(x=mean_eis, line_dash="dash", line_color="#2ECC71", annotation_text=f"Mean: {mean_eis:.1f}", annotation_position="top left")
                fig_dist.add_vline(x=median_eis, line_dash="dot", line_color="#3498DB", annotation_text=f"Median: {median_eis:.1f}", annotation_position="top right")
                st.plotly_chart(fig_dist, width="stretch")
                
            with col2:
                # 4. Manpower vs EIS Scatter
                # Clip barricades count for sizing
                df_filtered['barricades_size'] = np.clip(df_filtered['target_barricades'].values, 2, 50)
                
                fig_scatter = px.scatter(
                    df_filtered,
                    x='target_eis',
                    y='target_manpower',
                    color='event_cause',
                    size='barricades_size',
                    title='Resource Allocations: Manpower vs EIS Congestion',
                    labels={'target_eis': 'EIS Congestion Severity', 'target_manpower': 'Recommended Officers', 'event_cause': 'Cause'},
                    hover_data=['address', 'target_barricades'],
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_scatter.update_layout(**plotly_layout_args)
                st.plotly_chart(fig_scatter, width="stretch")
                st.caption("Note: Scatter point sizes correspond to recommended physical barricades count.")

        # TAB 3: Geospatial & Corridors
        with tab3:
            # 5. Top 10 Highest Impact Corridors (Exclude Non-corridor)
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                df_corr = df_filtered[df_filtered['corridor'] != 'Non-corridor']
                if df_corr.empty:
                    df_corr = df_filtered.copy() # fallback if no major corridor matches
                    
                agg_corr = df_corr.groupby('corridor')['target_eis'].mean().reset_index(name='mean_eis')
                agg_corr = agg_corr.sort_values(by='mean_eis', ascending=True).tail(10)
                
                fig_corr = px.bar(
                    agg_corr,
                    x='mean_eis',
                    y='corridor',
                    orientation='h',
                    color='mean_eis',
                    color_continuous_scale='Reds',
                    title='Top 10 Highest Impact Corridors (Mean EIS)',
                    labels={'mean_eis': 'Mean Event Impact Score', 'corridor': 'Corridor Name'}
                )
                fig_corr.update_layout(**plotly_layout_args)
                st.plotly_chart(fig_corr, width="stretch")
                
            with col2:
                # 6. Event Cause Frequency Distribution & Mean EIS
                agg_cause = df_filtered.groupby('event_cause').agg(
                    count=('id', 'count'),
                    avg_eis=('target_eis', 'mean')
                ).reset_index()
                agg_cause = agg_cause.sort_values(by='count', ascending=True)
                
                fig_cause = px.bar(
                    agg_cause,
                    x='count',
                    y='event_cause',
                    orientation='h',
                    color='avg_eis',
                    color_continuous_scale='YlOrRd',
                    title='Incident Cause Distribution & Average EIS Impact',
                    labels={'count': 'Volume of Incidents', 'event_cause': 'Incident Cause', 'avg_eis': 'Avg EIS'}
                )
                fig_cause.update_layout(**plotly_layout_args)
                st.plotly_chart(fig_cause, width="stretch")
                
            st.markdown("<hr style='border: 0; border-top: 1px solid #2C303E; margin: 1.5rem 0;'>", unsafe_allow_html=True)
            
            fig_mapbox = px.density_map(
                df_filtered,
                lat='latitude',
                lon='longitude',
                z='target_eis',
                radius=12,
                center=dict(lat=12.9716, lon=77.5946),
                zoom=10.5,
                map_style="carto-darkmatter", # Dark styling matching dashboard
                title="Geospatial Incident Density across Bengaluru (weighted by EIS)"
            )
            # Layout margins removal for full display
            fig_mapbox.update_layout(
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='#1A1D26',
                font_color='#FAFAFA',
                title_font=dict(size=16, family='Inter, sans-serif')
            )
            st.plotly_chart(fig_mapbox, width="stretch")
