import pandas as pd
import numpy as np
import os
import pygeohash as pgh

def run_analysis():
    print("==================================================")
    print("ASTraM Traffic Event Dataset Exploratory Analysis")
    print("==================================================")
    
    csv_path = os.path.join("data", "Astram event data_anonymized - Astram event data_anonymizedb40ac87 - Copy.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    # Load dataset
    df = pd.read_csv(csv_path, low_memory=False)
    total_rows = len(df)
    print(f"Loaded dataset successfully. Total Rows: {total_rows}, Columns: {len(df.columns)}")

    # 1. Coordinate Check (Bengaluru Bounds: lat [12.8, 13.25], lng [77.35, 77.85])
    print("\n--- 1. Geospatial Coordinate Inspection ---")
    valid_coords = df[(df['latitude'] > 0) & (df['longitude'] > 0)]
    print(f"  Rows with non-zero lat/lng: {len(valid_coords)} ({len(valid_coords)/total_rows*100:.2f}%)")
    print("  Latitude Range:", df['latitude'].min(), "to", df['latitude'].max())
    print("  Longitude Range:", df['longitude'].min(), "to", df['longitude'].max())
    
    # Bengaluru bounding box count
    blr_df = df[(df['latitude'] >= 12.8) & (df['latitude'] <= 13.25) & 
                (df['longitude'] >= 77.35) & (df['longitude'] <= 77.85)]
    print(f"  Rows within Greater Bengaluru Bounding Box: {len(blr_df)} ({len(blr_df)/total_rows*100:.2f}%)")
    
    # 2. Missingness Analysis
    print("\n--- 2. Missingness Analysis ---")
    missing_pct = df.isnull().mean() * 100
    missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)
    print("  Top columns with missing values:")
    for col, pct in missing_cols.head(15).items():
        print(f"    - {col}: {pct:.2f}% null")

    # 3. Temporal Coverage & Distributions
    print("\n--- 3. Temporal Coverage & Distributions ---")
    df['start_dt'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    print("  Start DateTime range:", df['start_dt'].min(), "to", df['start_dt'].max())
    
    # Extract hour and day of week
    df['hour'] = df['start_dt'].dt.hour
    df['day_of_week'] = df['start_dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    print("  Hour of day distribution (counts grouped):")
    hour_groups = pd.cut(df['hour'], bins=[-1, 5, 9, 13, 17, 21, 24], 
                         labels=['Night (00-05)', 'Morning Peak (06-09)', 'Midday (10-13)', 'Afternoon (14-17)', 'Evening Peak (18-21)', 'Late Night (22-23)'])
    print(hour_groups.value_counts().sort_index())

    # 4. Incident Duration Analysis (Target Feasibility)
    print("\n--- 4. Incident Duration Analysis ---")
    df['closed_dt'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
    df['duration_min'] = (df['closed_dt'] - df['start_dt']).dt.total_seconds() / 60.0
    
    valid_dur = df[df['duration_min'].notnull()]
    print(f"  Rows with valid closed duration: {len(valid_dur)} ({len(valid_dur)/total_rows*100:.2f}%)")
    print("  Duration Percentiles (Minutes):")
    print(valid_dur['duration_min'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))
    
    # Negative durations
    neg_count = (df['duration_min'] < 0).sum()
    print(f"  Negative duration counts: {neg_count}")

    # 5. Geohash Cardinality
    print("\n--- 5. Geospatial Clustering (Geohashes) ---")
    # Clean coordinates for geohashing
    geo_df = df[(df['latitude'] >= 12.8) & (df['latitude'] <= 13.25) & 
                (df['longitude'] >= 77.35) & (df['longitude'] <= 77.85)].copy()
    
    for prec in [5, 6, 7]:
        geo_df[f'geohash_{prec}'] = geo_df.apply(lambda r: pgh.encode(r['latitude'], r['longitude'], precision=prec), axis=1)
        print(f"  Unique geohashes at precision {prec}: {geo_df[f'geohash_{prec}'].nunique()}")

    # 6. Text Description Statistics
    print("\n--- 6. Incident Text Description Analysis ---")
    desc_series = df['description'].dropna().astype(str)
    print(f"  Total non-null descriptions: {len(desc_series)} ({len(desc_series)/total_rows*100:.2f}%)")
    lengths = desc_series.apply(len)
    print("  Text character length stats:")
    print(lengths.describe(percentiles=[0.5, 0.9, 0.95]))

    # 7. Proposed Target Variables Simulation
    print("\n--- 7. Derived Target Variables Distribution Simulation ---")
    # Median duration imputation by cause
    median_durs = df.groupby('event_cause')['duration_min'].transform('median').fillna(60.0)
    durations = df['duration_min'].fillna(median_durs)
    # Clip extreme durations at 24 hours (1440 minutes) and replace negative values
    durations = np.clip(durations.values, 5.0, 1440.0)
    
    # Priority score mapping
    prio_score = df['priority'].map({'High': 100, 'Low': 50}).fillna(50).values
    
    # Road closure score mapping
    closure_score = df['requires_road_closure'].apply(lambda x: 100 if x else 0).values
    
    # Event cause severity mapping
    cause_severity = df['event_cause'].map({
        'public_event': 100, 'protest': 100, 'rally': 100,
        'water_logging': 90, 'accident': 90,
        'construction': 70, 'road_conditions': 70,
        'vehicle_breakdown': 50, 'tree_fall': 50,
        'pot_holes': 30, 'others': 30, 'congestion': 30
    }).fillna(30).values
    
    # Corridor score mapping
    corridor_score = df['corridor'].apply(lambda x: 30 if pd.isna(x) or x == 'Non-corridor' else 100).values
    
    # Standard log-duration score between 0 and 100
    log_dur_score = 100 * (np.log1p(durations) - np.log1p(5.0)) / (np.log1p(1440.0) - np.log1p(5.0))
    
    # Event Impact Score (EIS)
    eis = 0.3 * log_dur_score + 0.25 * closure_score + 0.15 * prio_score + 0.2 * cause_severity + 0.1 * corridor_score
    eis = np.clip(eis, 0, 100)
    
    # Manpower Recommendation
    base_manpower = df['event_cause'].map({
        'public_event': 10, 'protest': 10, 'rally': 10,
        'accident': 4, 'water_logging': 3,
        'construction': 2, 'road_conditions': 2, 'tree_fall': 2,
        'vehicle_breakdown': 1, 'pot_holes': 1, 'others': 1, 'congestion': 1
    }).fillna(1).values
    prio_mult = df['priority'].map({'High': 1.5, 'Low': 1.0}).fillna(1.0).values
    closure_mult = df['requires_road_closure'].apply(lambda x: 2.0 if x else 1.0).values
    
    # Peak hours and corridor adjustment
    is_peak = df['hour'].isin([6, 7, 8, 9, 18, 19, 20, 21]).astype(int).values
    is_corridor = (corridor_score == 100).astype(int)
    
    manpower = np.round(base_manpower * prio_mult * closure_mult) + is_peak + is_corridor
    manpower = np.clip(manpower, 1, 30).astype(int)
    
    # Barricades Recommendation
    base_barricades = df['event_cause'].map({
        'construction': 20, 'public_event': 15, 'protest': 15, 'rally': 15,
        'water_logging': 10, 'pot_holes': 5, 'tree_fall': 5, 'accident': 5,
        'vehicle_breakdown': 0, 'others': 0, 'congestion': 0, 'road_conditions': 0
    }).fillna(0).values
    
    road_closure_flat = df['requires_road_closure'].apply(lambda x: 20 if x else 0).values
    barricades = np.round(base_barricades * prio_mult) + road_closure_flat
    barricades = np.clip(barricades, 0, 50).astype(int)
    
    # Diversion Recommendation
    diversion = ((df['requires_road_closure'] == True) | (eis >= 60)).astype(int).values
    
    print("\n  EIS Simulation Stats (0 to 100):")
    print(pd.Series(eis).describe())
    
    print("\n  Manpower Recommendation Stats:")
    print(pd.Series(manpower).describe())
    print(pd.Series(manpower).value_counts().head(5))
    
    print("\n  Barricades Recommendation Stats:")
    print(pd.Series(barricades).describe())
    print(pd.Series(barricades).value_counts().head(5))
    
    print("\n  Diversion Required Count (Binary):")
    print(pd.Series(diversion).value_counts())

    print("\n==================================================")
    print("Data analysis completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    run_analysis()
