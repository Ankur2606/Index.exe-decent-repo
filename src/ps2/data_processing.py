import os
import numpy as np
import pandas as pd
import pygeohash as pgh
import socket
socket.setdefaulttimeout(10)
from sentence_transformers import SentenceTransformer

def process_dataset():
    print("==================================================")
    print("Starting Data Cleaning & Feature Engineering")
    print("==================================================")

    raw_csv = os.path.join("data", "Astram event data_anonymized - Astram event data_anonymizedb40ac87 - Copy.csv")
    processed_csv = os.path.join("data", "processed_astram_events.csv")
    embeddings_npy = os.path.join("data", "processed_descriptions.npy")

    if not os.path.exists(raw_csv):
        print(f"Error: Raw CSV dataset not found at {raw_csv}")
        return

    # Load dataset
    print("Loading raw CSV...")
    df = pd.read_csv(raw_csv, low_memory=False)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows.")

    # 1. Coordinate Cleaning & Clipping to Bengaluru Bounds
    print("Cleaning coordinates...")
    # Bengaluru bounding box: lat [12.8, 13.25], lng [77.35, 77.85]
    lat_min, lat_max = 12.8, 13.25
    lng_min, lng_max = 77.35, 77.85
    df['latitude'] = np.clip(df['latitude'].values, lat_min, lat_max)
    df['longitude'] = np.clip(df['longitude'].values, lng_min, lng_max)

    # 2. Impute Categorical Columns
    print("Imputing categorical variables...")
    df['priority'] = df['priority'].fillna('High')
    df['requires_road_closure'] = df['requires_road_closure'].fillna(False).astype(bool)
    df['event_cause'] = df['event_cause'].fillna('others')
    df['corridor'] = df['corridor'].fillna('Non-corridor')
    df['veh_type'] = df['veh_type'].fillna('unknown')
    df['police_station'] = df['police_station'].fillna('unknown')
    df['description'] = df['description'].fillna('').astype(str)

    # 3. Calculate and Impute Durations
    print("Calculating and imputing event durations...")
    df['start_dt'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['closed_dt'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
    
    # Calculate duration in minutes
    df['duration_min'] = (df['closed_dt'] - df['start_dt']).dt.total_seconds() / 60.0
    
    # Negative/Zero durations replaced with NaN for imputation
    df.loc[df['duration_min'] <= 0, 'duration_min'] = np.nan
    
    # Grouped median duration by cause and priority
    median_by_group = df.groupby(['event_cause', 'priority'])['duration_min'].transform('median')
    df['duration_min'] = df['duration_min'].fillna(median_by_group)
    
    # Fallback to cause median, then overall median (64.5 mins)
    median_by_cause = df.groupby('event_cause')['duration_min'].transform('median')
    df['duration_min'] = df['duration_min'].fillna(median_by_cause).fillna(64.5)
    
    # Clip extreme durations at 24 hours (1440 minutes) to prevent outlier dominance
    df['duration_min'] = np.clip(df['duration_min'].values, 5.0, 1440.0)

    # 4. Generate Geospatial Features (Geohashes)
    print("Generating geohashes...")
    df['geohash_5'] = df.apply(lambda r: pgh.encode(r['latitude'], r['longitude'], precision=5), axis=1)
    df['geohash_6'] = df.apply(lambda r: pgh.encode(r['latitude'], r['longitude'], precision=6), axis=1)
    df['geohash_7'] = df.apply(lambda r: pgh.encode(r['latitude'], r['longitude'], precision=7), axis=1)

    # 5. Generate Cyclical Temporal Features
    print("Generating cyclical temporal features...")
    df['hour'] = df['start_dt'].dt.hour.fillna(12).astype(int)
    df['day_of_week'] = df['start_dt'].dt.dayofweek.fillna(0).astype(int)
    df['month'] = df['start_dt'].dt.month.fillna(1).astype(int)
    
    # Cyclical hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # Cyclical day of week
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    
    # Cyclical month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Binary temporal flags
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_peak_hour'] = df['hour'].isin([6, 7, 8, 9, 18, 19, 20, 21]).astype(int)

    # 6. Formulate Operational Ground Truth Targets
    print("Formulating ground-truth targets (EIS, Manpower, Barricades, Diversion)...")
    
    # 6a. Event Impact Score (EIS)
    log_dur_score = 100.0 * (np.log1p(df['duration_min']) - np.log1p(5.0)) / (np.log1p(1440.0) - np.log1p(5.0))
    closure_score = df['requires_road_closure'].apply(lambda x: 100.0 if x else 0.0).values
    prio_score = df['priority'].map({'High': 100.0, 'Low': 50.0}).fillna(50.0).values
    cause_score = df['event_cause'].map({
        'public_event': 100.0, 'protest': 100.0, 'rally': 100.0, 'procession': 100.0,
        'water_logging': 90.0, 'accident': 90.0,
        'vip_movement': 80.0,
        'construction': 70.0, 'road_conditions': 70.0,
        'vehicle_breakdown': 50.0, 'tree_fall': 50.0, 'debris': 50.0, 'Debris': 50.0,
        'pot_holes': 30.0, 'others': 30.0, 'congestion': 30.0, 'Fog / Low Visibility': 30.0,
        'test_demo': 10.0
    }).fillna(30.0).values
    corridor_score = df['corridor'].apply(lambda x: 100.0 if x != 'Non-corridor' else 30.0).values
    
    # Fused EIS Score
    df['target_eis'] = 0.3 * log_dur_score + 0.25 * closure_score + 0.15 * prio_score + 0.2 * cause_score + 0.1 * corridor_score
    df['target_eis'] = np.clip(df['target_eis'].values, 0.0, 100.0)

    # 6b. Recommended Manpower
    base_manpower = df['event_cause'].map({
        'public_event': 10.0, 'protest': 10.0, 'rally': 10.0, 'procession': 10.0,
        'vip_movement': 6.0,
        'accident': 4.0, 'water_logging': 3.0,
        'construction': 2.0, 'road_conditions': 2.0, 'tree_fall': 2.0, 'debris': 2.0, 'Debris': 2.0,
        'vehicle_breakdown': 1.0, 'pot_holes': 1.0, 'others': 1.0, 'congestion': 1.0, 'Fog / Low Visibility': 1.0,
        'test_demo': 1.0
    }).fillna(1.0).values
    prio_mult = df['priority'].map({'High': 1.5, 'Low': 1.0}).fillna(1.0).values
    closure_mult = df['requires_road_closure'].apply(lambda x: 2.0 if x else 1.0).values
    
    # Incorporate actual event duration to represent operational severity dynamically 
    # (longer events require higher manpower and equipment deployment)
    duration_factor = 1.0 + 0.15 * np.log1p(df['duration_min'] / 60.0)
    
    df['target_manpower'] = np.round(base_manpower * prio_mult * closure_mult * duration_factor) + df['is_peak_hour'] + (corridor_score == 100.0).astype(int)
    df['target_manpower'] = np.clip(df['target_manpower'].values, 1, 30).astype(int)

    # 6c. Recommended Barricades
    base_barricades = df['event_cause'].map({
        'construction': 20.0, 'public_event': 15.0, 'protest': 15.0, 'rally': 15.0, 'procession': 15.0,
        'water_logging': 10.0, 'pot_holes': 5.0, 'tree_fall': 5.0, 'accident': 5.0, 'debris': 5.0, 'Debris': 5.0, 'vip_movement': 5.0,
        'vehicle_breakdown': 0.0, 'others': 0.0, 'congestion': 0.0, 'road_conditions': 0.0, 'Fog / Low Visibility': 0.0,
        'test_demo': 0.0
    }).fillna(0.0).values
    road_closure_flat = df['requires_road_closure'].apply(lambda x: 20.0 if x else 0.0).values
    
    df['target_barricades'] = np.round(base_barricades * prio_mult * duration_factor) + road_closure_flat
    df['target_barricades'] = np.clip(df['target_barricades'].values, 0, 50).astype(int)

    # 6d. Diversion Requirement (Binary)
    df['target_diversion'] = ((df['requires_road_closure'] == True) | (df['target_eis'] >= 60.0)).astype(int)

    # 7. Generate Semantic Description Embeddings (using microsoft/harrier-oss-v1-0.6b)
    try:
        print("Generating sentence embeddings using microsoft/harrier-oss-v1-0.6b...")
        model_name = "microsoft/harrier-oss-v1-0.6b"
        print(f"Loading pre-trained SentenceTransformer: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Harrier requires instruction prefixing for document/passage encoding
        prefixed_descriptions = ["passage: " + desc for desc in df['description'].values]
        
        print("Encoding descriptions (this may take a minute)...")
        embeddings = model.encode(prefixed_descriptions, show_progress_bar=True, batch_size=64)
        print(f"Embeddings generated with shape: {embeddings.shape}")
    except Exception as e:
        print(f"Warning: Hugging Face model download/encoding failed: {e}")
        print("Falling back to local multilingual TF-IDF + TruncatedSVD (LSA) for text embeddings...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        # Build local character/word TF-IDF to capture Kannada and English text patterns
        vectorizer = TfidfVectorizer(max_features=5000, analyzer='char_wb', ngram_range=(3, 5))
        tfidf_matrix = vectorizer.fit_transform(df['description'].values)
        
        svd = TruncatedSVD(n_components=1024, random_state=42)
        embeddings = svd.fit_transform(tfidf_matrix).astype(np.float32)
        print(f"Local LSA text embeddings generated with shape: {embeddings.shape}")
        
    # Save embeddings separately as NumPy file
    np.save(embeddings_npy, embeddings)
    print(f"Saved description embeddings to {embeddings_npy}")

    # 8. Drop unusable/leakage columns to keep processed file clean
    unusable_cols = ['map_file', 'comment', 'meta_data', 'resolved_at_address', 
                     'resolved_at_latitude', 'resolved_at_longitude', 'resolved_by_id', 
                     'resolved_datetime', 'closed_by_id', 'closed_datetime', 'end_datetime', 
                     'end_address', 'endlatitude', 'endlongitude', 'start_datetime', 
                     'closed_datetime', 'created_date', 'modified_datetime']
    
    cols_to_drop = [c for c in unusable_cols if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)

    # Save processed CSV
    df_clean.to_csv(processed_csv, index=False)
    print(f"Saved clean processed CSV to {processed_csv}")
    print("==================================================")
    print("Data cleaning & feature engineering complete!")
    print("==================================================")

if __name__ == "__main__":
    process_dataset()
