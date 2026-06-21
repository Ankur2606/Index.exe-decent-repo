import numpy as np
import pandas as pd
import torch
import pygeohash as pgh

def build_features(
    lat: float,
    lon: float,
    event_type: str,
    event_cause: str,
    priority: str,
    veh_type: str,
    corridor: str,
    police_station: str,
    zone: str,
    date_val,
    time_val,
    text_embedding: np.ndarray,
    vocabularies: dict
) -> tuple:
    """
    Builds model-ready features for both PyTorch Two-Tower and LightGBM models.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        event_type (str): Event type
        event_cause (str): Event cause
        priority (str): Priority
        veh_type (str): Vehicle type
        corridor (str): Corridor
        police_station (str): Police station
        zone (str): Zone
        date_val: Date object
        time_val: Time object
        text_embedding (np.ndarray): 1024-dimensional description embedding
        vocabularies (dict): Categorical vocabularies from training data
        
    Returns:
        tuple: (nn_inputs, gbdt_df)
            - nn_inputs (dict): Dict of torch.Tensor for PyTorch Two-Tower forward pass
            - gbdt_df (pd.DataFrame): 1-row DataFrame for LightGBM prediction
    """
    # 1. Coordinate Clipping to Bengaluru Bounds
    lat_min, lat_max = 12.8, 13.25
    lng_min, lng_max = 77.35, 77.85
    lat_clean = np.clip(lat, lat_min, lat_max)
    lon_clean = np.clip(lon, lng_min, lng_max)

    # 2. Geohash Generation
    geohash_5 = pgh.encode(lat_clean, lon_clean, precision=5)
    geohash_6 = pgh.encode(lat_clean, lon_clean, precision=6)
    geohash_7 = pgh.encode(lat_clean, lon_clean, precision=7)

    # 3. Temporal Feature Calculations
    hour = int(time_val.hour)
    day_of_week = int(date_val.weekday()) # Monday=0, Sunday=6
    month = int(date_val.month)

    hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
    day_sin = float(np.sin(2 * np.pi * day_of_week / 7.0))
    day_cos = float(np.cos(2 * np.pi * day_of_week / 7.0))
    month_sin = float(np.sin(2 * np.pi * month / 12.0))
    month_cos = float(np.cos(2 * np.pi * month / 12.0))

    is_weekend = 1.0 if day_of_week in [5, 6] else 0.0
    is_peak_hour = 1.0 if hour in [6, 7, 8, 9, 18, 19, 20, 21] else 0.0

    raw_features = {
        'event_type': event_type,
        'event_cause': event_cause,
        'priority': priority,
        'veh_type': veh_type,
        'geohash_5': geohash_5,
        'geohash_6': geohash_6,
        'geohash_7': geohash_7,
        'corridor': corridor,
        'police_station': police_station,
        'zone': zone
    }

    # 4. Neural Network Inputs
    nn_inputs = {}
    for col, val in raw_features.items():
        vocab = vocabularies.get(col, [])
        # Convert all vocab elements and input value to strings to handle mixed-type lists (e.g. float nan)
        val_str = str(val)
        vocab_str = [str(v) for v in vocab]
        # NN expects 1-based index, 0 is unseen/unknown
        idx = vocab_str.index(val_str) + 1 if val_str in vocab_str else 0
        nn_inputs[col] = torch.tensor([idx], dtype=torch.long)

    nn_inputs['coordinates'] = torch.tensor([[lat_clean, lon_clean]], dtype=torch.float32)
    nn_inputs['temporal_cyclical'] = torch.tensor([[
        hour_sin, hour_cos,
        day_sin, day_cos,
        month_sin, month_cos
    ]], dtype=torch.float32)
    nn_inputs['temporal_flags'] = torch.tensor([[is_weekend, is_peak_hour]], dtype=torch.float32)
    nn_inputs['description_embedding'] = torch.tensor(text_embedding, dtype=torch.float32).unsqueeze(0)

    # 5. GBDT (LightGBM) Inputs
    gbdt_features = {}
    for col, val in raw_features.items():
        if col in ['geohash_5', 'geohash_6', 'geohash_7']:
            # GBDTs do not use geohashes, they use lat/lon directly
            continue
        vocab = vocabularies.get(col, [])
        val_str = str(val)
        vocab_str = [str(v) for v in vocab]
        # GBDTs were trained on category codes (which are sorted alphabetically in pandas)
        sorted_vocab = sorted(vocab_str)
        idx = sorted_vocab.index(val_str) if val_str in sorted_vocab else -1
        gbdt_features[col] = idx

    gbdt_features['latitude'] = lat_clean
    gbdt_features['longitude'] = lon_clean
    gbdt_features['hour_sin'] = hour_sin
    gbdt_features['hour_cos'] = hour_cos
    gbdt_features['day_sin'] = day_sin
    gbdt_features['day_cos'] = day_cos
    gbdt_features['month_sin'] = month_sin
    gbdt_features['month_cos'] = month_cos
    gbdt_features['is_weekend'] = is_weekend
    gbdt_features['is_peak_hour'] = is_peak_hour

    # Description embeddings columns
    desc_cols = [f'desc_emb_{i}' for i in range(len(text_embedding))]
    desc_dict = {col: text_embedding[i] for i, col in enumerate(desc_cols)}
    
    # Combine tabular and description features
    combined_features = {**gbdt_features, **desc_dict}
    
    # Create DataFrame with the exact features list in order
    tab_features_order = ['event_type', 'event_cause', 'priority', 'veh_type', 
                          'latitude', 'longitude', 'corridor', 'police_station', 'zone',
                          'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 
                          'is_weekend', 'is_peak_hour']
    
    ordered_features = {}
    for feat in tab_features_order:
        ordered_features[feat] = [combined_features[feat]]
    for col in desc_cols:
        ordered_features[col] = [combined_features[col]]
        
    gbdt_df = pd.DataFrame(ordered_features)
    
    return nn_inputs, gbdt_df
