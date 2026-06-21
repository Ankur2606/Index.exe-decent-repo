import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lightgbm as lgb

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Import PyTorch model definition
from models import PyTorchTwoTowerModel

class TrafficDataset(Dataset):
    def __init__(self, df, embeddings, vocabularies):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings.astype(np.float32)
        self.vocabularies = vocabularies

        # Pre-map categorical strings to integer index values
        self.cat_mappings = {}
        for col, vocab in vocabularies.items():
            # Build string to index dictionary (index starts at 1, 0 represents unseen/unknown)
            self.cat_mappings[col] = {val: idx + 1 for idx, val in enumerate(vocab)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Helper to map string category to index
        def get_cat_idx(col):
            val = str(row[col])
            return self.cat_mappings[col].get(val, 0)

        item = {
            # Tower 1 inputs
            'event_type': torch.tensor(get_cat_idx('event_type'), dtype=torch.long),
            'event_cause': torch.tensor(get_cat_idx('event_cause'), dtype=torch.long),
            'priority': torch.tensor(get_cat_idx('priority'), dtype=torch.long),
            'veh_type': torch.tensor(get_cat_idx('veh_type'), dtype=torch.long),
            'description_embedding': torch.tensor(self.embeddings[idx], dtype=torch.float32),
            
            # Tower 2 inputs
            'coordinates': torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32),
            'geohash_5': torch.tensor(get_cat_idx('geohash_5'), dtype=torch.long),
            'geohash_6': torch.tensor(get_cat_idx('geohash_6'), dtype=torch.long),
            'geohash_7': torch.tensor(get_cat_idx('geohash_7'), dtype=torch.long),
            'corridor': torch.tensor(get_cat_idx('corridor'), dtype=torch.long),
            'police_station': torch.tensor(get_cat_idx('police_station'), dtype=torch.long),
            'zone': torch.tensor(get_cat_idx('zone'), dtype=torch.long),
            'temporal_cyclical': torch.tensor([
                row['hour_sin'], row['hour_cos'],
                row['day_sin'], row['day_cos'],
                row['month_sin'], row['month_cos']
            ], dtype=torch.float32),
            'temporal_flags': torch.tensor([row['is_weekend'], row['is_peak_hour']], dtype=torch.float32),
            
            # Targets
            'target_eis': torch.tensor(row['target_eis'], dtype=torch.float32),
            'target_manpower': torch.tensor(row['target_manpower'], dtype=torch.float32),
            'target_barricades': torch.tensor(row['target_barricades'], dtype=torch.float32),
            'target_diversion': torch.tensor(row['target_diversion'], dtype=torch.float32)
        }
        return item

def train_pipeline():
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    original_stdout = sys.stdout
    sys.stdout = DualLogger(os.path.join(logs_dir, "train.log"))

    print("==================================================")
    print("Starting PyTorch GPU Training Pipeline")
    print("==================================================")

    # 1. Device configuration (CUDA GPU vs CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    processed_csv = os.path.join("data", "processed_astram_events.csv")
    embeddings_npy = os.path.join("data", "processed_descriptions.npy")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(processed_csv) or not os.path.exists(embeddings_npy):
        print("Error: Processed dataset or embeddings not found. Run data_processing.py first.")
        return

    # Load Data
    df = pd.read_csv(processed_csv)
    embeddings = np.load(embeddings_npy)
    
    # Pre-build categorical vocabularies
    categorical_cols = ['event_type', 'event_cause', 'priority', 'veh_type', 
                        'geohash_5', 'geohash_6', 'geohash_7', 'corridor', 
                        'police_station', 'zone']
    vocabularies = {}
    vocab_sizes = {}
    for col in categorical_cols:
        vocab = df[col].astype(str).unique().tolist()
        vocabularies[col] = vocab
        vocab_sizes[col] = len(vocab)
    
    vocab_path = os.path.join(models_dir, "vocabularies.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocabularies, f)

    # Setup Cross-Validation
    df['eis_bin'] = pd.qcut(df['target_eis'], q=5, labels=False, duplicates='drop')
    df['stratify_key'] = df['eis_bin'].astype(str) + "_" + df['event_cause'].astype(str)
    counts = df['stratify_key'].value_counts()
    singletons = counts[counts == 1].index
    df.loc[df['stratify_key'].isin(singletons), 'stratify_key'] = "misc"

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(df, df['stratify_key']))

    # NN Out-of-fold validation arrays
    nn_val_predictions = {
        'eis': np.zeros(len(df)),
        'manpower': np.zeros(len(df)),
        'barricades': np.zeros(len(df)),
        'diversion': np.zeros(len(df))
    }

    print("\n--- Training PyTorch Two-Tower Dual Encoder ---")
    
    # Losses
    criterion_eis = nn.HuberLoss(delta=1.0)
    criterion_manpower = nn.PoissonNLLLoss(log_input=False, full=True)
    criterion_barricades = nn.PoissonNLLLoss(log_input=False, full=True)
    criterion_diversion = nn.BCELoss()

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n👉 Training Fold {fold+1}/5...")
        
        train_sub_df = df.iloc[train_idx]
        val_sub_df = df.iloc[val_idx]
        
        train_dataset = TrafficDataset(train_sub_df, embeddings[train_idx], vocabularies)
        val_dataset = TrafficDataset(val_sub_df, embeddings[val_idx], vocabularies)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        model = PyTorchTwoTowerModel(vocab_sizes, text_embedding_dim=embeddings.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
        
        best_val_loss = float('inf')
        epochs = 25
        patience = 8
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            # tqdm loop for steps
            tqdm_desc = f"Epoch {epoch+1:02d}/{epochs:02d} [Train]"
            pbar = tqdm(train_loader, desc=tqdm_desc, leave=False)
            
            for batch in pbar:
                # Move tensors to device
                inputs = {k: v.to(device) for k, v in batch.items() if not k.startswith('target_')}
                t_eis = batch['target_eis'].to(device)
                t_manpower = batch['target_manpower'].to(device)
                t_barricades = batch['target_barricades'].to(device)
                t_diversion = batch['target_diversion'].to(device)
                
                optimizer.zero_grad()
                p_eis, p_manpower, p_barricades, p_diversion = model(inputs)
                
                loss_eis = criterion_eis(p_eis, t_eis)
                loss_manpower = criterion_manpower(p_manpower, t_manpower)
                loss_barricades = criterion_barricades(p_barricades, t_barricades)
                loss_diversion = criterion_diversion(p_diversion, t_diversion)
                
                loss = loss_eis + 2.0 * loss_manpower + 2.0 * loss_barricades + loss_diversion
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(t_eis)
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            train_loss /= len(train_dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {k: v.to(device) for k, v in batch.items() if not k.startswith('target_')}
                    t_eis = batch['target_eis'].to(device)
                    t_manpower = batch['target_manpower'].to(device)
                    t_barricades = batch['target_barricades'].to(device)
                    t_diversion = batch['target_diversion'].to(device)
                    
                    p_eis, p_manpower, p_barricades, p_diversion = model(inputs)
                    
                    loss_eis = criterion_eis(p_eis, t_eis)
                    loss_manpower = criterion_manpower(p_manpower, t_manpower)
                    loss_barricades = criterion_barricades(p_barricades, t_barricades)
                    loss_diversion = criterion_diversion(p_diversion, t_diversion)
                    
                    loss = loss_eis + 2.0 * loss_manpower + 2.0 * loss_barricades + loss_diversion
                    val_loss += loss.item() * len(t_eis)
                    
            val_loss /= len(val_dataset)
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            scheduler.step()
            
            # Early stopping and checkpoint saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(models_dir, f"two_tower_fold_{fold}.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break
        
        # Load best model for validation prediction
        best_model = PyTorchTwoTowerModel(vocab_sizes, text_embedding_dim=embeddings.shape[1]).to(device)
        best_model.load_state_dict(torch.load(os.path.join(models_dir, f"two_tower_fold_{fold}.pt")))
        best_model.eval()
        
        with torch.no_grad():
            start_idx = 0
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if not k.startswith('target_')}
                # Find indexes inside fold val set
                p_eis, p_manpower, p_barricades, p_diversion = best_model(inputs)
                
                # Unpack and store OOF predictions
                batch_size = len(p_eis)
                batch_val_idx = val_idx[start_idx : start_idx + batch_size]
                
                nn_val_predictions['eis'][batch_val_idx] = p_eis.cpu().numpy()
                nn_val_predictions['manpower'][batch_val_idx] = p_manpower.cpu().numpy()
                nn_val_predictions['barricades'][batch_val_idx] = p_barricades.cpu().numpy()
                nn_val_predictions['diversion'][batch_val_idx] = p_diversion.cpu().numpy()
                
                start_idx += batch_size

    # Save OOF
    nn_oof_path = os.path.join(models_dir, "nn_oof_predictions.pkl")
    with open(nn_oof_path, "wb") as f:
        pickle.dump(nn_val_predictions, f)
    print(f"OOF predictions saved to {nn_oof_path}")

    # 6. GBDT Training
    print("\n--- Training LightGBM baseline classifiers & regressors ---")
    df_gbdt = df.copy()
    for col in categorical_cols:
        df_gbdt[col] = df_gbdt[col].astype('category').cat.codes
        
    tab_features = ['event_type', 'event_cause', 'priority', 
                    'veh_type', 'latitude', 'longitude', 'corridor', 'police_station', 'zone',
                    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 
                    'is_weekend', 'is_peak_hour']
                    
    desc_cols = [f'desc_emb_{i}' for i in range(embeddings.shape[1])]
    desc_df = pd.DataFrame(embeddings, columns=desc_cols)
    X_all = pd.concat([df_gbdt[tab_features], desc_df], axis=1)
    
    gbdt_val_predictions = {
        'eis': np.zeros(len(df)),
        'manpower': np.zeros(len(df)),
        'barricades': np.zeros(len(df)),
        'diversion': np.zeros(len(df))
    }
    
    targets = {
        'eis': ('target_eis', 'regression'),
        'manpower': ('target_manpower', 'poisson'),
        'barricades': ('target_barricades', 'poisson'),
        'diversion': ('target_diversion', 'binary')
    }
    
    for target_key, (target_col, task_type) in targets.items():
        print(f"Training LightGBM for target: {target_col} ({task_type})...")
        y = df[target_col].values
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if task_type == 'regression':
                gbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31, 
                                        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                                        random_state=42, verbose=-1)
            elif task_type == 'poisson':
                gbm = lgb.LGBMRegressor(objective='poisson', n_estimators=500, learning_rate=0.04, num_leaves=31, 
                                        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                                        random_state=42, verbose=-1)
            else:
                gbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.04, num_leaves=31, 
                                         subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                                         random_state=42, verbose=-1)
                
            gbm.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            
            gbm_path = os.path.join(models_dir, f"lgb_{target_key}_fold_{fold}.pkl")
            with open(gbm_path, "wb") as f:
                pickle.dump(gbm, f)
                
            if task_type == 'binary':
                gbdt_val_predictions[target_key][val_idx] = gbm.predict_proba(X_val)[:, 1]
            else:
                gbdt_val_predictions[target_key][val_idx] = gbm.predict(X_val)
                
    gbdt_oof_path = os.path.join(models_dir, "gbdt_oof_predictions.pkl")
    with open(gbdt_oof_path, "wb") as f:
        pickle.dump(gbdt_val_predictions, f)
    print("GBDT training completed!")
    print("==================================================")
    
    # Restore stdout
    sys.stdout = original_stdout

if __name__ == "__main__":
    train_pipeline()
