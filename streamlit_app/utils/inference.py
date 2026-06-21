import os
import sys
import numpy as np
import torch
import streamlit as st

# Resolve project root and append paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "streamlit_app"))

from components.model_loader import (
    load_vocabularies,
    load_pytorch_models,
    load_lgbm_models,
    load_description_embedder
)
from utils.feature_builder import build_features

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InferenceEngine:
    def __init__(self):
        # Load vocabularies
        self.vocabularies = load_vocabularies()
        self.vocab_sizes = {col: len(vocab) for col, vocab in self.vocabularies.items()}
        
        # Load PyTorch Two-Tower models
        self.nn_models = load_pytorch_models(self.vocab_sizes)
        
        # Load LightGBM models
        self.lgb_models = load_lgbm_models()
        
        # Load text embedder
        self.embedder = load_description_embedder()
        
    def predict(self, raw_input: dict) -> dict:
        """
        Runs the full ensembled prediction pipeline on UI input.
        Args:
            raw_input (dict): Contains input fields:
                - latitude (float)
                - longitude (float)
                - event_type (str)
                - event_cause (str)
                - priority (str)
                - veh_type (str)
                - corridor (str)
                - police_station (str)
                - zone (str)
                - date (date object)
                - time (time object)
                - description (str)
        Returns:
            dict: predicted targets and confidence metadata
        """
        # 1. Text description embedding
        desc = raw_input.get('description', '')
        if self.embedder is not None and desc.strip():
            # Apply instruction prefix for Harrier embedder
            prefixed = "passage: " + desc.strip()
            text_emb = self.embedder.encode(prefixed, show_progress_bar=False)
        else:
            # Fallback to zero vector if offline/no text
            text_emb = np.zeros(1024, dtype=np.float32)
            
        # 2. Build features
        nn_inputs, gbdt_df = build_features(
            lat=raw_input['latitude'],
            lon=raw_input['longitude'],
            event_type=raw_input['event_type'],
            event_cause=raw_input['event_cause'],
            priority=raw_input['priority'],
            veh_type=raw_input['veh_type'],
            corridor=raw_input['corridor'],
            police_station=raw_input['police_station'],
            zone=raw_input['zone'],
            date_val=raw_input['date'],
            time_val=raw_input['time'],
            text_embedding=text_emb,
            vocabularies=self.vocabularies
        )
        
        # 3. Run PyTorch Two-Tower Models (5-Fold Ensemble)
        nn_eis_preds = []
        nn_manpower_preds = []
        nn_barricades_preds = []
        nn_diversion_preds = []
        
        if self.nn_models:
            # Move inputs to device
            inputs_device = {k: v.to(DEVICE) for k, v in nn_inputs.items()}
            with torch.no_grad():
                for model in self.nn_models:
                    p_eis, p_manpower, p_barricades, p_diversion = model(inputs_device)
                    nn_eis_preds.append(float(p_eis.cpu().numpy()[0]))
                    nn_manpower_preds.append(float(p_manpower.cpu().numpy()[0]))
                    nn_barricades_preds.append(float(p_barricades.cpu().numpy()[0]))
                    nn_diversion_preds.append(float(p_diversion.cpu().numpy()[0]))
        else:
            # Defaults if models failed to load
            nn_eis_preds = [30.0] * 5
            nn_manpower_preds = [2.0] * 5
            nn_barricades_preds = [5.0] * 5
            nn_diversion_preds = [0.0] * 5

        # 4. Run LightGBM Models (5-Fold Ensemble)
        lgb_eis_preds = []
        lgb_manpower_preds = []
        lgb_barricades_preds = []
        lgb_diversion_preds = []
        
        for fold in range(5):
            # EIS (Regression)
            if self.lgb_models['eis'] and len(self.lgb_models['eis']) > fold:
                lgb_eis_preds.append(float(self.lgb_models['eis'][fold].predict(gbdt_df)[0]))
            else:
                lgb_eis_preds.append(30.0)
                
            # Manpower (Poisson Regression)
            if self.lgb_models['manpower'] and len(self.lgb_models['manpower']) > fold:
                lgb_manpower_preds.append(float(self.lgb_models['manpower'][fold].predict(gbdt_df)[0]))
            else:
                lgb_manpower_preds.append(2.0)
                
            # Barricades (Poisson Regression)
            if self.lgb_models['barricades'] and len(self.lgb_models['barricades']) > fold:
                lgb_barricades_preds.append(float(self.lgb_models['barricades'][fold].predict(gbdt_df)[0]))
            else:
                lgb_barricades_preds.append(5.0)
                
            # Diversion (Binary Classification - predict probability)
            if self.lgb_models['diversion'] and len(self.lgb_models['diversion']) > fold:
                lgb_diversion_preds.append(float(self.lgb_models['diversion'][fold].predict_proba(gbdt_df)[0, 1]))
            else:
                lgb_diversion_preds.append(0.0)
                
        # 5. Average the predictions across folds
        avg_nn_eis = np.mean(nn_eis_preds)
        avg_nn_manpower = np.mean(nn_manpower_preds)
        avg_nn_barricades = np.mean(nn_barricades_preds)
        avg_nn_diversion = np.mean(nn_diversion_preds)
        
        avg_lgb_eis = np.mean(lgb_eis_preds)
        avg_lgb_manpower = np.mean(lgb_manpower_preds)
        avg_lgb_barricades = np.mean(lgb_barricades_preds)
        avg_lgb_diversion = np.mean(lgb_diversion_preds)
        
        # 6. Ensemble Blending (Target-specific weights)
        # - EIS: 40% NN + 60% GBDT
        # - Manpower: 10% NN + 90% GBDT
        # - Barricades: 50% NN + 50% GBDT
        # - Diversion: 50% NN + 50% GBDT
        fused_eis = 0.4 * avg_nn_eis + 0.6 * avg_lgb_eis
        fused_manpower = 0.1 * avg_nn_manpower + 0.9 * avg_lgb_manpower
        fused_barricades = 0.5 * avg_nn_barricades + 0.5 * avg_lgb_barricades
        fused_diversion_prob = 0.5 * avg_nn_diversion + 0.5 * avg_lgb_diversion
        
        # 7. Apply Physical Constraints and Post-processing
        eis_final = np.clip(fused_eis, 0.0, 100.0)
        manpower_final = int(np.clip(np.round(fused_manpower), 1, 30))
        barricades_final = int(np.clip(np.round(fused_barricades), 0, 50))
        diversion_final = bool(fused_diversion_prob >= 0.5)
        
        # Severity assignment
        if eis_final <= 30.0:
            severity = "LOW"
        elif eis_final <= 60.0:
            severity = "MODERATE"
        elif eis_final <= 80.0:
            severity = "HIGH"
        else:
            severity = "CRITICAL"
            
        # 8. Confidence Score (Standard deviation of ensembled EIS predictions scaled)
        all_eis_preds = nn_eis_preds + lgb_eis_preds
        eis_std = np.std(all_eis_preds)
        # Standard deviation of 0 means 100% agreement. Standard deviation of 15 or more means low agreement (around 50%).
        confidence_val = max(50.0, min(99.8, 100.0 - (eis_std * 3.5)))
        
        return {
            "eis": float(eis_final),
            "manpower": manpower_final,
            "barricades": barricades_final,
            "diversion": diversion_final,
            "eis_severity": severity,
            "confidence": float(confidence_val),
            # Detailed breakdown for debugging / explainability
            "nn_predictions": {
                "eis": float(avg_nn_eis),
                "manpower": float(avg_nn_manpower),
                "barricades": float(avg_nn_barricades),
                "diversion": float(avg_nn_diversion)
            },
            "lgb_predictions": {
                "eis": float(avg_lgb_eis),
                "manpower": float(avg_lgb_manpower),
                "barricades": float(avg_lgb_barricades),
                "diversion": float(avg_lgb_diversion)
            }
        }
