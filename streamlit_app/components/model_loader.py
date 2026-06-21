import os
import sys
import pickle
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer

# Resolve project root and append source package to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "event_driven_congestion"))

from models import PyTorchTwoTowerModel

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_vocabularies():
    """
    Loads categorical vocabularies dictionary.
    """
    vocab_path = os.path.join(MODELS_DIR, "vocabularies.pkl")
    if not os.path.exists(vocab_path):
        st.error(f"Vocabularies file not found at {vocab_path}")
        return {}
    with open(vocab_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_pytorch_models(vocab_sizes):
    """
    Loads 5-fold PyTorch Two-Tower model checkpoints.
    """
    models = []
    for fold in range(5):
        model_path = os.path.join(MODELS_DIR, f"two_tower_fold_{fold}.pt")
        if not os.path.exists(model_path):
            st.error(f"PyTorch model checkpoint not found for fold {fold} at {model_path}")
            continue
        
        # Instantiate Two-Tower model structure
        model = PyTorchTwoTowerModel(vocab_sizes, text_embedding_dim=1024)
        
        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            models.append(model)
        except Exception as e:
            st.error(f"Error loading PyTorch model fold {fold}: {e}")
            
    return models

@st.cache_resource
def load_lgbm_models():
    """
    Loads 5-fold LightGBM model checkpoints for each target.
    Returns:
        dict: target_key -> list of 5 estimators
    """
    targets = ['eis', 'manpower', 'barricades', 'diversion']
    lgb_models = {target: [] for target in targets}
    
    for target in targets:
        for fold in range(5):
            model_path = os.path.join(MODELS_DIR, f"lgb_{target}_fold_{fold}.pkl")
            if not os.path.exists(model_path):
                st.error(f"LightGBM model not found for target {target} fold {fold} at {model_path}")
                continue
            try:
                with open(model_path, "rb") as f:
                    gbm = pickle.load(f)
                lgb_models[target].append(gbm)
            except Exception as e:
                st.error(f"Error loading LightGBM model for target {target} fold {fold}: {e}")
                
    return lgb_models

@st.cache_resource
def load_description_embedder():
    """
    Loads SentenceTransformer model for Kannada/English description embedding.
    """
    model_name = "microsoft/harrier-oss-v1-0.6b"
    try:
        # st.info("Initializing SentenceTransformer...")
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.warning(f"Failed to load Hugging Face SentenceTransformer '{model_name}' due to: {e}. Falling back to zero-vector embedding for text.")
        return None
