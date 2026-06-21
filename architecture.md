# Model Architecture - Two-Tower Dual Encoder and Gradient Boosting Baselines

This document details the machine learning and deep learning architecture built to forecast event-driven traffic congestion (Event Impact Score) and recommend operational resources (Manpower, Barricades, and Diversions).

---

## 1. Architectural Overview

To solve the multi-faceted prediction problem, we use a hybrid modeling strategy combining:
1. **A Deep Spatio-Temporal Two-Tower Dual Encoder (PyTorch)**: Designed to decouple incident characteristics from the spatio-temporal context, mapping them into a shared latent space before cross-attention fusion.
2. **A SOTA Gradient Boosting Baseline (LightGBM/CatBoost)**: Used as a fast, high-performance tabular baseline and part of our final ensemble.

```mermaid
graph TD
    subgraph Incident Tower (Tower 1)
        A1[event_cause, event_type, priority, requires_road_closure] --> B1[Embedding Layers]
        A2[description text] --> B2[Harrier-OSS-v1-0.6b Embedding]
        B1 --> C1[Concat & Dense Layer]
        B2 --> C1
    end

    subgraph Context Tower (Tower 2)
        X1[latitude, longitude] --> Y1[RBF Spatial Encoding]
        X2[geohash_5, geohash_6, geohash_7] --> Y2[Embedding Layers]
        X3[hour, day_of_week] --> Y3[Sin/Cos Position Encodings]
        Y1 --> Z1[Concat & Dense Layer]
        Y2 --> Z1
        Y3 --> Z1
    end

    C1 --> Fusion[Multi-Layer Cross Fusion / Dot Product]
    Z1 --> Fusion
    
    Fusion --> SharedDense[Shared Dense Layers + Dropout]
    
    SharedDense --> Head1[EIS Head: Regression]
    SharedDense --> Head2[Manpower Head: Poisson Regressor]
    SharedDense --> Head3[Barricades Head: Poisson Regressor]
    SharedDense --> Head4[Diversion Head: Binary Classifier]
```

---

## 2. Two-Tower Neural Network Architecture Details

The model is defined in [models.py](file:///c:/Users/bhavy/Desktop/Kaam/Hackathon/Gridlock-flipkart/Gridlock%20Round%202/ps2/src/ps2/models.py). The detailed layers, parameter shapes, and fusion mechanisms are described below.

### Tower 1: Incident Encoder (Incident Tower)
Processes categorical and textual details of the incident:
*   **Categorical Embeddings**: Mapped to learned low-dimensional spaces:
    *   `event_type` $\rightarrow$ Embedding of size $(\text{vocab\_size} + 1, 8)$
    *   `event_cause` $\rightarrow$ Embedding of size $(\text{vocab\_size} + 1, 16)$
    *   `priority` $\rightarrow$ Embedding of size $(\text{vocab\_size} + 1, 8)$
    *   `veh_type` $\rightarrow$ Embedding of size $(\text{vocab\_size} + 1, 8)$
*   **Text Embedding**: Ingests a pre-computed 1024-dimensional dense semantic embedding generated from the raw text description using the pre-trained `microsoft/harrier-oss-v1-0.6b` model.
*   **Concatenated Incident Vector**: Concatenates categorical embeddings and description embeddings, yielding a combined dimension of:
    $$\text{incident\_in\_dim} = 8 + 16 + 8 + 8 + 1024 = 1064$$
*   **Dense Feedforward Pipeline**:
    *   `Linear(1064, 128)`
    *   `SELU` activation
    *   `LayerNorm(128)`
    *   `Linear(128, 128)`
    *   `SELU` activation
    Yields incident representation vector $\mathbf{u} \in \mathbb{R}^{128}$.

### Tower 2: Context Encoder (Context Tower)
Processes the spatial and temporal context of where and when the incident occurred:
*   **Spatial Context Embeddings**:
    *   Exact coordinates `latitude` and `longitude` are passed as raw continuous inputs (dim=2).
    *   `geohash_5` $\rightarrow$ Embedding of size $(\text{vocab\_size} + 1, 16)$
    *   `geohash_6` $\rightarrow$ Embedding of size $(\text{vocab\_size} + 1, 16)$
    *   `geohash_7` $\rightarrow$ Embedding of size $(\text{vocab\_size} + 1, 16)$
    *   `corridor` $\rightarrow$ Embedding of size $(\text{vocab\_size} + 1, 16)$
    *   `police_station` $\rightarrow$ Embedding of size $(\text{vocab\_size} + 1, 16)$
    *   `zone` $\rightarrow$ Embedding of size $(\text{vocab\_size} + 1, 16)$
*   **Temporal Context Encodings**:
    *   `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `month_sin`, `month_cos` continuous values representing cyclical temporal position (dim=6).
    *   `is_weekend`, `is_peak_hour` binary flags (dim=2).
*   **Concatenated Context Vector**: Concatenates spatial inputs, categorical spatial embeddings, cyclical features, and binary temporal flags, yielding a combined dimension of:
    $$\text{context\_in\_dim} = 2 + (16 \times 6) + 6 + 2 = 106$$
*   **Dense Feedforward Pipeline**:
    *   `Linear(106, 128)`
    *   `SELU` activation
    *   `LayerNorm(128)`
    *   `Linear(128, 128)`
    *   `SELU` activation
    Yields context representation vector $\mathbf{v} \in \mathbb{R}^{128}$.

### Interaction Fusion Layer
To capture complex non-linear interactions between the incident details ($\mathbf{u} \in \mathbb{R}^{128}$) and environmental context ($\mathbf{v} \in \mathbb{R}^{128}$), we calculate:
1.  **Joint Representation (Concatenation)**: $\mathbf{f}_{\text{cat}} = [\mathbf{u}; \mathbf{v}] \in \mathbb{R}^{256}$
2.  **Element-wise Similarity (Hadamard Product)**: $\mathbf{f}_{\text{prod}} = \mathbf{u} \odot \mathbf{v} \in \mathbb{R}^{128}$
3.  **Absolute Distance (Difference)**: $\mathbf{f}_{\text{diff}} = |\mathbf{u} - \mathbf{v}| \in \mathbb{R}^{128}$
All three vectors are concatenated to construct the interaction space representation $\mathbf{h}_{\text{fuse}} \in \mathbb{R}^{512}$:
$$\mathbf{h}_{\text{fuse}} = [\mathbf{f}_{\text{cat}}; \mathbf{f}_{\text{diff}}; \mathbf{f}_{\text{prod}}]$$

### Shared Hidden Layers
The fused interaction vector $\mathbf{h}_{\text{fuse}}$ passes through shared hidden layers to form general traffic representations:
*   `Linear(512, 128)`
*   `SELU` activation
*   `Dropout(p=0.3)`
*   `Linear(128, 64)`
*   `SELU` activation
Yields shared embedding $\mathbf{z} \in \mathbb{R}^{64}$.

### Multi-Task Prediction Heads
specialized heads branch off the shared representation $\mathbf{z}$ to make predictions for each target variable:
1.  **Event Impact Score (EIS) Head** (Continuous $[0, 100]$):
    *   `Linear(64, 1)` $\rightarrow$ `Sigmoid()` activation $\rightarrow$ Scale output by multiplying by $100.0$.
2.  **Recommended Manpower Head** (Poisson count $\ge 1$):
    *   `Linear(64, 1)` $\rightarrow$ `Softplus()` activation $\rightarrow$ Shift output by adding $1.0$ (ensures deployment is at least 1 officer).
3.  **Recommended Barricades Head** (Poisson count $\ge 0$):
    *   `Linear(64, 1)` $\rightarrow$ `Softplus()` activation.
4.  **Diversion Plan Head** (Binary classification $\{0, 1\}$):
    *   `Linear(64, 1)` $\rightarrow$ `Sigmoid()` activation.

---

## 3. PyTorch Model Code Definition

Below is the complete PyTorch implementation of the `PyTorchTwoTowerModel` class:

```python
import torch
import torch.nn as nn

class PyTorchTwoTowerModel(nn.Module):
    def __init__(self, vocab_sizes, text_embedding_dim=1024):
        super(PyTorchTwoTowerModel, self).__init__()
        
        # ------------------ TOWER 1: INCIDENT ENCODER ------------------
        # Embeddings for categorical inputs
        self.embed_event_type = nn.Embedding(vocab_sizes['event_type'] + 1, 8)
        self.embed_event_cause = nn.Embedding(vocab_sizes['event_cause'] + 1, 16)
        self.embed_priority = nn.Embedding(vocab_sizes['priority'] + 1, 8)
        self.embed_veh_type = nn.Embedding(vocab_sizes['veh_type'] + 1, 8)
        
        # Dense block for Incident Tower
        incident_in_dim = 8 + 16 + 8 + 8 + text_embedding_dim
        self.incident_dense = nn.Sequential(
            nn.Linear(incident_in_dim, 128),
            nn.SELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.SELU()
        )
        
        # ------------------ TOWER 2: CONTEXT ENCODER ------------------
        # Embeddings for spatial codes
        self.embed_geohash_5 = nn.Embedding(vocab_sizes['geohash_5'] + 1, 16)
        self.embed_geohash_6 = nn.Embedding(vocab_sizes['geohash_6'] + 1, 16)
        self.embed_geohash_7 = nn.Embedding(vocab_sizes['geohash_7'] + 1, 16)
        self.embed_corridor = nn.Embedding(vocab_sizes['corridor'] + 1, 16)
        self.embed_police_station = nn.Embedding(vocab_sizes['police_station'] + 1, 16)
        self.embed_zone = nn.Embedding(vocab_sizes['zone'] + 1, 16)
        
        # Dense block for Context Tower
        context_in_dim = 2 + 16*6 + 6 + 2
        self.context_dense = nn.Sequential(
            nn.Linear(context_in_dim, 128),
            nn.SELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.SELU()
        )
        
        # ------------------ FUSION & HEADS ------------------
        # Multi-task shared layers
        # fused dim: 128 (incident) + 128 (context) + 128 (abs diff) + 128 (product) = 512
        self.shared_dense = nn.Sequential(
            nn.Linear(512, 128),
            nn.SELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.SELU()
        )
        
        # Heads
        self.eis_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.manpower_head = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
        self.barricades_head = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
        self.diversion_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, inputs):
        # 1. Process Incident Tower
        e_type = self.embed_event_type(inputs['event_type'])
        e_cause = self.embed_event_cause(inputs['event_cause'])
        e_prio = self.embed_priority(inputs['priority'])
        e_veh = self.embed_veh_type(inputs['veh_type'])
        text_emb = inputs['description_embedding']
        
        inc_concat = torch.cat([e_type, e_cause, e_prio, e_veh, text_emb], dim=1)
        inc_out = self.incident_dense(inc_concat)
        
        # 2. Process Context Tower
        coords = inputs['coordinates']
        gh_5 = self.embed_geohash_5(inputs['geohash_5'])
        gh_6 = self.embed_geohash_6(inputs['geohash_6'])
        gh_7 = self.embed_geohash_7(inputs['geohash_7'])
        corr = self.embed_corridor(inputs['corridor'])
        ps = self.embed_police_station(inputs['police_station'])
        zone = self.embed_zone(inputs['zone'])
        temp_cyc = inputs['temporal_cyclical']
        temp_flags = inputs['temporal_flags']
        
        ctx_concat = torch.cat([coords, gh_5, gh_6, gh_7, corr, ps, zone, temp_cyc, temp_flags], dim=1)
        ctx_out = self.context_dense(ctx_concat)
        
        # 3. Decoupled Fusion Interaction
        fused_cat = torch.cat([inc_out, ctx_out], dim=1)
        fused_diff = torch.abs(inc_out - ctx_out)
        fused_prod = inc_out * ctx_out
        
        fused_all = torch.cat([fused_cat, fused_diff, fused_prod], dim=1)
        shared_out = self.shared_dense(fused_all)
        
        # 4. Predict Heads
        eis = self.eis_head(shared_out) * 100.0
        manpower = self.manpower_head(shared_out) + 1.0
        barricades = self.barricades_head(shared_out)
        diversion = self.diversion_head(shared_out)
        
        return eis.squeeze(1), manpower.squeeze(1), barricades.squeeze(1), diversion.squeeze(1)
```

---

## 4. Loss Formulation & Optimization Strategy

### Unified Multi-Task Loss
The total loss is minimized jointly using weighted sum aggregation:
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{EIS}} + \lambda_2 \mathcal{L}_{\text{manpower}} + \lambda_3 \mathcal{L}_{\text{barricades}} + \lambda_4 \mathcal{L}_{\text{diversion}}$$
Where:
- $\lambda_1 = 1.0$ (Huber Loss)
- $\lambda_2 = 2.0$ (Poisson Loss)
- $\lambda_3 = 2.0$ (Poisson Loss)
- $\lambda_4 = 1.0$ (Binary Cross-Entropy Loss)

### Optimization
- **Optimizer**: AdamW (Adam with Weight Decay) with an initial learning rate of $10^{-3}$ and weight decay of $10^{-4}$.
- **Learning Rate Scheduler**: Cosine Annealing learning rate decay.
- **Regularization**: Early Stopping on validation loss with patience of 6 epochs. Save best weights for each fold to `models/two_tower_fold_{fold}.pt`.

---

## 5. Validation & Observability Strategy

- **5-Fold Stratified Cross-Validation**: Folds are stratified based on the binned target Event Impact Score (EIS) and `event_cause` to ensure each fold has an identical distribution of incidents and severity levels.
- **Observability**: During training, all hyper-parameters, training losses per epoch, validation losses, and final fold metrics are logged to `logs/run_summary.log`.
