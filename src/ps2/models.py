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
        # total concat dim: 8 (type) + 16 (cause) + 8 (priority) + 1 (road closure flag) + 8 (veh_type) + text_embedding_dim
        incident_in_dim = 8 + 16 + 8 + 1 + 8 + text_embedding_dim
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
        # total concat dim: 2 (coords) + 16*6 (embeddings) + 6 (temporal sin/cos) + 2 (temporal flags)
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
        
        # 1. Event Impact Score (EIS) Head: [0, 100]
        self.eis_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 2. Recommended Manpower Head: >= 1
        self.manpower_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        # 3. Recommended Barricades Head: >= 0
        self.barricades_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        # 4. Diversion Plan Head: [0, 1]
        self.diversion_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # 1. Process Incident Tower
        e_type = self.embed_event_type(inputs['event_type'])
        e_cause = self.embed_event_cause(inputs['event_cause'])
        e_prio = self.embed_priority(inputs['priority'])
        e_veh = self.embed_veh_type(inputs['veh_type'])
        road_closure = inputs['requires_road_closure'].unsqueeze(1)
        text_emb = inputs['description_embedding']
        
        inc_concat = torch.cat([e_type, e_cause, e_prio, road_closure, e_veh, text_emb], dim=1)
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
