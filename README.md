# Event Driven Congestion (Planned and Unplanned)

Political rallies, festivals, sports events, construction activities, and sudden gatherings create localized traffic breakdowns.

Problem Statement 2 Submission by Team Insight.exe

Developed by Team Insight.exe
* Kavya Jain (Team Leader)
* Bhavya Pratap Singh Tomar
* Chitrakshi Gagrani
* Divyanshi Goyal

Deployed Application Prototype: https://index-exe-traffic-demand.streamlit.app/

## Live Demos & Deployments

| Interface | Link | Description |
|---|---|---|
| **Streamlit Prediction UI** | [index-exe-traffic-demand.streamlit.app](https://index-exe-traffic-demand.streamlit.app/) | Interactive traffic event prediction, heat maps, and analytics dashboard. |
| **Agentic Voice App** | [DecentSanage/astram_voice](https://huggingface.co/spaces/DecentSanage/astram_voice) | Voice-first agentic dispatch assistant. Speak your incident report and get instant ML-backed resource deployment recommendations on a call. |
| **Project Video Demo (Streamlit UI)** | [Drive Recording](https://drive.google.com/file/d/1u2x2mtRqVm9rYcwzN7HgfCLYb53Irdv7/view?usp=sharing) | Full walkthrough of the Streamlit prediction and analytics interface. |
| **Voice Web App Demo** | [Drive Recording](https://drive.google.com/file/d/1CtLlMR-YrlkuvRQhrqnKhOIeWE4lBdzd/view?usp=sharing) | Demonstration of the ASTraM voice dispatch web application. |


## Abstract
Bengaluru faces complex urban traffic challenges where real time events such as vehicle breakdowns, public processions, construction bottlenecks, and sudden weather anomalies cause cascading gridlocks. Traditional traffic control relies on reactive policing. This research presents the Active Spatio Temporal Resource Management (ASTraM) platform, a hybrid predictive system that forecasts event severity and automates operational resource dispatch recommendations for the Bengaluru Traffic Police. By combining deep semantic representation encoders with gradient boosted decision trees, the system estimates the Event Impact Score, calculates optimal personnel deployments, recommends barricade dispatches, and determines alternate route diversion protocols.

## Mathematical Formulation and Target Modeling
The system models traffic events as spatio temporal records containing location coordinates, cyclical time tags, categorical cause details, and unstructured text descriptions. We formulate this predictive problem under a multi task learning framework with four target variables:
1. Event Impact Score: A continuous regression target scaled from 0 (normal flow) to 100 (total road block) representing traffic queue severity.
2. Manpower Recommendation: A discrete integer count modeling the required number of traffic officers (scale 1 to 30) using Poisson loss functions.
3. Barricade Recommendation: A discrete integer count modeling the number of physical lane barriers to deploy (scale 0 to 50) using Poisson loss functions.
4. Diversion Requirement: A binary classification target (0 or 1) indicating if immediate route detour protocols must be activated.

## Spatio Temporal and Semantic Encoder Architecture
The core modeling engine uses a dual tower neural network constructed in PyTorch:

### Tower One: Incident Semantic Embeddings
Unstructured traffic log descriptions (in both Kannada and English) are processed using a multilingual Harrier sentence transformer to produce 1024 dimensional continuous embedding vectors. Categorical fields (such as event cause and priority) pass through independent categorical embedding layers. The text embeddings and categorical features are concatenated and projected through a dense linear layer with rectified linear activation (ReLU) and dropout to construct the incident semantic embedding vector.

### Tower Two: Spatio Temporal Context
Spatial coordinates (latitude and longitude) are mapped using Radial Basis Function spatial encodings to represent distance and neighborhood proximity. Geohash tokens (resolution levels 5, 6, and 7) pass through dense spatial embedding layers. Cyclical temporal variables (hour of day and day of week) are transformed using sine and cosine position encodings. These representations are concatenated and projected through a dense linear layer with ReLU and dropout to construct the spatio temporal context embedding vector.

### Multi Tower Representation Fusion
The incident embedding vector and the spatio temporal context embedding vector are fused using a multi layer cross fusion operator:
* Element wise Hadamard product to capture interactive dependencies.
* Vector absolute distance differences to capture representational separation.
* Original tower representations concatenation to retain absolute features.
The resulting fused vector is passed through shared dense representation blocks.

### Multi Task Decoupled Prediction Heads
The shared dense representation is split into four distinct task specific layers:
* Regression Head: Projecting to a single continuous unit using Mean Squared Error loss.
* Poisson Regressors: Two separate linear layers mapping to discrete personnel and barricade counts using Poisson negative log likelihood loss equipped with the Stirling approximation.
* Classification Head: Projecting to a binary logistic output using Binary Cross Entropy loss.

## Tabular Decision Forests
To capture categorical rules and threshold boundaries that neural networks might smooth over, a parallel 5 fold LightGBM decision forest ensemble is trained on the label encoded tabular features, coordinates, geohashes, and cyclical temporal embedders. The trees are regularized using row and feature bagging to prevent overfitting.

## Weighted Prediction Blending
Predictions from the 5 deep learning folds and the 5 gradient boosted forest folds are combined using validation optimized weights to maximize model generalizability:
* Event Impact Score: 40 percent Neural Network and 60 percent LightGBM
* Personnel Dispatch: 10 percent Neural Network and 90 percent LightGBM
* Barricade Dispatch: 50 percent Neural Network and 50 percent LightGBM
* Diversion Decisions: 50 percent Neural Network and 50 percent LightGBM

## Application Interfaces and Visualizations
### Prediction Page
<img width="1265" height="679" alt="image" src="https://github.com/user-attachments/assets/178e529e-c72f-47fa-91a9-83b8b9e9850d" />

---

### HeatMap Generation W.R.T Location and Cause
<img width="1121" height="615" alt="image" src="https://github.com/user-attachments/assets/74c5fde1-7bd1-468c-a256-9096431830a6" />

---

### Graphical Analytics
<img width="1213" height="622" alt="image" src="https://github.com/user-attachments/assets/0df0d205-8669-4585-9508-937879f328a8" />

---

### ML/DL Model Architecture
<img width="1022" height="522" alt="image" src="https://github.com/user-attachments/assets/74fba4a3-8dc8-4710-ab5e-9cd7cfcab7a0" />


## Source Code Directory Structure

### Core Research: Architecture & Feature Documentation
* `docs/` — Architecture notes, feature descriptions, and result analyses
  * `architecture.md` — Full ML/DL model architecture description
  * `feature_description.md` — Feature engineering methodology and column definitions
  * `analysis_results.md` — Detailed experimental results and ablation notes
  * `result_analysis.md` — Statistical breakdown of ensemble vs. baseline performance
  * `ps_description.md` — Problem statement framing and objectives
  * `submission_guidelines.md` — Submission structure and checklist
  * `ui_integration_idea.md` — Conceptual UI/UX design notes

### ML Pipeline Source
* `src/event_driven_congestion/` — Full ML training and evaluation pipeline
  * `data_processing.py` — Spatio-temporal feature extraction, geohash encoding, cyclical transforms
  * `models.py` — Dual-tower neural network and LightGBM model definitions
  * `train.py` — k-fold cross-validation training loop with weighted ensemble blending
  * `evaluate.py` — Multi-task evaluation metrics (MAE, RMSE, R², Accuracy, AUC)

### Training & Test Result Logs
* `logs/` — Full training run outputs and model evaluation reports
  * `run_summary.log` — **ML & DL model test results** — MAE, RMSE, R², AUC scores for all targets (EIS, Manpower, Barricades, Diversion)
  * `run_summary_baselines.log` — Baseline-only evaluation results for comparison
  * `train.log` — Epoch-level training loss curves and fold validation scores

### Inference & API
* `api.py` — FastAPI REST inference server (port 8000) serving the ensemble models
* `download_models.py` — Pre-downloads sentence transformer weights during Docker build

### Streamlit Dashboard
* `streamlit_app/`
  * `app.py` — Main Streamlit entry point
  * `components/` — Reusable UI components (map, model loader, prediction card)
  * `pages/` — 1_Predict.py, 2_Live_Map.py, 3_Analytics.py, 4_About.py
  * `utils/` — Feature builder, geocoder, inference engine

### Voice Agent (ASTraM Agentic Dispatch)
* `voice_agent/` — Google ADK multi-agent voice dispatch backend
  * `main.py` — FastAPI server entry point (port 7860), mounts static frontend
  * `agents.py` — Gemini ADK sequential agent: Requirements → RAG → Narrator
  * `websocket_bridge.py` — WebSocket router handling voice transcript and TTS streams
  * `knowledge_base.py` — ChromaDB RAG store for Bengaluru traffic guidelines
  * `shared_state.py` — Per-session incident state management
* `frontend/` — Next.js 16 voice dispatch web interface
  * `app/voice/page.tsx` — Main voice UI with 3-phase progressive wizard
  * `hooks/useVoiceSession.ts` — WebSocket + Web Speech API integration hook

### Configuration
* `pyproject.toml` — Project dependencies and build config
* `requirements.txt` — Pinned dependency lockfile for Docker builds
* `Dockerfile` — Multi-stage build: Next.js static export + Python FastAPI backend


## Setup and Execution

To run the application locally, ensure you have the uv package manager installed.

1. Build the virtual environment and install all dependencies:
   uv sync

2. Launch the Streamlit prediction dashboard:
   uv run streamlit run streamlit_app/app.py

### Running the Agentic Voice Dispatch App

The voice interface requires three concurrent processes. Open three separate terminal windows in the project root.

**Terminal 1 — ML Inference API (port 8000):**
```
uv run api.py
```

**Terminal 2 — Voice Agent WebSocket Server (port 7860):**
```
uv run .\voice_agent\main.py
```

**Terminal 3 — Next.js Frontend (port 3000):**
```
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000/voice` in your browser, grant microphone permissions, and click **Start Session** to begin a voice-guided incident report.

For external access (e.g., mobile testing), expose the voice server via ngrok:
```
ngrok http 127.0.0.1:7860
```
Then update `NEXT_PUBLIC_WS_URL` in the frontend environment to point to the ngrok WebSocket URL.
