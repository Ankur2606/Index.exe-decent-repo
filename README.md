# Event Driven Congestion (Planned and Unplanned)

Political rallies, festivals, sports events, construction activities, and sudden gatherings create localized traffic breakdowns.

Problem Statement 2 Submission by Team Insight.exe

Developed by Team Insight.exe
* Kavya Jain (Team Leader)
* Bhavya Pratap Singh Tomar
* Chitrakshi Gagrani
* Divyanshi Goyal

Deployed Application Prototype: https://huggingface.co/spaces/Insight-exe/astram-traffic-intel

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

[Insert Predict Dashboard Screenshot Here]

[Insert Live Operations Map Screenshot Here]

[Insert Analytics Dashboard Screenshot Here]

[Insert Project Context and Mermaid Flowchart Screenshot Here]

## Source Code Directory Structure

* docs/
  * analysis_results.md
  * architecture.md
  * feature_description.md
  * ps_description.md
  * result_analysis.md
  * submission_guidelines.md
  * ui_integration_idea.md
* src/
  * event_driven_congestion/
    * data_processing.py
    * evaluate.py
    * models.py
    * train.py
* download_models.py
* streamlit_app/
  * app.py
  * components/
    * map_component.py
    * model_loader.py
    * prediction_card.py
  * pages/
    * 1_Predict.py
    * 2_Live_Map.py
    * 3_Analytics.py
    * 4_About.py
  * utils/
    * feature_builder.py
    * geocoder.py
    * inference.py
* pyproject.toml
* requirements.txt
* setup.txt
* setup.sh
* setup.bash

## Setup and Execution

To run the application locally, ensure you have the uv package manager installed.

1. Build the virtual environment and install all dependencies:
   uv sync

2. Launch the Streamlit server:
   uv run streamlit run streamlit_app/app.py
