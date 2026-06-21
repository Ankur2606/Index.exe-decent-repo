import streamlit as st

st.title("Problem Statement 2: Event-Driven Congestion (Planned & Unplanned)")
st.caption("Political rallies, festivals, sports events, construction activities, and sudden gatherings create localized traffic breakdowns.")

# Section 1: Problem Statement & Context
st.header("Operational Context & Challenges")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Bengaluru faces complex urban traffic challenges where real-time events such as vehicle breakdowns, public processions, construction bottlenecks, and sudden weather anomalies cause localized traffic breakdowns. To maintain vehicular flow and respond to incidents, the **Bengaluru Traffic Police (BTP)** requires early-warning decision aids to forecast the congestion intensity and dispatch operational resources dynamically and precisely.
    
    Our predictive resource management platform addresses this critical need. Given a traffic incident's location, cause, priority, and textual log descriptions, the system predicts: (1) the **Event Impact Score (EIS)** measuring queue severity from 0 (low flow restriction) to 100 (total gridlock), (2) the **Manpower Recommendation** (number of traffic officers to deploy, scale 1-30), (3) the **Barricades Dispatch Count** (0-50 physical barricades), and (4) the **Diversion Plan** (whether an immediate detour protocol is required).
    
    By wrapping complex predictive modeling pipelines into an intuitive, real-time dispatch dashboard, this system empowers dispatchers to minimize delay response times, optimize traffic patrol schedules, and prevent cascading gridlocks across major corridors.
    """)

with col2:
    st.markdown("### Operational Objectives")
    st.markdown("""
    *   **Congestion Control**: Prevent bottlenecks before queues block major intersections.
    *   **Staff Optimization**: Deploy officers where their traffic-regulation impact is highest.
    *   **Resource Dispatch**: Minimize the cost of logistics by matching barricade dispatches to physical lane restrictions.
    *   **Detour Automation**: Automate early alerts for route diversions.
    """)

st.markdown("<hr style='border: 0; border-top: 1px solid #2C303E; margin: 1.5rem 0;'>", unsafe_allow_html=True)

# Section 2: Model Architecture
st.header("Predictive Modeling Architecture")
st.markdown("""
The ASTraM prediction framework is built upon an integrated estimation engine that combines deep spatio-temporal representations with robust decision trees:
*   **Integrated Spatial and Semantic Engine**: Processes geographical coordinates, spatial geohashes, temporal cyclical cycles, and textual logs to generate a continuous event representation context.
*   **Categorical Analytics Module**: Evaluates event parameters such as event causes, vehicle categories, and priority classes to resolve operational thresholds and categorical decision rules.
""")

# Show Mermaid Diagram
st.markdown("#### 📐 Decoupled Two-Tower & LightGBM Multi-Task Processing Architecture")

import streamlit.components.v1 as components

mermaid_html = """
<div style="background-color: #0E1117; color: #FAFAFA; font-family: 'Inter', sans-serif; padding: 10px;">
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({
            startOnLoad: true,
            theme: 'dark',
            themeVariables: {
                background: '#0E1117',
                primaryColor: '#1A1D26',
                primaryTextColor: '#FAFAFA',
                lineColor: '#2C303E',
                secondaryColor: '#2C303E',
                tertiaryColor: '#2C303E'
            }
        });
    </script>
    <div class="mermaid" style="background-color: #0E1117; display: flex; justify-content: center;">
        graph TD
            A1["Categorical Fields<br>(Event Type, Event Cause, Priority, Vehicle Type)"] --> B1["Categorical Embedding Layers"]
            A2["Raw Description Text<br>(Kannada / English Traffic Logs)"] --> B2["Harrier-OSS-v1-0.6b Embedder<br>(1024-dim Contextual Embeddings)"]
            B1 --> C1["Concatenation & Dense Linear Block<br>(256 Hidden Units, ReLU, Dropout: 0.2)"]
            B2 --> C1

            X1["GPS Coordinates<br>(Latitude, Longitude)"] --> Y1["RBF Spatial Encoding<br>(Radial Basis Proximity Matrix)"]
            X2["Geohash Codes<br>(Resolution 5, 6, 7 Strings)"] --> Y2["Geohash Embeddings"]
            X3["Cyclical Time Input<br>(Incident Hour, Day of Week)"] --> Y3["Sine / Cosine Position Encoding"]
            Y1 --> Z1["Concatenation & Dense Linear Block<br>(256 Hidden Units, ReLU, Dropout: 0.2)"]
            Y2 --> Z1
            Y3 --> Z1

            C1 --> Fusion["Multi-Layer Cross-Fusion Layer<br>(Hadamard Product + Vector Absolute Distance + Concatenation)"]
            Z1 --> Fusion
            
            Fusion --> SharedDense["Shared Dense Layers<br>(dim: 128 -> 64, ReLU, Dropout: 0.2)"]

            SharedDense --> Head1["Event Impact Score (EIS) Head<br>(Continuous regression with MSE loss)"]
            SharedDense --> Head2["Manpower Dispatch Head<br>(Discrete Poisson with Stirling NLL loss)"]
            SharedDense --> Head3["Barricade Dispatch Head<br>(Discrete Poisson with Stirling NLL loss)"]
            SharedDense --> Head4["Diversion Plan Head<br>(Binary sigmoid classifier with Weighted BCE loss handling data imabalance)"]

            TabularInputs["Tabular Features<br>(Label-Encoded Categories + Numerical Scales)"] --> LGBM["5-Fold LightGBM Ensemble<br>(Robust decision boundary estimators)"]

            Head1 --> CombineEIS["Weighted EIS Combiner<br>(40% Neural Network + 60% LightGBM)"]
            LGBM --> CombineEIS
            
            Head2 --> CombineManpower["Weighted Manpower Combiner<br>(10% Neural Network + 90% LightGBM)"]
            LGBM --> CombineManpower

            Head3 --> CombineBarricades["Weighted Barricades Combiner<br>(50% Neural Network + 50% LightGBM)"]
            LGBM --> CombineBarricades

            Head4 --> CombineDiversions["Weighted Diversion Combiner<br>(30% Neural Network + 70% LightGBM)"]
            LGBM --> CombineDiversions

            CombineEIS --> Out1["Final Event Impact Score (0 - 100)"]
            CombineManpower --> Out2["Recommended Officers (1 - 30)"]
            CombineBarricades --> Out3["Recommended Barricades (0 - 50)"]
            CombineDiversions --> Out4["Diversion Plan Decision (Yes / No)"]

            classDef tower1 fill:#2C303E,stroke:#4B5563,stroke-width:2px,color:#FAFAFA;
            classDef tower2 fill:#1F2937,stroke:#3B82F6,stroke-width:2px,color:#FAFAFA;
            classDef fusion fill:#311B92,stroke:#6200EA,stroke-width:2px,color:#FAFAFA;
            classDef heads fill:#1B5E20,stroke:#00C853,stroke-width:2px,color:#FAFAFA;
            classDef inputs fill:#0F172A,stroke:#1E293B,stroke-width:1px,color:#94A3B8;
            classDef ensemble fill:#880E4F,stroke:#FF4081,stroke-width:2px,color:#FAFAFA;
            classDef output fill:#E65100,stroke:#FF6D00,stroke-width:2px,color:#FAFAFA;

            class A1,A2,X1,X2,X3,TabularInputs inputs;
            class B1,B2,C1 tower1;
            class Y1,Y2,Y3,Z1 tower2;
            class Fusion,SharedDense fusion;
            class Head1,Head2,Head3,Head4 heads;
            class LGBM,CombineEIS,CombineManpower,CombineBarricades,CombineDiversions ensemble;
            class Out1,Out2,Out3,Out4 output;
    </div>
</div>
"""

st.iframe(mermaid_html, height=850)

st.markdown("<hr style='border: 0; border-top: 1px solid #2C303E; margin: 1.5rem 0;'>", unsafe_allow_html=True)

# Section 3: Performance Metrics
st.header("Model Performance & Operational Metrics")

# Render metrics grid as columns
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown("""
    <div style="background-color: #1A1D26; border-radius: 6px; padding: 1.5rem 1rem; border-top: 4px solid #2ECC71; text-align: center; min-height: 120px;">
        <span style="font-size: 0.8rem; color: #8A8F98; text-transform: uppercase; font-weight: bold; display: block; margin-bottom: 0.5rem;">Event Impact Score</span>
        <span style="font-size: 1.8rem; font-weight: bold; color: #FAFAFA; display: block; margin-bottom: 0.3rem;">95.97%</span>
        <span style="font-size: 0.8rem; color: #2ECC71; font-weight: 500;">R²: 0.7011</span>
    </div>
    """, unsafe_allow_html=True)
    
with m_col2:
    st.markdown("""
    <div style="background-color: #1A1D26; border-radius: 6px; padding: 1.5rem 1rem; border-top: 4px solid #3498DB; text-align: center; min-height: 120px;">
        <span style="font-size: 0.8rem; color: #8A8F98; text-transform: uppercase; font-weight: bold; display: block; margin-bottom: 0.5rem;">Manpower Dispatch</span>
        <span style="font-size: 1.8rem; font-weight: bold; color: #FAFAFA; display: block; margin-bottom: 0.3rem;">96.67%</span>
        <span style="font-size: 0.8rem; color: #3498DB; font-weight: 500;">R²: 0.8666</span>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown("""
    <div style="background-color: #1A1D26; border-radius: 6px; padding: 1.5rem 1rem; border-top: 4px solid #9B59B6; text-align: center; min-height: 120px;">
        <span style="font-size: 0.8rem; color: #8A8F98; text-transform: uppercase; font-weight: bold; display: block; margin-bottom: 0.5rem;">Barricades Dispatch</span>
        <span style="font-size: 1.8rem; font-weight: bold; color: #FAFAFA; display: block; margin-bottom: 0.3rem;">95.08%</span>
        <span style="font-size: 0.8rem; color: #9B59B6; font-weight: 500;">R²: 0.8302</span>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown("""
    <div style="background-color: #1A1D26; border-radius: 6px; padding: 1.5rem 1rem; border-top: 4px solid #E74C3C; text-align: center; min-height: 120px;">
        <span style="font-size: 0.8rem; color: #8A8F98; text-transform: uppercase; font-weight: bold; display: block; margin-bottom: 0.5rem;">Diversion Decisions</span>
        <span style="font-size: 1.8rem; font-weight: bold; color: #FAFAFA; display: block; margin-bottom: 0.3rem;">91.45%</span>
        <span style="font-size: 0.8rem; color: #E74C3C; font-weight: 500;">Accuracy Rate</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown("#### Validation Performance and Safety Bounds")
st.markdown(r"""
| Target Variable | Primary Metric | Model Reliability (Percentage) | Average Operational Error | Error Margin & Boundaries |
| :--- | :--- | :--- | :--- | :--- |
| **Event Impact Score (EIS)** | R²: `0.7011` | **`95.97%` Accuracy** | &plusmn; 4.03 points (Scale 0 to 100) | &plusmn; 7.38 points (RMSE) in extreme cases |
| **Manpower Recommendation** | R²: `0.8666` | **`96.67%` Accuracy** | &plusmn; 0.33 officers (Scale 1 to 30) | Off by >= 1 officer in only 20% of cases |
| **Barricade Recommendation** | R²: `0.8302` | **`95.08%` Accuracy** | &plusmn; 2.39 barricades (Scale 0 to 50) | Off by >= 5 barricades in only 10% of cases |
| **Diversion Requirement** | Accuracy: `91.45%` | **`91.45%` Accuracy** | Recall: `76.68%` (F1-Score: `0.7841`) | `8.55%` Error Rate (Correct 11 of 12 times) |
""")

st.markdown("<hr style='border: 0; border-top: 1px solid #2C303E; margin: 1.5rem 0;'>", unsafe_allow_html=True)

# Section 4: Tech Stack
st.header("System Technology Stack")
st.markdown("""
The dashboard and models are engineered using:
- **Core ML Frameworks**: `PyTorch` (Multi-Task Neural Networks), `LightGBM` (Gradient Boosting Decision Trees)
- **NLP / Embeddings**: `SentenceTransformers` (`microsoft/harrier-oss-v1-0.6b` multilingual embedder)
- **UI Platform**: `Streamlit` (Interactive Dashboard Layer)
- **Spatial Resolution & Geocoding**: `Folium` & `streamlit-folium` (Mapping engines), `pygeohash` (geohash precision codes 5, 6, 7), `geopy` (Nominatim OSM geocoding API)
- **Performance & Serialization**: `Joblib` / `Pickle` (model serialization), `Pandas` (tabular data manipulation), `NumPy` (vector math)
- **Package Management**: `uv` (Fast dependency compilation and execution)
""")

# Section 5: Team Credits
st.markdown("<hr style='border: 0; border-top: 1px solid #2C303E; margin: 1.5rem 0;'>", unsafe_allow_html=True)
st.header("Project Team")
st.markdown("""
Developed by **Insight.exe** for the Flipkart Gridlock Hackathon 2.0:
*   **Kavya Jain** (Team Leader)
*   **Bhavya Pratap Singh Tomar**
*   **Chitrakshi Gagrani**
*   **Divyanshi Goyal**
""")
