# ASTraM Traffic Event Intelligence: Data Analysis & Strategy Report

This report summarizes our findings from the Exploratory Data Analysis (EDA) of the ASTraM event dataset, our data cleaning and feature engineering strategy, the formulation of operational ground-truth targets, and our model architecture design.

---

## 1. Exploratory Data Analysis (EDA) Findings

The raw dataset contains **8,173 rows and 46 columns** representing traffic incident reports logged by traffic operations in Bengaluru.

### Geospatial Distribution
*   **Bounding Box**: 99.79% of coordinate entries reside within Greater Bengaluru bounds: Latitude $[12.80, 13.25]$ and Longitude $[77.35, 77.85]$. 
*   **Anomalies**: Only 17 rows fell slightly outside this bounding box, which we clip to the boundaries during preprocessing to prevent spatial distortion while preserving the rows.
*   **Hierarchical Clustering**: Geohashing at precisions 5, 6, and 7 yields:
    *   68 unique sectors (precision 5, ~4.9 km)
    *   685 neighborhoods (precision 6, ~1.2 km x 0.6 km)
    *   2,788 junctions/points (precision 7, ~150 m)

### Missingness Profiling
*   **Unusable Features**: `map_file`, `meta_data`, and `comment` are 100% null and have been dropped.
*   **Highly Sparse Features**: Columns such as `direction` (99.47% null), `resolved_at_*` (99.09%), and `route_path` (98.32%) represent detailed logs that are rarely populated.
*   **Categorical/Essential Features**: Key features like `event_type`, `event_cause`, `priority`, and `requires_road_closure` are 100% complete and highly reliable.

### Temporal Frequencies
*   **Seasonality**: The start datetimes span from November 2023 to April 2024.
*   **Hourly Peaks**: Incidents are heavily clustered during:
    *   **Night/Early Morning (00:00-05:59)**: 2,777 logs (reflecting freight/truck breakdowns during night logistics).
    *   **Morning Peak (06:00-09:59)**: 1,627 logs.
    *   **Evening Peak (18:00-21:59)**: 2,297 logs.

### Incident Duration Outliers
*   Only **38.28%** of rows have explicit duration logs (`closed_datetime` - `start_datetime`).
*   **Negative Durations**: 3 rows had negative durations (closed datetime set before start datetime), representing logging errors.
*   **Extreme Tail Outliers**: The median duration is **64.5 minutes**, but the mean is 6,352 minutes (~4.4 days) due to open-ended cases like construction or unresolved reports. 95% of events close within 27 days.
*   *Solution*: We clipped durations to a maximum of 24 hours (1,440 minutes) and imputed missing durations using the median duration of corresponding `event_cause` and `priority` groups.

### Semantic Text Logs
*   **83.36%** of reports include a textual `description`.
*   **Multilingual Mix**: The descriptions contain a mix of Kannada, English, and local slang (e.g., "diesel prablam", "punchure").
*   *Solution*: We utilize a multilingual SentenceTransformer (`microsoft/harrier-oss-v1-0.6b`) to generate 1024-dimensional dense vectors, with a robust local TF-IDF + SVD (LSA) fallback to handle any network connection failures.

---

## 2. Feature Engineering & Positional Encodings

To maximize prediction performance, we engineered the following feature classes:
1.  **Temporal Cyclical Encodings**: Map `hour`, `day_of_week`, and `month` to cyclical sine and cosine curves to preserve temporal continuity (e.g., hour 23 is adjacent to hour 0).
2.  **Geohashing**: Embed coordinates as string categories at three hierarchical scales to allow the deep learning model to learn localized spatial bottlenecks.
3.  **Temporal Flags**: Binary flags for `is_weekend` and `is_peak_hour` to capture traffic multipliers.

---

## 3. Label Formulation Framework (Ground Truth)

Since the raw dataset does not contain explicit target columns for resource requirements, we designed a robust, traffic-engineering framework to calculate consistent labels:

### 1. Event Impact Score (EIS) - Range $[0, 100]$
Combines log-duration, priority, road closure necessity, event cause severity, and corridor density:
$$\text{EIS} = 0.3 \cdot S_{\text{dur}} + 0.25 \cdot S_{\text{closure}} + 0.15 \cdot S_{\text{prio}} + 0.2 \cdot S_{\text{cause}} + 0.1 \cdot S_{\text{corridor}}$$
*Where $S_{\text{dur}}$ is log-normalized duration, and other scores represent weights based on operational severity (e.g. road closures, major corridors, high-severity causes).*

### 2. Recommended Manpower - Range $[1, 30]$
Predicts the number of traffic officers to deploy:
$$\text{Manpower} = \text{round}(\text{base\_manpower} \times \text{priority\_mult} \times \text{closure\_mult}) + \text{is\_peak\_hour} + \text{is\_corridor}$$
*Base manpower ranges from 1 (breakdown) to 10 (public rallies), scaled by multipliers and adjusted for peak times.*

### 3. Recommended Barricades - Range $[0, 50]$
Predicts physical barricades needed for traffic management:
$$\text{Barricades} = \text{round}(\text{base\_barricades} \times \text{priority\_mult}) + (20 \text{ if requires road closure else } 0)$$
*Construction and planned rallies receive high base barricade counts, while road closures require a flat +20 block.*

### 4. Diversion Plan Required - Binary $\{0, 1\}$
$$\text{Diversion} = 1 \text{ if (requires road closure OR } \text{EIS} \ge 60) \text{ else } 0$$

---

## 4. Modeling & Training Strategy

### SOTA Two-Tower Dual Encoder Network
Separates inputs into two independent pipelines:
1.  **Incident Tower**: Learns representations for incident categories (`event_cause`, `event_type`, `priority`, `requires_road_closure`) and textual logs.
2.  **Context Tower**: Learns representations for where and when the event occurs (coordinates, geohash embeddings, cyclical temporal features).
3.  **Interaction Fusion**: Merges both towers using concatenation, absolute differences, and element-wise multiplication to capture non-linear relationships.
4.  **Multi-Task Heads**: Predicts all 4 targets simultaneously using specialized loss functions (Poisson loss for count regression; Huber loss for continuous score; Binary Cross-Entropy for classification).

### Gradient Boosting Baselines (LightGBM/CatBoost)
Trains individual trees on flattened dense embeddings combined with tabular features. The predictions of GBDT and the Two-Tower neural network are aggregated in a final weighted ensemble.

### Validation Scheme
We use **5-fold Stratified Cross-Validation**, stratifying folds by the binned `target_eis` and `event_cause` to eliminate fold bias.
All results, losses, and final comparisons will be output to `logs/run_summary.log`.
