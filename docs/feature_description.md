# Feature Description - ASTraM Traffic Event Intelligence

This document describes all the features ingested into the machine learning and deep learning models to predict event-driven traffic congestion (Event Impact Score) and recommend operational resources (Manpower, Barricades, and Diversions).

---

## 1. Incident Metadata Features

| Feature Name | Data Type | Description | Operational Significance |
| :--- | :--- | :--- | :--- |
| `event_type` | Categorical (Binary) | Whether the event is `planned` (rallies, festivals) or `unplanned` (accidents, breakdowns). | Planned events have longer durations and wider spatial impact compared to sudden breakdowns. |
| `event_cause` | Categorical | The root cause of the incident (e.g. `vehicle_breakdown`, `accident`, `water_logging`, `construction`). | Directly governs base resource allocation (e.g. construction needs barricades; accidents need police). |
| `priority` | Categorical (Binary) | Operator-assigned priority level (`High` or `Low`). | Reflects the immediate urgency and severity of the road block. |
| `requires_road_closure` | Categorical (Binary) | Boolean flag indicating if the event requires blocking traffic flow. | The single strongest predictor of extreme congestion and the need for diversions. |
| `veh_type` | Categorical | The class of vehicle involved (e.g., `heavy_vehicle`, `lcv`, `two_wheeler`). | Heavy vehicles (busses, trucks) block lanes completely and require heavy towing resources. |

---

## 2. Spatio-Temporal Context Features

| Feature Name | Data Type | Description | Operational Significance |
| :--- | :--- | :--- | :--- |
| `latitude`, `longitude` | Continuous | Exact GPS coordinates of the incident. | Used to compute spatial proximity and proximity to dense intersections. |
| `geohash_5` | Categorical (String) | Geohash code of length 5 (precision ~4.9 km x 4.9 km). | Identifies regional spatial sectors of Bengaluru. |
| `geohash_6` | Categorical (String) | Geohash code of length 6 (precision ~1.2 km x 0.6 km). | Identifies neighborhood-level traffic cells. |
| `geohash_7` | Categorical (String) | Geohash code of length 7 (precision ~150 m x 150 m). | Captures exact junction-level local bottlenecks. |
| `corridor` | Categorical | The major high-volume arterial road corridor (e.g., ORR, Tumkur Road, Bellary Road). | Events on corridors have massive cascading effects compared to local residential streets. |
| `police_station` | Categorical | The jurisdictional police station handling the incident. | Captures administrative boundaries and local traffic management dispatch centers. |
| `zone` | Categorical | The administrative traffic zone (e.g. West, South, Central). | Used for regional resource aggregation. |
| `hour_of_day` | Cyclical (Continuous) | Hour when the event was logged (represented via cyclical sin/cos encodings). | Captures daily peak/off-peak congestion cycles. |
| `day_of_week` | Cyclical (Continuous) | Day of the week (represented via cyclical sin/cos encodings). | Captures weekly congestion cycles (weekday rush hours vs weekend patterns). |
| `is_weekend` | Binary | Indicates if the event started on Saturday or Sunday. | Accounts for different traffic baselines on weekends. |
| `is_peak_hour` | Binary | Indicates if the event started during morning (06:00-09:59) or evening (18:00-21:59) peaks. | Incidents during peak hours have multiplier effects on delay times. |

---

## 3. Semantic Text Features

| Feature Name | Data Type | Description | Operational Significance |
| :--- | :--- | :--- | :--- |
| `description_embedding` | Continuous (Vector) | 1024-dimensional dense embedding generated using `microsoft/harrier-oss-v1-0.6b`. | Encodes detailed text logs, capturing nuance (e.g., "tyre burst", "crane needed", "no problem for traffic"). |
