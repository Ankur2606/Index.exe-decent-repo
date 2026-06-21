#  UI Mapping Strategy: Using the `address` Column
Your idea of mapping the `address` column to latitude/longitude in the UI is **highly recommended and operationally standard**!
* **How it helps**: 
  1. **Dashboard Usability**: Raw coordinates (lat/lng) are unintuitive for human dispatch operators. Displaying the structured `address` text directly on event cards and tooltips makes the dispatching action immediate.
  2. **Interactive Search & Pin Placement**: In the UI, operators can search for addresses using geocoders (e.g., Google Maps or OpenStreetMap Nominatim APIs) to automatically pin events and query predictions for locations before coordinates are verified on-site.
  3. **Data Integrity**: By keeping the high-cardinality `address` text out of the ML model training inputs, we prevent overfitting, while keeping it in the database allows us to use it for human validation and UI mapping.



