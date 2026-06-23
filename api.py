import os
import sys
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Resolve project paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "streamlit_app"))

from utils.inference import InferenceEngine

app = FastAPI(
    title="ASTraM Event-Driven Congestion API",
    description="REST API for predicting traffic incident impact and resources deployment.",
    version="1.0"
)

# Global inference engine instance
engine = None

class PredictRequest(BaseModel):
    latitude: float = Field(..., description="Latitude of the incident (12.80 to 13.25)", example=12.9685753)
    longitude: float = Field(..., description="Longitude of the incident (77.35 to 77.85)", example=77.7011831)
    event_type: str = Field("unplanned", description="Event type (unplanned or planned)", example="planned")
    event_cause: str = Field("others", description="Cause of the incident (e.g. construction, water_logging)", example="construction")
    priority: str = Field("High", description="Priority level (High or Low)", example="High")
    veh_type: str = Field("unknown", description="Vehicle type involved", example="unknown")
    corridor: str = Field("Non-corridor", description="Corridor name", example="ORR East 2")
    police_station: str = Field("unknown", description="Police station jurisdiction", example="HAL Old Airport")
    zone: str = Field("unknown", description="Administrative zone (can be nan as string)", example="East Zone 1")
    date: datetime.date = Field(..., description="Date of the incident (YYYY-MM-DD)", example="2026-06-21")
    time: datetime.time = Field(..., description="Time of the incident (HH:MM:SS)", example="18:16:40")
    description: str = Field("", description="Raw Kannada or English description log of the event", example="[LOCATION] towards marathhalli and karthiknagara towards mahadevpura traffic movement will be slow due to metrostation work")

@app.on_event("startup")
def startup_event():
    global engine
    # Initialize the ensembled model inference engine on startup
    engine = InferenceEngine()

@app.post("/predict")
def predict(payload: PredictRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not loaded.")
    
    try:
        # Convert request payload to the raw dict format expected by InferenceEngine
        raw_input = {
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "event_type": payload.event_type,
            "event_cause": payload.event_cause,
            "priority": payload.priority,
            "veh_type": payload.veh_type,
            "corridor": payload.corridor,
            "police_station": payload.police_station,
            "zone": payload.zone,
            "date": payload.date,  # FastAPI automatically parses to date object
            "time": payload.time,  # FastAPI automatically parses to time object
            "description": payload.description
        }
        res = engine.predict(raw_input)
        return {
            "status": "success",
            "predictions": {
                "event_impact_score": res["eis"],
                "severity_band": res["eis_severity"],
                "recommended_officers": res["manpower"],
                "recommended_barricades": res["barricades"],
                "diversion_required": res["diversion"],
                "ensemble_confidence": f"{res['confidence']:.2f}%"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "engine_loaded": engine is not None}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
