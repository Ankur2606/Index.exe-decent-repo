import threading
import copy

class SharedSessionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {
            "fields": {
                "location": "",
                "event_type": "",
                "event_cause": "",
                "priority": "",
                "vehicle_type": ""
            },
            "resolved": {
                "latitude": 12.9685753,
                "longitude": 77.7011831,
                "corridor": "Non-corridor",
                "police_station": "unknown",
                "zone": "unknown",
                "date": "2026-06-21",
                "time": "18:16:40"
            },
            "prediction": None,
            "recommendations": [],
            "transcripts": []
        }

    def get_state(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def set_field(self, field_name: str, value: str):
        with self.lock:
            if field_name in self.state["fields"]:
                self.state["fields"][field_name] = value

    def set_resolved(self, key: str, value: str):
        with self.lock:
            if key in self.state["resolved"]:
                self.state["resolved"][key] = value

    def set_prediction(self, prediction_dict):
        with self.lock:
            self.state["prediction"] = prediction_dict

    def set_recommendations(self, recs):
        with self.lock:
            self.state["recommendations"] = recs

    def add_transcript(self, speaker: str, text: str, ts: str):
        with self.lock:
            self.state["transcripts"].append({
                "speaker": speaker,
                "text": text,
                "ts": ts
            })

    def reset(self):
        with self.lock:
            self.state["fields"] = {
                "location": "",
                "event_type": "",
                "event_cause": "",
                "priority": "",
                "vehicle_type": ""
            }
            self.state["resolved"] = {
                "latitude": 12.9685753,
                "longitude": 77.7011831,
                "corridor": "Non-corridor",
                "police_station": "unknown",
                "zone": "unknown",
                "date": "2026-06-21",
                "time": "18:16:40"
            }
            self.state["prediction"] = None
            self.state["recommendations"] = []
            self.state["transcripts"] = []
