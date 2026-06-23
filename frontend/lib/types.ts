export type Speaker = "agent" | "user";

export interface TranscriptLine {
  speaker: Speaker;
  text: string;
  ts: string;
}

export interface PredictionData {
  event_impact_score: number;
  severity_band: "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  recommended_officers: number;
  recommended_barricades: number;
  diversion_required: "YES" | "NO" | boolean;
  ensemble_confidence: string;
}

export type SessionState = "READY" | "LISTENING" | "PROCESSING" | "COMPLETE";

export interface FieldState {
  name: string;
  value: string;
  state: "empty" | "collecting" | "confirmed";
}

export interface ResolvedFields {
  corridor: string;
  police_station: string;
  zone: string;
  date: string;
  time: string;
}

export type WsMessage =
  | { type: "field_update"; field: string; value: string }
  | { type: "field_resolved"; field: string; value: string }
  | { type: "transcript"; speaker: Speaker; text: string; ts: string }
  | { type: "prediction_start" }
  | { type: "prediction_result"; data: PredictionData }
  | { type: "rag_recommendations"; items: string[] }
  | { type: "narration_start" }
  | { type: "session_complete" };
