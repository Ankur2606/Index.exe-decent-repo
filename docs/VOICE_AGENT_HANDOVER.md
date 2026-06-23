# ASTraM Voice Intelligence Project Walkthrough and Handover

This document outlines the voice first operational dispatch system built for the Bengaluru Traffic Police. The system transcribes operator voice incident reports, resolves spatial coordinate boundaries, runs ensembled machine learning predictions, and retrieves factual operational recommendations from a persistent vector database.

Hugging Face Space: DecentSanage/astram_voice

## System Architecture

The application is structured into a React NextJS frontend and a FastAPI backend server.

### Backend Components

All backend logic resides in the voice_agent package:
*   [main.py](file:///voice_agent/main.py): Entry point of the server. Inits the database on startup and mounts the compiled frontend static files directory.
*   [agents.py](file:///voice_agent/agents.py): Google ADK workflow compiled from three sequential agents:
    *   RequirementsAgent: Connects to the Gemini Live API to gather core operational variables and maps spatial names to coordinates.
    *   RAGAgent: Performs context lookups against the traffic corpus vector store.
    *   NarratorAgent: Prepares the dispatch summary.
*   [knowledge_base.py](file:///voice_agent/knowledge_base.py): ChromaDB client configuration. Stores traffic guidelines and embeds document chunks using Gemini embedding models. Caches embeddings to skip reindexing on system reload.
*   [websocket_bridge.py](file:///voice_agent/websocket_bridge.py): FastAPI WebSocket router managing the bidirectional communication stream.
*   [shared_state.py](file:///voice_agent/shared_state.py): Session storage keeping track of the resolved incident attributes and ML predictions.

### Frontend Dashboard Components

The client code resides in the frontend folder:
*   [page.tsx](file:///frontend/app/voice/page.tsx): Main operational dashboard layout displaying the active zones:
    *   Zone A (Header): Session controls and language selection.
    *   Zone B (Voice Visualizer): Animated SVG waveform ring and rolling live transcript.
    *   Zone C (Staged Fields): Real time confirmation status of target variables.
    *   Zone D (Predictions and Actions): Impact gauges, resource requirements, and alternate routes.
*   [useVoiceSession.ts](file:///frontend/hooks/useVoiceSession.ts): Custom React hook wrapping the Web Speech API and WebSocket protocol. Dynamically resolves host routes to adapt to cloud environments.
*   [types.ts](file:///frontend/lib/types.ts): Shared TypeScript declarations.

## Execution Flow

1. Operator clicks Start Session.
2. Browser Web Speech API captures voice and converts speech to text in English, Hindi, or Kannada.
3. WebSocket streams transcriptions to the backend.
4. Gemini extracts the five target variables (location, event type, cause, vehicle type, and priority).
5. Backend queries the ensembled ML model to predict incident impact, required traffic officers, barricades, and whether a detour route is required.
6. ChromaDB retrieves localized traffic guidelines and displays alternative route advice.
7. The final dispatch directive is narrated back to the operator.

## Hugging Face Spaces Deployment

The application is deployed on Hugging Face Spaces as a Docker container.

### Staged Commit Process
The codebase is pushed in three sequential commits using python HfApi to bypass local git connection resets:
1. Upload of description embeddings processed_descriptions.npy.
2. Upload of LightGBM and Neural Network model files.
3. Upload of application source code and frontend assets.

### Dockerfile Settings
The container build executes the following:
*   Frontend compilation to a static export.
*   Python dependency installation under python 3.12 slim base image to satisfy scipy dependencies.
*   Predownloading and caching sentence transformer model weights during the build phase.
*   Exposing default port 7860.

## Local Execution and Testing Guidelines

Follow these steps to run the application locally and verify functionality.

### Prerequisite Environment Setup

1. Create a file named `.env` in the root folder of the project.
2. Add your Google API key to enable Gemini connections:
   `GEMINI_API_KEY=your_actual_api_key_here`
3. Save the file.

### Running the Backend Server

1. Switch to the root folder of the project.
2. Verify that python packages are installed.
3. Run the following command to start the FastAPI server:
   `uv run python voice_agent/main.py`
4. This starts the backend on port 7860. The system will automatically build the vector database in the data directory if it is not already present.

### Running the Frontend Client

1. Navigate to the frontend directory:
   `cd frontend`
2. Install the necessary node modules:
   `npm install`
3. Launch the development server:
   `npm run dev`
4. This launches the NextJS dashboard on port 3000. Open your browser and navigate to the voice page:
   `http://localhost:3000/voice`

### Verification and Expected Behavior

Follow this verification flow to test the system:

1. Visit the voice page on your local server or Hugging Face Space.
2. Grant microphone access permissions in the browser when prompted.
3. Click the Start Session button in Zone A. The status indicator will transition to LISTENING.
4. Voice Transcription Test: Speak into the microphone. Voice transcription will stream in Zone B.
5. Preset Simulation Test: Click the button labeled ORR Construction Closure.
6. Expected Results:
   * The variables in Zone C (Location, Event Type, Cause, Priority, and Vehicle Type) will instantly update to confirmed.
   * The predictions in Zone D will display:
     * Event Impact Score circular gauge updates.
     * Recommended manpower count is populated.
     * Recommended barricades count is populated.
     * Route Diversion Banner displays True.
     * Detour recommendations from ChromaDB display alternate routes.
   * A synthesized dispatch summary is narrated back.
