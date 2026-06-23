import os
import sys
from fastapi.staticfiles import StaticFiles

# Resolve project paths and append to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "streamlit_app"))

# Import standard app instance from api
from api import app
from voice_agent.websocket_bridge import router as ws_router
from voice_agent.knowledge_base import build_knowledge_base

# Add the websocket router
app.include_router(ws_router)

@app.on_event("startup")
def startup_event():
    # Build RAG knowledge base on start
    print("Initializing voice intelligence backend services")
    try:
        build_knowledge_base()
    except Exception as e:
        print(f"Failed to initialize ChromaDB knowledge base: {e}")

# Mount static build folder containing compiled NextJS output
frontend_out = os.path.abspath(os.path.join(PROJECT_ROOT, "frontend", "out"))
if os.path.exists(frontend_out):
    app.mount("/", StaticFiles(directory=frontend_out, html=True), name="static")
else:
    print(f"Directory {frontend_out} not found. Serve backend without static files.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("voice_agent.main:app", host="0.0.0.0", port=7860, reload=True)
