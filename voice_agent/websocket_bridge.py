import os
import json
import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from voice_agent.agents import (
    get_session_state,
    set_websocket,
    geocode_address,
    update_field,
    submit_prediction,
    emit_ws_event
)

router = APIRouter()

# Map tools for manual invocation during chat sessions
TOOL_MAP = {
    "geocode_address": geocode_address,
    "update_field": update_field,
    "submit_prediction": submit_prediction
}

# Instruction for theRequirementsAgent Chat
SYSTEM_PROMPT = """You are ASTraM RequirementsAgent, a voice dispatcher for Bengaluru Traffic Police.
Your task is to converse with the operator to collect 5 fields:
1. LOCATION - Resolve the address description by calling geocode_address tool.
2. EVENT TYPE - Determine if it is planned or unplanned.
3. EVENT CAUSE - Identify the cause (construction, water_logging, accident, vehicle_breakdown, public_rally, vip_movement, tree_fallen, fire_incident).
4. PRIORITY - Set to High or Low.
5. VEHICLE TYPE - Type of vehicle involved.

Instructions:
* Speak professionally and concisely.
* If the operator provides an address, call geocode_address first.
* Once a field value is identified, call update_field with the field_name and value.
* Guide the operator to collect missing fields.
* Once you have collected all fields, confirm them with the operator and call submit_prediction."""

@router.websocket("/ws/voice-session")
async def websocket_endpoint(websocket: WebSocket, lang: str = "EN"):
    await websocket.accept()
    
    # Associate current websocket connection with tools
    set_websocket(websocket)
    
    # Reset shared session state for a new session
    state = get_session_state()
    state.reset()
    
    # Initialize the GenAI Client and Chat Session
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai_client = genai.Client(api_key=api_key)
    else:
        genai_client = genai.Client()
        
    chat = genai_client.chats.create(
        model="gemini-3.1-flash-live-preview",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[geocode_address, update_field, submit_prediction],
            temperature=0.2
        )
    )
    
    # Send a greeting transcript line from system
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    greeting = "ASTraM voice link established. State the traffic incident description."
    if lang == "HI":
        greeting = "यातायात घटना का विवरण प्रदान करें।"
    elif lang == "KA":
        greeting = "ಸಂಚಾರ ಘಟನೆಯ ವಿವರವನ್ನು ನೀಡಿ."
        
    state.add_transcript("agent", greeting, ts)
    await websocket.send_text(json.dumps({
        "type": "transcript",
        "speaker": "agent",
        "text": greeting,
        "ts": ts
    }))

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle voice/text input from user
            if message.get("type") == "voice_input":
                user_text = message.get("text", "").strip()
                if not user_text:
                    continue
                    
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                state.add_transcript("user", user_text, ts)
                await websocket.send_text(json.dumps({
                    "type": "transcript",
                    "speaker": "user",
                    "text": user_text,
                    "ts": ts
                }))
                
                # Send to Gemini chat session
                response = chat.send_message(user_text)
                
                # Tool calling resolution loop (handles synchronous function calls)
                while response.function_calls:
                    tool_responses = []
                    for call in response.function_calls:
                        name = call.name
                        args = call.args
                        
                        if name in TOOL_MAP:
                            tool_result = TOOL_MAP[name](**args)
                            tool_responses.append(
                                types.Part.from_function_response(
                                    name=name,
                                    response={"result": tool_result}
                                )
                            )
                            
                    response = chat.send_message(tool_responses)
                
                # Send agent response text back to client
                if response.text:
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    state.add_transcript("agent", response.text, ts)
                    await websocket.send_text(json.dumps({
                        "type": "transcript",
                        "speaker": "agent",
                        "text": response.text,
                        "ts": ts
                    }))
                    
            elif message.get("type") == "trigger_prediction":
                # Manual trigger of ML model prediction
                submit_prediction()
                
            elif message.get("type") == "inject_preset":
                # Inject a preset test case with 100 percent diversion required matching coordinates
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                
                # 1. Update shared state
                state.set_field("location", "HAL Old Airport Road near Marathahalli")
                state.set_field("event_type", "planned")
                state.set_field("event_cause", "construction")
                state.set_field("priority", "High")
                state.set_field("vehicle_type", "unknown")
                
                state.set_resolved("latitude", 12.9685753)
                state.set_resolved("longitude", 77.7011831)
                state.set_resolved("corridor", "ORR East 2")
                state.set_resolved("police_station", "HAL Old Airport")
                state.set_resolved("zone", "East Zone 1")
                
                # 2. Emit updates to client
                await emit_ws_event({"type": "field_update", "field": "location", "value": "HAL Old Airport Road near Marathahalli"})
                await emit_ws_event({"type": "field_update", "field": "event_type", "value": "planned"})
                await emit_ws_event({"type": "field_update", "field": "event_cause", "value": "construction"})
                await emit_ws_event({"type": "field_update", "field": "priority", "value": "High"})
                await emit_ws_event({"type": "field_update", "field": "vehicle_type", "value": "unknown"})
                
                await emit_ws_event({"type": "field_resolved", "field": "corridor", "value": "ORR East 2"})
                await emit_ws_event({"type": "field_resolved", "field": "police_station", "value": "HAL Old Airport"})
                await emit_ws_event({"type": "field_resolved", "field": "zone", "value": "East Zone 1"})
                
                # 3. Simulate dialogue transcript for feedback
                await websocket.send_text(json.dumps({
                    "type": "transcript",
                    "speaker": "user",
                    "text": "Injecting test preset: Planned Construction at HAL Old Airport Road (ORR East 2)",
                    "ts": ts
                }))
                
                agent_msg = "Preset values injected. Running predictions."
                await websocket.send_text(json.dumps({
                    "type": "transcript",
                    "speaker": "agent",
                    "text": agent_msg,
                    "ts": ts
                }))
                
                # 4. Trigger prediction calculation
                submit_prediction()
                
            elif message.get("type") == "replay_audio":
                # Simulated audio replay notification
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                await websocket.send_text(json.dumps({
                    "type": "transcript",
                    "speaker": "agent",
                    "text": "[REPLAYING LAST AUDIO SEGMENT]",
                    "ts": ts
                }))
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    finally:
        set_websocket(None)
