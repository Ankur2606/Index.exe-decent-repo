import os
import json
import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from groq import Groq
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

# Define schemas for Groq (OpenAI-compatible)
GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "geocode_address",
            "description": "Geocode an address description to coordinate points and administrative boundaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "The text description of the location (e.g. HAL Old Airport Road)."
                    }
                },
                "required": ["address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_field",
            "description": "Update a specific incident field in the shared state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "enum": ["location", "event_type", "event_cause", "priority", "vehicle_type"],
                        "description": "Name of the field to update."
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to assign to the field."
                    }
                },
                "required": ["field_name", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_prediction",
            "description": "Submit the collected incident details to run the ML congestion prediction. Call this once all required fields are collected.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

# Prompt instructing LLM on requirement gathering with options
SYSTEM_PROMPT = """You are ASTraM RequirementsAgent, an expert traffic incident voice dispatcher helper for Bengaluru Traffic Police.
Your primary objective is to gather exactly 5 core variables from the operator's voice reports through a back-and-forth conversation before triggering the dispatch calculation.

The 5 target variables and their exact allowed options are:
1. LOCATION: The address description. Call `geocode_address` immediately once the operator mentions a location.
2. EVENT_TYPE: Must be exactly one of: 'planned' or 'unplanned'.
3. EVENT_CAUSE: Must be exactly one of: 'construction', 'water_logging', 'accident', 'vehicle_breakdown', 'public_rally', 'vip_movement', 'tree_fallen', 'fire_incident'.
4. PRIORITY: Must be exactly one of: 'High' or 'Low'.
5. VEHICLE_TYPE: Type of vehicle involved (e.g. 'two_wheeler', 'bus', 'car', 'auto', 'truck', 'unknown').

Rules for Requirement Gathering:
* Speak briefly, professionally, and extremely concisely. Your text output will be read aloud via Text-to-Speech (TTS), so keep your response under 25-35 words to save time and API tokens.
* Check which of the 5 variables are missing.
* If variables are missing, ask for them one by one. Always list the exact allowed options in parentheses to guide the operator (e.g. "What is the priority (High or Low)?" or "What is the cause (construction, water_logging, accident, vehicle_breakdown, public_rally, vip_movement, tree_fallen, fire_incident)?").
* Once you identify a field value, call `update_field` immediately to confirm it in the system.
* DO NOT call `submit_prediction` until you have gathered and updated all 5 variables. Once all 5 variables are resolved, confirm the details with the operator and call `submit_prediction` to finalize the dispatch.
* When you call `submit_prediction`, simply respond with 'Calculating dispatch metrics now.' and nothing else. The system will automatically output and narrate the official dispatch report."""

async def generate_and_send_tts(websocket: WebSocket, text: str, groq_client: Groq):
    try:
        print(f"Generating TTS for: '{text}'")
        response = groq_client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="troy",
            input=text,
            response_format="wav"
        )
        audio_bytes = response.read()
        await websocket.send_bytes(audio_bytes)
        print("TTS audio bytes sent over WS.")
    except Exception as e:
        print(f"Failed to generate TTS: {e}")

@router.websocket("/ws/voice-session")
async def websocket_endpoint(websocket: WebSocket, lang: str = "EN"):
    await websocket.accept()
    
    # Associate current websocket connection with tools
    set_websocket(websocket)
    
    # Reset shared session state for a new session
    state = get_session_state()
    state.reset()
    
    # Initialize the Groq Client
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "GROQ_API_KEY not configured. Please set it in your .env or HF Space Settings > Secrets."
        }))
        await websocket.close(code=1008, reason="Missing GROQ_API_KEY")
        return
        
    groq_client = Groq(api_key=groq_key)
    chat_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
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
    await generate_and_send_tts(websocket, greeting, groq_client)

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
                
                chat_history.append({"role": "user", "content": user_text})
                
                try:
                    response = groq_client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=chat_history,
                        tools=GROQ_TOOLS,
                        tool_choice="auto",
                        temperature=0.2
                    )
                    
                    response_message = response.choices[0].message
                    chat_history.append(response_message)
                    
                    # Tool calling resolution loop
                    while response_message.tool_calls:
                        for tool_call in response_message.tool_calls:
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                            
                            if function_name in TOOL_MAP:
                                tool_result = TOOL_MAP[function_name](**function_args)
                                chat_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": function_name,
                                    "content": json.dumps({"result": tool_result})
                                })
                        
                        # Get new completion from Groq with tool outputs
                        response = groq_client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=chat_history,
                            tools=GROQ_TOOLS,
                            tool_choice="auto",
                            temperature=0.2
                        )
                        response_message = response.choices[0].message
                        chat_history.append(response_message)
                    
                    # Send agent response text back to client
                    agent_text = response_message.content
                    if agent_text:
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        state.add_transcript("agent", agent_text, ts)
                        await websocket.send_text(json.dumps({
                            "type": "transcript",
                            "speaker": "agent",
                            "text": agent_text,
                            "ts": ts
                        }))
                        await generate_and_send_tts(websocket, agent_text, groq_client)
                        
                except Exception as e:
                    print(f"Error in Groq Chat: {e}")
                    err_msg = "Error communicating with Groq API."
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    await websocket.send_text(json.dumps({
                        "type": "transcript",
                        "speaker": "agent",
                        "text": err_msg,
                        "ts": ts
                    }))
                    
            elif message.get("type") == "trigger_prediction":
                # Manual trigger of ML model prediction
                submit_prediction()
                
            elif message.get("type") == "inject_preset":
                # Inject a preset test case
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
                # Narrate the last agent transcript using Groq TTS
                last_agent_text = ""
                for line in reversed(state.get_state().get("transcripts", [])):
                    if line.get("speaker") == "agent":
                        last_agent_text = line.get("text", "")
                        break
                
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                if last_agent_text:
                    await websocket.send_text(json.dumps({
                        "type": "transcript",
                        "speaker": "agent",
                        "text": f"[REPLAYING AUDIO: \"{last_agent_text}\"]",
                        "ts": ts
                    }))
                    await generate_and_send_tts(websocket, last_agent_text, groq_client)
                else:
                    await websocket.send_text(json.dumps({
                        "type": "transcript",
                        "speaker": "agent",
                        "text": "[NO AUDIO TO REPLAY]",
                        "ts": ts
                    }))
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    finally:
        set_websocket(None)
