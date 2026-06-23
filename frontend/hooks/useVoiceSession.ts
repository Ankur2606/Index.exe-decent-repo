import { useEffect, useRef, useState, useCallback } from "react";
import { TranscriptLine, PredictionData, ResolvedFields, WsMessage, SessionState } from "../lib/types";

export function useVoiceSession() {
  const [sessionState, setSessionState] = useState<SessionState>("READY");
  const [transcripts, setTranscripts] = useState<TranscriptLine[]>([]);
  const [amplitude, setAmplitude] = useState<number>(0);
  const [sessionTime, setSessionTime] = useState<number>(900); // 15 minutes
  const [micActive, setMicActive] = useState<boolean>(false);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);

  // Field states tracking: name, value, status (empty, collecting, confirmed)
  const [fields, setFields] = useState({
    location: { value: "", state: "empty" },
    event_type: { value: "", state: "empty" },
    event_cause: { value: "", state: "empty" },
    priority: { value: "", state: "empty" },
    vehicle_type: { value: "", state: "empty" },
  });

  const [resolvedFields, setResolvedFields] = useState<ResolvedFields>({
    corridor: "",
    police_station: "",
    zone: "",
    date: "",
    time: "",
  });

  const socketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const ampIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const recognitionRef = useRef<any>(null);

  // Active language code
  const [language, setLanguage] = useState<string>("EN");

  // Timer countdown
  useEffect(() => {
    if (sessionState === "READY") {
      setSessionTime(900);
      return;
    }
    if (sessionState === "COMPLETE") {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }

    timerRef.current = setInterval(() => {
      setSessionTime((prev) => {
        if (prev <= 1) {
          clearInterval(timerRef.current!);
          // Auto close and submit prediction
          handleSubmitPrediction();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [sessionState]);

  // Simulate audio amplitude when mic is active or agent is speaking
  useEffect(() => {
    if (micActive || sessionState === "PROCESSING") {
      ampIntervalRef.current = setInterval(() => {
        setAmplitude(Math.random() * 80 + 20); // 20 to 100
      }, 100);
    } else {
      if (ampIntervalRef.current) clearInterval(ampIntervalRef.current);
      setAmplitude(0);
    }

    return () => {
      if (ampIntervalRef.current) clearInterval(ampIntervalRef.current);
    };
  }, [micActive, sessionState]);

  // Configure Web Speech API Recognition
  useEffect(() => {
    if (typeof window !== "undefined") {
      const SpeechRecognitionClass = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognitionClass) {
        const rec = new SpeechRecognitionClass();
        rec.continuous = false;
        rec.interimResults = false;
        
        // Map language pills to browser recognition locales
        rec.lang = language === "HI" ? "hi-IN" : language === "KA" ? "kn-IN" : "en-IN";

        rec.onstart = () => {
          console.log("Web Speech API active");
        };

        rec.onresult = (event: any) => {
          const text = event.results[0][0].transcript;
          if (text) {
            sendUserVoiceMessage(text);
          }
        };

        rec.onerror = (err: any) => {
          console.warn("Speech transcription warning:", err.error, err);
          setMicActive(false);
          
          let errMsg = "Speech recognition error occurred.";
          if (err.error === "not-allowed") {
            errMsg = "Microphone access blocked. Please use localhost:3000 / localhost:7860 or run over HTTPS.";
          } else if (err.error === "no-speech") {
            errMsg = "No speech detected. Please check your microphone input levels.";
          }
          
          setTranscripts((prev) => [
            ...prev,
            {
              speaker: "agent",
              text: `[SYSTEM: ${errMsg}]`,
              ts: new Date().toLocaleTimeString(),
            },
          ]);
        };

        rec.onend = () => {
          setMicActive(false);
        };

        recognitionRef.current = rec;
      }
    }
  }, [language]);

  // Activate Web Speech API on mic trigger
  useEffect(() => {
    if (micActive) {
      try {
        if (recognitionRef.current) {
          recognitionRef.current.start();
        }
      } catch (err) {
        console.error("Failed to start speech recognition", err);
      }
    } else {
      try {
        if (recognitionRef.current) {
          recognitionRef.current.stop();
        }
      } catch (err) {
        // Safe to ignore
      }
    }
  }, [micActive]);

  // Handle WebSocket connection
  const connectWs = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
    }

    let host = window.location.host || "localhost:7860";
    if (window.location.port === "3000") {
      host = `${window.location.hostname}:7860`;
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${host}/ws/voice-session?lang=${language}`;

    const ws = new WebSocket(wsUrl);
    ws.binaryType = "blob";
    socketRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connection established");
    };

    ws.onmessage = (event) => {
      // Handle binary audio frame playback from Groq TTS
      if (event.data instanceof Blob) {
        console.log("Received binary audio frame from backend");
        const url = URL.createObjectURL(event.data);
        const audio = new Audio(url);
        audio.play().catch((err) => {
          console.error("Audio playback error:", err);
        });
        return;
      }

      try {
        const msg = JSON.parse(event.data) as WsMessage;
        switch (msg.type) {
          case "field_update":
            setFields((prev) => {
              const updated = { ...prev };
              const key = msg.field as keyof typeof prev;
              if (updated[key]) {
                updated[key] = {
                  value: msg.value,
                  state: msg.value ? "confirmed" : "collecting",
                };
              }
              return updated;
            });
            break;
          case "field_resolved":
            setResolvedFields((prev) => ({
              ...prev,
              [msg.field]: msg.value,
            }));
            break;
          case "transcript":
            setTranscripts((prev) => {
              const nextTranscripts = [...prev, { speaker: msg.speaker, text: msg.text, ts: msg.ts }];
              // Keep only last 15 transcripts to maintain performance
              if (nextTranscripts.length > 15) {
                return nextTranscripts.slice(nextTranscripts.length - 15);
              }
              return nextTranscripts;
            });
            break;
          case "prediction_start":
            setSessionState("PROCESSING");
            break;
          case "prediction_result":
            setPrediction(msg.data);
            setSessionState("COMPLETE");
            break;
          case "rag_recommendations":
            setRecommendations(msg.items);
            break;
          case "narration_start":
            console.log("Agent started narration");
            break;
          case "session_complete":
            setSessionState("COMPLETE");
            break;
          default:
            break;
        }
      } catch (err) {
        console.error("Error parsing websocket message", err);
      }
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed");
    };

    ws.onerror = (err) => {
      console.error("WebSocket error", err);
    };
  }, [language]);

  const startSession = () => {
    setSessionState("LISTENING");
    setTranscripts([]);
    setPrediction(null);
    setRecommendations([]);
    setFields({
      location: { value: "", state: "empty" },
      event_type: { value: "", state: "empty" },
      event_cause: { value: "", state: "empty" },
      priority: { value: "", state: "empty" },
      vehicle_type: { value: "", state: "empty" },
    });
    setResolvedFields({
      corridor: "",
      police_station: "",
      zone: "",
      date: "",
      time: "",
    });
    connectWs();
  };

  const endSession = () => {
    if (socketRef.current) {
      socketRef.current.close();
    }
    setSessionState("READY");
    setMicActive(false);
  };

  const handleSubmitPrediction = () => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: "trigger_prediction" }));
    }
  };

  const changeLanguage = (lang: string) => {
    setLanguage(lang);
    if (sessionState === "LISTENING" || sessionState === "PROCESSING") {
      // Reconnect with new language context
      setTimeout(() => {
        connectWs();
      }, 100);
    }
  };

  // Push user text speech event directly (e.g. simulated input)
  const sendUserVoiceMessage = (text: string) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(
        JSON.stringify({
          type: "voice_input",
          text: text,
        })
      );
    }
  };

  // Inject preset test case that returns diversion_required: true
  const injectPresetTestCase = () => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(
        JSON.stringify({
          type: "inject_preset",
          preset: "ORR Construction Closure",
        })
      );
    } else {
      // If session is not active, auto-start first
      startSession();
      setTimeout(() => {
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
          socketRef.current.send(
            JSON.stringify({
              type: "inject_preset",
              preset: "ORR Construction Closure",
            })
          );
        }
      }, 1000);
    }
  };

  // Replay agent last audio output
  const replayLastAudio = () => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: "replay_audio" }));
    }
  };

  return {
    sessionState,
    transcripts,
    amplitude,
    sessionTime,
    micActive,
    setMicActive,
    fields,
    resolvedFields,
    prediction,
    recommendations,
    language,
    changeLanguage,
    startSession,
    endSession,
    sendUserVoiceMessage,
    injectPresetTestCase,
    replayLastAudio,
    handleSubmitPrediction,
  };
}
