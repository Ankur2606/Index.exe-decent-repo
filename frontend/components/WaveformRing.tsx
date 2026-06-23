"use client";

import React from "react";
import { Mic, Volume2 } from "lucide-react";
import { SessionState } from "../lib/types";

interface WaveformRingProps {
  sessionState: SessionState;
  micActive: boolean;
  amplitude: number;
}

export default function WaveformRing({ sessionState, micActive, amplitude }: WaveformRingProps) {
  // Determine dominant mode: agent speaking vs user speaking vs idle
  const isAgentSpeaking = sessionState === "PROCESSING";
  const isUserSpeaking = micActive;

  // Set colors based on state
  let strokeColor = "var(--border-active)";
  let glowColor = "rgba(45, 53, 85, 0.1)";
  let ringClass = "animate-pulse";

  if (isUserSpeaking) {
    strokeColor = "var(--accent-primary)";
    glowColor = "rgba(79, 110, 247, 0.3)";
    ringClass = "transition-all duration-75";
  } else if (isAgentSpeaking) {
    strokeColor = "var(--accent-voice)";
    glowColor = "rgba(124, 58, 237, 0.3)";
    ringClass = "transition-all duration-75";
  }

  // Calculate dynamic scale factor based on voice amplitude
  const scale = 1 + (amplitude / 100) * 0.25;

  return (
    <div className="flex flex-col items-center justify-center py-6">
      <div className="relative flex items-center justify-center w-52 h-52">
        {/* Glow Ring behind the SVG */}
        <div
          className="absolute inset-0 rounded-full blur-xl transition-all duration-300"
          style={{
            backgroundColor: glowColor,
            transform: `scale(${scale})`,
          }}
        />

        {/* Dynamic SVG Waveform Circle */}
        <svg
          viewBox="0 0 100 100"
          className={`absolute w-full h-full ${ringClass}`}
          style={{
            transform: `scale(${scale})`,
            transition: "transform 100ms ease-out",
          }}
        >
          <circle
            cx="50"
            cy="50"
            r="44"
            fill="none"
            stroke={strokeColor}
            strokeWidth="2"
            strokeDasharray={isUserSpeaking || isAgentSpeaking ? "4 2 1 2" : "0"}
            className="transition-all duration-300"
          />

          {isUserSpeaking && (
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke="var(--accent-primary)"
              strokeWidth="0.5"
              opacity="0.5"
              strokeDasharray="1 1"
            />
          )}

          {isAgentSpeaking && (
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke="var(--accent-voice)"
              strokeWidth="0.5"
              opacity="0.5"
              strokeDasharray="2 2"
            />
          )}
        </svg>

        {/* Center Button / Icon */}
        <div
          className="relative z-10 flex items-center justify-center w-28 h-28 rounded-full border shadow-lg transition-all duration-300"
          style={{
            backgroundColor: isUserSpeaking
              ? "var(--accent-primary)"
              : isAgentSpeaking
              ? "var(--accent-voice)"
              : "var(--bg-card)",
            borderColor: isUserSpeaking
              ? "var(--accent-primary)"
              : isAgentSpeaking
              ? "var(--accent-voice)"
              : "var(--border-active)",
            boxShadow:
              isUserSpeaking || isAgentSpeaking
                ? `0 0 25px ${isUserSpeaking ? "rgba(79, 110, 247, 0.6)" : "rgba(124, 58, 237, 0.6)"}`
                : "none",
          }}
        >
          {isAgentSpeaking ? (
            <Volume2 className="w-10 h-10 text-white animate-bounce" />
          ) : (
            <Mic
              className={`w-10 h-10 transition-colors duration-300 ${
                isUserSpeaking ? "text-white animate-pulse" : "text-[#F0F4FF]"
              }`}
            />
          )}
        </div>
      </div>

      <div className="mt-4 text-center">
        <span
          className="text-xs uppercase tracking-widest font-mono transition-all duration-300"
          style={{
            color: isUserSpeaking
              ? "var(--accent-primary)"
              : isAgentSpeaking
              ? "var(--accent-voice)"
              : "var(--text-muted)",
          }}
        >
          {isUserSpeaking ? "Microphone Active" : isAgentSpeaking ? "Agent Speaking" : "Idle"}
        </span>
      </div>
    </div>
  );
}
