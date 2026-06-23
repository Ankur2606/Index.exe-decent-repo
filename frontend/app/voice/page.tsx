"use client";

import React, { useState, useRef, useEffect } from "react";
import { useVoiceSession } from "../../hooks/useVoiceSession";
import { Mic, MicOff, Send, ChevronDown, Download, RotateCcw, Shield, Users, Construction, AlertTriangle, Layers, Radio } from "lucide-react";

type PageView = "gather" | "inferencing" | "results";

export default function VoiceDispatchPage() {
  const {
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
    handleSubmitPrediction,
  } = useVoiceSession();

  const [currentPage, setCurrentPage] = useState<PageView>("gather");
  const [inputText, setInputText] = useState("");
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcripts]);

  // Auto-transition to inferencing when processing starts
  useEffect(() => {
    if (sessionState === "PROCESSING") {
      setCurrentPage("inferencing");
    }
  }, [sessionState]);

  // Auto-transition to results when prediction arrives
  useEffect(() => {
    if (prediction && sessionState === "COMPLETE") {
      setCurrentPage("results");
    }
  }, [prediction, sessionState]);

  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const remainder = secs % 60;
    return `${mins}:${remainder < 10 ? "0" : ""}${remainder}`;
  };

  const confirmedCount = Object.values(fields).filter((f) => f.state === "confirmed").length;

  const handleSendText = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim()) {
      sendUserVoiceMessage(inputText.trim());
      setInputText("");
    }
  };

  const handleNewIncident = () => {
    endSession();
    setCurrentPage("gather");
  };

  const handleExportReport = () => {
    if (!prediction) return;
    const report = {
      timestamp: new Date().toISOString(),
      fields: {
        location: fields.location.value,
        event_type: fields.event_type.value,
        event_cause: fields.event_cause.value,
        priority: fields.priority.value,
        vehicle_type: fields.vehicle_type.value,
      },
      resolved: resolvedFields,
      predictions: prediction,
      recommendations,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `astram_dispatch_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Severity color mapping
  const getSeverityColor = (band: string) => {
    switch (band) {
      case "LOW": return "#22C55E";
      case "MODERATE": return "#EAB308";
      case "HIGH": return "#F97316";
      case "CRITICAL": return "#EF4444";
      default: return "#4F6EF7";
    }
  };

  const getSeverityBg = (band: string) => {
    switch (band) {
      case "LOW": return "rgba(34,197,94,0.12)";
      case "MODERATE": return "rgba(234,179,8,0.12)";
      case "HIGH": return "rgba(249,115,22,0.12)";
      case "CRITICAL": return "rgba(239,68,68,0.12)";
      default: return "rgba(79,110,247,0.12)";
    }
  };

  /* =================== PAGE 1: DATA GATHERING =================== */
  const renderDataGatheringPage = () => (
    <div className="flex flex-col h-full page-enter">
      {/* Top Bar */}
      <header className="shrink-0 px-4 pt-3 pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#4F6EF7] to-[#7C3AED] flex items-center justify-center">
              <Shield className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-wide">ASTraM</h1>
              <p className="text-[10px] text-[#6B7A99] font-mono">Voice Dispatch System</p>
            </div>
          </div>

          {/* Language Pills */}
          <div className="flex items-center gap-1">
            {["EN", "HI", "KA"].map((lang) => (
              <button
                key={lang}
                onClick={() => changeLanguage(lang)}
                className={`text-[11px] px-2.5 py-1 rounded-full font-semibold transition-all touch-target flex items-center justify-center ${
                  language === lang
                    ? "bg-[#7C3AED] text-white shadow-[0_0_12px_rgba(124,58,237,0.3)]"
                    : "bg-[#161B2E] text-[#6B7A99]"
                }`}
              >
                {lang === "HI" ? "हिं" : lang === "KA" ? "ಕ" : "EN"}
              </button>
            ))}
          </div>
        </div>

        {/* Session Status + Timer */}
        {sessionState !== "READY" && (
          <div className="flex items-center justify-between mt-2 animate-slide-down">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${
                sessionState === "LISTENING" ? "bg-[#22C55E] animate-pulse" : "bg-[#6B7A99]"
              }`} />
              <span className="text-[11px] font-mono text-[#6B7A99]">{sessionState}</span>
            </div>
            <span className={`text-[11px] font-mono px-2 py-0.5 rounded-full ${
              sessionTime < 120
                ? "bg-[#EF4444]/15 text-[#EF4444] animate-pulse"
                : "text-[#6B7A99]"
            }`}>
              {formatTime(sessionTime)}
            </span>
          </div>
        )}
      </header>

      {/* Field Status Chips */}
      {sessionState !== "READY" && (
        <div className="shrink-0 px-4 pb-2 animate-slide-down">
          <div className="flex gap-1.5 overflow-x-auto pb-1">
            {Object.entries(fields).map(([key, field]) => (
              <div
                key={key}
                className={`shrink-0 px-2.5 py-1 rounded-full text-[10px] font-mono border transition-all ${
                  field.state === "confirmed"
                    ? "bg-[#22C55E]/15 border-[#22C55E]/30 text-[#22C55E]"
                    : field.state === "collecting"
                    ? "bg-[#7C3AED]/15 border-[#7C3AED]/30 text-[#7C3AED] animate-pulse"
                    : "bg-[#161B2E] border-[#1E2436] text-[#6B7A99]"
                }`}
              >
                {field.state === "confirmed" ? "✓ " : ""}{key.replace("_", " ")}
              </div>
            ))}
          </div>
          {/* Progress bar */}
          <div className="mt-1.5 h-1 bg-[#1E2436] rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-[#4F6EF7] to-[#7C3AED] rounded-full transition-all duration-500"
              style={{ width: `${(confirmedCount / 5) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Chat / Transcript Area */}
      <div className="flex-1 overflow-y-auto px-4 py-2 thin-scrollbar">
        {sessionState === "READY" ? (
          /* Start Screen */
          <div className="h-full flex flex-col items-center justify-center text-center px-4 gap-6">
            {/* Animated Logo Ring */}
            <div className="relative w-32 h-32">
              <div className="absolute inset-0 rounded-full border-2 border-[#4F6EF7]/20 animate-pulse-ring-outer" />
              <div className="absolute inset-3 rounded-full border-2 border-[#7C3AED]/30 animate-pulse-ring" />
              <div className="absolute inset-6 rounded-full bg-gradient-to-br from-[#4F6EF7] to-[#7C3AED] flex items-center justify-center shadow-[0_0_40px_rgba(124,58,237,0.25)]">
                <Radio className="w-10 h-10 text-white" />
              </div>
            </div>

            <div>
              <h2 className="text-xl font-bold mb-1">Incident Dispatch</h2>
              <p className="text-[13px] text-[#6B7A99] leading-relaxed max-w-[260px]">
                Tap the mic or type to report a traffic incident. The AI agent will guide you through data collection.
              </p>
            </div>

            <button
              onClick={startSession}
              className="bg-gradient-to-r from-[#4F6EF7] to-[#7C3AED] text-white font-semibold text-sm px-8 py-3 rounded-full shadow-[0_4px_20px_rgba(79,110,247,0.3)] active:scale-95 transition-transform touch-target"
            >
              START SESSION
            </button>

            {/* Preset test button */}
            <button
              onClick={injectPresetTestCase}
              className="flex items-center gap-1.5 text-[11px] text-[#EF4444] font-mono border border-[#EF4444]/20 bg-[#EF4444]/5 px-3 py-1.5 rounded-full active:scale-95 transition-transform"
            >
              <Layers className="w-3 h-3" />
              DEMO PRESET
            </button>
          </div>
        ) : (
          /* Chat Bubbles */
          <div className="flex flex-col gap-2.5 py-1">
            {transcripts.map((line, i) => (
              <div
                key={i}
                className={`flex flex-col ${line.speaker === "user" ? "items-end" : "items-start"} animate-slide-up`}
                style={{ animationDelay: `${Math.min(i * 0.03, 0.15)}s` }}
              >
                <div className={`chat-bubble ${line.speaker === "user" ? "chat-bubble-user" : "chat-bubble-agent"}`}>
                  {line.text}
                </div>
                <span className="text-[9px] text-[#6B7A99] font-mono mt-0.5 px-1">{line.ts}</span>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
        )}
      </div>

      {/* Bottom Input Area */}
      {sessionState !== "READY" && (
        <div className="shrink-0 border-t border-[#1E2436] bg-[#0A0D14] px-4 pt-3 safe-bottom">
          {/* Text input row */}
          <form onSubmit={handleSendText} className="flex gap-2 mb-3">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Type incident details..."
              className="flex-1 bg-[#161B2E] border border-[#1E2436] rounded-full px-4 py-2.5 text-sm text-[#F0F4FF] focus:outline-none focus:border-[#4F6EF7] placeholder-[#6B7A99]"
            />
            <button
              type="submit"
              className="w-10 h-10 rounded-full bg-[#4F6EF7] flex items-center justify-center shrink-0 active:scale-90 transition-transform touch-target"
            >
              <Send className="w-4 h-4 text-white" />
            </button>
          </form>

          {/* Mic Button Row */}
          <div className="flex items-center justify-between pb-2">
            <button
              onClick={handleSubmitPrediction}
              disabled={confirmedCount < 3}
              className={`text-[11px] font-mono px-3 py-2 rounded-full border transition-all touch-target ${
                confirmedCount >= 3
                  ? "bg-[#7C3AED]/15 border-[#7C3AED]/30 text-[#7C3AED] active:scale-95"
                  : "bg-[#161B2E] border-[#1E2436] text-[#6B7A99]/40 cursor-not-allowed"
              }`}
            >
              SUBMIT ({confirmedCount}/5)
            </button>

            {/* Central Mic Button */}
            <button
              onMouseDown={() => setMicActive(true)}
              onMouseUp={() => setMicActive(false)}
              onTouchStart={(e) => { e.preventDefault(); setMicActive(true); }}
              onTouchEnd={(e) => { e.preventDefault(); setMicActive(false); }}
              className={`relative w-16 h-16 rounded-full flex items-center justify-center transition-all active:scale-90 ${
                micActive
                  ? "bg-[#7C3AED] shadow-[0_0_30px_rgba(124,58,237,0.5)]"
                  : "bg-gradient-to-br from-[#4F6EF7] to-[#7C3AED] shadow-[0_4px_20px_rgba(79,110,247,0.25)]"
              }`}
            >
              {/* Pulse rings when active */}
              {micActive && (
                <>
                  <span className="absolute inset-0 rounded-full border-2 border-[#7C3AED]/40 animate-pulse-ring" />
                  <span className="absolute -inset-2 rounded-full border border-[#7C3AED]/20 animate-pulse-ring-outer" />
                </>
              )}
              {micActive ? (
                <MicOff className="w-6 h-6 text-white relative z-10" />
              ) : (
                <Mic className="w-6 h-6 text-white relative z-10" />
              )}
            </button>

            <button
              onClick={endSession}
              className="text-[11px] font-mono px-3 py-2 rounded-full border border-[#EF4444]/20 text-[#EF4444] bg-[#EF4444]/5 touch-target active:scale-95 transition-transform"
            >
              END
            </button>
          </div>
        </div>
      )}
    </div>
  );

  /* =================== PAGE 2: INFERENCING =================== */
  const renderInferencingPage = () => (
    <div className="flex flex-col h-full items-center justify-center px-6 text-center page-enter">
      {/* Radar Animation */}
      <div className="relative w-44 h-44 mb-8">
        {/* Outer ring */}
        <div className="absolute inset-0 rounded-full border border-[#4F6EF7]/15" />
        {/* Middle ring */}
        <div className="absolute inset-6 rounded-full border border-[#4F6EF7]/25" />
        {/* Inner ring */}
        <div className="absolute inset-12 rounded-full border border-[#7C3AED]/35" />
        {/* Center dot */}
        <div className="absolute inset-[68px] rounded-full bg-[#7C3AED] shadow-[0_0_20px_rgba(124,58,237,0.5)] animate-breathe" />
        {/* Sweep line */}
        <div className="absolute inset-0 animate-radar" style={{ transformOrigin: "center" }}>
          <div className="absolute top-1/2 left-1/2 w-1/2 h-[2px] bg-gradient-to-r from-[#7C3AED] to-transparent origin-left" />
        </div>
        {/* Pulsing rings */}
        <div className="absolute inset-0 rounded-full border-2 border-[#4F6EF7]/20 animate-pulse-ring" />
        <div className="absolute -inset-4 rounded-full border border-[#4F6EF7]/10 animate-pulse-ring-outer" />
      </div>

      <h2 className="text-lg font-bold mb-1 animate-fade-in">Analyzing Incident</h2>
      <p className="text-[13px] text-[#6B7A99] mb-6 animate-fade-in">Running ML prediction pipeline...</p>

      {/* Loading dots */}
      <div className="flex gap-2 mb-8">
        <span className="w-2.5 h-2.5 rounded-full bg-[#4F6EF7] loading-dot" />
        <span className="w-2.5 h-2.5 rounded-full bg-[#7C3AED] loading-dot" />
        <span className="w-2.5 h-2.5 rounded-full bg-[#4F6EF7] loading-dot" />
      </div>

      {/* Collected fields summary */}
      <div className="w-full max-w-xs glass-card p-4 animate-slide-up">
        <p className="text-[10px] uppercase tracking-wider text-[#6B7A99] font-mono mb-3">Incident Details Collected</p>
        <div className="space-y-2">
          {Object.entries(fields).map(([key, field]) => (
            <div key={key} className="flex items-center justify-between text-xs">
              <span className="text-[#6B7A99] font-mono capitalize">{key.replace("_", " ")}</span>
              <span className={`font-medium ${field.value ? "text-[#22C55E]" : "text-[#6B7A99]/40"}`}>
                {field.value || "—"}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  /* =================== PAGE 3: RESULTS =================== */
  const renderResultsPage = () => {
    if (!prediction) return null;

    const score = prediction.event_impact_score;
    const severity = prediction.severity_band;
    const color = getSeverityColor(severity);
    const bgColor = getSeverityBg(severity);

    // SVG arc calculation for the score gauge
    const circumference = 2 * Math.PI * 40;
    const arcLength = (score / 100) * circumference * 0.75; // 270 degree arc
    const dashOffset = circumference * 0.75 - arcLength;

    return (
      <div className="flex flex-col h-full overflow-y-auto thin-scrollbar page-enter">
        {/* Header */}
        <header className="shrink-0 px-4 pt-3 pb-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#4F6EF7] to-[#7C3AED] flex items-center justify-center">
              <Shield className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold">Dispatch Report</h1>
              <p className="text-[10px] text-[#6B7A99] font-mono">ASTraM Intelligence</p>
            </div>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-[#6B7A99]">
            <span className="px-2 py-0.5 rounded-full bg-[#22C55E]/15 text-[#22C55E] border border-[#22C55E]/20">
              ✓ {prediction.ensemble_confidence}
            </span>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 px-4 pb-4 space-y-4 stagger-children">

          {/* Impact Score Gauge */}
          <div className="glass-card p-5 flex flex-col items-center">
            <div className="relative w-36 h-36">
              <svg viewBox="0 0 100 100" className="w-full h-full -rotate-[135deg]">
                {/* Background arc */}
                <circle
                  cx="50" cy="50" r="40"
                  fill="none"
                  stroke="#1E2436"
                  strokeWidth="6"
                  strokeDasharray={`${circumference * 0.75} ${circumference * 0.25}`}
                  strokeLinecap="round"
                />
                {/* Score arc */}
                <circle
                  cx="50" cy="50" r="40"
                  fill="none"
                  stroke={color}
                  strokeWidth="6"
                  strokeDasharray={`${arcLength} ${circumference - arcLength}`}
                  strokeLinecap="round"
                  className="score-arc"
                  style={{ filter: `drop-shadow(0 0 6px ${color}40)` }}
                />
              </svg>
              {/* Center score text */}
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-bold" style={{ color }}>{Math.round(score)}</span>
                <span className="text-[10px] text-[#6B7A99] font-mono">IMPACT</span>
              </div>
            </div>

            {/* Severity Badge */}
            <div
              className="mt-2 px-4 py-1.5 rounded-full text-xs font-bold tracking-wider"
              style={{ backgroundColor: bgColor, color, border: `1px solid ${color}30` }}
            >
              {severity} SEVERITY
            </div>
          </div>

          {/* Resource Cards Grid */}
          <div className="grid grid-cols-3 gap-2.5">
            {/* Officers */}
            <div className="glass-card p-3 flex flex-col items-center text-center">
              <div className="w-9 h-9 rounded-full bg-[#4F6EF7]/15 flex items-center justify-center mb-2">
                <Users className="w-4 h-4 text-[#4F6EF7]" />
              </div>
              <span className="text-xl font-bold text-[#F0F4FF]">{prediction.recommended_officers}</span>
              <span className="text-[9px] text-[#6B7A99] font-mono mt-0.5">OFFICERS</span>
            </div>

            {/* Barricades */}
            <div className="glass-card p-3 flex flex-col items-center text-center">
              <div className="w-9 h-9 rounded-full bg-[#F97316]/15 flex items-center justify-center mb-2">
                <Construction className="w-4 h-4 text-[#F97316]" />
              </div>
              <span className="text-xl font-bold text-[#F0F4FF]">{prediction.recommended_barricades}</span>
              <span className="text-[9px] text-[#6B7A99] font-mono mt-0.5">BARRICADES</span>
            </div>

            {/* Diversion */}
            <div className="glass-card p-3 flex flex-col items-center text-center">
              <div className={`w-9 h-9 rounded-full flex items-center justify-center mb-2 ${
                prediction.diversion_required === "YES" || prediction.diversion_required === true
                  ? "bg-[#EF4444]/15"
                  : "bg-[#22C55E]/15"
              }`}>
                <AlertTriangle className={`w-4 h-4 ${
                  prediction.diversion_required === "YES" || prediction.diversion_required === true
                    ? "text-[#EF4444]"
                    : "text-[#22C55E]"
                }`} />
              </div>
              <span className={`text-sm font-bold ${
                prediction.diversion_required === "YES" || prediction.diversion_required === true
                  ? "text-[#EF4444]"
                  : "text-[#22C55E]"
              }`}>
                {prediction.diversion_required === "YES" || prediction.diversion_required === true ? "YES" : "NO"}
              </span>
              <span className="text-[9px] text-[#6B7A99] font-mono mt-0.5">DIVERSION</span>
            </div>
          </div>

          {/* Incident Summary */}
          <div className="glass-card p-4">
            <p className="text-[10px] uppercase tracking-wider text-[#6B7A99] font-mono mb-3">Incident Summary</p>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2.5">
              <div>
                <span className="text-[10px] text-[#6B7A99] font-mono block">Location</span>
                <span className="text-[13px] text-[#F0F4FF]">{fields.location.value || "—"}</span>
              </div>
              <div>
                <span className="text-[10px] text-[#6B7A99] font-mono block">Event Type</span>
                <span className="text-[13px] text-[#F0F4FF] capitalize">{fields.event_type.value || "—"}</span>
              </div>
              <div>
                <span className="text-[10px] text-[#6B7A99] font-mono block">Cause</span>
                <span className="text-[13px] text-[#F0F4FF] capitalize">{(fields.event_cause.value || "—").replace("_", " ")}</span>
              </div>
              <div>
                <span className="text-[10px] text-[#6B7A99] font-mono block">Priority</span>
                <span className={`text-[13px] font-semibold ${
                  fields.priority.value === "High" ? "text-[#EF4444]" : "text-[#22C55E]"
                }`}>{fields.priority.value || "—"}</span>
              </div>
              <div>
                <span className="text-[10px] text-[#6B7A99] font-mono block">Vehicle</span>
                <span className="text-[13px] text-[#F0F4FF] capitalize">{fields.vehicle_type.value || "—"}</span>
              </div>
              <div>
                <span className="text-[10px] text-[#6B7A99] font-mono block">Zone</span>
                <span className="text-[13px] text-[#F0F4FF]">{resolvedFields.zone || "—"}</span>
              </div>
            </div>

            {/* Extra resolved metadata */}
            <div className="mt-3 pt-3 border-t border-[#1E2436] grid grid-cols-3 gap-2 text-[10px] font-mono">
              <div>
                <span className="text-[#6B7A99] block">Corridor</span>
                <span className="text-[#F0F4FF] truncate block">{resolvedFields.corridor || "—"}</span>
              </div>
              <div>
                <span className="text-[#6B7A99] block">Station</span>
                <span className="text-[#F0F4FF] truncate block">{resolvedFields.police_station || "—"}</span>
              </div>
              <div>
                <span className="text-[#6B7A99] block">Time</span>
                <span className="text-[#F0F4FF] block">{resolvedFields.time || "—"}</span>
              </div>
            </div>
          </div>

          {/* RAG Recommendations */}
          {recommendations.length > 0 && (
            <div className="glass-card p-4">
              <p className="text-[10px] uppercase tracking-wider text-[#6B7A99] font-mono mb-3">Traffic Intelligence</p>
              <div className="space-y-2.5">
                {recommendations.map((rec, i) => (
                  <div key={i} className="flex gap-2.5">
                    <div className="w-5 h-5 rounded-full bg-[#4F6EF7]/15 flex items-center justify-center shrink-0 mt-0.5">
                      <span className="text-[9px] font-bold text-[#4F6EF7]">{i + 1}</span>
                    </div>
                    <p className="text-[12px] text-[#F0F4FF]/80 leading-relaxed">{rec}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-2.5 safe-bottom pt-1 pb-2">
            <button
              onClick={handleExportReport}
              className="flex-1 flex items-center justify-center gap-2 bg-[#4F6EF7] text-white text-sm font-semibold py-3 rounded-2xl active:scale-95 transition-transform shadow-[0_4px_15px_rgba(79,110,247,0.25)]"
            >
              <Download className="w-4 h-4" />
              Export Report
            </button>
            <button
              onClick={handleNewIncident}
              className="flex-1 flex items-center justify-center gap-2 bg-[#161B2E] border border-[#1E2436] text-[#F0F4FF] text-sm font-semibold py-3 rounded-2xl active:scale-95 transition-transform"
            >
              <RotateCcw className="w-4 h-4" />
              New Incident
            </button>
          </div>
        </div>
      </div>
    );
  };

  /* =================== MAIN RENDER =================== */
  return (
    <div className="h-[100dvh] w-full max-w-md mx-auto bg-[#0A0D14] text-[#F0F4FF] flex flex-col overflow-hidden font-sans">
      {currentPage === "gather" && renderDataGatheringPage()}
      {currentPage === "inferencing" && renderInferencingPage()}
      {currentPage === "results" && renderResultsPage()}
    </div>
  );
}
