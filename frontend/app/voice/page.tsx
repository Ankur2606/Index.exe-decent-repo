"use client";

import React, { useState } from "react";
import { useVoiceSession } from "../../hooks/useVoiceSession";
import WaveformRing from "../../components/WaveformRing";
import TranscriptScroll from "../../components/TranscriptScroll";
import FieldCard from "../../components/FieldCard";
import GaugeCard from "../../components/GaugeCard";
import MetricCard from "../../components/MetricCard";
import RecommendationList from "../../components/RecommendationList";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ChevronDown, ChevronUp, Play, Square, RefreshCw, Download, Layers } from "lucide-react";

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
    replayLastAudio,
    handleSubmitPrediction,
  } = useVoiceSession();

  const [autoResolvedExpanded, setAutoResolvedExpanded] = useState(false);
  const [inputText, setInputText] = useState("");

  // Format session countdown timer
  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const remainder = secs % 60;
    return `${mins}:${remainder < 10 ? "0" : ""}${remainder}`;
  };

  // Count confirmed fields
  const getConfirmedFieldsCount = () => {
    return Object.values(fields).filter((f) => f.state === "confirmed").length;
  };

  const confirmedCount = getConfirmedFieldsCount();
  const progressPercent = (confirmedCount / 5) * 100;

  // Handle manual typing of transcript for fallback simulation
  const handleSendText = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim()) {
      sendUserVoiceMessage(inputText.trim());
      setInputText("");
    }
  };

  // Export predictions as JSON dispatch report
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
      recommendations: recommendations,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `astram_dispatch_report_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-[#0A0D14] text-[#F0F4FF] flex flex-col font-sans">
        {/* ZONE A: Header Bar */}
        <header className="h-14 border-b border-[#1E2436] bg-[#111520] px-6 flex items-center justify-between z-10 shrink-0">
          <div className="flex items-center gap-3">
            <span className="font-bold text-lg tracking-wider text-[#F0F4FF] flex items-center gap-1.5">
              ASTraM <span className="text-xs px-2 py-0.5 rounded bg-[#7C3AED]/20 text-[#7C3AED] border border-[#7C3AED]/30">Voice Dispatch</span>
            </span>
          </div>

          {/* Session Status Pill */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-[#161B2E] border border-[#1E2436] px-3 py-1 rounded-full text-xs font-mono">
              <span className={`w-2.5 h-2.5 rounded-full ${
                sessionState === "LISTENING"
                  ? "bg-[#22C55E] animate-ping"
                  : sessionState === "PROCESSING"
                  ? "bg-[#7C3AED] animate-pulse"
                  : sessionState === "COMPLETE"
                  ? "bg-[#4F6EF7]"
                  : "bg-[#6B7A99]"
              }`} />
              <span className="text-[#F0F4FF]">{sessionState}</span>
            </div>

            {sessionState !== "READY" && (
              <div className={`text-xs font-mono px-3 py-1 rounded-full border ${
                sessionTime < 120
                  ? "bg-[#EF4444]/15 text-[#EF4444] border-[#EF4444]/30 animate-pulse"
                  : "bg-[#161B2E] text-[#6B7A99] border-[#1E2436]"
              }`}>
                Timer: {formatTime(sessionTime)}
              </div>
            )}

            {/* PRESET INJECTOR BUTTON (Required to demonstrate diversion prediction) */}
            <Button
              size="sm"
              onClick={injectPresetTestCase}
              className="bg-[#EF4444]/10 hover:bg-[#EF4444]/20 text-[#EF4444] border border-[#EF4444]/30 font-mono text-xs h-7 px-3 flex items-center gap-1.5"
            >
              <Layers className="w-3.5 h-3.5" />
              <span>TEST PRESET DISPATCH</span>
            </Button>
          </div>

          {/* Language Selector */}
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => changeLanguage("HI")}
              className={`text-xs px-2.5 py-1 rounded font-semibold transition-all ${
                language === "HI"
                  ? "bg-[#7C3AED] text-white"
                  : "bg-[#161B2E] text-[#6B7A99] hover:bg-[#1E2436]"
              }`}
            >
              हिं
            </button>
            <button
              onClick={() => changeLanguage("EN")}
              className={`text-xs px-2.5 py-1 rounded font-semibold transition-all ${
                language === "EN"
                  ? "bg-[#7C3AED] text-white"
                  : "bg-[#161B2E] text-[#6B7A99] hover:bg-[#1E2436]"
              }`}
            >
              EN
            </button>
            <button
              onClick={() => changeLanguage("KA")}
              className={`text-xs px-2.5 py-1 rounded font-semibold transition-all ${
                language === "KA"
                  ? "bg-[#7C3AED] text-white"
                  : "bg-[#161B2E] text-[#6B7A99] hover:bg-[#1E2436]"
              }`}
            >
              ಕ
            </button>
          </div>
        </header>

        {/* 12-Column Main Workspace */}
        <main className="flex-1 p-6 grid grid-cols-12 gap-6 min-h-0 overflow-y-auto">
          
          {/* ZONE B: Voice Interface Panel (left 5 cols) */}
          <section className="col-span-5 flex flex-col gap-4 bg-[#111520] border border-[#1E2436] p-4 rounded-md min-h-[500px]">
            <div className="text-xs uppercase tracking-wider text-[#6B7A99] font-mono border-b border-[#1E2436] pb-2">
              Voice Interface Controller
            </div>

            {sessionState === "READY" ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-6 border border-dashed border-[#1E2436] rounded-md">
                <span className="text-sm font-mono text-[#6B7A99] mb-4">
                  Session offline. Establish communication link.
                </span>
                <Button
                  onClick={startSession}
                  className="bg-[#4F6EF7] hover:bg-[#4F6EF7]/80 text-white font-mono flex items-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  <span>START DISPATCH SESSION</span>
                </Button>
              </div>
            ) : (
              <div className="flex-1 flex flex-col min-h-0">
                {/* Waveform Ring */}
                <WaveformRing
                  sessionState={sessionState}
                  micActive={micActive}
                  amplitude={amplitude}
                />

                {/* Live rolling transcript log */}
                <TranscriptScroll lines={transcripts} />

                {/* Quick Simulation input bar for test environment */}
                <form onSubmit={handleSendText} className="mt-3 flex gap-2">
                  <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Simulate operator speech..."
                    className="flex-1 bg-[#161B2E] border border-[#1E2436] rounded px-3 py-1.5 text-sm text-[#F0F4FF] focus:outline-none focus:border-[#4F6EF7] font-mono placeholder-[#6B7A99]"
                  />
                  <Button type="submit" size="sm" className="bg-[#4F6EF7] hover:bg-[#4F6EF7]/80 text-white font-mono text-xs">
                    SEND
                  </Button>
                </form>

                {/* Bottom voice control actions bar */}
                <div className="mt-4 pt-3 border-t border-[#1E2436] flex items-center justify-between">
                  <Button
                    size="sm"
                    onClick={replayLastAudio}
                    className="bg-[#161B2E] hover:bg-[#1E2436] text-[#F0F4FF] border border-[#1E2436] text-xs font-mono flex items-center gap-1.5"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>REPLAY LAST</span>
                  </Button>

                  <div className="flex items-center gap-2">
                    {/* Simulated push to talk action */}
                    <button
                      onMouseDown={() => setMicActive(true)}
                      onMouseUp={() => setMicActive(false)}
                      onTouchStart={() => setMicActive(true)}
                      onTouchEnd={() => setMicActive(false)}
                      className={`px-4 py-2 rounded font-mono text-xs font-semibold select-none border transition-all ${
                        micActive
                          ? "bg-[#4F6EF7] text-white border-[#4F6EF7] shadow-[0_0_15px_rgba(79,110,247,0.4)]"
                          : "bg-[#161B2E] text-[#F0F4FF] border-[#1E2436] hover:bg-[#1E2436]"
                      }`}
                    >
                      {micActive ? "RELEASE TO SEND" : "HOLD SPACE / CLICK TO SPEAK"}
                    </button>
                  </div>

                  <Button
                    size="sm"
                    onClick={endSession}
                    className="bg-[#EF4444]/10 hover:bg-[#EF4444]/20 text-[#EF4444] border border-[#EF4444]/20 text-xs font-mono flex items-center gap-1.5"
                  >
                    <Square className="w-3.5 h-3.5" />
                    <span>END SESSION</span>
                  </Button>
                </div>
              </div>
            )}
          </section>

          {/* ZONE C: Field Collection Status (center 3 cols) */}
          <section className="col-span-3 flex flex-col gap-4 bg-[#111520] border border-[#1E2436] p-4 rounded-md">
            <div className="text-xs uppercase tracking-wider text-[#6B7A99] font-mono border-b border-[#1E2436] pb-2">
              Field Collection Status
            </div>

            {/* Status cards stack */}
            <div className="flex-1 flex flex-col gap-3">
              <FieldCard name="Location" value={fields.location.value} state={fields.location.state as any} />
              <FieldCard name="Event Type" value={fields.event_type.value} state={fields.event_type.state as any} />
              <FieldCard name="Event Cause" value={fields.event_cause.value} state={fields.event_cause.state as any} />
              <FieldCard name="Priority" value={fields.priority.value} state={fields.priority.state as any} />
              <FieldCard name="Vehicle Type" value={fields.vehicle_type.value} state={fields.vehicle_type.state as any} />

              {/* Auto-resolved Fields collapsible section */}
              <div className="border border-[#1E2436] rounded bg-[#161B2E] mt-2 overflow-hidden">
                <button
                  onClick={() => setAutoResolvedExpanded(!autoResolvedExpanded)}
                  className="w-full px-3 py-2 flex items-center justify-between text-xs font-mono text-[#6B7A99] hover:bg-[#1E2436]/50 transition-colors"
                >
                  <span>Auto Resolved Metadata</span>
                  {autoResolvedExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                </button>

                {autoResolvedExpanded && (
                  <div className="p-3 border-t border-[#1E2436] grid grid-cols-2 gap-2 text-[10px] font-mono bg-[#111520]">
                    <div className="flex flex-col">
                      <span className="text-[#6B7A99]">Corridor</span>
                      <span className="text-[#F0F4FF] truncate">{resolvedFields.corridor || "Not resolved"}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-[#6B7A99]">Police Station</span>
                      <span className="text-[#F0F4FF] truncate">{resolvedFields.police_station || "Not resolved"}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-[#6B7A99]">Zone</span>
                      <span className="text-[#F0F4FF] truncate">{resolvedFields.zone || "Not resolved"}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-[#6B7A99]">Date</span>
                      <span className="text-[#F0F4FF]">{resolvedFields.date || "Not resolved"}</span>
                    </div>
                    <div className="flex flex-col col-span-2">
                      <span className="text-[#6B7A99]">Time</span>
                      <span className="text-[#F0F4FF]">{resolvedFields.time || "Not resolved"}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Progress indicator */}
            <div className="pt-3 border-t border-[#1E2436] space-y-2">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-[#6B7A99]">Extraction Progress:</span>
                <span className="text-[#4F6EF7]">{confirmedCount}/5 Fields</span>
              </div>
              <Progress value={progressPercent} className="h-1.5 bg-[#111520] border border-[#1E2436]" />
            </div>

            {/* Run dispatch dispatch control */}
            {sessionState !== "READY" && (
              <Button
                disabled={sessionState === "PROCESSING"}
                onClick={handleSubmitPrediction}
                className="w-full mt-2 bg-[#7C3AED] hover:bg-[#7C3AED]/80 text-white font-mono text-xs h-9"
              >
                SUBMIT FOR ML PREDICTION
              </Button>
            )}
          </section>

          {/* ZONE D: Prediction Results Panel (right 4 cols) */}
          <section className="col-span-4 flex flex-col gap-4 bg-[#111520] border border-[#1E2436] p-4 rounded-md">
            <div className="text-xs uppercase tracking-wider text-[#6B7A99] font-mono border-b border-[#1E2436] pb-2">
              Incident Prediction & RAG Directives
            </div>

            {!prediction ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-6 border border-dashed border-[#1E2436] rounded-md">
                <span className="text-xs font-mono text-[#6B7A99]">
                  Awaiting dispatch submission. Collect incident variables above and trigger ML prediction pipeline.
                </span>
              </div>
            ) : (
              <div className="flex-1 flex flex-col gap-4 animate-fade-in">
                {/* 4 Metrics display */}
                <div className="grid grid-cols-1 gap-3">
                  <GaugeCard
                    score={prediction.event_impact_score}
                    severity={prediction.severity_band}
                  />
                  <MetricCard
                    type="diversion"
                    value={prediction.diversion_required}
                  />
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard
                      type="officers"
                      value={prediction.recommended_officers}
                    />
                    <MetricCard
                      type="barricades"
                      value={prediction.recommended_barricades}
                    />
                  </div>
                </div>

                {/* Ensemble confidence banner */}
                <div className="bg-[#161B2E] border border-[#1E2436] rounded-md px-3 py-1.5 flex items-center justify-between text-xs font-mono text-[#6B7A99]">
                  <span>System Reliability Match</span>
                  <span className="text-[#22C55E] font-semibold">
                    {prediction.ensemble_confidence}
                  </span>
                </div>

                {/* RAG Recommendations List */}
                <RecommendationList recommendations={recommendations} />

                {/* Export Button */}
                <Button
                  onClick={handleExportReport}
                  className="w-full bg-[#4F6EF7] hover:bg-[#4F6EF7]/80 text-white font-mono text-xs flex items-center justify-center gap-1.5 mt-auto"
                >
                  <Download className="w-4 h-4" />
                  <span>EXPORT DISPATCH REPORT</span>
                </Button>
              </div>
            )}
          </section>
        </main>
      </div>
    </TooltipProvider>
  );
}
