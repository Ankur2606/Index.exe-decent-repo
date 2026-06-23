"use client";

import React, { useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TranscriptLine } from "../lib/types";

interface TranscriptScrollProps {
  lines: TranscriptLine[];
}

export default function TranscriptScroll({ lines }: TranscriptScrollProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [lines]);

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-[#111520] border border-[#1E2436] rounded-md p-4">
      <div className="text-xs uppercase tracking-wider text-[#6B7A99] font-mono mb-3 border-b border-[#1E2436] pb-2">
        Operational Log Transcript
      </div>

      <ScrollArea className="flex-1 overflow-y-auto pr-2">
        <div className="flex flex-col gap-3 min-h-[160px] justify-end">
          {lines.length === 0 ? (
            <div className="text-xs text-[#6B7A99] font-mono italic text-center py-8">
              Waiting for voice transmission log...
            </div>
          ) : (
            lines.map((line, idx) => {
              const isAgent = line.speaker === "agent";
              return (
                <div
                  key={idx}
                  className={`flex flex-col w-full animate-fade-in ${
                    isAgent ? "items-start" : "items-end"
                  }`}
                >
                  <div className="flex items-center gap-2 max-w-[85%]">
                    {isAgent ? (
                      <div className="flex flex-col items-start">
                        <div className="flex items-center gap-1.5 mb-1">
                          <span className="text-[10px] bg-[#7C3AED] text-white font-mono px-1.5 py-0.5 rounded">
                            ASTraM System
                          </span>
                          <span className="text-[10px] text-[#6B7A99] font-mono">
                            {line.ts}
                          </span>
                        </div>
                        <div className="bg-[#161B2E] border border-[#2D3555] rounded-lg p-3 text-sm text-[#7C3AED] leading-relaxed font-sans shadow-sm">
                          {line.text}
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col items-end">
                        <div className="flex items-center gap-1.5 mb-1">
                          <span className="text-[10px] text-[#6B7A99] font-mono">
                            {line.ts}
                          </span>
                          <span className="text-[10px] bg-[#4F6EF7] text-white font-mono px-1.5 py-0.5 rounded">
                            Operator
                          </span>
                        </div>
                        <div className="bg-[#4F6EF7]/10 border border-[#4F6EF7]/20 rounded-lg p-3 text-sm text-[#F0F4FF] leading-relaxed font-sans shadow-sm">
                          {line.text}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
          <div ref={containerRef} />
        </div>
      </ScrollArea>
    </div>
  );
}
