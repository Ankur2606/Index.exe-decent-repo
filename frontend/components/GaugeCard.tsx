"use client";

import React from "react";
import { Badge } from "@/components/ui/badge";

interface GaugeCardProps {
  score: number;
  severity: "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
}

export default function GaugeCard({ score, severity }: GaugeCardProps) {
  // SVG circular properties
  const radius = 38;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  // Determine colors based on score
  let arcColor = "#22C55E"; // low
  let glowColor = "rgba(34, 197, 94, 0.1)";
  let badgeStyle = "bg-[#22C55E]/20 text-[#22C55E] border-[#22C55E]/30";

  if (score > 80) {
    arcColor = "#EF4444"; // critical
    glowColor = "rgba(239, 68, 68, 0.1)";
    badgeStyle = "bg-[#EF4444]/20 text-[#EF4444] border-[#EF4444]/30";
  } else if (score > 60) {
    arcColor = "#F97316"; // high
    glowColor = "rgba(249, 115, 22, 0.1)";
    badgeStyle = "bg-[#F97316]/20 text-[#F97316] border-[#F97316]/30";
  } else if (score > 30) {
    arcColor = "#EAB308"; // moderate
    glowColor = "rgba(234, 179, 8, 0.1)";
    badgeStyle = "bg-[#EAB308]/20 text-[#EAB308] border-[#EAB308]/30";
  }

  return (
    <div
      className="flex flex-col items-center justify-center p-4 rounded-md border border-[#1E2436] transition-all duration-300 relative overflow-hidden bg-[#161B2E]"
      style={{
        boxShadow: `inset 0 0 40px ${glowColor}`,
      }}
    >
      <div className="text-[10px] tracking-wider text-[#6B7A99] font-mono uppercase mb-2">
        Event Impact Score
      </div>

      <div className="relative w-32 h-32 flex items-center justify-center">
        {/* SVG Circular progress */}
        <svg viewBox="0 0 100 100" className="w-full h-full transform -rotate-90">
          {/* Track Circle */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke="#1E2436"
            strokeWidth="8"
          />
          {/* Progress Arc */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke={arcColor}
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            style={{
              transition: "stroke-dashoffset 800ms cubic-bezier(0.4, 0, 0.2, 1)",
            }}
          />
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold font-mono text-[#F0F4FF]">
            {Math.round(score)}
          </span>
          <span className="text-[10px] font-mono text-[#6B7A99] mt-0.5">/100</span>
        </div>
      </div>

      <Badge className={`mt-3 font-mono text-xs uppercase px-2.5 py-0.5 border ${badgeStyle}`}>
        {severity}
      </Badge>
    </div>
  );
}
