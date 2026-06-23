"use client";

import React from "react";
import { Info } from "lucide-react";

interface RecommendationListProps {
  recommendations: string[];
}

export default function RecommendationList({ recommendations }: RecommendationListProps) {
  return (
    <div className="flex flex-col p-4 rounded-md border border-[#1E2436] bg-[#161B2E]">
      <div className="text-[10px] tracking-wider text-[#6B7A99] font-mono uppercase mb-3 border-b border-[#1E2436] pb-2 flex items-center justify-between">
        <span>Operational Recommendations</span>
        <span className="text-[9px] bg-[#7C3AED]/20 text-[#7C3AED] px-1 rounded font-normal normal-case">
          Grounded RAG Knowledge
        </span>
      </div>

      {recommendations.length === 0 ? (
        <div className="flex items-center gap-2 text-xs text-[#6B7A99] font-mono italic py-4">
          <Info className="w-4 h-4 text-[#6B7A99]" />
          <span>Awaiting incident verification...</span>
        </div>
      ) : (
        <div className="space-y-3">
          {recommendations.map((rec, idx) => {
            // Give each bullet a colored left border to separate visually
            const borderColors = ["border-l-[#4F6EF7]", "border-l-[#7C3AED]", "border-l-[#EAB308]"];
            const borderColor = borderColors[idx % borderColors.length];

            return (
              <div
                key={idx}
                className={`pl-3 border-l-2 ${borderColor} py-1 text-xs text-[#F0F4FF] leading-relaxed`}
              >
                {rec}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
