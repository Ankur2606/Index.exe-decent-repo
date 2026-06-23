"use client";

import React from "react";
import { Check, Dot, Circle } from "lucide-react";
import { Card } from "@/components/ui/card";

interface FieldCardProps {
  name: string;
  value: string;
  state: "empty" | "collecting" | "confirmed";
}

export default function FieldCard({ name, value, state }: FieldCardProps) {
  let borderColor = "border-[#1E2436]";
  let statusIcon = <Circle className="w-4 h-4 text-[#2D3555]" />;
  let leftBorderColor = "border-l-transparent";

  if (state === "collecting") {
    borderColor = "border-[#4F6EF7]/40 shadow-[0_0_10px_rgba(79,110,247,0.15)]";
    statusIcon = <Dot className="w-4 h-4 text-[#4F6EF7] animate-ping" />;
    leftBorderColor = "border-l-[#4F6EF7]";
  } else if (state === "confirmed") {
    borderColor = "border-[#1E2436]";
    statusIcon = <Check className="w-4 h-4 text-[#22C55E]" />;
    leftBorderColor = "border-l-[#22C55E]";
  }

  return (
    <Card
      className={`bg-[#161B2E] ${borderColor} border border-l-4 ${leftBorderColor} p-3 rounded-md transition-all duration-300`}
    >
      <div className="flex items-center justify-between">
        <div className="flex flex-col">
          <span className="text-[10px] tracking-wider text-[#6B7A99] font-mono uppercase">
            {name}
          </span>
          <span className="text-sm font-semibold text-[#F0F4FF] mt-0.5 min-h-[20px]">
            {value || "Waiting for signal..."}
          </span>
        </div>
        <div className="flex items-center justify-center w-6 h-6 rounded-full bg-[#111520] border border-[#1E2436]">
          {statusIcon}
        </div>
      </div>
    </Card>
  );
}
