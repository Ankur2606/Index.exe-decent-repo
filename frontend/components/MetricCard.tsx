"use client";

import React from "react";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ShieldAlert, AlertTriangle } from "lucide-react";

interface MetricCardProps {
  type: "officers" | "barricades" | "diversion";
  value: number | string | boolean;
}

export default function MetricCard({ type, value }: MetricCardProps) {
  if (type === "officers") {
    const count = Number(value) || 0;
    const iconsCount = Math.min(count, 10);
    const remainder = count - iconsCount;

    // Estimate shift timing based on the current hour (just a presentation detail)
    const currentHour = new Date().getHours();
    const isPeak = (currentHour >= 8 && currentHour <= 11) || (currentHour >= 17 && currentHour <= 21);
    const shiftText = isPeak ? "Peak Hour Force" : "Off Peak Force";

    return (
      <div className="flex flex-col p-4 rounded-md border border-[#1E2436] bg-[#161B2E] transition-all duration-300">
        <div className="text-[10px] tracking-wider text-[#6B7A99] font-mono uppercase mb-2">
          Officer Deployment
        </div>
        <div className="text-3xl font-bold font-sans text-[#F0F4FF] mb-2">
          {count} <span className="text-xs font-normal text-[#6B7A99]">officers</span>
        </div>
        <div className="flex flex-wrap items-center gap-1 min-h-[24px] mb-3">
          {Array.from({ length: iconsCount }).map((_, i) => (
            <span key={i} className="text-sm">👮</span>
          ))}
          {remainder > 0 && (
            <span className="text-[10px] font-mono text-[#6B7A99] ml-1">
              +{remainder} more
            </span>
          )}
        </div>
        <div className="mt-auto">
          <Badge className="bg-[#4F6EF7]/20 text-[#4F6EF7] border-[#4F6EF7]/30 text-[10px] tracking-wider font-mono">
            {shiftText}
          </Badge>
        </div>
      </div>
    );
  }

  if (type === "barricades") {
    const count = Number(value) || 0;
    const percent = Math.min((count / 50) * 100, 100);

    // Color code barricade load
    let barColor = "bg-[#22C55E]"; // low
    if (count > 30) {
      barColor = "bg-[#EF4444]";
    } else if (count > 15) {
      barColor = "bg-[#F97316]";
    }

    return (
      <div className="flex flex-col p-4 rounded-md border border-[#1E2436] bg-[#161B2E] transition-all duration-300">
        <div className="text-[10px] tracking-wider text-[#6B7A99] font-mono uppercase mb-2">
          Barricade Allocation
        </div>
        <div className="text-3xl font-bold font-sans text-[#F0F4FF] mb-2">
          {count} <span className="text-xs font-normal text-[#6B7A99]">units</span>
        </div>
        <div className="space-y-1 mb-3">
          <Progress value={percent} className="h-2 bg-[#111520] border border-[#1E2436]" style={{
            color: barColor // Wait, Progress shadcn uses standard styling, but we can style the container or inner indicator.
          }} />
          <div className="flex justify-between text-[8px] font-mono text-[#6B7A99]">
            <span>0</span>
            <span>25</span>
            <span>50 MAX</span>
          </div>
        </div>
        <div className="mt-auto">
          {count > 20 ? (
            <div className="flex items-center gap-1 text-[10px] font-semibold text-[#F97316] bg-[#F97316]/10 px-2 py-1 rounded border border-[#F97316]/20">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Request Additional Truck</span>
            </div>
          ) : (
            <span className="text-[10px] font-mono text-[#6B7A99]">
              Standard depot dispatch
            </span>
          )}
        </div>
      </div>
    );
  }

  if (type === "diversion") {
    // Check if value is yes or true
    const isYes = value === "YES" || value === "yes" || value === true || String(value).toUpperCase() === "TRUE";

    // Style according to diversion need
    const bgColor = isYes ? "rgba(239, 68, 68, 0.15)" : "rgba(34, 197, 94, 0.15)";
    const borderColor = isYes ? "border-[#EF4444]/30" : "border-[#22C55E]/30";
    const textStyle = isYes ? "text-[#EF4444]" : "text-[#22C55E]";

    return (
      <div
        className="flex flex-col p-4 rounded-md border transition-all duration-300"
        style={{
          backgroundColor: bgColor,
          borderColor: borderColor,
        }}
      >
        <div className="text-[10px] tracking-wider text-[#6B7A99] font-mono uppercase mb-2">
          Diversion Required
        </div>
        <div className={`text-4xl font-extrabold font-mono tracking-wider mb-2 ${textStyle}`}>
          {isYes ? "YES" : "NO"}
        </div>
        <div className="flex items-start gap-1 text-[10px] text-[#F0F4FF] mt-auto">
          <ShieldAlert className="w-3.5 h-3.5 mt-0.5 text-[#6B7A99]" />
          <span>
            {isYes
              ? "Activate alternate route protocol now"
              : "Standard transit corridors remain operational"}
          </span>
        </div>
      </div>
    );
  }

  return null;
}
