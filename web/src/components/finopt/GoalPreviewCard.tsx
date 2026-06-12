/**
 * GoalPreviewCard — a compact, illustrative "goal" card for the login hero.
 * Shows a goal name, a feasibility status pill, and a confidence bar. Static,
 * presentational content (no data fetching) used purely to communicate the
 * product idea on the marketing surface.
 */
import { Target } from "lucide-react";
import { cn } from "@/lib/utils";

interface GoalPreviewCardProps {
  goal: string;
  status: string;
  confidence: number; // 0..1
  className?: string;
  style?: React.CSSProperties;
}

export function GoalPreviewCard({
  goal,
  status,
  confidence,
  className,
  style,
}: GoalPreviewCardProps) {
  const pct = Math.round(confidence * 100);

  return (
    <div
      style={style}
      className={cn(
        "w-60 rounded-2xl border border-white/50 bg-white/90 p-4 shadow-xl shadow-[#005C99]/15 backdrop-blur-md",
        className,
      )}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#E6F4FB] text-[#008FD3]">
            <Target className="h-4 w-4" />
          </span>
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wide text-[#6B7280]">
              Goal
            </p>
            <p className="text-sm font-semibold text-[#111827]">{goal}</p>
          </div>
        </div>
        <span className="rounded-full bg-[#DCFCE7] px-2.5 py-1 text-[11px] font-semibold text-[#15803D]">
          {status}
        </span>
      </div>

      <div className="mt-4">
        <div className="mb-1 flex items-center justify-between text-[11px] font-medium text-[#6B7280]">
          <span>Confidence</span>
          <span className="tabular text-[#111827]">{pct}%</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-[#E6F4FB]">
          <div
            className="h-full rounded-full bg-gradient-to-r from-[#008FD3] to-[#12BFEF]"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
