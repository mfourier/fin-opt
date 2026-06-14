/**
 * MiniMetricCard — a small floating glass card used to decorate the login hero.
 * Presentational only: a label, a value, and an optional icon/tone.
 */
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

type Tone = "default" | "success";

interface MiniMetricCardProps {
  label: string;
  value: string;
  icon?: LucideIcon;
  tone?: Tone;
  className?: string;
  style?: React.CSSProperties;
}

export function MiniMetricCard({
  label,
  value,
  icon: Icon,
  tone = "default",
  className,
  style,
}: MiniMetricCardProps) {
  return (
    <div
      style={style}
      className={cn(
        "flex items-center gap-3 rounded-2xl border border-white/20 bg-card/82 p-3 pr-4 text-card-foreground shadow-lg shadow-black/10 backdrop-blur-md",
        className,
      )}
    >
      {Icon && (
        <span
          className={cn(
            "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl",
            tone === "success" ? "bg-success-soft text-success" : "bg-accent text-primary",
          )}
        >
          <Icon className="h-4 w-4" />
        </span>
      )}
      <div className="min-w-0">
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </p>
        <p className="truncate text-sm font-semibold text-foreground">{value}</p>
      </div>
    </div>
  );
}
