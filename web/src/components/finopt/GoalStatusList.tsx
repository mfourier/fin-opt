import { CheckCircle2, AlertTriangle } from "lucide-react";
import type { GoalStatus } from "@/mocks/types";
import { formatCLP, describeConfidence, formatMonthYear, formatPercent } from "@/lib/format";
import { cn } from "@/lib/utils";

export type GoalDetail = {
  type: GoalStatus["type"];
  account: string;
  threshold: number;
  requiredConfidence: number;
  targetDate?: string;
};

type Props = {
  goals: GoalStatus[];
  accountDisplayNames?: Record<string, string>;
  goalDetails?: GoalDetail[];
};

export function GoalStatusList({ goals, accountDisplayNames = {}, goalDetails = [] }: Props) {
  return (
    <div className="rounded-2xl border bg-card p-5 sm:p-6">
      <div className="flex items-end justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">Your goals</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            How likely each goal is to be reached at your chosen confidence.
          </p>
        </div>
      </div>

      <ul className="mt-4 space-y-3">
        {goals.map((g) => {
          const actual = g.actual_probability ?? g.empirical_probability ?? 0;
          const required = g.required_confidence;
          const tone = g.satisfied ? "success" : "warning";
          const accName = accountDisplayNames[g.account] ?? g.account;
          const detail = goalDetails.find((candidate) => matchesGoal(candidate, g));
          return (
            <li
              key={g.goal}
              className="grid gap-4 rounded-xl border bg-muted/30 p-4 sm:grid-cols-[1fr_auto]"
            >
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  {/* Title from structured fields — g.goal is a backend string and
                      may contain account slugs ("cuenta_vivienda at horizon T=27"). */}
                  <h3 className="font-medium text-foreground">{accName}</h3>
                  <span className="rounded-full bg-secondary px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
                    {g.type === "terminal" ? "By the end of the plan" : intermediateChip(detail?.targetDate)}
                  </span>
                </div>
                <p className="mt-1 text-sm text-muted-foreground">
                  Reach <span className="tabular font-medium text-foreground">{formatCLP(g.threshold)}</span>{" "}
                  in <span className="font-medium text-foreground">{accName}</span>.{" "}
                  <span className="text-foreground/80">{describeConfidence(required)}.</span>
                </p>
                {g.note && <p className="mt-2 text-sm text-muted-foreground">{g.note}</p>}

                <div className="mt-3">
                  <ProbabilityBar required={required} actual={actual} tone={tone} />
                  <div className="mt-1.5 flex items-center justify-between text-xs text-muted-foreground">
                    <span>
                      Required:{" "}
                      <span className="tabular font-medium text-foreground">{formatPercent(required)}</span>
                    </span>
                    <span>
                      Achieved:{" "}
                      <span className="tabular font-medium text-foreground">{formatPercent(actual)}</span>
                    </span>
                  </div>
                </div>
              </div>

              <div className="sm:text-right">
                <Pill tone={tone}>
                  {tone === "success" ? (
                    <CheckCircle2 className="size-3.5" aria-hidden />
                  ) : (
                    <AlertTriangle className="size-3.5" aria-hidden />
                  )}
                  {g.satisfied ? "On track" : "Needs adjustment"}
                </Pill>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function matchesGoal(detail: GoalDetail, goal: GoalStatus): boolean {
  return detail.type === goal.type
    && detail.account === goal.account
    && detail.threshold === goal.threshold
    && detail.requiredConfidence === goal.required_confidence;
}

function intermediateChip(targetDate?: string): string {
  return targetDate ? `By ${formatMonthYear(targetDate)}` : "By a specific date";
}

function ProbabilityBar({
  required,
  actual,
  tone,
}: {
  required: number;
  actual: number;
  tone: "success" | "warning";
}) {
  const fillColor = tone === "success" ? "bg-success" : "bg-warning";
  const pct = Math.max(0, Math.min(1, actual));
  const reqPct = Math.max(0, Math.min(1, required));
  return (
    <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
      <div
        className={cn("h-full rounded-full transition-all", fillColor)}
        style={{ width: `${pct * 100}%` }}
      />
      <div
        className="absolute top-1/2 h-3 w-px -translate-y-1/2 bg-foreground/60"
        style={{ left: `${reqPct * 100}%` }}
        aria-label={`Required confidence ${formatPercent(required)}`}
      />
    </div>
  );
}

function Pill({ tone, children }: { tone: "success" | "warning"; children: React.ReactNode }) {
  const cls = {
    success: "bg-success-soft text-success ring-success/30",
    warning: "bg-warning-soft text-warning ring-warning/30",
  } as const;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1",
        cls[tone],
      )}
    >
      {children}
    </span>
  );
}
