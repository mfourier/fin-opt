import { CheckCircle2, AlertTriangle, ShieldCheck } from "lucide-react";
import { Trans, useTranslation } from "react-i18next";
import type { GoalStatus } from "@/mocks/types";
import {
  formatCLP,
  describeConfidence,
  formatMonthYear,
  formatPercent,
  isFloorGoal,
} from "@/lib/format";
import { cn } from "@/lib/utils";

/** Measured empirical success rate for a goal (never the hidden CVaR level). */
function goalEmpirical(g: GoalStatus): number {
  return g.actual_probability ?? g.empirical_probability ?? 0;
}

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
  const { t } = useTranslation("plan");
  // Plan-level collapse: every goal sits on its floor (reached in ~all scenarios
  // even at a relaxed setting). The certainty choice can't move such a plan, so we
  // say so up front instead of presenting near-identical numbers as a real choice.
  const allFloor =
    goals.length > 0 && goals.every((g) => isFloorGoal(goalEmpirical(g), g.required_confidence));

  return (
    <div className="rounded-2xl border bg-card p-5 sm:p-6">
      <div className="flex items-end justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">{t("goals.title")}</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {t("goals.subtitle")}
          </p>
        </div>
      </div>

      {allFloor && (
        <div className="mt-4 flex items-start gap-2.5 rounded-xl bg-success-soft p-3 text-sm text-success ring-1 ring-success/30">
          <ShieldCheck className="mt-0.5 size-4 shrink-0" aria-hidden />
          <p>{t("goals.allFloor", { count: goals.length })}</p>
        </div>
      )}

      <ul className="mt-4 space-y-3">
        {goals.map((g) => {
          // The only certainty figure we show is the *measured* empirical success
          // rate across the simulations — never the hidden CVaR level the optimizer
          // was given (that level is camouflaged behind the wizard's presets).
          const actual = goalEmpirical(g);
          // Per-goal floor note: reached in ~every scenario despite a relaxed
          // request → the preset can't push it higher; say so honestly.
          const onFloor = !allFloor && isFloorGoal(actual, g.required_confidence);
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
                    {g.type === "terminal"
                      ? t("goals.chipTerminal")
                      : detail?.targetDate
                        ? t("goals.chipBy", { date: formatMonthYear(detail.targetDate) })
                        : t("goals.chipBySpecific")}
                  </span>
                </div>
                <p className="mt-1 text-sm text-muted-foreground">
                  <Trans
                    i18nKey="goals.reachLine"
                    t={t}
                    values={{ amount: formatCLP(g.threshold), account: accName }}
                    components={{
                      amount: <span className="tabular font-medium text-foreground" />,
                      acct: <span className="font-medium text-foreground" />,
                    }}
                  />
                </p>

                <div className="mt-3">
                  <ProbabilityBar actual={actual} tone={tone} />
                  <div className="mt-1.5 flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">{describeConfidence(actual)}.</span>
                    <span className="tabular font-medium text-foreground">{formatPercent(actual)}</span>
                  </div>
                </div>

                {onFloor && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    {t("goals.onFloor")}
                  </p>
                )}
              </div>

              <div className="sm:text-right">
                <Pill tone={tone}>
                  {tone === "success" ? (
                    <CheckCircle2 className="size-3.5" aria-hidden />
                  ) : (
                    <AlertTriangle className="size-3.5" aria-hidden />
                  )}
                  {g.satisfied ? t("goals.onTrack") : t("goals.needsAdjustment")}
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

function ProbabilityBar({
  actual,
  tone,
}: {
  actual: number;
  tone: "success" | "warning";
}) {
  const fillColor = tone === "success" ? "bg-success" : "bg-warning";
  const pct = Math.max(0, Math.min(1, actual));
  return (
    <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
      <div
        className={cn("h-full rounded-full transition-all", fillColor)}
        style={{ width: `${pct * 100}%` }}
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
