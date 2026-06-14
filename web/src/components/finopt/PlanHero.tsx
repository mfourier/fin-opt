import { CheckCircle2, AlertTriangle, XCircle } from "lucide-react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";
import { formatDateShort, formatMonthsLong } from "@/lib/format";

export type FeasibilityStatus = "feasible" | "tight" | "infeasible";

type Props = {
  scenarioName: string;
  profileName: string;
  optimalHorizonMonths: number | null;
  feasibility: FeasibilityStatus;
  solveTimeSeconds?: number | null;
  goalsAchieved: number;
  goalsTotal: number;
  updatedAt?: string | null;
};

// Icon + tone per status; label/description come from `plan:hero.status.<id>`.
const STATUS = {
  feasible: { icon: CheckCircle2, tone: "success" },
  tight: { icon: AlertTriangle, tone: "warning" },
  infeasible: { icon: XCircle, tone: "danger" },
} as const;

export function PlanHero({
  scenarioName,
  profileName,
  optimalHorizonMonths,
  feasibility,
  solveTimeSeconds,
  goalsAchieved,
  goalsTotal,
  updatedAt,
}: Props) {
  const { t } = useTranslation("plan");
  const s = STATUS[feasibility];
  const Icon = s.icon;
  const statusLabel = t(`hero.status.${feasibility}.label`);
  const statusDescription = t(`hero.status.${feasibility}.description`);

  return (
    <section
      aria-labelledby="plan-hero-title"
      className={cn(
        "relative overflow-hidden rounded-2xl border bg-card p-6 sm:p-10",
        "shadow-[0_1px_0_oklch(0_0_0/0.03),0_30px_60px_-30px_oklch(0.4_0.1_254/0.25)]",
      )}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-0 bg-[radial-gradient(80%_60%_at_85%_-10%,var(--color-accent)_0%,transparent_60%)] opacity-70"
      />
      <div className="relative z-10 flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2 text-xs font-medium text-muted-foreground">
            <span className="rounded-full bg-secondary px-2.5 py-1">{profileName}</span>
            <span aria-hidden>·</span>
            <span className="truncate">{scenarioName}</span>
          </div>
          <h1
            id="plan-hero-title"
            className="mt-3 text-sm font-medium uppercase tracking-wider text-muted-foreground"
          >
            {t("hero.minHorizon")}
          </h1>
          <p className="tabular mt-1 text-5xl font-semibold leading-none text-foreground sm:text-6xl md:text-7xl">
            {formatMonthsLong(optimalHorizonMonths)}
          </p>
          <p className="mt-3 max-w-prose text-sm text-muted-foreground">
            {t("hero.horizonHint")}
          </p>
        </div>

        <div className="flex flex-col items-start gap-3 sm:items-end">
          <StatusBadge tone={s.tone}>
            <Icon className="size-4" aria-hidden />
            {statusLabel}
          </StatusBadge>
          <p className="max-w-xs text-sm text-muted-foreground sm:text-right">{statusDescription}</p>
        </div>
      </div>

      <dl className="relative z-10 mt-8 grid grid-cols-2 gap-4 border-t pt-6 sm:grid-cols-4">
        <Stat label={t("hero.stats.goalsAchieved")} value={`${goalsAchieved} / ${goalsTotal}`} />
        <Stat label={t("hero.stats.planStatus")} value={statusLabel} />
        <Stat label={t("hero.stats.computeTime")} value={solveTimeSeconds ? `${solveTimeSeconds.toFixed(2)}s` : "—"} />
        <Stat label={t("hero.stats.updated")} value={formatDateShort(updatedAt)} />
      </dl>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className="tabular mt-1 text-lg font-semibold text-foreground">{value}</dd>
    </div>
  );
}

function StatusBadge({
  tone,
  children,
}: {
  tone: "success" | "warning" | "danger";
  children: React.ReactNode;
}) {
  const toneClasses = {
    success: "bg-success-soft text-success ring-success/30",
    warning: "bg-warning-soft text-warning ring-warning/30",
    danger: "bg-danger-soft text-danger ring-danger/30",
  } as const;
  const dot = {
    success: "bg-success",
    warning: "bg-warning",
    danger: "bg-danger",
  } as const;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium ring-1",
        toneClasses[tone],
      )}
    >
      <span className={cn("size-1.5 rounded-full", dot[tone])} aria-hidden />
      {children}
    </span>
  );
}
