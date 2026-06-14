import { useState } from "react";
import { Download, RotateCw, Settings2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { Button } from "@/components/ui/button";
import type { Profile, Scenario, Result, JobStatus } from "@/mocks/types";
import { formatCLPCompact } from "@/lib/format";
import { PlanHero, type FeasibilityStatus } from "./PlanHero";
import { PlanExplainers } from "./PlanExplainers";
import { WealthFanChart, type WithdrawalMarker } from "./WealthFanChart";
import { ContributionPlan } from "./ContributionPlan";
import { GoalStatusList, type GoalDetail } from "./GoalStatusList";
import { JobStatusView } from "./JobStatusView";
import type { ExplainerFocus } from "./plan-explainer-focus";

type Props = {
  profile: Profile;
  scenario: Scenario;
  result: Result;
  jobStatus: JobStatus;
  jobProgress?: number;
  onExportJSON?: () => void;
  onExportCSV?: () => void;
  onRecalculate?: () => void;
  onAdjustGoals?: () => void;
  updatedAt?: string | null;
  isRecalculating?: boolean;
  actionError?: string | null;
};

export function PlanResults({
  profile,
  scenario,
  result,
  jobStatus,
  jobProgress,
  onExportJSON,
  onExportCSV,
  onRecalculate,
  onAdjustGoals,
  updatedAt,
  isRecalculating,
  actionError,
}: Props) {
  const { t } = useTranslation("plan");
  const [hoverFocus, setHoverFocus] = useState<ExplainerFocus | null>(null);
  const [pinnedFocus, setPinnedFocus] = useState<ExplainerFocus | null>(null);
  const activeFocus = pinnedFocus ?? hoverFocus;

  const handleHoverFocusChange = (focus: ExplainerFocus | null) => {
    setHoverFocus(focus);
  };

  const handleTogglePin = (focus: ExplainerFocus) => {
    setPinnedFocus((current) => (current === focus ? null : focus));
    setHoverFocus(null);
  };

  const handleClearPin = () => {
    setPinnedFocus(null);
  };

  if (jobStatus !== "completed") {
    return (
      <div className="mx-auto w-full max-w-3xl">
        <JobStatusView
          status={jobStatus}
          progress={jobProgress}
          onAdjust={onAdjustGoals}
          onRetry={onRecalculate}
        />
      </div>
    );
  }

  const feasibility: FeasibilityStatus = result.feasible
    ? (result.goal_status ?? []).every((g) => g.satisfied)
      ? "feasible"
      : "tight"
    : "infeasible";

  const goals = result.goal_status ?? [];
  const achieved = goals.filter((g) => g.satisfied).length;

  const accountDisplay = Object.fromEntries(
    profile.accounts_config.map((a) => [a.name, a.display_name ?? a.name]),
  );

  // One dashed goal line per goal (terminal + dated), colored per account.
  const goalLines = [...scenario.terminal_goals, ...scenario.intermediate_goals].map((g) => ({
    account: g.account,
    threshold: g.threshold,
    label: accountDisplay[g.account] ?? g.account,
  }));

  const goalDetails: GoalDetail[] = [
    ...scenario.terminal_goals.map((g) => ({
      type: "terminal" as const,
      account: g.account,
      threshold: g.threshold,
      requiredConfidence: g.confidence,
    })),
    ...scenario.intermediate_goals.map((g) => ({
      type: "intermediate" as const,
      account: g.account,
      threshold: g.threshold,
      requiredConfidence: g.confidence,
      targetDate: g.date,
    })),
  ];

  const withdrawalMarkers: WithdrawalMarker[] = buildWithdrawalMarkers(scenario, t);

  return (
    <div className="space-y-6">
      <PlanHero
        scenarioName={scenario.name}
        profileName={profile.name}
        optimalHorizonMonths={result.optimal_horizon}
        feasibility={feasibility}
        solveTimeSeconds={result.solve_time}
        goalsAchieved={achieved}
        goalsTotal={goals.length}
        updatedAt={updatedAt}
      />

      <PlanExplainers
        profile={profile}
        scenario={scenario}
        result={result}
        activeFocus={activeFocus}
        pinnedFocus={pinnedFocus}
        onFocusChange={handleHoverFocusChange}
        onTogglePin={handleTogglePin}
        onClearPin={handleClearPin}
      />

      {/* Action bar */}
      <div className="rounded-xl border bg-card px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-muted-foreground">
            {t("actions.hint")}
          </p>
          <div className="flex flex-wrap items-center gap-2">
            {onAdjustGoals && (
              <Button variant="ghost" size="sm" onClick={onAdjustGoals} disabled={isRecalculating}>
                <Settings2 className="size-4" aria-hidden /> {t("actions.adjust")}
              </Button>
            )}
            {onExportCSV && (
              <Button variant="outline" size="sm" onClick={onExportCSV} disabled={isRecalculating}>
                <Download className="size-4" aria-hidden /> CSV
              </Button>
            )}
            {onExportJSON && (
              <Button variant="outline" size="sm" onClick={onExportJSON} disabled={isRecalculating}>
                <Download className="size-4" aria-hidden /> JSON
              </Button>
            )}
            {onRecalculate && (
              <Button size="sm" onClick={onRecalculate} disabled={isRecalculating}>
                <RotateCw className="size-4" aria-hidden />
                {isRecalculating ? t("actions.recalculating") : t("actions.recalculate")}
              </Button>
            )}
          </div>
        </div>
        {actionError && (
          <p className="mt-3 rounded-lg border border-danger/30 bg-danger-soft px-3 py-2 text-sm text-danger">
            {actionError}
          </p>
        )}
      </div>

      {result.summary_stats?.total_wealth && (
        <WealthFanChart
          percentiles={result.summary_stats.total_wealth}
          accounts={result.summary_stats.per_account}
          goals={goalLines}
          startDate={scenario.start_date}
          withdrawals={withdrawalMarkers}
          activeFocus={activeFocus}
          pinnedFocus={pinnedFocus}
          onFocusChange={handleHoverFocusChange}
          onTogglePin={handleTogglePin}
        />
      )}

      {goals.length > 0 && (
        <GoalStatusList
          goals={goals}
          accountDisplayNames={accountDisplay}
          goalDetails={goalDetails}
        />
      )}

      {result.summary_stats?.cash_flow && (
        <ContributionPlan
          cashFlow={result.summary_stats.cash_flow}
          startDate={scenario.start_date}
          withdrawals={withdrawalMarkers}
          activeFocus={activeFocus}
          pinnedFocus={pinnedFocus}
          onFocusChange={handleHoverFocusChange}
          onTogglePin={handleTogglePin}
        />
      )}
    </div>
  );
}

function buildWithdrawalMarkers(scenario: Scenario, t: TFunction): WithdrawalMarker[] {
  const start = parseIsoMonthStart(scenario.start_date);
  if (!start || !scenario.withdrawals) return [];

  const markers: WithdrawalMarker[] = [];
  for (const withdrawal of scenario.withdrawals.scheduled) {
    const month = monthDiff(start, parseIsoMonthStart(withdrawal.date));
    if (month === null || month < 0) continue;
    markers.push({
      month,
      label: `${displayWithdrawalLabel(withdrawal.description, withdrawal.account, t)} ${formatCLPCompact(withdrawal.amount)}`,
    });
  }

  for (const withdrawal of scenario.withdrawals.stochastic) {
    // Stochastic withdrawals carry either a calendar `date` or a 1-indexed
    // `month` offset from start_date (the core's convention); the chart's
    // x-axis is 0-indexed months from start.
    const month = withdrawal.date
      ? monthDiff(start, parseIsoMonthStart(withdrawal.date))
      : typeof withdrawal.month === "number"
        ? withdrawal.month - 1
        : null;
    if (month === null || month < 0) continue;
    markers.push({
      month,
      label: `${displayWithdrawalLabel(withdrawal.description, withdrawal.account, t)} ${formatCLPCompact(withdrawal.base_amount)}`,
    });
  }

  return markers.sort((a, b) => a.month - b.month);
}

function parseIsoMonthStart(value?: string | null): Date | null {
  if (!value) return null;
  const match = value.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) return null;
  return new Date(Number(match[1]), Number(match[2]) - 1, 1);
}

function monthDiff(start: Date, target: Date | null): number | null {
  if (!target) return null;
  return (target.getFullYear() - start.getFullYear()) * 12 + (target.getMonth() - start.getMonth());
}

function displayWithdrawalLabel(description: string | undefined, account: string, t: TFunction): string {
  return description?.trim() || t("withdrawalFallback", { account });
}
