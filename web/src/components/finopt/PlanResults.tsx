import { Download, RotateCw, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Profile, Scenario, Result, JobStatus } from "@/mocks/types";
import { PlanHero, type FeasibilityStatus } from "./PlanHero";
import { WealthFanChart } from "./WealthFanChart";
import { ContributionPlan } from "./ContributionPlan";
import { GoalStatusList } from "./GoalStatusList";
import { JobStatusView } from "./JobStatusView";

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
}: Props) {
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

  // Pick the largest terminal goal threshold to show as the chart goal line.
  const heroGoal = scenario.terminal_goals[0];
  const heroGoalLabel = heroGoal
    ? `Goal: ${accountDisplay[heroGoal.account] ?? heroGoal.account}`
    : undefined;

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
      />

      {/* Action bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-card px-4 py-3">
        <p className="text-sm text-muted-foreground">
          Update your situation or goals and recalculate to see a new plan.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          {onAdjustGoals && (
            <Button variant="ghost" size="sm" onClick={onAdjustGoals}>
              <Settings2 className="size-4" aria-hidden /> Adjust
            </Button>
          )}
          {onExportCSV && (
            <Button variant="outline" size="sm" onClick={onExportCSV}>
              <Download className="size-4" aria-hidden /> CSV
            </Button>
          )}
          {onExportJSON && (
            <Button variant="outline" size="sm" onClick={onExportJSON}>
              <Download className="size-4" aria-hidden /> JSON
            </Button>
          )}
          {onRecalculate && (
            <Button size="sm" onClick={onRecalculate}>
              <RotateCw className="size-4" aria-hidden /> Recalculate
            </Button>
          )}
        </div>
      </div>

      {result.summary_stats?.total_wealth && (
        <WealthFanChart
          percentiles={result.summary_stats.total_wealth}
          goalThreshold={heroGoal?.threshold}
          goalLabel={heroGoal ? `Goal · ${heroGoalLabel}` : undefined}
        />
      )}

      {goals.length > 0 && <GoalStatusList goals={goals} accountDisplayNames={accountDisplay} />}

      {result.summary_stats?.cash_flow && (
        <ContributionPlan cashFlow={result.summary_stats.cash_flow} />
      )}
    </div>
  );
}
