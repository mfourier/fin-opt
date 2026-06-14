import type { ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  ArrowRight,
  BriefcaseBusiness,
  Clock3,
  FolderKanban,
  Loader2,
  Plus,
  Target,
  Trash2,
  TrendingUp,
  Wallet,
} from 'lucide-react'
import { supabase } from '../lib/supabase'
import { useAuthStore } from '../lib/store'
import { useToast } from '../components/Toast'
import type { Job, Profile, Scenario } from '../types/database'
import { formatCLP, formatDateShort, formatMonthsLong, formatPercent } from '../lib/format'
import {
  type PlanHealth,
  type ResultPreviewLite,
  describeProfileRisk,
  getPlanHealth,
  getProfileMonthlyContributionCapacity,
  getProfileMonthlyIncome,
  getProfileStartingBalance,
  getProfileTopAccounts,
  getProfileWeightedReturn,
  summarizeGoalStatus,
} from '../lib/finance'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

type DashboardScenario = Pick<Scenario, 'id' | 'name' | 'is_demo' | 'start_date'> & {
  profiles: Pick<Profile, 'name'> | null
}
type DashboardJob = Pick<
  Job,
  'id' | 'scenario_id' | 'job_type' | 'status' | 'progress' | 'created_at' | 'completed_at'
> & {
  scenarios: DashboardScenario | null
}
type DashboardScenarioRow = Pick<Scenario, 'id' | 'name' | 'is_demo' | 'start_date'> & {
  profiles: Array<Pick<Profile, 'name'>> | null
}
type DashboardJobRow = Pick<
  Job,
  'id' | 'scenario_id' | 'job_type' | 'status' | 'progress' | 'created_at' | 'completed_at'
> & {
  scenarios: DashboardScenarioRow[] | DashboardScenarioRow | null
}

const toneStyles = {
  positive: 'bg-success-soft text-success',
  warning: 'bg-warning-soft text-warning',
  danger: 'bg-danger-soft text-danger',
  neutral: 'bg-muted text-muted-foreground',
  active: 'bg-accent text-accent-foreground',
} as const

export default function DashboardPage() {
  const queryClient = useQueryClient()
  const user = useAuthStore((state) => state.user)
  const toast = useToast()

  const { data: profiles, isLoading: profilesLoading } = useQuery({
    queryKey: ['profiles', 'dashboard', user?.id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .order('created_at', { ascending: false })
      if (error) throw error
      return data as Profile[]
    },
    enabled: !!user,
  })

  const { data: scenarios, isLoading: scenariosLoading } = useQuery({
    queryKey: ['scenarios', 'dashboard', user?.id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('scenarios')
        .select('id, name, is_demo, start_date, profiles(name)')
        .order('created_at', { ascending: false })
      if (error) throw error
      return ((data ?? []) as DashboardScenarioRow[]).map((scenario) => ({
        ...scenario,
        profiles: relationToOne(scenario.profiles),
      }))
    },
    enabled: !!user,
  })

  const { data: allJobs, isLoading: jobsLoading } = useQuery({
    queryKey: ['dashboard-jobs', user?.id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('jobs')
        .select('id, scenario_id, job_type, status, progress, created_at, completed_at, scenarios(id, name, is_demo, start_date, profiles(name))')
        .order('created_at', { ascending: false })
      if (error) throw error
      return ((data ?? []) as DashboardJobRow[]).map((job) => {
        const scenario = relationToOne(job.scenarios)
        return {
          ...job,
          scenarios: scenario
            ? {
                ...scenario,
                profiles: relationToOne(scenario.profiles),
              }
            : null,
        }
      })
    },
    enabled: !!user,
  })

  const recentJobs = (allJobs ?? []).slice(0, 10)
  const recentJobIds = recentJobs.map((job) => job.id)

  const { data: recentResults } = useQuery({
    queryKey: ['dashboard-recent-results', recentJobIds],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('results')
        .select('job_id, optimal_horizon, feasible, goal_status')
        .in('job_id', recentJobIds)
      if (error) throw error
      return data as ResultPreviewLite[]
    },
    enabled: recentJobIds.length > 0,
  })

  const latestJobsByScenarioId = Object.fromEntries(
    (allJobs ?? []).reduce<[string, DashboardJob][]>((entries, job) => {
      if (!entries.find(([scenarioId]) => scenarioId === job.scenario_id)) {
        entries.push([job.scenario_id, job])
      }
      return entries
    }, []),
  ) as Record<string, DashboardJob>

  const latestCompletedJobIds = Object.values(latestJobsByScenarioId)
    .filter((job) => job.status === 'completed')
    .map((job) => job.id)

  const { data: latestScenarioResults } = useQuery({
    queryKey: ['dashboard-latest-plan-results', latestCompletedJobIds],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('results')
        .select('job_id, optimal_horizon, feasible, goal_status')
        .in('job_id', latestCompletedJobIds)
      if (error) throw error
      return data as ResultPreviewLite[]
    },
    enabled: latestCompletedJobIds.length > 0,
  })

  const deleteJobMutation = useMutation({
    mutationFn: async (jobId: string) => {
      const { error } = await supabase.from('jobs').delete().eq('id', jobId)
      if (error) throw error
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dashboard-jobs'] })
      queryClient.invalidateQueries({ queryKey: ['dashboard-recent-results'] })
      queryClient.invalidateQueries({ queryKey: ['dashboard-latest-plan-results'] })
      toast.success('Result deleted', 'The run and its saved result were removed from your dashboard.')
    },
    onError: (error: Error) => {
      toast.error('Failed to delete result', error.message)
    },
  })

  const ownProfiles = profiles?.filter((profile) => !profile.is_demo) ?? []
  const ownScenarios = scenarios?.filter((scenario) => !scenario.is_demo) ?? []
  const latestSituation = ownProfiles[0] ?? null
  const scenariosById = Object.fromEntries((scenarios ?? []).map((scenario) => [scenario.id, scenario])) as Record<string, DashboardScenario>
  const recentResultsByJobId = Object.fromEntries((recentResults ?? []).map((result) => [result.job_id, result])) as Record<string, ResultPreviewLite>
  const latestResultsByJobId = Object.fromEntries((latestScenarioResults ?? []).map((result) => [result.job_id, result])) as Record<string, ResultPreviewLite>

  const totalNetWorth = latestSituation ? getProfileStartingBalance(latestSituation) : 0
  const totalMonthlyContribution = latestSituation ? getProfileMonthlyContributionCapacity(latestSituation) : 0
  const aggregatedReturn = latestSituation ? getProfileWeightedReturn(latestSituation) : 0

  const planEntries = ownScenarios.map((scenario) => {
    const latestJob = latestJobsByScenarioId[scenario.id]
    const latestResult = latestJob ? latestResultsByJobId[latestJob.id] : undefined
    const health = getPlanHealth(latestJob, latestResult)
    const goals = summarizeGoalStatus(latestResult?.goal_status)
    return {
      scenario,
      latestJob,
      latestResult,
      health,
      goals,
    }
  })

  const healthCounts = planEntries.reduce(
    (counts, entry) => {
      counts[entry.health] += 1
      return counts
    },
    {
      on_track: 0,
      tight: 0,
      needs_changes: 0,
      running: 0,
      queued: 0,
      failed: 0,
      draft: 0,
      completed: 0,
    } satisfies Record<PlanHealth, number>,
  )

  const activeRuns = healthCounts.running + healthCounts.queued
  const plansOnTrack = healthCounts.on_track
  const plansNeedingReview = healthCounts.tight + healthCounts.needs_changes + healthCounts.failed

  const priorityPlan = [...planEntries].sort((left, right) => {
    const rankDelta = planHealthPriority(left.health) - planHealthPriority(right.health)
    if (rankDelta !== 0) return rankDelta
    const leftHorizon = left.latestResult?.optimal_horizon ?? Number.POSITIVE_INFINITY
    const rightHorizon = right.latestResult?.optimal_horizon ?? Number.POSITIVE_INFINITY
    return leftHorizon - rightHorizon
  })[0]

  const nextMilestonePlan = [...planEntries]
    .filter((entry) => entry.latestResult?.optimal_horizon)
    .sort((left, right) => (left.latestResult?.optimal_horizon ?? Infinity) - (right.latestResult?.optimal_horizon ?? Infinity))[0]

  const statCards = [
    {
      label: 'Net worth',
      value: profilesLoading ? '—' : formatCLP(totalNetWorth),
      detail:
        latestSituation
          ? `Based on your primary situation, ${latestSituation.name}.`
          : 'Add a situation to start tracking your balance sheet.',
      icon: Wallet,
    },
    {
      label: 'Monthly investing power',
      value: profilesLoading ? '—' : `${formatCLP(totalMonthlyContribution)}/mo`,
      detail:
        latestSituation
          ? 'Estimated from income and contribution settings in your primary situation.'
          : 'No contribution plan yet.',
      icon: TrendingUp,
    },
    {
      label: 'Plans on track',
      value: scenariosLoading || jobsLoading ? '—' : String(plansOnTrack),
      detail:
        plansOnTrack > 0
          ? `${healthCounts.tight} tight plan${healthCounts.tight === 1 ? '' : 's'} still worth reviewing.`
          : 'No fully on-track plans yet.',
      icon: Target,
    },
    {
      label: 'Need attention',
      value: scenariosLoading || jobsLoading ? '—' : String(plansNeedingReview),
      detail:
        activeRuns > 0
          ? `${activeRuns} run${activeRuns === 1 ? '' : 's'} currently in progress.`
          : 'No active calculations right now.',
      icon: Clock3,
    },
  ]

  return (
    <div className="space-y-8">
      <Card className="overflow-hidden border-primary/10 bg-[radial-gradient(120%_120%_at_100%_0%,var(--color-accent)_0%,transparent_48%),linear-gradient(180deg,var(--color-card)_0%,color-mix(in_oklab,var(--color-card)_92%,var(--color-accent))_100%)] p-6 shadow-[0_1px_0_oklch(0_0_0/0.03),0_24px_48px_-24px_oklch(0.4_0.1_254/0.28)] sm:p-8">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <span className="inline-flex rounded-full bg-secondary px-3 py-1 text-xs font-medium text-secondary-foreground">
              Financial planning overview
            </span>
            <h1 className="mt-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
              Dashboard
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground sm:text-base">
              See your balance sheet, current savings capacity, and which plans are closest to the finish line.
            </p>

            {latestSituation ? (
              <div className="mt-6 flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
                <span className="rounded-full bg-card px-3 py-1 font-medium text-foreground shadow-sm">
                  Primary situation: {latestSituation.name}
                </span>
                <span>{formatCLP(getProfileStartingBalance(latestSituation))} starting balance</span>
                <span>{formatCLP(getProfileMonthlyContributionCapacity(latestSituation))}/mo investing power</span>
                <span>{formatPercent(aggregatedReturn, 1)} expected return</span>
                <span>{describeProfileRisk(latestSituation)}</span>
              </div>
            ) : (
              <div className="mt-6 rounded-2xl border border-dashed border-border bg-card/70 p-4 text-sm text-muted-foreground">
                Add a situation first so FinOpt can turn your income, assets, and goals into a plan.
              </div>
            )}
          </div>

          <div className="flex w-full max-w-md flex-col gap-3 rounded-2xl border border-border bg-card/90 p-4 shadow-sm">
            {priorityPlan?.latestJob ? (
              <>
                <div className="flex items-center justify-between gap-3">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Priority plan
                  </p>
                  <StatusPill tone={dashboardTone(priorityPlan.health)}>
                    {dashboardLabel(priorityPlan.health)}
                  </StatusPill>
                </div>
                <div>
                  <p className="font-semibold text-foreground">{priorityPlan.scenario.name}</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {priorityPlan.scenario.profiles?.name ?? 'Unknown situation'}
                  </p>
                </div>
                <p className="text-sm text-muted-foreground">
                  {priorityPlanSummary(priorityPlan.health, priorityPlan.latestResult, priorityPlan.goals)}
                </p>
                <Button asChild className="rounded-xl">
                  <Link to={`/results/${priorityPlan.latestJob.id}`}>
                    {priorityPlan.latestJob.status === 'completed' ? 'Open priority plan' : 'Open active run'}
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
              </>
            ) : (
              <>
                <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Next best step
                </p>
                <p className="text-sm text-muted-foreground">
                  {latestSituation
                    ? 'Create or run a plan to see which goals are already within reach.'
                    : 'Create your first situation to start building goal-based plans.'}
                </p>
                <div className="flex flex-wrap gap-2">
                  <Button asChild className="rounded-xl">
                    <Link to="/profiles">
                      <Plus className="h-4 w-4" />
                      Add a situation
                    </Link>
                  </Button>
                  <Button asChild variant="outline" className="rounded-xl">
                    <Link to="/scenarios">
                      <Target className="h-4 w-4" />
                      New plan
                    </Link>
                  </Button>
                </div>
              </>
            )}
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        {statCards.map(({ label, value, detail, icon: Icon }) => (
          <Card key={label} className="p-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-muted-foreground">{label}</p>
                <p className="mt-2 text-2xl font-semibold tabular text-foreground">{value}</p>
                <p className="mt-2 text-sm text-muted-foreground">{detail}</p>
              </div>
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-accent text-primary">
                <Icon className="h-5 w-5" />
              </span>
            </div>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.3fr,1fr]">
        <Card className="p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold text-foreground">Portfolio snapshot</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Use your latest saved situation as the baseline for future plans.
              </p>
            </div>
            <BriefcaseBusiness className="h-5 w-5 text-muted-foreground" />
          </div>

          {latestSituation ? (
            <div className="mt-6 space-y-5">
              <div className="grid gap-3 sm:grid-cols-2">
                <SnapshotMetric label="Starting balance" value={formatCLP(getProfileStartingBalance(latestSituation))} />
                <SnapshotMetric label="Monthly income" value={`${formatCLP(getProfileMonthlyIncome(latestSituation))}/mo`} />
                <SnapshotMetric label="Investing power" value={`${formatCLP(getProfileMonthlyContributionCapacity(latestSituation))}/mo`} />
                <SnapshotMetric label="Expected annual return" value={formatPercent(getProfileWeightedReturn(latestSituation), 1)} />
              </div>
              <p className="text-sm text-muted-foreground">
                These metrics are based on your primary situation, <span className="font-medium text-foreground">{latestSituation.name}</span>.
              </p>

              <div className="rounded-2xl bg-muted/50 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">Top accounts</p>
                    <p className="text-sm text-muted-foreground">
                      Highest-balance accounts inside {latestSituation.name}.
                    </p>
                  </div>
                  <span className="rounded-full bg-secondary px-2.5 py-1 text-xs font-medium text-secondary-foreground">
                    {describeProfileRisk(latestSituation)}
                  </span>
                </div>
                <div className="mt-4 space-y-3">
                  {getProfileTopAccounts(latestSituation).map((account) => (
                    <div key={account.id}>
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium text-foreground">{account.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {formatPercent(account.annualReturn, 1)} return · {formatPercent(account.annualVolatility, 1)} vol
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="tabular text-sm font-semibold text-foreground">{formatCLP(account.balance)}</p>
                          <p className="text-xs text-muted-foreground">{formatPercent(account.share, 0)} of balance</p>
                        </div>
                      </div>
                      <div className="mt-2 h-2 overflow-hidden rounded-full bg-background">
                        <div
                          className="h-full rounded-full bg-primary"
                          style={{ width: `${Math.max(account.share * 100, 6)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <EmptyPanel
              title="No portfolio snapshot yet"
              message="Create a situation with income and account balances to unlock a financial baseline."
              ctaLabel="Add a situation"
              ctaHref="/profiles"
            />
          )}
        </Card>

        <Card className="p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold text-foreground">Plan health</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Track where your plans stand before you open each result in detail.
              </p>
            </div>
            <Target className="h-5 w-5 text-muted-foreground" />
          </div>

          {ownScenarios.length > 0 ? (
            <div className="mt-6 space-y-5">
              <div className="grid grid-cols-2 gap-3">
                <HealthMetric label="On track" value={healthCounts.on_track} tone="positive" />
                <HealthMetric label="Tight" value={healthCounts.tight} tone="warning" />
                <HealthMetric label="Needs changes" value={healthCounts.needs_changes + healthCounts.failed} tone="danger" />
                <HealthMetric label="Pending or running" value={activeRuns} tone="active" />
              </div>

              <div className="rounded-2xl bg-muted/50 p-4">
                <p className="text-sm font-medium text-foreground">Next milestone</p>
                {nextMilestonePlan?.latestResult?.optimal_horizon ? (
                  <>
                    <p className="mt-2 text-2xl font-semibold text-foreground">
                      {formatMonthsLong(nextMilestonePlan.latestResult.optimal_horizon)}
                    </p>
                    <p className="mt-2 text-sm text-muted-foreground">
                      {nextMilestonePlan.scenario.name} is currently the closest plan with a visible result.
                    </p>
                  </>
                ) : (
                  <p className="mt-2 text-sm text-muted-foreground">
                    Run a plan to surface a likely time horizon here.
                  </p>
                )}
              </div>
            </div>
          ) : (
            <EmptyPanel
              title="No plans to evaluate yet"
              message="Create a plan and run FinOpt to start comparing what is on track and what needs review."
              ctaLabel="Build a plan"
              ctaHref="/scenarios"
            />
          )}
        </Card>
      </div>

      <div className="flex flex-wrap gap-3">
        <Button asChild className="rounded-xl">
          <Link to="/profiles">
            <Plus className="h-4 w-4" />
            Add a situation
          </Link>
        </Button>
        <Button asChild variant="outline" className="rounded-xl">
          <Link to="/scenarios">
            <Target className="h-4 w-4" />
            Build a plan
          </Link>
        </Button>
      </div>

      <Card className="overflow-hidden">
        <div className="flex items-center gap-2 border-b border-border px-6 py-4">
          <FolderKanban className="h-4 w-4 text-muted-foreground" />
          <div>
            <h2 className="text-base font-semibold text-foreground">Recent results and runs</h2>
            <p className="text-sm text-muted-foreground">
              Review the latest optimizer activity and remove runs you no longer need.
            </p>
          </div>
        </div>
        <div className="divide-y divide-border">
          {jobsLoading ? (
            <div className="flex items-center gap-2 px-6 py-5 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading your latest runs…
            </div>
          ) : recentJobs.length === 0 ? (
            <div className="px-6 py-10 text-center">
              <p className="text-sm text-muted-foreground">
                No runs yet. Create a plan and launch a calculation to start seeing results here.
              </p>
              <Button asChild variant="outline" size="sm" className="mt-4 rounded-xl">
                <Link to="/scenarios">
                  <Target className="h-4 w-4" />
                  Create your first plan
                </Link>
              </Button>
            </div>
          ) : (
            recentJobs.map((job) => {
              const result = recentResultsByJobId[job.id]
              const health = getPlanHealth(job, result)
              const status = recentRunStatusCopy(health, result)
              const isDeleting = deleteJobMutation.isPending && deleteJobMutation.variables === job.id
              const scenario = job.scenarios ?? scenariosById[job.scenario_id] ?? null
              const goals = summarizeGoalStatus(result?.goal_status)

              return (
                <div
                  key={job.id}
                  className="flex flex-col gap-4 px-6 py-5 transition-colors hover:bg-muted/40 lg:flex-row lg:items-center lg:justify-between"
                >
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="truncate font-medium text-foreground">
                        {scenario?.name ?? 'Untitled plan'}
                      </p>
                      <StatusPill tone={status.tone}>{status.label}</StatusPill>
                    </div>
                    <p className="mt-1 truncate text-sm text-muted-foreground">
                      {scenario?.profiles?.name ?? 'Unknown situation'} · {formatJobTimestamp(job)}
                    </p>
                    <p className="mt-2 text-sm text-muted-foreground">{status.summary}</p>
                    {(result?.optimal_horizon || goals.total > 0) && (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {goals.total > 0 && (
                          <InlineMetric label="Goals met" value={`${goals.met}/${goals.total}`} />
                        )}
                        {result?.optimal_horizon && (
                          <InlineMetric label="Horizon" value={formatMonthsLong(result.optimal_horizon)} />
                        )}
                      </div>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center gap-2 lg:justify-end">
                    {(job.status === 'running' || job.status === 'pending') && (
                      <span className="rounded-full bg-accent px-3 py-1 text-xs font-medium text-accent-foreground">
                        {job.status === 'running' ? `${job.progress}% complete` : 'Queued'}
                      </span>
                    )}
                    <Button asChild variant="outline" size="sm" className="rounded-xl">
                      <Link to={`/results/${job.id}`}>
                        {job.status === 'completed' ? 'View result' : 'Open run'}
                        <ArrowRight className="h-3.5 w-3.5" />
                      </Link>
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="rounded-xl border-danger/30 text-danger hover:bg-danger-soft hover:text-danger"
                      disabled={isDeleting}
                      onClick={() => {
                        const itemLabel = job.status === 'completed' ? 'result' : 'run'
                        if (confirm(`Delete this ${itemLabel}? This action cannot be undone.`)) {
                          deleteJobMutation.mutate(job.id)
                        }
                      }}
                    >
                      {isDeleting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                      Delete
                    </Button>
                  </div>
                </div>
              )
            })
          )}
        </div>
      </Card>
    </div>
  )
}

function planHealthPriority(health: PlanHealth) {
  const priorities: Record<PlanHealth, number> = {
    needs_changes: 0,
    tight: 1,
    failed: 2,
    running: 3,
    queued: 4,
    completed: 5,
    on_track: 6,
    draft: 7,
  }
  return priorities[health]
}

function dashboardLabel(health: PlanHealth) {
  switch (health) {
    case 'needs_changes':
      return 'Needs changes'
    case 'tight':
      return 'Tight plan'
    case 'failed':
      return 'Failed'
    case 'running':
      return 'Calculating'
    case 'queued':
      return 'Queued'
    case 'completed':
      return 'Completed'
    case 'on_track':
      return 'On track'
    default:
      return 'Needs run'
  }
}

function dashboardTone(health: PlanHealth): keyof typeof toneStyles {
  switch (health) {
    case 'needs_changes':
    case 'failed':
      return 'danger'
    case 'tight':
      return 'warning'
    case 'running':
    case 'queued':
      return 'active'
    case 'completed':
      return 'neutral'
    case 'on_track':
      return 'positive'
    default:
      return 'neutral'
  }
}

function priorityPlanSummary(
  health: PlanHealth,
  result: ResultPreviewLite | undefined,
  goals: { met: number; total: number },
) {
  if (health === 'needs_changes') {
    return 'This plan is currently outside reach and likely needs adjusted contributions, goals, or timing.'
  }
  if (health === 'tight') {
    return result?.optimal_horizon
      ? `It may work in about ${formatMonthsLong(result.optimal_horizon)}, but the margin is still tight.`
      : 'This plan may work, but it still has little room for error.'
  }
  if (health === 'running' || health === 'queued') {
    return 'This is the most recent active run and should give you fresh guidance soon.'
  }
  if (health === 'completed') {
    return 'The latest run finished, but the saved result summary is not available yet.'
  }
  if (health === 'on_track') {
    return result?.optimal_horizon
      ? `${goals.met}/${goals.total || goals.met} goals are currently satisfied with an estimated horizon of ${formatMonthsLong(result.optimal_horizon)}.`
      : 'This completed plan currently looks achievable.'
  }
  if (health === 'failed') {
    return 'The last calculation failed, so this plan is the best candidate to revisit next.'
  }
  return 'Run this plan to surface a realistic timeline.'
}

function recentRunStatusCopy(health: PlanHealth, result?: ResultPreviewLite) {
  if (health === 'running') {
    return {
      label: 'Calculating',
      tone: 'active' as const,
      summary: 'FinOpt is simulating paths and optimizing allocations for this plan.',
    }
  }
  if (health === 'queued') {
    return {
      label: 'Queued',
      tone: 'neutral' as const,
      summary: 'This run is waiting to start.',
    }
  }
  if (health === 'completed') {
    return {
      label: 'Completed',
      tone: 'neutral' as const,
      summary: 'The run finished, but its saved result summary is not available here yet.',
    }
  }
  if (health === 'failed') {
    return {
      label: 'Failed',
      tone: 'danger' as const,
      summary: 'This run did not finish. Review the inputs and try again.',
    }
  }
  if (health === 'needs_changes') {
    return {
      label: 'Needs changes',
      tone: 'danger' as const,
      summary: 'The current inputs are not enough to reach every goal.',
    }
  }
  if (health === 'tight') {
    return {
      label: 'Tight plan',
      tone: 'warning' as const,
      summary: result?.optimal_horizon
        ? `Goals are close, but the plan still looks tight over ${formatMonthsLong(result.optimal_horizon)}.`
        : 'Goals are close, but this plan still has little margin.',
    }
  }
  return {
    label: 'On track',
    tone: 'positive' as const,
    summary: result?.optimal_horizon
      ? `All goals are currently achievable in about ${formatMonthsLong(result.optimal_horizon)}.`
      : 'The run finished successfully.',
  }
}

function formatJobTimestamp(job: Pick<Job, 'status' | 'created_at' | 'completed_at'>) {
  if (job.status === 'completed' && job.completed_at) {
    return `Completed ${formatDateShort(job.completed_at)}`
  }
  return `Started ${formatDateShort(job.created_at)}`
}

function SnapshotMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-muted/60 px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-base font-semibold tabular text-foreground">{value}</p>
    </div>
  )
}

function HealthMetric({
  label,
  value,
  tone,
}: {
  label: string
  value: number
  tone: keyof typeof toneStyles
}) {
  return (
    <div className="rounded-2xl border border-border bg-card px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-medium text-foreground">{label}</p>
        <StatusPill tone={tone}>{value}</StatusPill>
      </div>
    </div>
  )
}

function StatusPill({
  tone,
  children,
}: {
  tone: keyof typeof toneStyles
  children: ReactNode
}) {
  return (
    <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${toneStyles[tone]}`}>
      {children}
    </span>
  )
}

function EmptyPanel({
  title,
  message,
  ctaLabel,
  ctaHref,
}: {
  title: string
  message: string
  ctaLabel: string
  ctaHref: string
}) {
  return (
    <div className="mt-6 rounded-2xl border border-dashed border-border bg-muted/30 p-5">
      <p className="font-medium text-foreground">{title}</p>
      <p className="mt-2 text-sm text-muted-foreground">{message}</p>
      <Button asChild variant="outline" size="sm" className="mt-4 rounded-xl">
        <Link to={ctaHref}>{ctaLabel}</Link>
      </Button>
    </div>
  )
}

function InlineMetric({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-muted-foreground">
      {label}: <span className="text-foreground">{value}</span>
    </span>
  )
}

function relationToOne<T>(value: T[] | T | null | undefined): T | null {
  if (!value) return null
  return Array.isArray(value) ? value[0] ?? null : value
}
