import type { ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { Trans, useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
import {
  ArrowRight,
  BriefcaseBusiness,
  CheckCircle2,
  Circle,
  Clock3,
  Flag,
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
import { formatCLP, formatDateShort, formatMonthsLong, formatPercent, monthLabel } from '../lib/format'
import {
  type PlanHealth,
  type ResultPreviewLite,
  describeProfileRisk,
  getPlanHealth,
  getProfileMonthlyContributionCapacity,
  getProfileStartingBalance,
  getProfileTopAccounts,
  getProfileWeightedReturn,
  summarizeGoalStatus,
} from '../lib/finance'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

type DashboardScenario = Pick<Scenario, 'id' | 'name' | 'is_demo' | 'start_date'> & {
  profiles: Pick<Profile, 'name' | 'accounts_config'> | null
}
type DashboardJob = Pick<
  Job,
  'id' | 'scenario_id' | 'job_type' | 'status' | 'progress' | 'created_at' | 'completed_at'
> & {
  scenarios: DashboardScenario | null
}
type DashboardScenarioRow = Pick<Scenario, 'id' | 'name' | 'is_demo' | 'start_date'> & {
  profiles: Array<Pick<Profile, 'name' | 'accounts_config'>> | null
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
  const { t } = useTranslation(['dashboard', 'scenarios', 'common'])
  const queryClient = useQueryClient()
  const user = useAuthStore((state) => state.user)
  const toast = useToast()
  const perMonth = (value: string) => `${value}${t('common:perMonth')}`

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
        .select('id, name, is_demo, start_date, profiles(name, accounts_config)')
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
        .select('id, scenario_id, job_type, status, progress, created_at, completed_at, scenarios(id, name, is_demo, start_date, profiles(name, accounts_config))')
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
      toast.success(t('toast.deleted'), t('toast.deletedDetail'))
    },
    onError: (error: Error) => {
      toast.error(t('toast.deleteFailed'), error.message)
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

  const milestoneAccountNames = Object.fromEntries(
    (nextMilestonePlan?.scenario.profiles?.accounts_config ?? []).map((account) => [
      account.name,
      account.display_name ?? account.name,
    ]),
  ) as Record<string, string>

  const statCards = [
    {
      label: t('stats.netWorth'),
      value: profilesLoading ? '—' : formatCLP(totalNetWorth),
      detail:
        latestSituation
          ? t('stats.netWorthDetail', { name: latestSituation.name })
          : t('stats.netWorthEmpty'),
      icon: Wallet,
    },
    {
      label: t('stats.investingPower'),
      value: profilesLoading ? '—' : perMonth(formatCLP(totalMonthlyContribution)),
      detail:
        latestSituation
          ? t('stats.investingPowerDetail')
          : t('stats.investingPowerEmpty'),
      icon: TrendingUp,
    },
    {
      label: t('stats.plansOnTrack'),
      value: scenariosLoading || jobsLoading ? '—' : String(plansOnTrack),
      detail:
        plansOnTrack > 0
          ? t('stats.plansOnTrackDetail', { count: healthCounts.tight })
          : t('stats.plansOnTrackEmpty'),
      icon: Target,
    },
    {
      label: t('stats.needAttention'),
      value: scenariosLoading || jobsLoading ? '—' : String(plansNeedingReview),
      detail:
        activeRuns > 0
          ? t('stats.needAttentionDetail', { count: activeRuns })
          : t('stats.needAttentionEmpty'),
      icon: Clock3,
    },
  ]

  return (
    <div className="space-y-8">
      <Card className="overflow-hidden border-primary/10 bg-[radial-gradient(120%_120%_at_100%_0%,var(--color-accent)_0%,transparent_48%),linear-gradient(180deg,var(--color-card)_0%,color-mix(in_oklab,var(--color-card)_92%,var(--color-accent))_100%)] p-6 shadow-[0_1px_0_oklch(0_0_0/0.03),0_24px_48px_-24px_oklch(0.4_0.1_254/0.28)] sm:p-8">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <span className="inline-flex rounded-full bg-secondary px-3 py-1 text-xs font-medium text-secondary-foreground">
              {t('badge')}
            </span>
            <h1 className="mt-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
              {t('title')}
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground sm:text-base">
              {t('subtitle')}
            </p>

            {latestSituation ? (
              <div className="mt-6 flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
                <span className="rounded-full bg-card px-3 py-1 font-medium text-foreground shadow-sm">
                  {t('primarySituation', { name: latestSituation.name })}
                </span>
                <span>{t('startingBalanceInline', { amount: formatCLP(getProfileStartingBalance(latestSituation)) })}</span>
                <span>{t('investingPowerInline', { amount: perMonth(formatCLP(getProfileMonthlyContributionCapacity(latestSituation))) })}</span>
                <span>{t('expectedReturnInline', { pct: formatPercent(aggregatedReturn, 1) })}</span>
                <span>{t(`common:profileRisk.${describeProfileRisk(latestSituation)}`)}</span>
              </div>
            ) : (
              <div className="mt-6 rounded-2xl border border-dashed border-border bg-card/70 p-4 text-sm text-muted-foreground">
                {t('addSituationFirst')}
              </div>
            )}
          </div>

          <div className="flex w-full max-w-md flex-col gap-3 rounded-2xl border border-border bg-card/90 p-4 shadow-sm">
            {priorityPlan?.latestJob ? (
              <>
                <div className="flex items-center justify-between gap-3">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    {t('priorityPlan')}
                  </p>
                  <StatusPill tone={dashboardTone(priorityPlan.health)}>
                    {dashboardLabel(priorityPlan.health, t)}
                  </StatusPill>
                </div>
                <div>
                  <p className="font-semibold text-foreground">{priorityPlan.scenario.name}</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {priorityPlan.scenario.profiles?.name ?? t('unknownSituation')}
                  </p>
                </div>
                <p className="text-sm text-muted-foreground">
                  {priorityPlanSummary(priorityPlan.health, priorityPlan.latestResult, priorityPlan.goals, t)}
                </p>
                <Button asChild className="rounded-xl">
                  <Link to={`/results/${priorityPlan.latestJob.id}`}>
                    {priorityPlan.latestJob.status === 'completed' ? t('openPriorityPlan') : t('openActiveRun')}
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
              </>
            ) : (
              <>
                <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  {t('nextBestStep')}
                </p>
                <p className="text-sm text-muted-foreground">
                  {latestSituation
                    ? t('nextStepWithSituation')
                    : t('nextStepNoSituation')}
                </p>
                <div className="flex flex-wrap gap-2">
                  <Button asChild className="rounded-xl">
                    <Link to="/profiles">
                      <Plus className="h-4 w-4" />
                      {t('addSituation')}
                    </Link>
                  </Button>
                  <Button asChild variant="outline" className="rounded-xl">
                    <Link to="/scenarios">
                      <Target className="h-4 w-4" />
                      {t('newPlan')}
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
              <h2 className="text-lg font-semibold text-foreground">{t('portfolio.title')}</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                {t('portfolio.subtitle')}
              </p>
            </div>
            <BriefcaseBusiness className="h-5 w-5 text-muted-foreground" />
          </div>

          {latestSituation ? (
            <div className="mt-6 space-y-5">
              <div className="grid gap-3 sm:grid-cols-2">
                <SnapshotMetric label={t('portfolio.startingBalance')} value={formatCLP(getProfileStartingBalance(latestSituation))} />
                <SnapshotMetric label={t('portfolio.monthlyInvestment')} value={perMonth(formatCLP(getProfileMonthlyContributionCapacity(latestSituation)))} />
                <SnapshotMetric label={t('portfolio.expectedReturn')} value={formatPercent(getProfileWeightedReturn(latestSituation), 1)} />
              </div>
              <p className="text-sm text-muted-foreground">
                <Trans
                  i18nKey="portfolio.basedOn"
                  t={t}
                  values={{ name: latestSituation.name }}
                  components={{ name: <span className="font-medium text-foreground" /> }}
                />
              </p>

              <div className="rounded-2xl bg-muted/50 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">{t('portfolio.topAccounts')}</p>
                    <p className="text-sm text-muted-foreground">
                      {t('portfolio.topAccountsSubtitle', { name: latestSituation.name })}
                    </p>
                  </div>
                  <span className="rounded-full bg-secondary px-2.5 py-1 text-xs font-medium text-secondary-foreground">
                    {t(`common:profileRisk.${describeProfileRisk(latestSituation)}`)}
                  </span>
                </div>
                <div className="mt-4 space-y-3">
                  {getProfileTopAccounts(latestSituation).map((account) => (
                    <div key={account.id}>
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium text-foreground">{account.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {t('portfolio.returnVol', { ret: formatPercent(account.annualReturn, 1), vol: formatPercent(account.annualVolatility, 1) })}
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="tabular text-sm font-semibold text-foreground">{formatCLP(account.balance)}</p>
                          <p className="text-xs text-muted-foreground">{t('portfolio.ofBalance', { pct: formatPercent(account.share, 0) })}</p>
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
              title={t('portfolio.emptyTitle')}
              message={t('portfolio.emptyMessage')}
              ctaLabel={t('portfolio.emptyCta')}
              ctaHref="/profiles"
            />
          )}
        </Card>

        <Card className="p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold text-foreground">{t('health.nextMilestone')}</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                {t('milestone.subtitle')}
              </p>
            </div>
            <div className="flex items-center gap-2">
              {nextMilestonePlan?.latestResult?.optimal_horizon && (
                <StatusPill tone={dashboardTone(nextMilestonePlan.health)}>
                  {dashboardLabel(nextMilestonePlan.health, t)}
                </StatusPill>
              )}
              <Flag className="h-5 w-5 text-muted-foreground" />
            </div>
          </div>

          {nextMilestonePlan?.latestResult?.optimal_horizon ? (
            <div className="mt-6 space-y-4">
              <div className="rounded-2xl bg-muted/50 p-5">
                <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  {t('milestone.estimatedHorizon')}
                </p>
                <p className="mt-2 text-3xl font-semibold text-foreground">
                  {formatMonthsLong(nextMilestonePlan.latestResult.optimal_horizon)}
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  {t('health.milestoneClosest', { name: nextMilestonePlan.scenario.name })}
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <InlineMetric
                    label={t('milestone.targetDate')}
                    value={monthLabel(nextMilestonePlan.latestResult.optimal_horizon, nextMilestonePlan.scenario.start_date)}
                  />
                  {nextMilestonePlan.goals.total > 0 && (
                    <InlineMetric
                      label={t('recent.goalsMet')}
                      value={`${nextMilestonePlan.goals.met}/${nextMilestonePlan.goals.total}`}
                    />
                  )}
                </div>
              </div>
              {nextMilestonePlan.latestResult.goal_status && nextMilestonePlan.latestResult.goal_status.length > 0 && (
                <div className="rounded-2xl border border-border p-4">
                  <p className="text-sm font-medium text-foreground">{t('milestone.goalsTitle')}</p>
                  <ul className="mt-3 space-y-2.5">
                    {nextMilestonePlan.latestResult.goal_status.map((goal, index) => (
                      <li key={index} className="flex items-center justify-between gap-3 text-sm">
                        <span className="flex min-w-0 items-center gap-2">
                          {goal.satisfied ? (
                            <CheckCircle2 className="h-4 w-4 shrink-0 text-success" />
                          ) : (
                            <Circle className="h-4 w-4 shrink-0 text-muted-foreground" />
                          )}
                          <span className="truncate text-foreground">{milestoneAccountNames[goal.account] ?? goal.account}</span>
                          <span className="shrink-0 rounded-full bg-secondary px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                            {goal.type === 'terminal' ? t('milestone.typeTerminal') : t('milestone.typeIntermediate')}
                          </span>
                        </span>
                        <span className="shrink-0 tabular text-muted-foreground">{formatCLP(goal.threshold)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {nextMilestonePlan.latestJob && (
                <Button asChild variant="outline" className="rounded-xl">
                  <Link to={`/results/${nextMilestonePlan.latestJob.id}`}>
                    {t('recent.viewResult')}
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
              )}
            </div>
          ) : (
            <EmptyPanel
              title={t('health.emptyTitle')}
              message={t('health.milestoneEmpty')}
              ctaLabel={t('health.emptyCta')}
              ctaHref="/scenarios"
            />
          )}
        </Card>
      </div>

      <div className="flex flex-wrap gap-3">
        <Button asChild className="rounded-xl">
          <Link to="/profiles">
            <Plus className="h-4 w-4" />
            {t('addSituation')}
          </Link>
        </Button>
        <Button asChild variant="outline" className="rounded-xl">
          <Link to="/scenarios">
            <Target className="h-4 w-4" />
            {t('buildPlan')}
          </Link>
        </Button>
      </div>

      <Card className="overflow-hidden">
        <div className="flex items-center gap-2 border-b border-border px-6 py-4">
          <FolderKanban className="h-4 w-4 text-muted-foreground" />
          <div>
            <h2 className="text-base font-semibold text-foreground">{t('recent.title')}</h2>
            <p className="text-sm text-muted-foreground">
              {t('recent.subtitle')}
            </p>
          </div>
        </div>
        <div className="divide-y divide-border">
          {jobsLoading ? (
            <div className="flex items-center gap-2 px-6 py-5 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              {t('recent.loading')}
            </div>
          ) : recentJobs.length === 0 ? (
            <div className="px-6 py-10 text-center">
              <p className="text-sm text-muted-foreground">
                {t('recent.empty')}
              </p>
              <Button asChild variant="outline" size="sm" className="mt-4 rounded-xl">
                <Link to="/scenarios">
                  <Target className="h-4 w-4" />
                  {t('recent.createFirst')}
                </Link>
              </Button>
            </div>
          ) : (
            recentJobs.map((job) => {
              const result = recentResultsByJobId[job.id]
              const health = getPlanHealth(job, result)
              const status = recentRunStatusCopy(health, t, result)
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
                        {scenario?.name ?? t('recent.untitled')}
                      </p>
                      <StatusPill tone={status.tone}>{status.label}</StatusPill>
                    </div>
                    <p className="mt-1 truncate text-sm text-muted-foreground">
                      {scenario?.profiles?.name ?? t('unknownSituation')} · {formatJobTimestamp(job, t)}
                    </p>
                    <p className="mt-2 text-sm text-muted-foreground">{status.summary}</p>
                    {(result?.optimal_horizon || goals.total > 0) && (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {goals.total > 0 && (
                          <InlineMetric label={t('recent.goalsMet')} value={`${goals.met}/${goals.total}`} />
                        )}
                        {result?.optimal_horizon && (
                          <InlineMetric label={t('recent.horizon')} value={formatMonthsLong(result.optimal_horizon)} />
                        )}
                      </div>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center gap-2 lg:justify-end">
                    {(job.status === 'running' || job.status === 'pending') && (
                      <span className="rounded-full bg-accent px-3 py-1 text-xs font-medium text-accent-foreground">
                        {job.status === 'running' ? t('recent.completeProgress', { pct: job.progress }) : t('recent.queued')}
                      </span>
                    )}
                    <Button asChild variant="outline" size="sm" className="rounded-xl">
                      <Link to={`/results/${job.id}`}>
                        {job.status === 'completed' ? t('recent.viewResult') : t('recent.openRun')}
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
                        const message = job.status === 'completed' ? t('recent.confirmDeleteResult') : t('recent.confirmDeleteRun')
                        if (confirm(message)) {
                          deleteJobMutation.mutate(job.id)
                        }
                      }}
                    >
                      {isDeleting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                      {t('recent.delete')}
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

function dashboardLabel(health: PlanHealth, t: TFunction) {
  switch (health) {
    case 'needs_changes':
      return t('scenarios:status.needsChanges')
    case 'tight':
      return t('scenarios:status.tight')
    case 'failed':
      return t('scenarios:status.failed')
    case 'running':
      return t('scenarios:status.calculating')
    case 'queued':
      return t('scenarios:status.queued')
    case 'completed':
      return t('scenarios:status.completed')
    case 'on_track':
      return t('scenarios:status.onTrack')
    default:
      return t('scenarios:status.needsRun')
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
  t: TFunction,
) {
  if (health === 'needs_changes') {
    return t('prioritySummary.needsChanges')
  }
  if (health === 'tight') {
    return result?.optimal_horizon
      ? t('prioritySummary.tightHorizon', { horizon: formatMonthsLong(result.optimal_horizon) })
      : t('prioritySummary.tight')
  }
  if (health === 'running' || health === 'queued') {
    return t('prioritySummary.running')
  }
  if (health === 'completed') {
    return t('prioritySummary.completed')
  }
  if (health === 'on_track') {
    return result?.optimal_horizon
      ? t('prioritySummary.onTrackHorizon', {
          met: goals.met,
          total: goals.total || goals.met,
          horizon: formatMonthsLong(result.optimal_horizon),
        })
      : t('prioritySummary.onTrack')
  }
  if (health === 'failed') {
    return t('prioritySummary.failed')
  }
  return t('prioritySummary.default')
}

function recentRunStatusCopy(health: PlanHealth, t: TFunction, result?: ResultPreviewLite) {
  if (health === 'running') {
    return {
      label: t('scenarios:status.calculating'),
      tone: 'active' as const,
      summary: t('runSummary.running'),
    }
  }
  if (health === 'queued') {
    return {
      label: t('scenarios:status.queued'),
      tone: 'neutral' as const,
      summary: t('runSummary.queued'),
    }
  }
  if (health === 'completed') {
    return {
      label: t('scenarios:status.completed'),
      tone: 'neutral' as const,
      summary: t('runSummary.completed'),
    }
  }
  if (health === 'failed') {
    return {
      label: t('scenarios:status.failed'),
      tone: 'danger' as const,
      summary: t('runSummary.failed'),
    }
  }
  if (health === 'needs_changes') {
    return {
      label: t('scenarios:status.needsChanges'),
      tone: 'danger' as const,
      summary: t('runSummary.needsChanges'),
    }
  }
  if (health === 'tight') {
    return {
      label: t('scenarios:status.tight'),
      tone: 'warning' as const,
      summary: result?.optimal_horizon
        ? t('runSummary.tightHorizon', { horizon: formatMonthsLong(result.optimal_horizon) })
        : t('runSummary.tight'),
    }
  }
  return {
    label: t('scenarios:status.onTrack'),
    tone: 'positive' as const,
    summary: result?.optimal_horizon
      ? t('runSummary.onTrackHorizon', { horizon: formatMonthsLong(result.optimal_horizon) })
      : t('runSummary.onTrack'),
  }
}

function formatJobTimestamp(job: Pick<Job, 'status' | 'created_at' | 'completed_at'>, t: TFunction) {
  if (job.status === 'completed' && job.completed_at) {
    return t('timestamp.completed', { date: formatDateShort(job.completed_at) })
  }
  return t('timestamp.started', { date: formatDateShort(job.created_at) })
}

function SnapshotMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-muted/60 px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-base font-semibold tabular text-foreground">{value}</p>
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
