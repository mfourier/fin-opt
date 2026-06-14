import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  ArrowRight,
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
import type { Job, Profile, Result, Scenario } from '../types/database'
import { formatCLP, formatDateShort, formatMonthsLong } from '../lib/format'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

type DashboardScenario = Scenario & { profiles: Pick<Profile, 'name'> }
type DashboardJob = Job & { scenarios: DashboardScenario | null }
type ResultPreview = Pick<Result, 'job_id' | 'optimal_horizon' | 'feasible' | 'goal_status'>

const statusStyles = {
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
        .select('id, profile_id, is_demo')
      if (error) throw error
      return data as Pick<Scenario, 'id' | 'profile_id' | 'is_demo'>[]
    },
    enabled: !!user,
  })

  const { data: recentJobs, isLoading: jobsLoading } = useQuery({
    queryKey: ['recent-jobs', user?.id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('jobs')
        .select('*, scenarios(name, profiles(name))')
        .order('created_at', { ascending: false })
        .limit(10)
      if (error) throw error
      return data as DashboardJob[]
    },
    enabled: !!user,
  })

  const recentJobIds = recentJobs?.map((job) => job.id) ?? []

  const { data: recentResults } = useQuery({
    queryKey: ['dashboard-results', recentJobIds],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('results')
        .select('job_id, optimal_horizon, feasible, goal_status')
        .in('job_id', recentJobIds)
      if (error) throw error
      return data as ResultPreview[]
    },
    enabled: recentJobIds.length > 0,
  })

  const deleteJobMutation = useMutation({
    mutationFn: async (jobId: string) => {
      const { error } = await supabase.from('jobs').delete().eq('id', jobId)
      if (error) throw error
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['recent-jobs'] })
      queryClient.invalidateQueries({ queryKey: ['dashboard-results'] })
      toast.success('Result deleted', 'The run and its saved result were removed from your dashboard.')
    },
    onError: (error: Error) => {
      toast.error('Failed to delete result', error.message)
    },
  })

  const ownProfiles = profiles?.filter((profile) => !profile.is_demo) ?? []
  const latestSituation = ownProfiles[0] ?? profiles?.[0] ?? null
  const totalBalance = latestSituation ? getStartingBalance(latestSituation) : null
  const monthlyIncome = latestSituation ? getMonthlyIncome(latestSituation) : null
  const activeRuns = recentJobs?.filter((job) => job.status === 'running' || job.status === 'pending').length ?? 0
  const totalPlans = scenarios?.filter((scenario) => !scenario.is_demo).length ?? 0
  const resultsByJobId = Object.fromEntries((recentResults ?? []).map((result) => [result.job_id, result]))
  const latestCompletedJob = recentJobs?.find((job) => job.status === 'completed') ?? null
  const latestCompletedResult = latestCompletedJob ? resultsByJobId[latestCompletedJob.id] : undefined

  const statCards = [
    {
      label: 'Starting balance',
      value: profilesLoading ? '—' : totalBalance === null ? '—' : formatCLP(totalBalance),
      detail: latestSituation ? latestSituation.name : 'Create a situation to start planning.',
      icon: Wallet,
    },
    {
      label: 'Monthly income',
      value: profilesLoading ? '—' : monthlyIncome === null ? '—' : `${formatCLP(monthlyIncome)}/mo`,
      detail: latestSituation ? 'Fixed and variable income combined.' : 'Add income to unlock plan guidance.',
      icon: TrendingUp,
    },
    {
      label: 'Plans built',
      value: scenariosLoading ? '—' : String(totalPlans),
      detail: totalPlans > 0 ? 'Goal-based plans linked to your situations.' : 'No plans yet.',
      icon: Target,
    },
    {
      label: 'Active runs',
      value: jobsLoading ? '—' : String(activeRuns),
      detail: activeRuns > 0 ? 'Calculations currently in progress.' : 'No calculations running right now.',
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
              Keep an eye on your latest situation, active plan runs, and the results that matter most.
            </p>

            {latestSituation ? (
              <div className="mt-6 flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
                <span className="rounded-full bg-card px-3 py-1 font-medium text-foreground shadow-sm">
                  Primary situation: {latestSituation.name}
                </span>
                <span>{latestSituation.accounts_config.length} account{latestSituation.accounts_config.length === 1 ? '' : 's'}</span>
                <span>{formatCLP(getStartingBalance(latestSituation))} starting balance</span>
                <span>{formatCLP(getMonthlyIncome(latestSituation))}/mo income</span>
              </div>
            ) : (
              <div className="mt-6 rounded-2xl border border-dashed border-border bg-card/70 p-4 text-sm text-muted-foreground">
                Add a situation first so FinOpt can turn your income, assets, and goals into a plan.
              </div>
            )}
          </div>

          <div className="flex w-full max-w-md flex-col gap-3 rounded-2xl border border-border bg-card/90 p-4 shadow-sm">
            {latestCompletedJob && latestCompletedResult ? (
              <>
                <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Latest result
                </p>
                <div>
                  <p className="font-semibold text-foreground">
                    {latestCompletedJob.scenarios?.name ?? 'Untitled plan'}
                  </p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {dashboardStatusCopy(latestCompletedJob, latestCompletedResult).summary}
                  </p>
                </div>
                <Button asChild className="rounded-xl">
                  <Link to={`/results/${latestCompletedJob.id}`}>
                    View latest result
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
                    ? 'Create or run a plan to see when your goals become achievable.'
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
          ) : recentJobs?.length === 0 ? (
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
            (recentJobs ?? []).map((job) => {
              const result = resultsByJobId[job.id]
              const status = dashboardStatusCopy(job, result)
              const isDeleting = deleteJobMutation.isPending && deleteJobMutation.variables === job.id
              return (
                <div
                  key={job.id}
                  className="flex flex-col gap-4 px-6 py-5 transition-colors hover:bg-muted/40 lg:flex-row lg:items-center lg:justify-between"
                >
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="truncate font-medium text-foreground">
                        {job.scenarios?.name ?? 'Untitled plan'}
                      </p>
                      <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${status.badgeClass}`}>
                        {status.label}
                      </span>
                    </div>
                    <p className="mt-1 truncate text-sm text-muted-foreground">
                      {job.scenarios?.profiles?.name ?? 'Unknown situation'} · {formatJobTimestamp(job)}
                    </p>
                    <p className="mt-2 text-sm text-muted-foreground">{status.summary}</p>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 lg:justify-end">
                    {job.status === 'running' && (
                      <span className="rounded-full bg-accent px-3 py-1 text-xs font-medium text-accent-foreground">
                        {job.progress}% complete
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

function getStartingBalance(profile: Pick<Profile, 'accounts_config'>) {
  return profile.accounts_config.reduce((sum, account) => sum + (account.initial_wealth ?? 0), 0)
}

function getMonthlyIncome(profile: Pick<Profile, 'income_config'>) {
  const fixedIncome = profile.income_config.fixed?.base ?? 0
  const variableIncome = profile.income_config.variable?.base ?? 0
  return fixedIncome + variableIncome
}

function formatJobTimestamp(job: Pick<Job, 'status' | 'created_at' | 'completed_at'>) {
  if (job.status === 'completed' && job.completed_at) {
    return `Completed ${formatDateShort(job.completed_at)}`
  }
  return `Started ${formatDateShort(job.created_at)}`
}

function dashboardStatusCopy(job: Pick<Job, 'status'>, result?: ResultPreview) {
  if (job.status === 'running') {
    return {
      label: 'Calculating',
      badgeClass: statusStyles.active,
      summary: 'FinOpt is simulating paths and optimizing allocations for this plan.',
    }
  }

  if (job.status === 'pending') {
    return {
      label: 'Queued',
      badgeClass: statusStyles.neutral,
      summary: 'This run is waiting to start.',
    }
  }

  if (job.status === 'failed') {
    return {
      label: 'Failed',
      badgeClass: statusStyles.danger,
      summary: 'This run did not finish. Review the inputs and try again.',
    }
  }

  if (!result) {
    return {
      label: 'Completed',
      badgeClass: statusStyles.positive,
      summary: 'The run finished successfully.',
    }
  }

  const achievedAllGoals = (result.goal_status ?? []).every((goal) => goal.satisfied)

  if (result.feasible === false) {
    return {
      label: 'Needs changes',
      badgeClass: statusStyles.danger,
      summary: 'The current inputs are not enough to reach every goal.',
    }
  }

  if (!achievedAllGoals) {
    return {
      label: 'Tight plan',
      badgeClass: statusStyles.warning,
      summary: result.optimal_horizon
        ? `Goals are close, but the plan still looks tight over ${formatMonthsLong(result.optimal_horizon)}.`
        : 'Goals are close, but this plan still has little margin.',
    }
  }

  return {
    label: 'On track',
    badgeClass: statusStyles.positive,
    summary: result.optimal_horizon
      ? `All goals are currently achievable in about ${formatMonthsLong(result.optimal_horizon)}.`
      : 'All goals are currently achievable with this plan.',
  }
}
