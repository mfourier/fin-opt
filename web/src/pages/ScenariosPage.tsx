import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
import { ArrowRight, BriefcaseBusiness, PlayCircle, Plus, Target, TrendingUp } from 'lucide-react'
import { supabase } from '../lib/supabase'
import { useAuthStore } from '../lib/store'
import { queueOptimization } from '../lib/api'
import { useToast } from '../components/Toast'
import { GoalsWizard } from '@/components/finopt/GoalsWizard'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { formatDateShort, formatMonthYear, formatMonthsLong } from '@/lib/format'
import { type PlanHealth, getPlanHealth, summarizeGoalStatus } from '@/lib/finance'
import type { Job, Profile, Result, Scenario, ScenarioInsert } from '../types/database'
import type { ScenarioDraft } from '@/mocks/types'

// Plain-language objective labels live in `common:objectives.<id>.title`.

// Hidden optimization defaults — the user never sets these. The bracketed
// horizon search infers the horizon, so T_min/T_max are just a safe cap.
// t_max is capped at 180 (15y): the optimizer's accumulation-factor tensor is
// O(n_sims * T^2 * M), so a larger ceiling risks OOM on small instances if a
// plan ever needs a long horizon (e.g. ~1.5GB at T=360, n_sims=500). Plans
// needing >15y fail gracefully (infeasible) instead of crashing the worker.
const HIDDEN_DEFAULTS = {
  n_sims: 500,
  seed: 42,
  t_max: 180,
  solver: 'ECOS',
} as const

export default function ScenariosPage() {
  const { t } = useTranslation(['scenarios', 'common'])
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const user = useAuthStore((state) => state.user)
  const toast = useToast()
  const objectiveLabel = (objective: string) =>
    t(`common:objectives.${objective}.title`, { defaultValue: objective })
  const formatGoals = (terminal: number, dated: number) =>
    t('metrics.goalsTerminal', { count: terminal }) +
    (dated > 0 ? ` · ${t('metrics.goalsDated', { count: dated })}` : '')
  const [showForm, setShowForm] = useState(false)
  const [editingScenario, setEditingScenario] = useState<Scenario | null>(null)

  const { data: profiles } = useQuery({
    queryKey: ['profiles', user?.id],
    queryFn: async () => {
      const { data, error } = await supabase.from('profiles').select('*').order('created_at', { ascending: false })
      if (error) throw error
      return data as Profile[]
    },
    enabled: !!user,
  })

  // The shared demo profile is read-only (owned by no user), so it can't back a
  // new plan — keep it out of the wizard's profile picker and the empty-state checks.
  const ownProfiles = profiles?.filter((p) => !p.is_demo)

  const { data: scenarios, isLoading } = useQuery({
    queryKey: ['scenarios', user?.id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('scenarios')
        .select('*, profiles(name)')
        .order('created_at', { ascending: false })
      if (error) throw error
      return data as (Scenario & { profiles: { name: string } })[]
    },
    enabled: !!user,
  })

  const { data: jobs } = useQuery({
    queryKey: ['scenario-jobs', user?.id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('jobs')
        .select('id, scenario_id, status, progress, created_at, completed_at')
        .order('created_at', { ascending: false })
      if (error) throw error
      return data as Pick<Job, 'id' | 'scenario_id' | 'status' | 'progress' | 'created_at' | 'completed_at'>[]
    },
    enabled: !!user,
  })

  const latestJobsByScenarioId = Object.fromEntries(
    (jobs ?? []).reduce<[string, Pick<Job, 'id' | 'scenario_id' | 'status' | 'progress' | 'created_at' | 'completed_at'>][]>((entries, job) => {
      if (!entries.find(([scenarioId]) => scenarioId === job.scenario_id)) {
        entries.push([job.scenario_id, job])
      }
      return entries
    }, []),
  )

  const latestCompletedJobIds = Object.values(latestJobsByScenarioId)
    .filter((job) => job.status === 'completed')
    .map((job) => job.id)

  const { data: latestResults } = useQuery({
    queryKey: ['scenario-latest-results', latestCompletedJobIds],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('results')
        .select('job_id, feasible, optimal_horizon, goal_status')
        .in('job_id', latestCompletedJobIds)
      if (error) throw error
      return data as Pick<Result, 'job_id' | 'feasible' | 'optimal_horizon' | 'goal_status'>[]
    },
    enabled: latestCompletedJobIds.length > 0,
  })

  const latestResultsByJobId = Object.fromEntries((latestResults ?? []).map((result) => [result.job_id, result]))

  const createMutation = useMutation({
    mutationFn: async (scenario: ScenarioInsert) => {
      const { data, error } = await supabase.from('scenarios').insert(scenario).select().single()
      if (error) throw error
      return data as Scenario
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] })
    },
    onError: (error: Error) => {
      toast.error(t('toast.createFailed'), error.message)
    },
  })

  const updateMutation = useMutation({
    mutationFn: async ({ id, ...scenario }: Partial<Scenario> & { id: string }) => {
      const { data, error } = await supabase
        .from('scenarios')
        .update(scenario)
        .eq('id', id)
        .select()
        .single()
      if (error) throw error
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] })
    },
    onError: (error: Error) => {
      toast.error(t('toast.updateFailed'), error.message)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase.from('scenarios').delete().eq('id', id)
      if (error) throw error
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] })
      toast.success(t('toast.deleted'), t('toast.deletedDetail'))
    },
    onError: (error: Error) => {
      toast.error(t('toast.deleteFailed'), error.message)
    },
  })

  const runOptimization = async (scenarioId: string) => {
    const { data: job, error: jobError } = await supabase
      .from('jobs')
      .insert({ scenario_id: scenarioId, job_type: 'optimization', status: 'pending', progress: 0 })
      .select()
      .single()

    if (jobError) {
      toast.error(t('toast.jobFailed'), jobError.message)
      return
    }

    try {
      await queueOptimization({ scenario_id: scenarioId, job_id: job.id })
      queryClient.invalidateQueries({ queryKey: ['recent-jobs'] })
      toast.info(t('toast.calculating'), t('toast.calculatingDetail'))
      navigate(`/results/${job.id}`)
    } catch (err) {
      await supabase
        .from('jobs')
        .update({
          status: 'failed',
          error_message: err instanceof Error ? err.message : t('toast.reachFailed'),
        })
        .eq('id', job.id)
      toast.error(t('toast.startFailed'), err instanceof Error ? err.message : t('toast.unknownError'))
    }
  }

  // Demo plans are read-only and pre-computed: jump straight to their existing
  // completed job instead of queuing a new (RLS-blocked) optimization.
  const viewResults = async (scenarioId: string) => {
    const { data, error } = await supabase
      .from('jobs')
      .select('id')
      .eq('scenario_id', scenarioId)
      .eq('status', 'completed')
      .order('created_at', { ascending: false })
      .limit(1)
      .maybeSingle()
    if (error || !data) {
      toast.error(t('toast.noResults'), t('toast.noResultsDetail'))
      return
    }
    navigate(`/results/${data.id}`)
  }

  const resetForm = () => {
    setShowForm(false)
    setEditingScenario(null)
  }

  // Close the wizard overlay with the Escape key.
  useEffect(() => {
    if (!showForm) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') resetForm()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [showForm])

  const handleEdit = (scenario: Scenario) => {
    setEditingScenario(scenario)
    setShowForm(true)
  }

  // Wizard "Calculate my plan": create/update the scenario (filling the hidden
  // optimization defaults) and immediately kick off the optimization.
  const handleCalculate = async (draft: ScenarioDraft) => {
    const scenarioData: ScenarioInsert = {
      profile_id: draft.profile_id,
      name: draft.name,
      description: draft.description,
      start_date: draft.start_date,
      objective: draft.objective,
      terminal_goals: draft.terminal_goals,
      intermediate_goals: draft.intermediate_goals,
      withdrawals: draft.withdrawals,
      ...HIDDEN_DEFAULTS,
    }

    try {
      let scenarioId: string
      if (editingScenario) {
        await updateMutation.mutateAsync({ id: editingScenario.id, ...scenarioData })
        scenarioId = editingScenario.id
      } else {
        const created = await createMutation.mutateAsync(scenarioData)
        scenarioId = created.id
      }
      resetForm()
      await runOptimization(scenarioId)
    } catch {
      // Mutations already surface errors via toast.
    }
  }

  const initialDraft: Partial<ScenarioDraft> | undefined = editingScenario
    ? {
        profile_id: editingScenario.profile_id,
        name: editingScenario.name,
        description: editingScenario.description,
        start_date: editingScenario.start_date,
        objective: editingScenario.objective as ScenarioDraft['objective'],
        terminal_goals: editingScenario.terminal_goals ?? [],
        intermediate_goals: editingScenario.intermediate_goals ?? [],
        withdrawals: editingScenario.withdrawals ?? null,
      }
    : undefined

  const scenarioEntries = (scenarios ?? [])
    .map((scenario) => {
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
    .sort((left, right) => {
      const rankDelta = planPriority(left.health) - planPriority(right.health)
      if (rankDelta !== 0) return rankDelta
      const leftHorizon = left.latestResult?.optimal_horizon ?? Number.POSITIVE_INFINITY
      const rightHorizon = right.latestResult?.optimal_horizon ?? Number.POSITIVE_INFINITY
      return leftHorizon - rightHorizon
    })

  const ownScenarioEntries = scenarioEntries.filter((entry) => !entry.scenario.is_demo)
  const demoScenarioEntries = scenarioEntries.filter((entry) => entry.scenario.is_demo)
  const healthCounts = ownScenarioEntries.reduce(
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

  const summaryCards = [
    {
      label: t('summary.onTrack'),
      value: healthCounts.on_track,
      detail: healthCounts.on_track > 0 ? t('summary.onTrackDetail') : t('summary.onTrackEmpty'),
      icon: Target,
    },
    {
      label: t('summary.needReview'),
      value: healthCounts.tight + healthCounts.needs_changes + healthCounts.failed,
      detail: t('summary.needReviewDetail'),
      icon: TrendingUp,
    },
    {
      label: t('summary.runningNow'),
      value: healthCounts.running + healthCounts.queued,
      detail: t('summary.runningNowDetail'),
      icon: PlayCircle,
    },
    {
      label: t('summary.needFirstRun'),
      value: healthCounts.draft,
      detail: t('summary.needFirstRunDetail'),
      icon: BriefcaseBusiness,
    },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">{t('title')}</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {t('subtitle')}
          </p>
        </div>
        <Button
          type="button"
          onClick={() => setShowForm(true)}
          disabled={!ownProfiles?.length}
          className="rounded-xl"
        >
          <Plus className="h-4 w-4" />
          {t('newPlan')}
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        {summaryCards.map(({ label, value, detail, icon: Icon }) => (
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

      {!ownProfiles?.length && (
        <div className="rounded-xl border border-warning/30 bg-warning-soft p-4 text-sm text-warning">
          {t('needSituation')}
          <Button asChild variant="link" className="ml-2 h-auto px-0 text-warning">
            <Link to="/profiles">{t('goToSituation')}</Link>
          </Button>
        </div>
      )}

      {/* Wizard overlay (redesigned) */}
      {showForm && (
        <div
          className="fixed inset-0 z-50 overflow-y-auto bg-background"
          style={{ fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif' }}
        >
          <div className="mx-auto w-full max-w-4xl px-4 py-6 sm:px-6">
            <div className="mb-2 flex justify-end">
              <Button variant="ghost" size="sm" onClick={resetForm}>
                {t('close')}
              </Button>
            </div>
            <GoalsWizard
              profiles={ownProfiles ?? []}
              initialDraft={initialDraft}
              onCalculate={handleCalculate}
              onCancel={resetForm}
            />
          </div>
        </div>
      )}

      {/* Plans list */}
      <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
        {isLoading ? (
          <div className="p-6 text-center text-muted-foreground">{t('loading')}</div>
        ) : scenarioEntries.length === 0 ? (
          <div className="p-8 text-center">
            <p className="text-base font-medium text-foreground">{t('emptyTitle')}</p>
            <p className="mt-2 text-sm text-muted-foreground">
              {t('emptyBody')}
            </p>
            <Button
              type="button"
              className="mt-4 rounded-xl"
              onClick={() => setShowForm(true)}
              disabled={!ownProfiles?.length}
            >
              <Plus className="h-4 w-4" />
              {t('createFirst')}
            </Button>
          </div>
        ) : (
          <div>
            {ownScenarioEntries.length > 0 ? (
              <div className="border-b border-border px-6 py-4">
                <h2 className="text-base font-semibold text-foreground">{t('yourPlans')}</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  {t('yourPlansSubtitle')}
                </p>
              </div>
            ) : (
              <div className="border-b border-border px-6 py-4">
                <h2 className="text-base font-semibold text-foreground">{t('yourPlans')}</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  {t('yourPlansEmptySubtitle')}
                </p>
              </div>
            )}

            <div className="divide-y divide-border">
              {ownScenarioEntries.map(({ scenario, latestJob, latestResult, health, goals }) => (
              <div
                key={scenario.id}
                className="flex flex-col gap-5 p-6 transition-colors hover:bg-muted/20 lg:flex-row lg:items-start lg:justify-between"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-lg font-semibold text-foreground">{scenario.name}</h3>
                    {scenario.is_demo && (
                      <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                        {t('demo')}
                      </span>
                    )}
                    <span className="rounded-full bg-secondary px-2.5 py-1 text-xs font-medium text-secondary-foreground">
                      {objectiveLabel(scenario.objective)}
                    </span>
                    {!scenario.is_demo && (
                      <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${planStatusTone(health)}`}>
                        {planStatusLabel(health, t)}
                      </span>
                    )}
                  </div>
                  <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                    {scenario.description || t('descriptionFallback', { name: scenario.profiles?.name ?? t('savedSituation') })}
                  </p>
                  <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                    <PlanMetricPill label={t('metrics.situation')} value={scenario.profiles?.name ?? t('situationUnknown')} />
                    <PlanMetricPill
                      label={t('metrics.goals')}
                      value={formatGoals(scenario.terminal_goals?.length ?? 0, scenario.intermediate_goals?.length ?? 0)}
                    />
                    <PlanMetricPill
                      label={t('metrics.withdrawals')}
                      value={`${(scenario.withdrawals?.scheduled?.length ?? 0) + (scenario.withdrawals?.stochastic?.length ?? 0)}`}
                    />
                    <PlanMetricPill label={t('metrics.startDate')} value={formatMonthYear(scenario.start_date)} />
                  </div>
                </div>

                <div className="flex w-full max-w-md flex-col gap-3 lg:items-end">
                  <div className="w-full rounded-2xl bg-muted/50 px-4 py-3 lg:max-w-sm">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        {t('latestRun')}
                      </p>
                      {!scenario.is_demo && (
                        <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${planStatusTone(health)}`}>
                          {planStatusLabel(health, t)}
                        </span>
                      )}
                    </div>
                    <p className="mt-1 font-medium text-foreground">
                      {describeLatestRun(scenario, latestJob, latestResultsByJobId, t)}
                    </p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {describeLatestRunDetail(scenario, latestJob, latestResultsByJobId, t)}
                    </p>
                    {!scenario.is_demo && (
                      <div className="mt-4 grid grid-cols-3 gap-2 rounded-xl border border-border/70 bg-card/70 p-3">
                        <PlanRunMetric
                          label={t('runMetrics.goalsMet')}
                          value={goals.total > 0 ? `${goals.met}/${goals.total}` : '—'}
                        />
                        <PlanRunMetric
                          label={t('runMetrics.horizon')}
                          value={latestResult?.optimal_horizon ? formatMonthsLong(latestResult.optimal_horizon) : '—'}
                        />
                        <PlanRunMetric
                          label={t('runMetrics.updated')}
                          value={latestJob ? formatDateShort(latestJob.completed_at ?? latestJob.created_at) : '—'}
                        />
                      </div>
                    )}
                  </div>

                  {scenario.is_demo ? (
                    <Button type="button" size="sm" className="rounded-xl" onClick={() => viewResults(scenario.id)}>
                      {t('actions.viewDemoResult')}
                      <ArrowRight className="h-3.5 w-3.5" />
                    </Button>
                  ) : (
                    <div className="flex flex-wrap gap-2 lg:justify-end">
                      {latestJob && (
                        <Button asChild size="sm" className="rounded-xl">
                          <Link to={`/results/${latestJob.id}`}>
                            {latestJob.status === 'completed' ? t('actions.viewLatestResult') : t('actions.openLatestRun')}
                            <ArrowRight className="h-3.5 w-3.5" />
                          </Link>
                        </Button>
                      )}
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="rounded-xl"
                        onClick={() => handleEdit(scenario)}
                      >
                        {t('actions.edit')}
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        className="rounded-xl bg-success text-success-foreground hover:bg-success/90"
                        onClick={() => runOptimization(scenario.id)}
                      >
                        {t('actions.run')}
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="rounded-xl border-danger/30 text-danger hover:bg-danger-soft hover:text-danger"
                        onClick={() => {
                          if (confirm(t('actions.confirmDelete'))) {
                            deleteMutation.mutate(scenario.id)
                          }
                        }}
                      >
                        {t('actions.delete')}
                      </Button>
                    </div>
                  )}
                </div>
              </div>
              ))}
            </div>

            {demoScenarioEntries.length > 0 && (
              <>
                <div className="border-y border-border bg-muted/20 px-6 py-4">
                  <h2 className="text-base font-semibold text-foreground">{t('examples')}</h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {t('examplesSubtitle')}
                  </p>
                </div>
                <div className="divide-y divide-border">
                  {demoScenarioEntries.map(({ scenario, latestJob }) => (
                    <div
                      key={scenario.id}
                      className="flex flex-col gap-5 px-6 py-5 transition-colors hover:bg-muted/10 lg:flex-row lg:items-start lg:justify-between"
                    >
                      <div className="min-w-0 flex-1 opacity-90">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="text-lg font-semibold text-foreground">{scenario.name}</h3>
                          <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                            {t('demo')}
                          </span>
                          <span className="rounded-full bg-secondary px-2.5 py-1 text-xs font-medium text-secondary-foreground">
                            {objectiveLabel(scenario.objective)}
                          </span>
                        </div>
                        <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                          {scenario.description || t('descriptionFallback', { name: scenario.profiles?.name ?? t('demoSituation') })}
                        </p>
                        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                          <PlanMetricPill label={t('metrics.situation')} value={scenario.profiles?.name ?? t('demoSituationName')} />
                          <PlanMetricPill
                            label={t('metrics.goals')}
                            value={formatGoals(scenario.terminal_goals?.length ?? 0, scenario.intermediate_goals?.length ?? 0)}
                          />
                          <PlanMetricPill
                            label={t('metrics.withdrawals')}
                            value={`${(scenario.withdrawals?.scheduled?.length ?? 0) + (scenario.withdrawals?.stochastic?.length ?? 0)}`}
                          />
                          <PlanMetricPill label={t('metrics.startDate')} value={formatMonthYear(scenario.start_date)} />
                        </div>
                      </div>

                      <div className="flex w-full max-w-md flex-col gap-3 lg:items-end">
                        <div className="w-full rounded-2xl bg-muted/40 px-4 py-3 lg:max-w-sm">
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                            {t('latestRun')}
                          </p>
                          <p className="mt-1 font-medium text-foreground">
                            {describeLatestRun(scenario, latestJob, latestResultsByJobId, t)}
                          </p>
                          <p className="mt-1 text-sm text-muted-foreground">
                            {describeLatestRunDetail(scenario, latestJob, latestResultsByJobId, t)}
                          </p>
                        </div>

                        <Button type="button" size="sm" className="rounded-xl" onClick={() => viewResults(scenario.id)}>
                          {t('actions.viewDemoResult')}
                          <ArrowRight className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function PlanMetricPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-muted/60 px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-base font-semibold text-foreground">{value}</p>
    </div>
  )
}

function PlanRunMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-card px-3 py-2">
      <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-sm font-semibold text-foreground">{value}</p>
    </div>
  )
}

function planPriority(health: PlanHealth) {
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

function planStatusLabel(health: PlanHealth, t: TFunction) {
  switch (health) {
    case 'needs_changes':
      return t('status.needsChanges')
    case 'tight':
      return t('status.tight')
    case 'failed':
      return t('status.failed')
    case 'running':
      return t('status.calculating')
    case 'queued':
      return t('status.queued')
    case 'completed':
      return t('status.completed')
    case 'on_track':
      return t('status.onTrack')
    default:
      return t('status.needsRun')
  }
}

function planStatusTone(health: PlanHealth) {
  switch (health) {
    case 'needs_changes':
    case 'failed':
      return 'bg-danger-soft text-danger'
    case 'tight':
      return 'bg-warning-soft text-warning'
    case 'running':
    case 'queued':
      return 'bg-accent text-accent-foreground'
    case 'completed':
      return 'bg-muted text-muted-foreground'
    case 'on_track':
      return 'bg-success-soft text-success'
    default:
      return 'bg-muted text-muted-foreground'
  }
}

function describeLatestRun(
  scenario: Pick<Scenario, 'is_demo'>,
  job: Pick<Job, 'id' | 'status'> | undefined,
  resultsByJobId: Record<string, Pick<Result, 'job_id' | 'feasible' | 'optimal_horizon' | 'goal_status'>>,
  t: TFunction,
) {
  if (scenario.is_demo) return t('latest.demoAvailable')
  if (!job) return t('latest.noRuns')

  if (job.status === 'completed') {
    const result = resultsByJobId[job.id]
    if (!result) return t('latest.completed')
    if (result?.feasible === false) return t('latest.needsChanges')
    if (result?.optimal_horizon) return t('latest.result', { horizon: formatMonthsLong(result.optimal_horizon) })
    return t('latest.ready')
  }

  if (job.status === 'failed') return t('latest.failed')
  if (job.status === 'running') return t('latest.running')
  return t('latest.queued')
}

function describeLatestRunDetail(
  scenario: Pick<Scenario, 'is_demo'>,
  job: Pick<Job, 'id' | 'status' | 'created_at' | 'completed_at'> | undefined,
  resultsByJobId: Record<string, Pick<Result, 'job_id' | 'feasible' | 'optimal_horizon' | 'goal_status'>>,
  t: TFunction,
) {
  if (scenario.is_demo) return t('latestDetail.demo')
  if (!job) return t('latestDetail.noRuns')

  if (job.status === 'completed') {
    const result = resultsByJobId[job.id]
    const completedLabel = job.completed_at ? formatDateShort(job.completed_at) : formatDateShort(job.created_at)
    if (!result) return t('latestDetail.completedNoSummary', { date: completedLabel })
    if (result?.feasible === false) return t('latestDetail.completedNeedsChanges', { date: completedLabel })
    return t('latestDetail.completedReady', { date: completedLabel })
  }

  if (job.status === 'failed') {
    return t('latestDetail.failed', { date: formatDateShort(job.created_at) })
  }

  if (job.status === 'running') {
    return t('latestDetail.running', { date: formatDateShort(job.created_at) })
  }

  return t('latestDetail.queued', { date: formatDateShort(job.created_at) })
}
