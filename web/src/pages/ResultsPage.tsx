import { useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { AlertCircle } from 'lucide-react'
import { useJobProgress } from '../hooks/useJobProgress'
import { supabase } from '../lib/supabase'
import { queueOptimization } from '../lib/api'
import type { Result, Scenario, Profile } from '../types/database'
import type { Scenario as PlanScenario, Result as PlanResult } from '@/mocks/types'
import { PlanResults } from '@/components/finopt/PlanResults'
import { Button } from '@/components/ui/button'

export default function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const { job, loading: jobLoading, error: jobError } = useJobProgress(jobId ?? null)
  const [recalculating, setRecalculating] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)

  const {
    data: result,
    isLoading: resultLoading,
    error: resultError,
  } = useQuery({
    queryKey: ['result', jobId] as const,
    queryFn: async () => {
      const { data, error } = await supabase
        .from('results')
        .select('*')
        .eq('job_id', jobId!)
        .single()
      if (error) throw error
      return data as Result
    },
    enabled: !!jobId && job?.status === 'completed',
  })

  const {
    data: scenario,
    isLoading: scenarioLoading,
    error: scenarioError,
  } = useQuery({
    queryKey: ['scenario', job?.scenario_id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('scenarios')
        .select('*, profiles(*)')
        .eq('id', job!.scenario_id)
        .single()
      if (error) throw error
      return data as Scenario & { profiles: Profile }
    },
    enabled: !!job?.scenario_id,
  })

  if (jobLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="text-center">
          <div className="mx-auto h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
          <p className="mt-4 text-sm text-muted-foreground">Loading job status...</p>
        </div>
      </div>
    )
  }

  if (!job) {
    return (
      <EmptyState
        title="Job not found"
        message={jobError ?? 'We could not find this result. It may have been removed or the link may be incomplete.'}
      />
    )
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-success-soft text-success'
      case 'running': return 'bg-accent text-accent-foreground'
      case 'failed': return 'bg-danger-soft text-danger'
      default: return 'bg-muted text-muted-foreground'
    }
  }

  const accountNames = scenario?.profiles?.accounts_config?.map(a => a.display_name || a.name) ?? []

  const handleExportJSON = () => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `finopt-result-${jobId}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleExportCSV = () => {
    if (!result?.allocation_policy) return
    const headers = ['Month', ...accountNames]
    const rows = result.allocation_policy.map((row, t) => [
      t.toString(),
      ...row.map(v => (v * 100).toFixed(2)),
    ])
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `finopt-allocation-${jobId}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleRecalculate = async () => {
    setRecalculating(true)
    setActionError(null)
    let newJobId: string | null = null
    try {
      const { data: newJob, error } = await supabase
        .from('jobs')
        .insert({ scenario_id: job.scenario_id, job_type: 'optimization', status: 'pending', progress: 0 })
        .select()
        .single()
      if (error) throw error
      newJobId = newJob.id
      await queueOptimization({ scenario_id: job.scenario_id, job_id: newJob.id })
      navigate(`/results/${newJob.id}`)
    } catch (e) {
      if (newJobId) {
        await supabase
          .from('jobs')
          .update({
            status: 'failed',
            error_message: e instanceof Error ? e.message : 'Failed to reach the compute service',
          })
          .eq('id', newJobId)
      }
      setActionError(e instanceof Error ? e.message : 'Failed to recalculate')
    } finally {
      setRecalculating(false)
    }
  }

  if (job.status === 'completed' && (resultLoading || scenarioLoading)) {
    return (
      <LoadingState message="Loading your completed plan and charts..." />
    )
  }

  if (job.status === 'completed' && (resultError || scenarioError)) {
    return (
      <EmptyState
        title="We couldn't load this plan"
        message={(resultError ?? scenarioError)?.message ?? 'Something went wrong while loading the result.'}
      />
    )
  }

  if (job.status === 'completed' && (!result || !scenario)) {
    return (
      <EmptyState
        title="Plan data unavailable"
        message="The job finished, but the scenario or result payload is missing."
      />
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold text-foreground">
            {job.job_type === 'optimization' ? 'Optimization' : 'Simulation'} Results
          </h1>
          <span className={`rounded-full px-3 py-1 text-sm font-medium ${getStatusColor(job.status)}`}>
            {job.status}
          </span>
        </div>
        <Button asChild variant="outline">
          <Link to="/scenarios">Back to Scenarios</Link>
        </Button>
      </div>

      {/* Progress - for pending/running jobs */}
      {(job.status === 'pending' || job.status === 'running') && (
        <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
              <span className="text-sm font-medium text-foreground">
                {job.current_step || 'Initializing...'}
              </span>
            </div>
            <span className="text-lg font-semibold text-primary">{job.progress}%</span>
          </div>
          <div className="h-3 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-primary transition-all duration-500"
              style={{ width: `${job.progress}%` }}
            />
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            This may take a few minutes depending on the number of simulations and horizon length.
          </p>
        </div>
      )}

      {/* Error - for failed jobs */}
      {job.status === 'failed' && (
        <div className="rounded-2xl border border-danger/30 bg-danger-soft p-6">
          <div className="flex items-start gap-4">
            <AlertCircle className="h-6 w-6 text-danger" />
            <div>
              <h3 className="font-medium text-danger">Optimization Failed</h3>
              <p className="mt-2 text-sm text-danger">{job.error_message || 'An unknown error occurred'}</p>
              <div className="mt-4">
                <Link
                  to={`/scenarios`}
                  className="text-sm font-medium text-danger transition-opacity hover:opacity-80"
                >
                  Try adjusting your scenario parameters →
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results - redesigned "My plan" view (Phase B: wired to real data) */}
      {job.status === 'completed' && result && scenario && (
        <div style={{ fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif' }}>
          <PlanResults
            profile={scenario.profiles}
            // database Scenario/Result widen some literal-union fields to
            // `string` (objective, goal_status[].type); otherwise identical.
            scenario={scenario as unknown as PlanScenario}
            result={result as unknown as PlanResult}
            jobStatus="completed"
            onExportJSON={handleExportJSON}
            onExportCSV={handleExportCSV}
            onRecalculate={handleRecalculate}
            onAdjustGoals={() => navigate('/scenarios')}
            updatedAt={job.completed_at ?? result.created_at}
            isRecalculating={recalculating}
            actionError={actionError}
          />
        </div>
      )}
    </div>
  )
}

function LoadingState({ message }: { message: string }) {
  return (
    <div className="flex h-64 items-center justify-center">
      <div className="text-center">
        <div className="mx-auto h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        <p className="mt-4 text-sm text-muted-foreground">{message}</p>
      </div>
    </div>
  )
}

function EmptyState({ title, message }: { title: string; message: string }) {
  return (
    <div className="rounded-2xl border border-border bg-card p-8 text-center shadow-sm">
      <svg className="mx-auto h-12 w-12 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <h2 className="mt-4 text-lg font-semibold text-foreground">{title}</h2>
      <p className="mt-2 text-sm text-muted-foreground">{message}</p>
      <div className="mt-5">
        <Button asChild variant="outline">
          <Link to="/scenarios">Back to Scenarios</Link>
        </Button>
      </div>
    </div>
  )
}
