import { useParams, Link, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useJobProgress } from '../hooks/useJobProgress'
import { supabase } from '../lib/supabase'
import { queueOptimization } from '../lib/api'
import type { Result, Scenario, Profile } from '../types/database'
import type { Scenario as PlanScenario, Result as PlanResult } from '@/mocks/types'
import { PlanResults } from '@/components/finopt/PlanResults'

export default function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const { job, loading: jobLoading } = useJobProgress(jobId ?? null)

  const { data: result } = useQuery({
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

  const { data: scenario } = useQuery({
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
          <div className="mx-auto h-10 w-10 animate-spin rounded-full border-4 border-primary-500 border-t-transparent"></div>
          <p className="mt-4 text-sm text-gray-500">Loading job status...</p>
        </div>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="flex h-64 flex-col items-center justify-center">
        <svg className="h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="mt-4 text-gray-500">Job not found</p>
        <Link to="/scenarios" className="mt-4 text-primary-600 hover:text-primary-500">
          Back to Scenarios
        </Link>
      </div>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800'
      case 'running': return 'bg-blue-100 text-blue-800'
      case 'failed': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
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
    try {
      const { data: newJob, error } = await supabase
        .from('jobs')
        .insert({ scenario_id: job.scenario_id, job_type: 'optimization', status: 'pending', progress: 0 })
        .select()
        .single()
      if (error) throw error
      await queueOptimization({ scenario_id: job.scenario_id, job_id: newJob.id })
      navigate(`/results/${newJob.id}`)
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Failed to recalculate')
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold text-gray-900">
            {job.job_type === 'optimization' ? 'Optimization' : 'Simulation'} Results
          </h1>
          <span className={`rounded-full px-3 py-1 text-sm font-medium ${getStatusColor(job.status)}`}>
            {job.status}
          </span>
        </div>
        <Link
          to="/scenarios"
          className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
        >
          ← Back to Scenarios
        </Link>
      </div>

      {/* Progress - for pending/running jobs */}
      {(job.status === 'pending' || job.status === 'running') && (
        <div className="rounded-lg bg-white p-6 shadow">
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary-500 border-t-transparent"></div>
              <span className="text-sm font-medium text-gray-700">
                {job.current_step || 'Initializing...'}
              </span>
            </div>
            <span className="text-lg font-semibold text-primary-600">{job.progress}%</span>
          </div>
          <div className="h-3 w-full overflow-hidden rounded-full bg-gray-200">
            <div
              className="h-full bg-gradient-to-r from-primary-400 to-primary-600 transition-all duration-500"
              style={{ width: `${job.progress}%` }}
            />
          </div>
          <p className="mt-3 text-xs text-gray-500">
            This may take a few minutes depending on the number of simulations and horizon length.
          </p>
        </div>
      )}

      {/* Error - for failed jobs */}
      {job.status === 'failed' && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-6">
          <div className="flex items-start gap-4">
            <svg className="h-6 w-6 text-red-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <div>
              <h3 className="font-medium text-red-800">Optimization Failed</h3>
              <p className="mt-2 text-sm text-red-700">{job.error_message || 'An unknown error occurred'}</p>
              <div className="mt-4">
                <Link
                  to={`/scenarios`}
                  className="text-sm font-medium text-red-700 hover:text-red-600"
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
          />
        </div>
      )}
    </div>
  )
}
