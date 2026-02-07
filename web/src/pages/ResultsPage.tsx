import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useJobProgress } from '../hooks/useJobProgress'
import { supabase } from '../lib/supabase'
import type { Result, Scenario } from '../types/database'
import AllocationHeatmap from '../components/AllocationHeatmap'
import AllocationChart from '../components/AllocationChart'
import GoalProgressCard from '../components/GoalProgressCard'

type ViewTab = 'overview' | 'allocation' | 'goals'

export default function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const { job, loading: jobLoading } = useJobProgress(jobId ?? null)
  const [activeTab, setActiveTab] = useState<ViewTab>('overview')

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
      return data as Scenario & { profiles: { name: string; accounts_config: { name: string; display_name?: string }[] } }
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

  // Calculate years and months from horizon
  const formatHorizon = (months: number) => {
    const years = Math.floor(months / 12)
    const remainingMonths = months % 12
    if (years === 0) return `${months} months`
    if (remainingMonths === 0) return `${years} year${years > 1 ? 's' : ''}`
    return `${years}y ${remainingMonths}m`
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-gray-900">
              {job.job_type === 'optimization' ? 'Optimization' : 'Simulation'} Results
            </h1>
            <span className={`rounded-full px-3 py-1 text-sm font-medium ${getStatusColor(job.status)}`}>
              {job.status}
            </span>
          </div>
          {scenario && (
            <p className="mt-1 text-sm text-gray-500">
              {scenario.name} • {scenario.profiles?.name}
            </p>
          )}
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

      {/* Results - for completed jobs */}
      {job.status === 'completed' && result && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="rounded-lg bg-white p-5 shadow">
              <p className="text-sm font-medium text-gray-500">Optimal Horizon</p>
              <p className="mt-2 text-3xl font-bold text-gray-900">
                {result.optimal_horizon ? formatHorizon(result.optimal_horizon) : '-'}
              </p>
              {result.optimal_horizon && (
                <p className="mt-1 text-xs text-gray-400">{result.optimal_horizon} months</p>
              )}
            </div>
            <div className="rounded-lg bg-white p-5 shadow">
              <p className="text-sm font-medium text-gray-500">Feasibility</p>
              <div className="mt-2 flex items-center gap-2">
                {result.feasible ? (
                  <>
                    <svg className="h-8 w-8 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <span className="text-2xl font-bold text-green-600">Feasible</span>
                  </>
                ) : (
                  <>
                    <svg className="h-8 w-8 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                    <span className="text-2xl font-bold text-red-600">Infeasible</span>
                  </>
                )}
              </div>
            </div>
            <div className="rounded-lg bg-white p-5 shadow">
              <p className="text-sm font-medium text-gray-500">Objective Value</p>
              <p className="mt-2 text-3xl font-bold text-gray-900">
                {result.objective_value?.toFixed(4) ?? '-'}
              </p>
              {scenario && (
                <p className="mt-1 text-xs text-gray-400">{scenario.objective}</p>
              )}
            </div>
            <div className="rounded-lg bg-white p-5 shadow">
              <p className="text-sm font-medium text-gray-500">Solve Time</p>
              <p className="mt-2 text-3xl font-bold text-gray-900">
                {result.solve_time?.toFixed(2) ?? '-'}
                <span className="ml-1 text-lg font-normal text-gray-500">sec</span>
              </p>
              {result.diagnostics?.n_iterations !== undefined && (
                <p className="mt-1 text-xs text-gray-400">
                  {String(result.diagnostics.n_iterations)} iterations
                </p>
              )}
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              {[
                { id: 'overview', label: 'Overview' },
                { id: 'allocation', label: 'Allocation Policy' },
                { id: 'goals', label: 'Goals' },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as ViewTab)}
                  className={`whitespace-nowrap border-b-2 px-1 py-4 text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'border-primary-500 text-primary-600'
                      : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          {/* Tab Content */}
          <div className="rounded-lg bg-white p-6 shadow">
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Quick Stats */}
                {result.goal_status && result.goal_status.length > 0 && (
                  <div>
                    <h3 className="mb-4 text-lg font-medium text-gray-900">Goal Summary</h3>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="rounded-lg bg-gray-50 p-4">
                        <p className="text-sm text-gray-500">Goals Achieved</p>
                        <p className="text-2xl font-bold text-gray-900">
                          {result.goal_status.filter(g => g.satisfied).length} / {result.goal_status.length}
                        </p>
                      </div>
                      <div className="rounded-lg bg-gray-50 p-4">
                        <p className="text-sm text-gray-500">Accounts Used</p>
                        <p className="text-2xl font-bold text-gray-900">{accountNames.length}</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Allocation Chart Preview */}
                {result.allocation_policy && (
                  <div>
                    <div className="mb-4 flex items-center justify-between">
                      <h3 className="text-lg font-medium text-gray-900">Allocation Over Time</h3>
                      <button
                        onClick={() => setActiveTab('allocation')}
                        className="text-sm text-primary-600 hover:text-primary-500"
                      >
                        View Details →
                      </button>
                    </div>
                    <AllocationChart
                      allocation={result.allocation_policy}
                      accountNames={accountNames}
                      startDate={scenario?.start_date}
                    />
                  </div>
                )}
              </div>
            )}

            {activeTab === 'allocation' && result.allocation_policy && (
              <div className="space-y-6">
                <div>
                  <h3 className="mb-2 text-lg font-medium text-gray-900">Allocation Timeline</h3>
                  <p className="mb-4 text-sm text-gray-500">
                    Stacked area chart showing how allocation changes over the {result.optimal_horizon}-month horizon.
                  </p>
                  <AllocationChart
                    allocation={result.allocation_policy}
                    accountNames={accountNames}
                    startDate={scenario?.start_date}
                  />
                </div>
                <hr className="border-gray-200" />
                <div>
                  <h3 className="mb-2 text-lg font-medium text-gray-900">Allocation Heatmap</h3>
                  <p className="mb-4 text-sm text-gray-500">
                    Detailed view of monthly allocation percentages. Hover for exact values.
                  </p>
                  <AllocationHeatmap
                    allocation={result.allocation_policy}
                    accountNames={accountNames}
                    startDate={scenario?.start_date}
                  />
                </div>
              </div>
            )}

            {activeTab === 'goals' && (
              <div>
                <h3 className="mb-4 text-lg font-medium text-gray-900">Goal Achievement Status</h3>
                {result.goal_status && result.goal_status.length > 0 ? (
                  <GoalProgressCard goals={result.goal_status} />
                ) : (
                  <p className="text-gray-500">No goal information available</p>
                )}
              </div>
            )}
          </div>

          {/* Export Options */}
          <div className="flex gap-4">
            <button
              onClick={() => {
                const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `finopt-result-${jobId}.json`
                a.click()
                URL.revokeObjectURL(url)
              }}
              className="flex items-center gap-2 rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Export JSON
            </button>
            {result.allocation_policy && (
              <button
                onClick={() => {
                  const headers = ['Month', ...accountNames]
                  const rows = result.allocation_policy!.map((row, t) => [
                    t.toString(),
                    ...row.map(v => (v * 100).toFixed(2))
                  ])
                  const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
                  const blob = new Blob([csv], { type: 'text/csv' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `finopt-allocation-${jobId}.csv`
                  a.click()
                  URL.revokeObjectURL(url)
                }}
                className="flex items-center gap-2 rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Export Allocation CSV
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
