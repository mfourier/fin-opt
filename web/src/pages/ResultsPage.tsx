import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useJobProgress } from '../hooks/useJobProgress'
import { supabase } from '../lib/supabase'
import type { Result, Scenario } from '../types/database'
import AllocationHeatmap from '../components/AllocationHeatmap'

export default function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>()
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
      return data as Scenario & { profiles: { name: string; accounts_config: { name: string }[] } }
    },
    enabled: !!job?.scenario_id,
  })

  if (jobLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary-500 border-t-transparent"></div>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="text-center">
        <p className="text-gray-500">Job not found</p>
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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            {job.job_type === 'optimization' ? 'Optimization' : 'Simulation'} Results
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            {scenario?.name} - {scenario?.profiles?.name}
          </p>
        </div>
        <span className={`rounded-full px-3 py-1 text-sm font-medium ${getStatusColor(job.status)}`}>
          {job.status}
        </span>
      </div>

      {/* Progress */}
      {(job.status === 'pending' || job.status === 'running') && (
        <div className="rounded-lg bg-white p-6 shadow">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">
              {job.current_step || 'Waiting to start...'}
            </span>
            <span className="text-sm text-gray-500">{job.progress}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
            <div
              className="h-full bg-primary-500 transition-all duration-500"
              style={{ width: `${job.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error */}
      {job.status === 'failed' && (
        <div className="rounded-lg bg-red-50 p-6">
          <h3 className="font-medium text-red-800">Optimization Failed</h3>
          <p className="mt-2 text-sm text-red-700">{job.error_message || 'Unknown error'}</p>
        </div>
      )}

      {/* Results */}
      {job.status === 'completed' && result && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
            <div className="rounded-lg bg-white p-6 shadow">
              <p className="text-sm font-medium text-gray-500">Optimal Horizon</p>
              <p className="mt-2 text-3xl font-semibold text-gray-900">
                {result.optimal_horizon} <span className="text-lg text-gray-500">months</span>
              </p>
            </div>
            <div className="rounded-lg bg-white p-6 shadow">
              <p className="text-sm font-medium text-gray-500">Feasible</p>
              <p className={`mt-2 text-3xl font-semibold ${result.feasible ? 'text-green-600' : 'text-red-600'}`}>
                {result.feasible ? 'Yes' : 'No'}
              </p>
            </div>
            <div className="rounded-lg bg-white p-6 shadow">
              <p className="text-sm font-medium text-gray-500">Objective Value</p>
              <p className="mt-2 text-3xl font-semibold text-gray-900">
                {result.objective_value?.toFixed(4) ?? '-'}
              </p>
            </div>
            <div className="rounded-lg bg-white p-6 shadow">
              <p className="text-sm font-medium text-gray-500">Solve Time</p>
              <p className="mt-2 text-3xl font-semibold text-gray-900">
                {result.solve_time?.toFixed(2) ?? '-'} <span className="text-lg text-gray-500">sec</span>
              </p>
            </div>
          </div>

          {/* Allocation Heatmap */}
          {result.allocation_policy && (
            <div className="rounded-lg bg-white p-6 shadow">
              <h3 className="mb-4 text-lg font-medium text-gray-900">Allocation Policy</h3>
              <AllocationHeatmap
                allocation={result.allocation_policy}
                accountNames={scenario?.profiles?.accounts_config?.map(a => a.name) ?? []}
              />
            </div>
          )}

          {/* Goal Status */}
          {result.goal_status && result.goal_status.length > 0 && (
            <div className="rounded-lg bg-white p-6 shadow">
              <h3 className="mb-4 text-lg font-medium text-gray-900">Goal Status</h3>
              <div className="divide-y divide-gray-200">
                {result.goal_status.map((goal, index) => (
                  <div key={index} className="flex items-center justify-between py-3">
                    <div>
                      <p className="font-medium text-gray-900">
                        {goal.type} - {goal.account}
                      </p>
                      <p className="text-sm text-gray-500">
                        Target: ${goal.threshold.toLocaleString()} at {(goal.required_confidence * 100).toFixed(0)}% confidence
                      </p>
                    </div>
                    <div className="text-right">
                      <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                        goal.satisfied ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                      }`}>
                        {goal.satisfied ? 'Achieved' : 'Not Met'}
                      </span>
                      {goal.actual_probability !== undefined && (
                        <p className="mt-1 text-sm text-gray-500">
                          Actual: {(goal.actual_probability * 100).toFixed(1)}%
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Export */}
          <div className="flex gap-4">
            <button
              onClick={() => {
                const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `finopt-result-${jobId}.json`
                a.click()
              }}
              className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              Export JSON
            </button>
          </div>
        </div>
      )}

      <Link to="/scenarios" className="inline-block text-sm text-primary-600 hover:text-primary-500">
        Back to Scenarios
      </Link>
    </div>
  )
}
