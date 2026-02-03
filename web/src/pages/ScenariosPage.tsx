import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabase'
import { useAuthStore } from '../lib/store'
import { queueOptimization } from '../lib/api'
import type { Profile, Scenario, ScenarioInsert, Goal } from '../types/database'

export default function ScenariosPage() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const user = useAuthStore((state) => state.user)
  const [showForm, setShowForm] = useState(false)

  const [formData, setFormData] = useState({
    profile_id: '',
    name: '',
    description: '',
    start_date: new Date().toISOString().split('T')[0],
    n_sims: 500,
    seed: 42,
    t_max: 120,
    solver: 'ECOS',
    objective: 'balanced',
    terminal_goals: [{ account: 0, threshold: 50000000, confidence: 0.80 }] as Goal[],
    intermediate_goals: [] as Goal[],
  })

  const { data: profiles } = useQuery({
    queryKey: ['profiles'],
    queryFn: async () => {
      const { data, error } = await supabase.from('profiles').select('*').order('created_at', { ascending: false })
      if (error) throw error
      return data as Profile[]
    },
    enabled: !!user,
  })

  const { data: scenarios, isLoading } = useQuery({
    queryKey: ['scenarios'],
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

  const createMutation = useMutation({
    mutationFn: async (scenario: ScenarioInsert) => {
      const { data, error } = await supabase.from('scenarios').insert(scenario).select().single()
      if (error) throw error
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] })
      resetForm()
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase.from('scenarios').delete().eq('id', id)
      if (error) throw error
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] })
    },
  })

  const runOptimization = async (scenario: Scenario) => {
    // Create job in Supabase
    const { data: job, error: jobError } = await supabase
      .from('jobs')
      .insert({
        scenario_id: scenario.id,
        job_type: 'optimization',
        status: 'pending',
        progress: 0,
      })
      .select()
      .single()

    if (jobError) {
      alert('Failed to create job: ' + jobError.message)
      return
    }

    // Queue with Python API
    try {
      await queueOptimization({
        scenario_id: scenario.id,
        job_id: job.id,
      })
      queryClient.invalidateQueries({ queryKey: ['recent-jobs'] })
      navigate(`/results/${job.id}`)
    } catch (err) {
      alert('Failed to queue optimization: ' + (err instanceof Error ? err.message : 'Unknown error'))
    }
  }

  const resetForm = () => {
    setShowForm(false)
    setFormData({
      profile_id: '',
      name: '',
      description: '',
      start_date: new Date().toISOString().split('T')[0],
      n_sims: 500,
      seed: 42,
      t_max: 120,
      solver: 'ECOS',
      objective: 'balanced',
      terminal_goals: [{ account: 'Conservative', threshold: 50000000, confidence: 0.80 }],
      intermediate_goals: [],
    })
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    createMutation.mutate({
      ...formData,
      withdrawals: null,
    })
  }

  const updateTerminalGoal = (index: number, field: keyof Goal, value: number | string) => {
    const newGoals = [...formData.terminal_goals]
    newGoals[index] = { ...newGoals[index], [field]: value }
    setFormData({ ...formData, terminal_goals: newGoals })
  }

  const selectedProfile = profiles?.find(p => p.id === formData.profile_id)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Scenarios</h1>
          <p className="mt-1 text-sm text-gray-500">
            Define optimization scenarios with goals and parameters.
          </p>
        </div>
        <button
          onClick={() => setShowForm(true)}
          disabled={!profiles?.length}
          className="rounded-md bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50"
        >
          New Scenario
        </button>
      </div>

      {!profiles?.length && (
        <div className="rounded-md bg-yellow-50 p-4 text-sm text-yellow-700">
          Create a profile first before creating scenarios.
        </div>
      )}

      {/* Form Modal */}
      {showForm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
          <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-lg bg-white p-6 shadow-xl">
            <h2 className="mb-4 text-lg font-medium text-gray-900">Create Scenario</h2>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Profile</label>
                  <select
                    value={formData.profile_id}
                    onChange={(e) => {
                      const profileId = e.target.value
                      const profile = profiles?.find(p => p.id === profileId)
                      const firstAccountName = profile?.accounts_config[0]?.name ?? 'Account'
                      setFormData({
                        ...formData,
                        profile_id: profileId,
                        terminal_goals: [{ account: firstAccountName, threshold: 50000000, confidence: 0.80 }],
                      })
                    }}
                    required
                    className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                  >
                    <option value="">Select profile</option>
                    {profiles?.map((p) => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Name</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    required
                    className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Description</label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  rows={2}
                  className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                />
              </div>

              {/* Parameters */}
              <div>
                <h3 className="mb-2 font-medium text-gray-900">Parameters</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm text-gray-600">Start Date</label>
                    <input
                      type="date"
                      value={formData.start_date}
                      onChange={(e) => setFormData({ ...formData, start_date: e.target.value })}
                      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600">Simulations</label>
                    <input
                      type="number"
                      value={formData.n_sims}
                      onChange={(e) => setFormData({ ...formData, n_sims: Number(e.target.value) })}
                      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600">Max Horizon (months)</label>
                    <input
                      type="number"
                      value={formData.t_max}
                      onChange={(e) => setFormData({ ...formData, t_max: Number(e.target.value) })}
                      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    />
                  </div>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-gray-600">Solver</label>
                    <select
                      value={formData.solver}
                      onChange={(e) => setFormData({ ...formData, solver: e.target.value })}
                      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    >
                      <option value="ECOS">ECOS</option>
                      <option value="SCS">SCS</option>
                      <option value="CLARABEL">CLARABEL</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600">Objective</label>
                    <select
                      value={formData.objective}
                      onChange={(e) => setFormData({ ...formData, objective: e.target.value })}
                      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    >
                      <option value="balanced">Balanced</option>
                      <option value="risky">Risky (Max Wealth)</option>
                      <option value="conservative">Conservative</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Terminal Goals */}
              <div>
                <h3 className="mb-2 font-medium text-gray-900">Terminal Goals</h3>
                {formData.terminal_goals.map((goal, index) => (
                  <div key={index} className="rounded-md border border-gray-200 p-4">
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="block text-sm text-gray-600">Account</label>
                        <select
                          value={goal.account as string}
                          onChange={(e) => updateTerminalGoal(index, 'account', e.target.value)}
                          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                        >
                          {selectedProfile?.accounts_config.map((acc) => (
                            <option key={acc.name} value={acc.name}>{acc.name}</option>
                          )) ?? <option value="Conservative">Conservative</option>}
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm text-gray-600">Target Amount</label>
                        <input
                          type="number"
                          value={goal.threshold}
                          onChange={(e) => updateTerminalGoal(index, 'threshold', Number(e.target.value))}
                          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm text-gray-600">Confidence</label>
                        <input
                          type="number"
                          step="0.01"
                          min="0"
                          max="1"
                          value={goal.confidence}
                          onChange={(e) => updateTerminalGoal(index, 'confidence', Number(e.target.value))}
                          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="flex justify-end gap-3">
                <button
                  type="button"
                  onClick={resetForm}
                  className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={createMutation.isPending}
                  className="rounded-md bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50"
                >
                  Create
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Scenarios List */}
      <div className="rounded-lg bg-white shadow">
        {isLoading ? (
          <div className="p-6 text-center text-gray-500">Loading...</div>
        ) : scenarios?.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            No scenarios yet. Create one to get started.
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {scenarios?.map((scenario) => (
              <div key={scenario.id} className="flex items-center justify-between p-6">
                <div>
                  <h3 className="font-medium text-gray-900">{scenario.name}</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Profile: {scenario.profiles?.name} | {scenario.terminal_goals.length} terminal goals
                  </p>
                  <p className="mt-1 text-xs text-gray-400">
                    {scenario.n_sims} sims | T_max={scenario.t_max} | {scenario.objective}
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => runOptimization(scenario)}
                    className="rounded-md bg-green-600 px-3 py-1.5 text-sm text-white hover:bg-green-700"
                  >
                    Run Optimization
                  </button>
                  <button
                    onClick={() => {
                      if (confirm('Delete this scenario?')) {
                        deleteMutation.mutate(scenario.id)
                      }
                    }}
                    className="rounded-md border border-red-300 bg-white px-3 py-1.5 text-sm text-red-700 hover:bg-red-50"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
