import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabase'
import { useAuthStore } from '../lib/store'
import { queueOptimization } from '../lib/api'
import type {
  Profile,
  Scenario,
  ScenarioInsert,
  IntermediateGoal,
  TerminalGoal,
  WithdrawalsConfig,
  ScheduledWithdrawal,
  StochasticWithdrawal,
} from '../types/database'

interface FormData {
  profile_id: string
  name: string
  description: string
  start_date: string
  n_sims: number
  seed: number | null
  t_max: number
  t_min?: number
  solver: string
  objective: string
  terminal_goals: TerminalGoal[]
  intermediate_goals: IntermediateGoal[]
  withdrawals: WithdrawalsConfig
}

const emptyWithdrawals: WithdrawalsConfig = {
  scheduled: [],
  stochastic: [],
}

const getDefaultFormData = (): FormData => ({
  profile_id: '',
  name: '',
  description: '',
  start_date: new Date().toISOString().split('T')[0],
  n_sims: 500,
  seed: 42,
  t_max: 120,
  solver: 'ECOS',
  objective: 'balanced',
  terminal_goals: [],
  intermediate_goals: [],
  withdrawals: { ...emptyWithdrawals },
})

export default function ScenariosPage() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const user = useAuthStore((state) => state.user)
  const [showForm, setShowForm] = useState(false)
  const [editingScenario, setEditingScenario] = useState<Scenario | null>(null)

  // Section toggles
  const [showWithdrawals, setShowWithdrawals] = useState(false)
  const [showIntermediateGoals, setShowIntermediateGoals] = useState(false)

  const [formData, setFormData] = useState<FormData>(getDefaultFormData())

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
    setEditingScenario(null)
    setShowWithdrawals(false)
    setShowIntermediateGoals(false)
    setFormData(getDefaultFormData())
  }

  const handleEdit = (scenario: Scenario) => {
    setEditingScenario(scenario)
    setShowWithdrawals(
      !!scenario.withdrawals &&
      ((scenario.withdrawals.scheduled?.length ?? 0) > 0 || (scenario.withdrawals.stochastic?.length ?? 0) > 0)
    )
    setShowIntermediateGoals((scenario.intermediate_goals?.length ?? 0) > 0)
    setFormData({
      profile_id: scenario.profile_id,
      name: scenario.name,
      description: scenario.description,
      start_date: scenario.start_date,
      n_sims: scenario.n_sims,
      seed: scenario.seed,
      t_max: scenario.t_max,
      t_min: scenario.t_min,
      solver: scenario.solver,
      objective: scenario.objective,
      terminal_goals: scenario.terminal_goals ?? [],
      intermediate_goals: scenario.intermediate_goals ?? [],
      withdrawals: scenario.withdrawals ?? { ...emptyWithdrawals },
    })
    setShowForm(true)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    const scenarioData: ScenarioInsert = {
      profile_id: formData.profile_id,
      name: formData.name,
      description: formData.description,
      start_date: formData.start_date,
      n_sims: formData.n_sims,
      seed: formData.seed,
      t_max: formData.t_max,
      t_min: formData.t_min,
      solver: formData.solver,
      objective: formData.objective,
      terminal_goals: formData.terminal_goals,
      intermediate_goals: showIntermediateGoals ? formData.intermediate_goals : [],
      withdrawals: showWithdrawals ? formData.withdrawals : null,
    }

    if (editingScenario) {
      updateMutation.mutate({ id: editingScenario.id, ...scenarioData })
    } else {
      createMutation.mutate(scenarioData)
    }
  }

  const selectedProfile = profiles?.find(p => p.id === formData.profile_id)
  const accountOptions = selectedProfile?.accounts_config ?? []

  // Terminal Goal helpers
  const addTerminalGoal = () => {
    const accountName = accountOptions[0]?.name ?? 'Account'
    setFormData({
      ...formData,
      terminal_goals: [...formData.terminal_goals, { account: accountName, threshold: 50000000, confidence: 0.80 }],
    })
  }

  const updateTerminalGoal = (index: number, field: keyof TerminalGoal, value: number | string) => {
    const newGoals = [...formData.terminal_goals]
    newGoals[index] = { ...newGoals[index], [field]: value }
    setFormData({ ...formData, terminal_goals: newGoals })
  }

  const removeTerminalGoal = (index: number) => {
    setFormData({
      ...formData,
      terminal_goals: formData.terminal_goals.filter((_, i) => i !== index),
    })
  }

  // Intermediate Goal helpers
  const addIntermediateGoal = () => {
    const accountName = accountOptions[0]?.name ?? 'Account'
    const futureDate = new Date()
    futureDate.setMonth(futureDate.getMonth() + 12)
    setFormData({
      ...formData,
      intermediate_goals: [
        ...formData.intermediate_goals,
        { account: accountName, threshold: 10000000, confidence: 0.90, date: futureDate.toISOString().split('T')[0] },
      ],
    })
  }

  const updateIntermediateGoal = (index: number, field: keyof IntermediateGoal, value: number | string) => {
    const newGoals = [...formData.intermediate_goals]
    newGoals[index] = { ...newGoals[index], [field]: value }
    setFormData({ ...formData, intermediate_goals: newGoals })
  }

  const removeIntermediateGoal = (index: number) => {
    setFormData({
      ...formData,
      intermediate_goals: formData.intermediate_goals.filter((_, i) => i !== index),
    })
  }

  // Scheduled Withdrawal helpers
  const addScheduledWithdrawal = () => {
    const accountName = accountOptions[0]?.name ?? 'Account'
    const futureDate = new Date()
    futureDate.setMonth(futureDate.getMonth() + 6)
    const newWithdrawal: ScheduledWithdrawal = {
      account: accountName,
      amount: 1000000,
      date: futureDate.toISOString().split('T')[0],
    }
    setFormData({
      ...formData,
      withdrawals: {
        ...formData.withdrawals,
        scheduled: [...formData.withdrawals.scheduled, newWithdrawal],
      },
    })
  }

  const updateScheduledWithdrawal = (index: number, field: keyof ScheduledWithdrawal, value: string | number) => {
    const newScheduled = [...formData.withdrawals.scheduled]
    newScheduled[index] = { ...newScheduled[index], [field]: value }
    setFormData({
      ...formData,
      withdrawals: { ...formData.withdrawals, scheduled: newScheduled },
    })
  }

  const removeScheduledWithdrawal = (index: number) => {
    setFormData({
      ...formData,
      withdrawals: {
        ...formData.withdrawals,
        scheduled: formData.withdrawals.scheduled.filter((_, i) => i !== index),
      },
    })
  }

  // Stochastic Withdrawal helpers
  const addStochasticWithdrawal = () => {
    const accountName = accountOptions[0]?.name ?? 'Account'
    const newWithdrawal: StochasticWithdrawal = {
      account: accountName,
      base_amount: 500000,
      sigma: 0.2,
      month: 11, // December
    }
    setFormData({
      ...formData,
      withdrawals: {
        ...formData.withdrawals,
        stochastic: [...formData.withdrawals.stochastic, newWithdrawal],
      },
    })
  }

  const updateStochasticWithdrawal = (
    index: number,
    field: keyof StochasticWithdrawal,
    value: string | number | undefined
  ) => {
    const newStochastic = [...formData.withdrawals.stochastic]
    newStochastic[index] = { ...newStochastic[index], [field]: value }
    setFormData({
      ...formData,
      withdrawals: { ...formData.withdrawals, stochastic: newStochastic },
    })
  }

  const removeStochasticWithdrawal = (index: number) => {
    setFormData({
      ...formData,
      withdrawals: {
        ...formData.withdrawals,
        stochastic: formData.withdrawals.stochastic.filter((_, i) => i !== index),
      },
    })
  }

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
          <div className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-lg bg-white p-6 shadow-xl">
            <h2 className="mb-4 text-lg font-medium text-gray-900">
              {editingScenario ? 'Edit Scenario' : 'Create Scenario'}
            </h2>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Basic Info */}
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
                        terminal_goals: formData.terminal_goals.length > 0
                          ? formData.terminal_goals.map(g => ({ ...g, account: firstAccountName }))
                          : [{ account: firstAccountName, threshold: 50000000, confidence: 0.80 }],
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

              {/* Simulation Parameters */}
              <div className="rounded-lg border border-gray-200 p-4">
                <h3 className="mb-3 font-medium text-gray-900">Simulation Parameters</h3>
                <div className="grid grid-cols-4 gap-4">
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
                    <label className="block text-sm text-gray-600">Seed (optional)</label>
                    <input
                      type="number"
                      value={formData.seed ?? ''}
                      onChange={(e) => setFormData({ ...formData, seed: e.target.value ? Number(e.target.value) : null })}
                      placeholder="Random"
                      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600">T Max (months)</label>
                    <input
                      type="number"
                      value={formData.t_max}
                      onChange={(e) => setFormData({ ...formData, t_max: Number(e.target.value) })}
                      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    />
                  </div>
                </div>
                <div className="mt-4 grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm text-gray-600">T Min (optional)</label>
                    <input
                      type="number"
                      value={formData.t_min ?? ''}
                      onChange={(e) => setFormData({ ...formData, t_min: e.target.value ? Number(e.target.value) : undefined })}
                      placeholder="Auto"
                      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    />
                    <p className="mt-1 text-xs text-gray-500">Minimum horizon to search</p>
                  </div>
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
                      <option value="balanced">Balanced (min turnover)</option>
                      <option value="risky">Risky (max wealth)</option>
                      <option value="conservative">Conservative (mean-variance)</option>
                      <option value="risky_turnover">Risky + Turnover penalty</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Terminal Goals */}
              <div className="rounded-lg border border-gray-200 p-4">
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="font-medium text-gray-900">Terminal Goals (at horizon T*)</h3>
                  <button
                    type="button"
                    onClick={addTerminalGoal}
                    disabled={!formData.profile_id}
                    className="text-sm text-primary-600 hover:text-primary-500 disabled:text-gray-400"
                  >
                    + Add Goal
                  </button>
                </div>
                {formData.terminal_goals.length === 0 ? (
                  <p className="text-sm text-gray-500">No terminal goals. Add at least one goal.</p>
                ) : (
                  <div className="space-y-3">
                    {formData.terminal_goals.map((goal, index) => (
                      <div key={index} className="flex items-end gap-3 rounded-md border border-gray-100 bg-gray-50 p-3">
                        <div className="flex-1">
                          <label className="block text-xs text-gray-500">Account</label>
                          <select
                            value={goal.account}
                            onChange={(e) => updateTerminalGoal(index, 'account', e.target.value)}
                            className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                          >
                            {accountOptions.map((acc) => (
                              <option key={acc.name} value={acc.name}>
                                {acc.display_name || acc.name}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="flex-1">
                          <label className="block text-xs text-gray-500">Target Amount</label>
                          <input
                            type="number"
                            value={goal.threshold}
                            onChange={(e) => updateTerminalGoal(index, 'threshold', Number(e.target.value))}
                            className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                          />
                        </div>
                        <div className="w-24">
                          <label className="block text-xs text-gray-500">Confidence</label>
                          <input
                            type="number"
                            step="0.01"
                            min="0"
                            max="1"
                            value={goal.confidence}
                            onChange={(e) => updateTerminalGoal(index, 'confidence', Number(e.target.value))}
                            className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                          />
                        </div>
                        <button
                          type="button"
                          onClick={() => removeTerminalGoal(index)}
                          className="mb-1 text-red-500 hover:text-red-600"
                        >
                          &times;
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Intermediate Goals */}
              <div className="rounded-lg border border-gray-200 p-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium text-gray-900">Intermediate Goals (at specific dates)</h3>
                  {!showIntermediateGoals ? (
                    <button
                      type="button"
                      onClick={() => setShowIntermediateGoals(true)}
                      className="text-sm text-primary-600 hover:text-primary-500"
                    >
                      + Enable
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => {
                        setShowIntermediateGoals(false)
                        setFormData({ ...formData, intermediate_goals: [] })
                      }}
                      className="text-sm text-red-600 hover:text-red-500"
                    >
                      Disable
                    </button>
                  )}
                </div>

                {showIntermediateGoals && (
                  <div className="mt-3">
                    <div className="mb-2 flex justify-end">
                      <button
                        type="button"
                        onClick={addIntermediateGoal}
                        disabled={!formData.profile_id}
                        className="text-sm text-primary-600 hover:text-primary-500 disabled:text-gray-400"
                      >
                        + Add Intermediate Goal
                      </button>
                    </div>
                    {formData.intermediate_goals.length === 0 ? (
                      <p className="text-sm text-gray-500">No intermediate goals.</p>
                    ) : (
                      <div className="space-y-3">
                        {formData.intermediate_goals.map((goal, index) => (
                          <div key={index} className="flex items-end gap-3 rounded-md border border-gray-100 bg-gray-50 p-3">
                            <div className="flex-1">
                              <label className="block text-xs text-gray-500">Account</label>
                              <select
                                value={goal.account}
                                onChange={(e) => updateIntermediateGoal(index, 'account', e.target.value)}
                                className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                              >
                                {accountOptions.map((acc) => (
                                  <option key={acc.name} value={acc.name}>
                                    {acc.display_name || acc.name}
                                  </option>
                                ))}
                              </select>
                            </div>
                            <div className="flex-1">
                              <label className="block text-xs text-gray-500">Target Date</label>
                              <input
                                type="date"
                                value={goal.date}
                                onChange={(e) => updateIntermediateGoal(index, 'date', e.target.value)}
                                className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                              />
                            </div>
                            <div className="flex-1">
                              <label className="block text-xs text-gray-500">Target Amount</label>
                              <input
                                type="number"
                                value={goal.threshold}
                                onChange={(e) => updateIntermediateGoal(index, 'threshold', Number(e.target.value))}
                                className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                              />
                            </div>
                            <div className="w-24">
                              <label className="block text-xs text-gray-500">Confidence</label>
                              <input
                                type="number"
                                step="0.01"
                                min="0"
                                max="1"
                                value={goal.confidence}
                                onChange={(e) => updateIntermediateGoal(index, 'confidence', Number(e.target.value))}
                                className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                              />
                            </div>
                            <button
                              type="button"
                              onClick={() => removeIntermediateGoal(index)}
                              className="mb-1 text-red-500 hover:text-red-600"
                            >
                              &times;
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Withdrawals */}
              <div className="rounded-lg border border-gray-200 p-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium text-gray-900">Withdrawals</h3>
                  {!showWithdrawals ? (
                    <button
                      type="button"
                      onClick={() => setShowWithdrawals(true)}
                      className="text-sm text-primary-600 hover:text-primary-500"
                    >
                      + Enable
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => {
                        setShowWithdrawals(false)
                        setFormData({ ...formData, withdrawals: { ...emptyWithdrawals } })
                      }}
                      className="text-sm text-red-600 hover:text-red-500"
                    >
                      Disable
                    </button>
                  )}
                </div>

                {showWithdrawals && (
                  <div className="mt-4 space-y-6">
                    {/* Scheduled Withdrawals */}
                    <div>
                      <div className="mb-2 flex items-center justify-between">
                        <h4 className="text-sm font-medium text-gray-700">Scheduled (Deterministic)</h4>
                        <button
                          type="button"
                          onClick={addScheduledWithdrawal}
                          disabled={!formData.profile_id}
                          className="text-xs text-primary-600 hover:text-primary-500 disabled:text-gray-400"
                        >
                          + Add
                        </button>
                      </div>
                      {formData.withdrawals.scheduled.length === 0 ? (
                        <p className="text-xs text-gray-500">No scheduled withdrawals.</p>
                      ) : (
                        <div className="space-y-2">
                          {formData.withdrawals.scheduled.map((w, index) => (
                            <div key={index} className="flex items-end gap-2 rounded-md bg-gray-50 p-2">
                              <div className="flex-1">
                                <label className="block text-xs text-gray-500">Account</label>
                                <select
                                  value={w.account}
                                  onChange={(e) => updateScheduledWithdrawal(index, 'account', e.target.value)}
                                  className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-xs focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                                >
                                  {accountOptions.map((acc) => (
                                    <option key={acc.name} value={acc.name}>
                                      {acc.display_name || acc.name}
                                    </option>
                                  ))}
                                </select>
                              </div>
                              <div className="flex-1">
                                <label className="block text-xs text-gray-500">Date</label>
                                <input
                                  type="date"
                                  value={w.date}
                                  onChange={(e) => updateScheduledWithdrawal(index, 'date', e.target.value)}
                                  className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-xs focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                                />
                              </div>
                              <div className="flex-1">
                                <label className="block text-xs text-gray-500">Amount</label>
                                <input
                                  type="number"
                                  value={w.amount}
                                  onChange={(e) => updateScheduledWithdrawal(index, 'amount', Number(e.target.value))}
                                  className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-xs focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                                />
                              </div>
                              <div className="flex-1">
                                <label className="block text-xs text-gray-500">Description</label>
                                <input
                                  type="text"
                                  value={w.description ?? ''}
                                  onChange={(e) => updateScheduledWithdrawal(index, 'description', e.target.value)}
                                  placeholder="Optional"
                                  className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-xs focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                                />
                              </div>
                              <button
                                type="button"
                                onClick={() => removeScheduledWithdrawal(index)}
                                className="mb-1 text-red-500 hover:text-red-600"
                              >
                                &times;
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Stochastic Withdrawals */}
                    <div>
                      <div className="mb-2 flex items-center justify-between">
                        <h4 className="text-sm font-medium text-gray-700">Stochastic (Recurring/Variable)</h4>
                        <button
                          type="button"
                          onClick={addStochasticWithdrawal}
                          disabled={!formData.profile_id}
                          className="text-xs text-primary-600 hover:text-primary-500 disabled:text-gray-400"
                        >
                          + Add
                        </button>
                      </div>
                      {formData.withdrawals.stochastic.length === 0 ? (
                        <p className="text-xs text-gray-500">No stochastic withdrawals.</p>
                      ) : (
                        <div className="space-y-2">
                          {formData.withdrawals.stochastic.map((w, index) => (
                            <div key={index} className="flex items-end gap-2 rounded-md bg-gray-50 p-2">
                              <div className="flex-1">
                                <label className="block text-xs text-gray-500">Account</label>
                                <select
                                  value={w.account}
                                  onChange={(e) => updateStochasticWithdrawal(index, 'account', e.target.value)}
                                  className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-xs focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                                >
                                  {accountOptions.map((acc) => (
                                    <option key={acc.name} value={acc.name}>
                                      {acc.display_name || acc.name}
                                    </option>
                                  ))}
                                </select>
                              </div>
                              <div className="w-24">
                                <label className="block text-xs text-gray-500">Base Amount</label>
                                <input
                                  type="number"
                                  value={w.base_amount}
                                  onChange={(e) => updateStochasticWithdrawal(index, 'base_amount', Number(e.target.value))}
                                  className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-xs focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                                />
                              </div>
                              <div className="w-16">
                                <label className="block text-xs text-gray-500">Sigma</label>
                                <input
                                  type="number"
                                  step="0.01"
                                  value={w.sigma}
                                  onChange={(e) => updateStochasticWithdrawal(index, 'sigma', Number(e.target.value))}
                                  className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-xs focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                                />
                              </div>
                              <div className="w-20">
                                <label className="block text-xs text-gray-500">Month (0-11)</label>
                                <input
                                  type="number"
                                  min="0"
                                  max="11"
                                  value={w.month ?? ''}
                                  onChange={(e) => updateStochasticWithdrawal(index, 'month', e.target.value ? Number(e.target.value) : undefined)}
                                  placeholder="All"
                                  className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-xs focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                                />
                              </div>
                              <div className="w-20">
                                <label className="block text-xs text-gray-500">Floor</label>
                                <input
                                  type="number"
                                  value={w.floor ?? ''}
                                  onChange={(e) => updateStochasticWithdrawal(index, 'floor', e.target.value ? Number(e.target.value) : undefined)}
                                  placeholder="None"
                                  className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-xs focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                                />
                              </div>
                              <div className="w-20">
                                <label className="block text-xs text-gray-500">Cap</label>
                                <input
                                  type="number"
                                  value={w.cap ?? ''}
                                  onChange={(e) => updateStochasticWithdrawal(index, 'cap', e.target.value ? Number(e.target.value) : undefined)}
                                  placeholder="None"
                                  className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-xs focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                                />
                              </div>
                              <button
                                type="button"
                                onClick={() => removeStochasticWithdrawal(index)}
                                className="mb-1 text-red-500 hover:text-red-600"
                              >
                                &times;
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Form Actions */}
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
                  disabled={createMutation.isPending || updateMutation.isPending}
                  className="rounded-md bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50"
                >
                  {editingScenario ? 'Update' : 'Create'}
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
                    Profile: {scenario.profiles?.name}
                  </p>
                  <div className="mt-1 flex flex-wrap gap-2 text-xs text-gray-400">
                    <span>{scenario.terminal_goals?.length ?? 0} terminal goals</span>
                    {(scenario.intermediate_goals?.length ?? 0) > 0 && (
                      <span>| {scenario.intermediate_goals.length} intermediate</span>
                    )}
                    {scenario.withdrawals && (
                      <span>
                        | {(scenario.withdrawals.scheduled?.length ?? 0) + (scenario.withdrawals.stochastic?.length ?? 0)} withdrawals
                      </span>
                    )}
                    <span>| {scenario.n_sims} sims</span>
                    <span>| T_max={scenario.t_max}</span>
                    <span>| {scenario.objective}</span>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleEdit(scenario)}
                    className="rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => runOptimization(scenario)}
                    className="rounded-md bg-green-600 px-3 py-1.5 text-sm text-white hover:bg-green-700"
                  >
                    Run
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
