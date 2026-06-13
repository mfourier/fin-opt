import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabase'
import { useAuthStore } from '../lib/store'
import { queueOptimization } from '../lib/api'
import { useToast } from '../components/Toast'
import { GoalsWizard } from '@/components/finopt/GoalsWizard'
import { Button } from '@/components/ui/button'
import type { Profile, Scenario, ScenarioInsert } from '../types/database'
import type { ScenarioDraft } from '@/mocks/types'

// Plain-language labels for the optimization objective (no solver jargon).
const OBJECTIVE_LABELS: Record<string, string> = {
  risky: 'Maximum growth',
  balanced: 'Balanced',
  conservative: 'Conservative',
  risky_turnover: 'Growth (stable)',
  proportional: 'Steady & even',
}

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
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const user = useAuthStore((state) => state.user)
  const toast = useToast()
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
      toast.error('Failed to create plan', error.message)
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
      toast.error('Failed to update plan', error.message)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase.from('scenarios').delete().eq('id', id)
      if (error) throw error
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] })
      toast.success('Plan deleted', 'The plan has been removed.')
    },
    onError: (error: Error) => {
      toast.error('Failed to delete plan', error.message)
    },
  })

  const runOptimization = async (scenarioId: string) => {
    const { data: job, error: jobError } = await supabase
      .from('jobs')
      .insert({ scenario_id: scenarioId, job_type: 'optimization', status: 'pending', progress: 0 })
      .select()
      .single()

    if (jobError) {
      toast.error('Failed to create job', jobError.message)
      return
    }

    try {
      await queueOptimization({ scenario_id: scenarioId, job_id: job.id })
      queryClient.invalidateQueries({ queryKey: ['recent-jobs'] })
      toast.info('Calculating your plan', 'Redirecting to results…')
      navigate(`/results/${job.id}`)
    } catch (err) {
      toast.error('Failed to start calculation', err instanceof Error ? err.message : 'Unknown error')
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
      toast.error('No results yet', 'This demo has no computed results.')
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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Plans</h1>
          <p className="mt-1 text-sm text-gray-500">
            Tell us your goals and we'll find the shortest path to reach them.
          </p>
        </div>
        <button
          onClick={() => setShowForm(true)}
          disabled={!ownProfiles?.length}
          className="rounded-md bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50"
        >
          New Plan
        </button>
      </div>

      {!ownProfiles?.length && (
        <div className="rounded-md bg-yellow-50 p-4 text-sm text-yellow-700">
          Create a profile first before creating a plan.
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
                Close
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
      <div className="rounded-lg bg-white shadow">
        {isLoading ? (
          <div className="p-6 text-center text-gray-500">Loading...</div>
        ) : scenarios?.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            No plans yet. Create one to get started.
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {scenarios?.map((scenario) => (
              <div key={scenario.id} className="flex items-center justify-between p-6">
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-medium text-gray-900">{scenario.name}</h3>
                    {scenario.is_demo && (
                      <span className="rounded-full bg-primary-100 px-2 py-0.5 text-xs font-medium text-primary-700">
                        Demo
                      </span>
                    )}
                  </div>
                  <p className="mt-1 text-sm text-gray-500">
                    Profile: {scenario.profiles?.name}
                  </p>
                  <div className="mt-1 flex flex-wrap gap-2 text-xs text-gray-400">
                    <span>{scenario.terminal_goals?.length ?? 0} goals</span>
                    {(scenario.intermediate_goals?.length ?? 0) > 0 && (
                      <span>| {scenario.intermediate_goals.length} dated</span>
                    )}
                    {scenario.withdrawals && (
                      <span>
                        | {(scenario.withdrawals.scheduled?.length ?? 0) + (scenario.withdrawals.stochastic?.length ?? 0)} withdrawals
                      </span>
                    )}
                    <span>| {OBJECTIVE_LABELS[scenario.objective] ?? scenario.objective}</span>
                  </div>
                </div>
                <div className="flex gap-2">
                  {scenario.is_demo ? (
                    <button
                      onClick={() => viewResults(scenario.id)}
                      className="rounded-md bg-primary-600 px-3 py-1.5 text-sm text-white hover:bg-primary-700"
                    >
                      View results
                    </button>
                  ) : (
                  <>
                  <button
                    onClick={() => handleEdit(scenario)}
                    className="rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => runOptimization(scenario.id)}
                    className="rounded-md bg-green-600 px-3 py-1.5 text-sm text-white hover:bg-green-700"
                  >
                    Run
                  </button>
                  <button
                    onClick={() => {
                      if (confirm('Delete this plan?')) {
                        deleteMutation.mutate(scenario.id)
                      }
                    }}
                    className="rounded-md border border-red-300 bg-white px-3 py-1.5 text-sm text-red-700 hover:bg-red-50"
                  >
                    Delete
                  </button>
                  </>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
