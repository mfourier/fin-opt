import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { ArrowRight, BriefcaseBusiness, Plus, TrendingUp, Wallet } from 'lucide-react'
import { supabase } from '../lib/supabase'
import { useAuthStore } from '../lib/store'
import { useToast } from '../components/Toast'
import { SituationForm } from '@/components/finopt/SituationForm'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { formatCLP, formatPercent } from '@/lib/format'
import {
  describeProfileRisk,
  getProfileIncomeMix,
  getProfileMonthlyContributionCapacity,
  getProfileMonthlyIncome,
  getProfileStartingBalance,
  getProfileTopAccounts,
  getProfileWeightedReturn,
} from '@/lib/finance'
import type { Profile, ProfileInsert, Scenario } from '../types/database'
import type { ProfileDraft } from '@/mocks/types'

export default function ProfilesPage() {
  const queryClient = useQueryClient()
  const user = useAuthStore((state) => state.user)
  const toast = useToast()
  const [showForm, setShowForm] = useState(false)
  const [editingProfile, setEditingProfile] = useState<Profile | null>(null)

  const { data: profiles, isLoading } = useQuery({
    // Scoped by user so a cached list can never leak across accounts.
    queryKey: ['profiles', user?.id],
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

  const { data: scenarios } = useQuery({
    queryKey: ['profile-scenarios', user?.id],
    queryFn: async () => {
      const { data, error } = await supabase.from('scenarios').select('id, profile_id')
      if (error) throw error
      return data as Pick<Scenario, 'id' | 'profile_id'>[]
    },
    enabled: !!user,
  })

  const createMutation = useMutation({
    mutationFn: async (profile: ProfileInsert) => {
      const { data, error } = await supabase.from('profiles').insert(profile).select().single()
      if (error) throw error
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] })
      resetForm()
      toast.success('Situation saved', 'Your situation has been created.')
    },
    onError: (error: unknown) => {
      toast.error('Failed to save situation', getProfileSaveErrorMessage(error))
    },
  })

  const updateMutation = useMutation({
    mutationFn: async ({ id, ...profile }: Partial<Profile> & { id: string }) => {
      const { data, error } = await supabase
        .from('profiles')
        .update(profile)
        .eq('id', id)
        .select()
        .single()
      if (error) throw error
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] })
      resetForm()
      toast.success('Situation updated', 'Your changes have been saved.')
    },
    onError: (error: unknown) => {
      toast.error('Failed to update situation', getProfileSaveErrorMessage(error))
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase.from('profiles').delete().eq('id', id)
      if (error) throw error
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] })
      toast.success('Situation deleted', 'The situation has been removed.')
    },
    onError: (error: Error) => {
      toast.error('Failed to delete situation', error.message)
    },
  })

  const resetForm = () => {
    setShowForm(false)
    setEditingProfile(null)
  }

  // Close the form overlay with the Escape key.
  useEffect(() => {
    if (!showForm) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') resetForm()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [showForm])

  const handleEdit = (profile: Profile) => {
    setEditingProfile(profile)
    setShowForm(true)
  }

  const handleSave = async (draft: ProfileDraft) => {
    const nextName = draft.name.trim()
    const duplicate = profiles?.find((profile) =>
      profile.name.trim() === nextName && profile.id !== editingProfile?.id,
    )

    if (duplicate) {
      toast.error(
        editingProfile ? 'Name already in use' : 'Situation already exists',
        'Choose a different situation name or edit the existing one.',
      )
      return
    }

    try {
      if (editingProfile) {
        await updateMutation.mutateAsync({
          id: editingProfile.id,
          name: nextName,
          description: draft.description,
          income_config: draft.income_config,
          accounts_config: draft.accounts_config,
          correlation_matrix: draft.correlation_matrix,
        })
      } else {
        await createMutation.mutateAsync({
          user_id: user!.id,
          name: nextName,
          description: draft.description,
          income_config: draft.income_config,
          accounts_config: draft.accounts_config,
          correlation_matrix: draft.correlation_matrix,
        })
      }
    } catch {
      // Mutations surface errors via toast.
    }
  }

  const ownProfiles = profiles?.filter((profile) => !profile.is_demo) ?? []
  const latestProfile = ownProfiles[0] ?? profiles?.[0] ?? null
  const summaryCards = [
    {
      label: 'Saved situations',
      value: ownProfiles.length,
      detail: ownProfiles.length > 0 ? 'Reusable financial setups for future plans.' : 'Create your first situation.',
      icon: BriefcaseBusiness,
    },
    {
      label: 'Primary balance',
      value: latestProfile ? formatCLP(getProfileStartingBalance(latestProfile)) : '—',
      detail: latestProfile ? `From ${latestProfile.name}.` : 'Add account balances to build a baseline.',
      icon: Wallet,
    },
    {
      label: 'Monthly investing power',
      value: latestProfile ? `${formatCLP(getProfileMonthlyContributionCapacity(latestProfile))}/mo` : '—',
      detail: latestProfile ? `Estimated from ${latestProfile.name}.` : 'Add income details to unlock better plans.',
      icon: TrendingUp,
    },
    {
      label: 'Expected annual return',
      value: latestProfile ? formatPercent(getProfileWeightedReturn(latestProfile), 1) : '—',
      detail: latestProfile ? `Weighted by the balances in ${latestProfile.name}.` : 'Add accounts to estimate portfolio growth.',
      icon: TrendingUp,
    },
  ]

  const scenarioCountByProfileId = Object.fromEntries(
    (scenarios ?? []).reduce<[string, number][]>((entries, scenario) => {
      const current = entries.find(([profileId]) => profileId === scenario.profile_id)
      if (current) {
        current[1] += 1
        return entries
      }
      entries.push([scenario.profile_id, 1])
      return entries
    }, []),
  )

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">My situation</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Organize the income, balances, and accounts that power every plan you build in FinOpt.
          </p>
        </div>
        <Button
          type="button"
          onClick={() => {
            setEditingProfile(null)
            setShowForm(true)
          }}
          className="rounded-xl"
        >
          <Plus className="h-4 w-4" />
          New situation
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
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

      {latestProfile && (
        <div className="rounded-2xl border border-border/70 bg-muted/20 px-4 py-3 text-sm text-muted-foreground">
          Top-level metrics on this page are based on your primary situation, <span className="font-medium text-foreground">{latestProfile.name}</span>.
        </div>
      )}

      {/* Form overlay (redesigned) */}
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
            <SituationForm
              initialProfile={editingProfile ?? undefined}
              onSave={handleSave}
              onCancel={resetForm}
            />
          </div>
        </div>
      )}

      {/* Situations list */}
      <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
        {isLoading ? (
          <div className="p-6 text-center text-muted-foreground">Loading your situations...</div>
        ) : profiles?.length === 0 ? (
          <div className="p-8 text-center">
            <p className="text-base font-medium text-foreground">No situations yet.</p>
            <p className="mt-2 text-sm text-muted-foreground">
              Add your income and account balances first, then use them across as many plans as you want.
            </p>
            <Button
              type="button"
              className="mt-4 rounded-xl"
              onClick={() => {
                setEditingProfile(null)
                setShowForm(true)
              }}
            >
              <Plus className="h-4 w-4" />
              Create your first situation
            </Button>
          </div>
        ) : (
          <div className="divide-y divide-border">
            {profiles?.map((profile) => (
              <div
                key={profile.id}
                className="flex flex-col gap-5 p-6 transition-colors hover:bg-muted/20 lg:flex-row lg:items-start lg:justify-between"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-lg font-semibold text-foreground">{profile.name}</h3>
                    {profile.is_demo && (
                      <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                        Demo
                      </span>
                    )}
                    {!profile.is_demo && (
                      <span className="rounded-full bg-secondary px-2.5 py-1 text-xs font-medium text-secondary-foreground">
                        {describeProfileRisk(profile)}
                      </span>
                    )}
                  </div>
                  <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                    {profile.description || 'A reusable financial setup for future plans.'}
                  </p>
                  <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                    <MetricPill
                      label="Starting balance"
                      value={formatCLP(getProfileStartingBalance(profile))}
                    />
                    <MetricPill
                      label="Monthly income"
                      value={`${formatCLP(getProfileMonthlyIncome(profile))}/mo`}
                    />
                    <MetricPill
                      label="Investing power"
                      value={`${formatCLP(getProfileMonthlyContributionCapacity(profile))}/mo`}
                    />
                    <MetricPill
                      label="Linked plans"
                      value={`${scenarioCountByProfileId[profile.id] ?? 0}`}
                    />
                  </div>

                  <div className="mt-4 grid gap-4 xl:grid-cols-[0.9fr,1.1fr]">
                    <div className="rounded-2xl bg-muted/50 p-4">
                      <p className="text-sm font-medium text-foreground">Funding snapshot</p>
                      <p className="mt-1 text-sm text-muted-foreground">
                        How this situation turns income into long-term savings.
                      </p>
                      <div className="mt-4 grid gap-3 sm:grid-cols-2">
                        <MetricPill
                          label="Fixed income"
                          value={`${formatCLP(getProfileIncomeMix(profile).fixedIncome)}/mo`}
                        />
                        <MetricPill
                          label="Variable income"
                          value={`${formatCLP(getProfileIncomeMix(profile).variableIncome)}/mo`}
                        />
                        <MetricPill
                          label="Expected return"
                          value={formatPercent(getProfileWeightedReturn(profile), 1)}
                        />
                        <MetricPill
                          label="Income mix"
                          value={`${formatPercent(getProfileIncomeMix(profile).fixedShare, 0)} fixed`}
                        />
                      </div>
                    </div>

                    <div className="rounded-2xl bg-muted/50 p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="text-sm font-medium text-foreground">Account mix</p>
                          <p className="mt-1 text-sm text-muted-foreground">
                            The highest-balance accounts inside this situation.
                          </p>
                        </div>
                        <span className="rounded-full bg-secondary px-2.5 py-1 text-xs font-medium text-secondary-foreground">
                          {describeProfileRisk(profile)}
                        </span>
                      </div>
                      <div className="mt-4 space-y-3">
                        {getProfileTopAccounts(profile, 4).map((account) => (
                          <div key={account.id}>
                            <div className="flex items-center justify-between gap-3">
                              <div className="min-w-0">
                                <p className="truncate text-sm font-medium text-foreground">{account.name}</p>
                                <p className="text-xs text-muted-foreground">
                                  {formatPercent(account.annualReturn, 1)} return · {formatPercent(account.annualVolatility, 1)} vol
                                </p>
                              </div>
                              <div className="text-right">
                                <p className="tabular text-sm font-semibold text-foreground">{formatCLP(account.balance)}</p>
                                <p className="text-xs text-muted-foreground">{formatPercent(account.share, 0)} of total</p>
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
                </div>
                {profile.is_demo ? (
                  <div className="flex flex-col items-start gap-2 lg:items-end">
                    <span className="text-xs text-muted-foreground">Read-only example</span>
                    <Button asChild variant="outline" size="sm" className="rounded-xl">
                      <Link to="/scenarios">
                        View demo plan
                        <ArrowRight className="h-3.5 w-3.5" />
                      </Link>
                    </Button>
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-2 lg:justify-end">
                    <Button asChild size="sm" className="rounded-xl">
                      <Link to="/scenarios">
                        Create plan
                        <ArrowRight className="h-3.5 w-3.5" />
                      </Link>
                    </Button>
                    <Button type="button" variant="outline" size="sm" className="rounded-xl" onClick={() => handleEdit(profile)}>
                      Edit
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="rounded-xl border-danger/30 text-danger hover:bg-danger-soft hover:text-danger"
                      onClick={() => {
                        if (confirm('Delete this situation?')) {
                          deleteMutation.mutate(profile.id)
                        }
                      }}
                    >
                      Delete
                    </Button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function MetricPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-muted/60 px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-base font-semibold tabular text-foreground">{value}</p>
    </div>
  )
}

function getProfileSaveErrorMessage(error: unknown): string {
  if (isUniqueConstraintError(error)) {
    return 'A situation with that name already exists in your account. Use a different name or edit the existing one.'
  }

  if (error instanceof Error) {
    return error.message
  }

  return 'An unexpected error occurred while saving your situation.'
}

function isUniqueConstraintError(error: unknown): error is { code?: string; message?: string } {
  if (!error || typeof error !== 'object') return false
  const maybeError = error as { code?: string; message?: string }
  return maybeError.code === '23505'
    || maybeError.message?.includes('profiles_name_user_unique') === true
}
