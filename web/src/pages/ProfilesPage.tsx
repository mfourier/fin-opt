import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { Trans, useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
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
  getProfileStartingBalance,
  getProfileTopAccounts,
  getProfileWeightedReturn,
} from '@/lib/finance'
import type { Profile, ProfileInsert, Scenario } from '../types/database'
import type { ProfileDraft } from '@/mocks/types'

export default function ProfilesPage() {
  const { t } = useTranslation(['profiles', 'common'])
  const queryClient = useQueryClient()
  const user = useAuthStore((state) => state.user)
  const toast = useToast()
  const perMonth = (value: string) => `${value}${t('common:perMonth')}`
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
      toast.success(t('toast.saved'), t('toast.savedDetail'))
    },
    onError: (error: unknown) => {
      toast.error(t('toast.saveFailed'), getProfileSaveErrorMessage(error, t))
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
      toast.success(t('toast.updated'), t('toast.updatedDetail'))
    },
    onError: (error: unknown) => {
      toast.error(t('toast.updateFailed'), getProfileSaveErrorMessage(error, t))
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase.from('profiles').delete().eq('id', id)
      if (error) throw error
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] })
      toast.success(t('toast.deleted'), t('toast.deletedDetail'))
    },
    onError: (error: Error) => {
      toast.error(t('toast.deleteFailed'), error.message)
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
        editingProfile ? t('toast.nameInUse') : t('toast.alreadyExists'),
        t('toast.duplicateDetail'),
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
      label: t('summary.saved'),
      value: ownProfiles.length,
      detail: ownProfiles.length > 0 ? t('summary.savedDetail') : t('summary.savedEmpty'),
      icon: BriefcaseBusiness,
    },
    {
      label: t('summary.primaryBalance'),
      value: latestProfile ? formatCLP(getProfileStartingBalance(latestProfile)) : '—',
      detail: latestProfile ? t('summary.primaryBalanceDetail', { name: latestProfile.name }) : t('summary.primaryBalanceEmpty'),
      icon: Wallet,
    },
    {
      label: t('summary.investingPower'),
      value: latestProfile ? perMonth(formatCLP(getProfileMonthlyContributionCapacity(latestProfile))) : '—',
      detail: latestProfile ? t('summary.investingPowerDetail', { name: latestProfile.name }) : t('summary.investingPowerEmpty'),
      icon: TrendingUp,
    },
    {
      label: t('summary.expectedReturn'),
      value: latestProfile ? formatPercent(getProfileWeightedReturn(latestProfile), 1) : '—',
      detail: latestProfile ? t('summary.expectedReturnDetail', { name: latestProfile.name }) : t('summary.expectedReturnEmpty'),
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
          <h1 className="text-2xl font-bold text-foreground">{t('title')}</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {t('subtitle')}
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
          {t('newSituation')}
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
          <Trans
            i18nKey="primaryNote"
            t={t}
            values={{ name: latestProfile.name }}
            components={{ name: <span className="font-medium text-foreground" /> }}
          />
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
                {t('close')}
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
          <div className="p-6 text-center text-muted-foreground">{t('loading')}</div>
        ) : profiles?.length === 0 ? (
          <div className="p-8 text-center">
            <p className="text-base font-medium text-foreground">{t('emptyTitle')}</p>
            <p className="mt-2 text-sm text-muted-foreground">
              {t('emptyBody')}
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
              {t('createFirst')}
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
                        {t('demo')}
                      </span>
                    )}
                    {!profile.is_demo && (
                      <span className="rounded-full bg-secondary px-2.5 py-1 text-xs font-medium text-secondary-foreground">
                        {t(`common:profileRisk.${describeProfileRisk(profile)}`)}
                      </span>
                    )}
                  </div>
                  <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                    {profile.description || t('descriptionFallback')}
                  </p>
                  <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                    <MetricPill
                      label={t('metrics.startingBalance')}
                      value={formatCLP(getProfileStartingBalance(profile))}
                    />
                    <MetricPill
                      label={t('metrics.monthlyInvestment')}
                      value={perMonth(formatCLP(getProfileMonthlyContributionCapacity(profile)))}
                    />
                    <MetricPill
                      label={t('metrics.linkedPlans')}
                      value={`${scenarioCountByProfileId[profile.id] ?? 0}`}
                    />
                  </div>

                  <div className="mt-4 grid gap-4 xl:grid-cols-[0.9fr,1.1fr]">
                    <div className="rounded-2xl bg-muted/50 p-4">
                      <p className="text-sm font-medium text-foreground">{t('funding.title')}</p>
                      <p className="mt-1 text-sm text-muted-foreground">
                        {t('funding.subtitle')}
                      </p>
                      <div className="mt-4 grid gap-3 sm:grid-cols-2">
                        <MetricPill
                          label={t('funding.fixedIncome')}
                          value={perMonth(formatCLP(getProfileIncomeMix(profile).fixedIncome))}
                        />
                        <MetricPill
                          label={t('funding.variableIncome')}
                          value={perMonth(formatCLP(getProfileIncomeMix(profile).variableIncome))}
                        />
                        <MetricPill
                          label={t('funding.expectedReturn')}
                          value={formatPercent(getProfileWeightedReturn(profile), 1)}
                        />
                        <MetricPill
                          label={t('funding.incomeMix')}
                          value={t('funding.incomeMixValue', { pct: formatPercent(getProfileIncomeMix(profile).fixedShare, 0) })}
                        />
                      </div>
                    </div>

                    <div className="rounded-2xl bg-muted/50 p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="text-sm font-medium text-foreground">{t('accountMix.title')}</p>
                          <p className="mt-1 text-sm text-muted-foreground">
                            {t('accountMix.subtitle')}
                          </p>
                        </div>
                        <span className="rounded-full bg-secondary px-2.5 py-1 text-xs font-medium text-secondary-foreground">
                          {t(`common:profileRisk.${describeProfileRisk(profile)}`)}
                        </span>
                      </div>
                      <div className="mt-4 space-y-3">
                        {getProfileTopAccounts(profile, 4).map((account) => (
                          <div key={account.id}>
                            <div className="flex items-center justify-between gap-3">
                              <div className="min-w-0">
                                <p className="truncate text-sm font-medium text-foreground">{account.name}</p>
                                <p className="text-xs text-muted-foreground">
                                  {t('accountMix.returnVol', { ret: formatPercent(account.annualReturn, 1), vol: formatPercent(account.annualVolatility, 1) })}
                                </p>
                              </div>
                              <div className="text-right">
                                <p className="tabular text-sm font-semibold text-foreground">{formatCLP(account.balance)}</p>
                                <p className="text-xs text-muted-foreground">{t('accountMix.ofTotal', { pct: formatPercent(account.share, 0) })}</p>
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
                    <span className="text-xs text-muted-foreground">{t('actions.readOnly')}</span>
                    <Button asChild variant="outline" size="sm" className="rounded-xl">
                      <Link to="/scenarios">
                        {t('actions.viewDemoPlan')}
                        <ArrowRight className="h-3.5 w-3.5" />
                      </Link>
                    </Button>
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-2 lg:justify-end">
                    <Button asChild size="sm" className="rounded-xl">
                      <Link to="/scenarios">
                        {t('actions.createPlan')}
                        <ArrowRight className="h-3.5 w-3.5" />
                      </Link>
                    </Button>
                    <Button type="button" variant="outline" size="sm" className="rounded-xl" onClick={() => handleEdit(profile)}>
                      {t('actions.edit')}
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="rounded-xl border-danger/30 text-danger hover:bg-danger-soft hover:text-danger"
                      onClick={() => {
                        if (confirm(t('actions.confirmDelete'))) {
                          deleteMutation.mutate(profile.id)
                        }
                      }}
                    >
                      {t('actions.delete')}
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

function getProfileSaveErrorMessage(error: unknown, t: TFunction): string {
  if (isUniqueConstraintError(error)) {
    return t('error.duplicate')
  }

  if (error instanceof Error) {
    return error.message
  }

  return t('error.unexpected')
}

function isUniqueConstraintError(error: unknown): error is { code?: string; message?: string } {
  if (!error || typeof error !== 'object') return false
  const maybeError = error as { code?: string; message?: string }
  return maybeError.code === '23505'
    || maybeError.message?.includes('profiles_name_user_unique') === true
}
