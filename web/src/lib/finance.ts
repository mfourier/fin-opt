import type { Job, Profile, Result } from '../types/database'

export type ResultPreviewLite = Pick<Result, 'job_id' | 'feasible' | 'optimal_horizon' | 'goal_status'>
export type PlanHealth = 'on_track' | 'tight' | 'needs_changes' | 'running' | 'queued' | 'failed' | 'draft' | 'completed'

function averageRate(rate: number | number[] | undefined) {
  if (Array.isArray(rate)) {
    if (rate.length === 0) return 0
    return rate.reduce((sum, value) => sum + value, 0) / rate.length
  }
  return rate ?? 0
}

function weightedAverage(
  values: Array<{ weight: number; value: number }>,
  fallback: number,
) {
  const totalWeight = values.reduce((sum, item) => sum + item.weight, 0)
  if (totalWeight > 0) {
    return values.reduce((sum, item) => sum + item.value * item.weight, 0) / totalWeight
  }
  if (values.length === 0) return fallback
  return values.reduce((sum, item) => sum + item.value, 0) / values.length
}

export function getProfileStartingBalance(profile: Pick<Profile, 'accounts_config'>) {
  return profile.accounts_config.reduce((sum, account) => sum + (account.initial_wealth ?? 0), 0)
}

export function getProfileMonthlyIncome(profile: Pick<Profile, 'income_config'>) {
  const fixedIncome = profile.income_config.fixed?.base ?? 0
  const variableIncome = profile.income_config.variable?.base ?? 0
  return fixedIncome + variableIncome
}

export function getProfileMonthlyContributionCapacity(profile: Pick<Profile, 'income_config'>) {
  const fixedIncome = profile.income_config.fixed?.base ?? 0
  const variableIncome = profile.income_config.variable?.base ?? 0
  return (
    fixedIncome * averageRate(profile.income_config.contribution_rate_fixed)
    + variableIncome * averageRate(profile.income_config.contribution_rate_variable)
  )
}

export function getProfileWeightedReturn(profile: Pick<Profile, 'accounts_config'>) {
  return weightedAverage(
    profile.accounts_config.map((account) => ({
      weight: account.initial_wealth ?? 0,
      value: account.annual_return ?? 0,
    })),
    0,
  )
}

export function getProfileWeightedVolatility(profile: Pick<Profile, 'accounts_config'>) {
  return weightedAverage(
    profile.accounts_config.map((account) => ({
      weight: account.initial_wealth ?? 0,
      value: account.annual_volatility ?? 0,
    })),
    0,
  )
}

export function getProfileIncomeMix(profile: Pick<Profile, 'income_config'>) {
  const fixedIncome = profile.income_config.fixed?.base ?? 0
  const variableIncome = profile.income_config.variable?.base ?? 0
  const totalIncome = fixedIncome + variableIncome
  return {
    fixedIncome,
    variableIncome,
    fixedShare: totalIncome > 0 ? fixedIncome / totalIncome : 0,
    variableShare: totalIncome > 0 ? variableIncome / totalIncome : 0,
  }
}

export function describeProfileRisk(profile: Pick<Profile, 'accounts_config'>) {
  if (profile.accounts_config.length === 0) return 'No accounts'
  const weightedVolatility = getProfileWeightedVolatility(profile)
  if (weightedVolatility >= 0.14) return 'Growth oriented'
  if (weightedVolatility >= 0.08) return 'Balanced risk'
  return 'Lower volatility'
}

export function getProfileTopAccounts(profile: Pick<Profile, 'accounts_config'>, limit = 3) {
  const totalBalance = getProfileStartingBalance(profile)
  return [...profile.accounts_config]
    .sort((left, right) => (right.initial_wealth ?? 0) - (left.initial_wealth ?? 0))
    .slice(0, limit)
    .map((account) => ({
      id: account.name,
      name: account.display_name || account.name,
      balance: account.initial_wealth ?? 0,
      share: totalBalance > 0 ? (account.initial_wealth ?? 0) / totalBalance : 0,
      annualReturn: account.annual_return ?? 0,
      annualVolatility: account.annual_volatility ?? 0,
    }))
}

export function summarizeGoalStatus(goalStatus?: Result['goal_status']) {
  const total = goalStatus?.length ?? 0
  const met = goalStatus?.filter((goal) => goal.satisfied).length ?? 0
  return { met, total }
}

export function getPlanHealth(
  job?: Pick<Job, 'status'>,
  result?: Pick<Result, 'feasible' | 'goal_status'>,
): PlanHealth {
  if (!job) return 'draft'
  if (job.status === 'running') return 'running'
  if (job.status === 'pending') return 'queued'
  if (job.status === 'failed') return 'failed'
  if (!result) return 'completed'
  if (result.feasible === false) return 'needs_changes'

  const { met, total } = summarizeGoalStatus(result.goal_status)
  if (total > 0 && met < total) return 'tight'
  return 'on_track'
}
