/**
 * Form validation utilities for FinOpt
 */

export interface ValidationError {
  field: string
  message: string
}

export interface ValidationResult {
  valid: boolean
  errors: ValidationError[]
}

// ============================================================================
// Profile Validation
// ============================================================================

export interface ProfileFormData {
  name: string
  income_config: {
    fixed?: {
      base: number
      annual_growth: number
    }
    variable?: {
      base: number
      sigma: number
      annual_growth?: number
      floor?: number
      cap?: number
    }
    contribution_rate_fixed: number | number[]
    contribution_rate_variable: number | number[]
  }
  accounts_config: Array<{
    name: string
    annual_return: number
    annual_volatility: number
    initial_wealth: number
  }>
}

export function validateProfile(data: ProfileFormData): ValidationResult {
  const errors: ValidationError[] = []

  // Name validation
  if (!data.name || data.name.trim().length === 0) {
    errors.push({ field: 'name', message: 'Profile name is required' })
  } else if (data.name.length > 100) {
    errors.push({ field: 'name', message: 'Profile name must be less than 100 characters' })
  }

  // Fixed income validation
  if (data.income_config.fixed) {
    if (data.income_config.fixed.base < 0) {
      errors.push({ field: 'fixed.base', message: 'Base income cannot be negative' })
    }
    if (data.income_config.fixed.annual_growth < -1 || data.income_config.fixed.annual_growth > 1) {
      errors.push({ field: 'fixed.annual_growth', message: 'Annual growth must be between -100% and 100%' })
    }
  }

  // Variable income validation
  if (data.income_config.variable) {
    if (data.income_config.variable.base < 0) {
      errors.push({ field: 'variable.base', message: 'Variable income base cannot be negative' })
    }
    if (data.income_config.variable.sigma < 0 || data.income_config.variable.sigma > 2) {
      errors.push({ field: 'variable.sigma', message: 'Volatility (sigma) must be between 0 and 200%' })
    }
    if (data.income_config.variable.floor !== undefined &&
        data.income_config.variable.cap !== undefined &&
        data.income_config.variable.floor > data.income_config.variable.cap) {
      errors.push({ field: 'variable.floor', message: 'Floor cannot be greater than cap' })
    }
  }

  // Contribution rate validation
  const validateContributionRate = (rate: number | number[], field: string) => {
    const rates = Array.isArray(rate) ? rate : [rate]
    rates.forEach((r, i) => {
      if (r < 0 || r > 1) {
        const suffix = Array.isArray(rate) ? ` (month ${i + 1})` : ''
        errors.push({ field, message: `Contribution rate${suffix} must be between 0 and 1` })
      }
    })
  }
  validateContributionRate(data.income_config.contribution_rate_fixed, 'contribution_rate_fixed')
  validateContributionRate(data.income_config.contribution_rate_variable, 'contribution_rate_variable')

  // Accounts validation
  if (data.accounts_config.length === 0) {
    errors.push({ field: 'accounts', message: 'At least one account is required' })
  }

  const accountNames = new Set<string>()
  data.accounts_config.forEach((account, i) => {
    if (!account.name || account.name.trim().length === 0) {
      errors.push({ field: `accounts[${i}].name`, message: `Account ${i + 1} name is required` })
    } else if (accountNames.has(account.name)) {
      errors.push({ field: `accounts[${i}].name`, message: `Duplicate account name: ${account.name}` })
    } else {
      accountNames.add(account.name)
    }

    if (account.annual_return < -1 || account.annual_return > 2) {
      errors.push({ field: `accounts[${i}].annual_return`, message: `Account ${i + 1} return must be between -100% and 200%` })
    }
    if (account.annual_volatility < 0 || account.annual_volatility > 2) {
      errors.push({ field: `accounts[${i}].annual_volatility`, message: `Account ${i + 1} volatility must be between 0 and 200%` })
    }
    if (account.initial_wealth < 0) {
      errors.push({ field: `accounts[${i}].initial_wealth`, message: `Account ${i + 1} initial wealth cannot be negative` })
    }
  })

  return { valid: errors.length === 0, errors }
}

// ============================================================================
// Scenario Validation
// ============================================================================

export interface ScenarioFormData {
  profile_id: string
  name: string
  start_date: string
  n_sims: number
  t_max: number
  t_min?: number
  terminal_goals: Array<{
    account: string
    threshold: number
    confidence: number
  }>
  intermediate_goals: Array<{
    account: string
    threshold: number
    confidence: number
    date: string
  }>
}

export function validateScenario(data: ScenarioFormData): ValidationResult {
  const errors: ValidationError[] = []

  // Required fields
  if (!data.profile_id) {
    errors.push({ field: 'profile_id', message: 'Please select a profile' })
  }
  if (!data.name || data.name.trim().length === 0) {
    errors.push({ field: 'name', message: 'Scenario name is required' })
  } else if (data.name.length > 100) {
    errors.push({ field: 'name', message: 'Scenario name must be less than 100 characters' })
  }

  // Start date validation
  if (!data.start_date) {
    errors.push({ field: 'start_date', message: 'Start date is required' })
  }

  // Simulation parameters
  if (data.n_sims < 100 || data.n_sims > 10000) {
    errors.push({ field: 'n_sims', message: 'Number of simulations must be between 100 and 10,000' })
  }
  if (data.t_max < 12 || data.t_max > 600) {
    errors.push({ field: 't_max', message: 'Maximum horizon must be between 12 and 600 months' })
  }
  if (data.t_min !== undefined && (data.t_min < 1 || data.t_min > data.t_max)) {
    errors.push({ field: 't_min', message: 'Minimum horizon must be between 1 and T_max' })
  }

  // Goals validation
  if (data.terminal_goals.length === 0 && data.intermediate_goals.length === 0) {
    errors.push({ field: 'goals', message: 'At least one goal (terminal or intermediate) is required' })
  }

  data.terminal_goals.forEach((goal, i) => {
    if (goal.threshold <= 0) {
      errors.push({ field: `terminal_goals[${i}].threshold`, message: `Terminal goal ${i + 1} threshold must be positive` })
    }
    if (goal.confidence <= 0 || goal.confidence > 1) {
      errors.push({ field: `terminal_goals[${i}].confidence`, message: `Terminal goal ${i + 1} confidence must be between 0 and 1` })
    }
  })

  data.intermediate_goals.forEach((goal, i) => {
    if (goal.threshold <= 0) {
      errors.push({ field: `intermediate_goals[${i}].threshold`, message: `Intermediate goal ${i + 1} threshold must be positive` })
    }
    if (goal.confidence <= 0 || goal.confidence > 1) {
      errors.push({ field: `intermediate_goals[${i}].confidence`, message: `Intermediate goal ${i + 1} confidence must be between 0 and 1` })
    }
    if (!goal.date) {
      errors.push({ field: `intermediate_goals[${i}].date`, message: `Intermediate goal ${i + 1} date is required` })
    } else {
      const goalDate = new Date(goal.date)
      const startDate = new Date(data.start_date)
      if (goalDate <= startDate) {
        errors.push({ field: `intermediate_goals[${i}].date`, message: `Intermediate goal ${i + 1} date must be after start date` })
      }
    }
  })

  return { valid: errors.length === 0, errors }
}

// ============================================================================
// Helper functions
// ============================================================================

export function getFieldError(errors: ValidationError[], field: string): string | undefined {
  return errors.find(e => e.field === field)?.message
}

export function hasFieldError(errors: ValidationError[], field: string): boolean {
  return errors.some(e => e.field === field || e.field.startsWith(`${field}[`) || e.field.startsWith(`${field}.`))
}
