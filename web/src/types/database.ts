// =============================================================================
// FinOpt Database Types
// Matches the JSON schema expected by finopt.serialization
// =============================================================================

// -----------------------------------------------------------------------------
// Income Configuration
// -----------------------------------------------------------------------------

export interface FixedIncomeConfig {
  base: number                    // Base monthly salary
  annual_growth: number           // Annual growth rate (e.g., 0.03 = 3%)
  salary_raises?: Record<string, number>  // Date string -> amount (e.g., "2026-03-01": 400000)
}

export interface VariableIncomeConfig {
  base: number                    // Base monthly variable income
  sigma: number                   // Monthly volatility (e.g., 0.1 = 10%)
  annual_growth?: number          // Annual growth rate
  seasonality?: number[]          // 12 monthly factors (e.g., [0, 0, 0, 0.6, 1, 1.16, ...])
  floor?: number                  // Minimum income
  cap?: number                    // Maximum income
  seed?: number                   // Random seed for reproducibility
}

export interface IncomeConfig {
  fixed?: FixedIncomeConfig
  variable?: VariableIncomeConfig
  // Contribution rates: scalar (same all months) or array[12] (per month)
  contribution_rate_fixed: number | number[]
  contribution_rate_variable: number | number[]
}

// -----------------------------------------------------------------------------
// Account Configuration
// -----------------------------------------------------------------------------

export interface AccountConfig {
  name: string                    // Unique identifier (used in goals/withdrawals)
  display_name?: string           // Human-readable name for UI
  annual_return: number           // Expected annual return (e.g., 0.08 = 8%)
  annual_volatility: number       // Annual volatility (e.g., 0.12 = 12%)
  initial_wealth: number          // Starting balance
}

// -----------------------------------------------------------------------------
// Withdrawal Configuration
// -----------------------------------------------------------------------------

export interface ScheduledWithdrawal {
  account: string                 // Account name
  amount: number                  // Withdrawal amount
  date: string                    // ISO date string (e.g., "2025-07-01")
  description?: string            // Optional description
}

export interface StochasticWithdrawal {
  account: string                 // Account name
  base_amount: number             // Expected withdrawal amount
  sigma: number                   // Volatility
  date?: string                   // Specific date (ISO string)
  month?: number                  // Or recurring month (0-11)
  floor?: number                  // Minimum withdrawal
  cap?: number                    // Maximum withdrawal
  seed?: number                   // Random seed
  description?: string            // Optional description
}

export interface WithdrawalsConfig {
  scheduled: ScheduledWithdrawal[]
  stochastic: StochasticWithdrawal[]
}

// -----------------------------------------------------------------------------
// Goal Configuration
// -----------------------------------------------------------------------------

export interface IntermediateGoal {
  account: string                 // Account name
  threshold: number               // Target amount
  confidence: number              // Required probability (e.g., 0.9 = 90%)
  date: string                    // Target date (ISO string)
}

export interface TerminalGoal {
  account: string                 // Account name
  threshold: number               // Target amount
  confidence: number              // Required probability (e.g., 0.8 = 80%)
}

// Legacy type for backward compatibility
export interface Goal {
  account: string
  threshold: number
  confidence: number
  date?: string                   // Present for intermediate, absent for terminal
}

// -----------------------------------------------------------------------------
// Summary and Result Types
// -----------------------------------------------------------------------------

export interface WealthPercentiles {
  mean: number[]
  p10: number[]
  p25: number[]
  p50: number[]
  p75: number[]
  p90: number[]
}

export interface PerAccountStats extends WealthPercentiles {
  account: string
  display_name: string
}

export interface CashFlowAccountStats {
  account: string
  display_name: string
  mean: number[]
}

export interface CashFlowStats {
  contributions_mean: number[]
  contributions_by_account: CashFlowAccountStats[]
  withdrawals_mean?: number[]
  withdrawals_by_account?: CashFlowAccountStats[]
}

export interface SampledAccountTrajectories {
  account: string
  display_name: string
  trajectories: number[][]
}

export interface SampledTrajectories {
  total: number[][]
  per_account: SampledAccountTrajectories[]
  n_sampled: number
  n_total: number
}

export interface SummaryStats {
  // Optimization results: wealth trajectory percentiles
  total_wealth?: WealthPercentiles
  per_account?: PerAccountStats[]
  // Sampled Monte Carlo trajectories
  trajectories?: SampledTrajectories
  // Cash flow statistics
  cash_flow?: CashFlowStats
  // Simulation results (legacy format)
  mean_wealth?: number[]
  std_wealth?: number[]
  percentiles?: Record<string, number[]>
}

export interface GoalStatus {
  goal: string
  type: string
  account: string
  threshold: number
  required_confidence: number
  satisfied: boolean
  actual_probability?: number
}

// -----------------------------------------------------------------------------
// Database Row Types
// -----------------------------------------------------------------------------

export interface Profile {
  id: string
  user_id: string
  name: string
  description: string
  income_config: IncomeConfig
  accounts_config: AccountConfig[]
  correlation_matrix: number[][] | null
  created_at: string
  updated_at: string
}

export interface Scenario {
  id: string
  profile_id: string
  name: string
  description: string
  intermediate_goals: IntermediateGoal[]
  terminal_goals: TerminalGoal[]
  withdrawals: WithdrawalsConfig | null
  start_date: string
  n_sims: number
  seed: number | null
  t_max: number
  t_min?: number
  solver: string
  objective: string
  created_at: string
  updated_at: string
}

export interface Job {
  id: string
  scenario_id: string
  job_type: 'simulation' | 'optimization'
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  current_step: string | null
  error_message: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string
}

export interface Result {
  id: string
  job_id: string
  result_type: string
  allocation_policy: number[][] | null
  optimal_horizon: number | null
  objective_value: number | null
  feasible: boolean | null
  solve_time: number | null
  diagnostics: Record<string, unknown> | null
  summary_stats: SummaryStats | null
  goal_status: GoalStatus[] | null
  created_at: string
}

// -----------------------------------------------------------------------------
// Insert Types (omit auto-generated fields)
// -----------------------------------------------------------------------------

export type ProfileInsert = Omit<Profile, 'id' | 'created_at' | 'updated_at'>
export type ScenarioInsert = Omit<Scenario, 'id' | 'created_at' | 'updated_at'>
export type JobInsert = Omit<Job, 'id' | 'created_at'>
export type ResultInsert = Omit<Result, 'id' | 'created_at'>

// -----------------------------------------------------------------------------
// Database Schema (for Supabase typed client)
// -----------------------------------------------------------------------------

export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: Profile
        Insert: ProfileInsert
        Update: Partial<ProfileInsert>
      }
      scenarios: {
        Row: Scenario
        Insert: ScenarioInsert
        Update: Partial<ScenarioInsert>
      }
      jobs: {
        Row: Job
        Insert: JobInsert
        Update: Partial<JobInsert>
      }
      results: {
        Row: Result
        Insert: ResultInsert
        Update: Partial<ResultInsert>
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Helper Types for Forms
// -----------------------------------------------------------------------------

export interface SalaryRaise {
  date: string
  amount: number
}

// Default values for forms
export const DEFAULT_SEASONALITY = [0, 0, 0, 0.6, 1.0, 1.16, 1.0, 1.1, 0.5, 0.9, 0.85, 1.0]
export const DEFAULT_CONTRIBUTION_FIXED = 0.3
export const DEFAULT_CONTRIBUTION_VARIABLE = 1.0
