// Base interfaces
export interface IncomeConfig {
  fixed?: {
    base: number
    annual_growth: number
    scheduled_raises?: Array<{ month: number; amount: number }>
  }
  variable?: {
    base: number
    sigma: number
    annual_growth?: number
    seasonality?: number[]
    floor?: number
    cap?: number
  }
  // These match the finopt serialization format
  contribution_rate_fixed?: number | number[]
  contribution_rate_variable?: number | number[]
}

export interface AccountConfig {
  name: string
  annual_return: number
  annual_volatility: number
  initial_wealth: number
}

export interface Goal {
  account: string | number
  threshold: number
  confidence: number
  date?: string
}

export interface Withdrawal {
  account: string | number
  amount: number
  date: string
  confidence?: number
}

export interface SummaryStats {
  mean_wealth: number[]
  std_wealth: number[]
  percentiles: { [key: string]: number[] }
}

export interface GoalStatus {
  goal_type: string
  account: string
  threshold: number
  confidence: number
  achieved: boolean
  actual_probability: number
}

// Row types (full database records)
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
  intermediate_goals: Goal[]
  terminal_goals: Goal[]
  withdrawals: Withdrawal[] | null
  start_date: string
  n_sims: number
  seed: number | null
  t_max: number
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
  summary_stats: SummaryStats | null
  goal_status: GoalStatus[] | null
  created_at: string
}

// Insert types (omit auto-generated fields)
export type ProfileInsert = Omit<Profile, 'id' | 'created_at' | 'updated_at'>
export type ScenarioInsert = Omit<Scenario, 'id' | 'created_at' | 'updated_at'>
export type JobInsert = Omit<Job, 'id' | 'created_at'>
export type ResultInsert = Omit<Result, 'id' | 'created_at'>

// Database schema for Supabase client
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
