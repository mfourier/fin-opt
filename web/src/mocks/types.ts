// Canonical FinOpt domain types. Do not rename or change shape.

export type AccountConfig = {
  name: string;
  display_name?: string;
  annual_return: number;
  annual_volatility: number;
  initial_wealth: number;
};

export type IncomeConfig = {
  fixed?: {
    base: number;
    annual_growth: number;
    salary_raises?: Record<string, number>;
  };
  variable?: {
    base: number;
    sigma: number;
    annual_growth?: number;
    seasonality?: number[];
    floor?: number;
    cap?: number;
  };
  contribution_rate_fixed: number | number[];
  contribution_rate_variable: number | number[];
};

export type Profile = {
  id: string;
  name: string;
  description: string;
  income_config: IncomeConfig;
  accounts_config: AccountConfig[];
  correlation_matrix: number[][] | null;
  is_demo?: boolean;
};

export type TerminalGoal = {
  account: string;
  threshold: number;
  confidence: number;
};

export type IntermediateGoal = {
  account: string;
  threshold: number;
  confidence: number;
  date: string;
};

export type ScheduledWithdrawal = {
  account: string;
  amount: number;
  date: string;
  description?: string;
};

export type StochasticWithdrawal = {
  account: string;
  base_amount: number;
  sigma: number;
  date?: string;
  month?: number;
  floor?: number;
  cap?: number;
  description?: string;
};

export type WithdrawalsConfig = {
  scheduled: ScheduledWithdrawal[];
  stochastic: StochasticWithdrawal[];
};

export type Scenario = {
  id: string;
  profile_id: string;
  name: string;
  description: string;
  start_date: string;
  terminal_goals: TerminalGoal[];
  intermediate_goals: IntermediateGoal[];
  withdrawals: WithdrawalsConfig | null;
  objective: "risky" | "balanced" | "conservative" | "proportional";
  is_demo?: boolean;
};

export type WealthPercentiles = {
  mean: number[];
  p10: number[];
  p25: number[];
  p50: number[];
  p75: number[];
  p90: number[];
};

/** Per-account wealth percentiles (summary_stats.per_account entries). */
export type AccountWealthSeries = WealthPercentiles & {
  account: string;
  display_name: string;
};

export type GoalStatus = {
  goal: string;
  type: "terminal" | "intermediate";
  account: string;
  threshold: number;
  required_confidence: number;
  satisfied: boolean;
  actual_probability?: number;
  empirical_probability?: number;
  confidence_gap?: number;
  note?: string;
};

export type Result = {
  allocation_policy: number[][] | null;
  optimal_horizon: number | null;
  objective_value: number | null;
  feasible: boolean | null;
  solve_time: number | null;
  summary_stats: {
    total_wealth?: WealthPercentiles;
    per_account?: AccountWealthSeries[];
    cash_flow?: {
      contributions_mean: number[];
      contributions_by_account: {
        account: string;
        display_name: string;
        mean: number[];
      }[];
      withdrawals_mean?: number[];
      withdrawals_by_account?: {
        account: string;
        display_name: string;
        mean: number[];
      }[];
    };
  } | null;
  goal_status: GoalStatus[] | null;
};

export type JobStatus = "pending" | "running" | "completed" | "failed";

export type ScenarioDraft = {
  profile_id: string;
  name: string;
  description: string;
  start_date: string;
  objective: "risky" | "balanced" | "conservative" | "proportional";
  terminal_goals: TerminalGoal[];
  intermediate_goals: IntermediateGoal[];
  withdrawals: WithdrawalsConfig | null;
};

export type ProfileDraft = {
  name: string;
  description: string;
  income_config: IncomeConfig;
  accounts_config: AccountConfig[];
  correlation_matrix: number[][] | null;
};
