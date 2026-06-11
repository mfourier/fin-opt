import type { Profile, Scenario, Result } from "./types";

export const mockProfile: Profile = {
  id: "p-1",
  name: "My situation",
  description: "Base salary plus a modest bonus, saving 30%.",
  income_config: {
    fixed: { base: 1_500_000, annual_growth: 0.03 },
    variable: { base: 200_000, sigma: 0.4 },
    contribution_rate_fixed: 0.3,
    contribution_rate_variable: 1.0,
  },
  accounts_config: [
    {
      name: "conservative",
      display_name: "Safe savings",
      annual_return: 0.08,
      annual_volatility: 0.09,
      initial_wealth: 2_000_000,
    },
    {
      name: "aggressive",
      display_name: "Growth ETF",
      annual_return: 0.14,
      annual_volatility: 0.18,
      initial_wealth: 2_400_000,
    },
  ],
  correlation_matrix: null,
};

export const mockProfileFreelancer: Profile = {
  id: "p-2",
  name: "Freelance income",
  description: "Variable monthly income with a smaller fixed retainer.",
  income_config: {
    fixed: { base: 600_000, annual_growth: 0.02 },
    variable: { base: 900_000, sigma: 0.5 },
    contribution_rate_fixed: 0.4,
    contribution_rate_variable: 0.6,
  },
  accounts_config: [
    {
      name: "buffer",
      display_name: "Cash buffer",
      annual_return: 0.05,
      annual_volatility: 0.04,
      initial_wealth: 1_500_000,
    },
    {
      name: "long_term",
      display_name: "Long-term portfolio",
      annual_return: 0.12,
      annual_volatility: 0.16,
      initial_wealth: 3_200_000,
    },
  ],
  correlation_matrix: null,
};

export const mockProfiles: Profile[] = [mockProfile, mockProfileFreelancer];

export const mockScenario: Scenario = {
  id: "s-1",
  profile_id: "p-1",
  name: "Apartment down payment",
  description: "Reach $50,000,000 with 80% confidence.",
  start_date: new Date().toISOString().slice(0, 10),
  terminal_goals: [
    { account: "aggressive", threshold: 50_000_000, confidence: 0.8 },
  ],
  intermediate_goals: [
    {
      account: "conservative",
      threshold: 8_000_000,
      confidence: 0.9,
      date: new Date(Date.now() + 365 * 24 * 3600 * 1000).toISOString().slice(0, 10),
    },
  ],
  withdrawals: {
    scheduled: [
      {
        account: "conservative",
        amount: 1_800_000,
        date: new Date(Date.now() + 8 * 30 * 24 * 3600 * 1000).toISOString().slice(0, 10),
        description: "Emergency expense",
      },
    ],
    stochastic: [
      {
        account: "aggressive",
        base_amount: 450_000,
        sigma: 0.2,
        date: new Date(Date.now() + 20 * 30 * 24 * 3600 * 1000).toISOString().slice(0, 10),
        description: "Annual tax payment",
      },
    ],
  },
  objective: "proportional",
};

// --- Build a coherent mock Result with monotone fan + smooth allocation drift.

const H = 50; // months: optimal horizon

function buildPercentiles() {
  const startCons = 2_000_000;
  const startAgg = 2_400_000;
  const contribMonthlyMean = 700_000; // saved per month
  const meanGrowthAnnual = 0.11;
  const sigmaAnnual = 0.14;

  const mean: number[] = [];
  const p10: number[] = [];
  const p25: number[] = [];
  const p50: number[] = [];
  const p75: number[] = [];
  const p90: number[] = [];

  let acc = startCons + startAgg;
  for (let t = 0; t <= H; t++) {
    if (t > 0) {
      const monthlyGrowth = Math.pow(1 + meanGrowthAnnual, 1 / 12) - 1;
      acc = acc * (1 + monthlyGrowth) + contribMonthlyMean;
    }
    const spread = sigmaAnnual * Math.sqrt(t / 12) * acc * 0.6;
    mean.push(acc);
    p50.push(acc);
    p10.push(Math.max(0, acc - 1.28 * spread));
    p25.push(Math.max(0, acc - 0.67 * spread));
    p75.push(acc + 0.67 * spread);
    p90.push(acc + 1.28 * spread);
  }
  return { mean, p10, p25, p50, p75, p90 };
}

function buildAllocation(): number[][] {
  // Drift from aggressive-heavy (~80%) at t=0 to conservative-heavy (~70%) at horizon.
  const rows: number[][] = [];
  for (let t = 0; t < H; t++) {
    const frac = t / (H - 1);
    const aggressive = 0.8 - 0.5 * frac;
    const conservative = 1 - aggressive;
    rows.push([conservative, aggressive]);
  }
  return rows;
}

function buildContributions() {
  const consMean: number[] = [];
  const aggMean: number[] = [];
  const totalMean: number[] = [];
  const consWithdrawals: number[] = [];
  const aggWithdrawals: number[] = [];
  const totalWithdrawals: number[] = [];
  for (let t = 0; t < H; t++) {
    const frac = t / (H - 1);
    const total = 700_000 * (1 + 0.0025 * t); // slow growth
    const aggressive = total * (0.8 - 0.5 * frac);
    const conservative = total - aggressive;
    const conservativeWithdrawal = t === 8 ? 1_800_000 : 0;
    const aggressiveWithdrawal = t === 20 ? 450_000 : 0;
    consMean.push(Math.round(conservative));
    aggMean.push(Math.round(aggressive));
    totalMean.push(Math.round(total));
    consWithdrawals.push(conservativeWithdrawal);
    aggWithdrawals.push(aggressiveWithdrawal);
    totalWithdrawals.push(conservativeWithdrawal + aggressiveWithdrawal);
  }
  return { consMean, aggMean, totalMean, consWithdrawals, aggWithdrawals, totalWithdrawals };
}

const wealth = buildPercentiles();
const allocation = buildAllocation();
const { consMean, aggMean, totalMean, consWithdrawals, aggWithdrawals, totalWithdrawals } = buildContributions();

export const mockResult: Result = {
  allocation_policy: allocation,
  optimal_horizon: H,
  objective_value: -0.2289,
  feasible: true,
  solve_time: 1.57,
  summary_stats: {
    total_wealth: wealth,
    cash_flow: {
      contributions_mean: totalMean,
      contributions_by_account: [
        { account: "conservative", display_name: "Safe savings", mean: consMean },
        { account: "aggressive", display_name: "Growth ETF", mean: aggMean },
      ],
      withdrawals_mean: totalWithdrawals,
      withdrawals_by_account: [
        { account: "conservative", display_name: "Safe savings", mean: consWithdrawals },
        { account: "aggressive", display_name: "Growth ETF", mean: aggWithdrawals },
      ],
    },
  },
  goal_status: [
    {
      goal: "Apartment down payment",
      type: "terminal",
      account: "aggressive",
      threshold: 50_000_000,
      required_confidence: 0.8,
      satisfied: true,
      actual_probability: 0.84,
      empirical_probability: 0.84,
      note: "On track. Your Growth ETF reaches the goal in about 4 years 2 months.",
    },
    {
      goal: "Emergency fund",
      type: "intermediate",
      account: "conservative",
      threshold: 8_000_000,
      required_confidence: 0.9,
      satisfied: true,
      actual_probability: 0.93,
      empirical_probability: 0.93,
      note: "Safe savings will cover this within the first year.",
    },
  ],
};

export const mockJobStatus: "pending" | "running" | "completed" | "failed" = "completed";
