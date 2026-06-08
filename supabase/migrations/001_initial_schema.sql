-- ============================================================================
-- FinOpt Database Schema v0.1.0
-- Compatible with serialization.py schema version 0.2.0
-- Idempotent: safe to run multiple times
-- ============================================================================

-- ============================================================================
-- PROFILES TABLE
-- Stores FinancialModel configuration (income + accounts + correlation)
-- JSON columns match the format from src/serialization.py
-- ============================================================================
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT DEFAULT '',

    -- Income configuration (matches income_to_dict() output)
    -- Example: {
    --   "fixed": {"base": 1500000, "annual_growth": 0.03, "salary_raises": null},
    --   "variable": {"base": 200000, "sigma": 0.1, "annual_growth": 0.02, ...},
    --   "contribution_rate_fixed": 0.3,
    --   "contribution_rate_variable": 1.0
    -- }
    income_config JSONB NOT NULL,

    -- Accounts configuration (list of account dicts)
    -- Example: [
    --   {"name": "Emergency", "annual_return": 0.04, "annual_volatility": 0.05, "initial_wealth": 0},
    --   {"name": "Retirement", "annual_return": 0.10, "annual_volatility": 0.12, "initial_wealth": 5000000}
    -- ]
    accounts_config JSONB NOT NULL,

    -- Correlation matrix (optional, M x M where M = number of accounts)
    -- Example: [[1.0, 0.3], [0.3, 1.0]]
    correlation_matrix JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,

    -- Constraints
    CONSTRAINT profiles_name_user_unique UNIQUE (user_id, name)
);

-- Index for faster user lookups
CREATE INDEX IF NOT EXISTS idx_profiles_user_id ON profiles(user_id);

-- ============================================================================
-- SCENARIOS TABLE
-- Stores optimization scenarios (goals + parameters) linked to a profile
-- ============================================================================
CREATE TABLE IF NOT EXISTS scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT DEFAULT '',

    -- Goals configuration (matches goals_to_dict() output)
    -- Example intermediate: [
    --   {"account": "Emergency", "threshold": 5500000, "confidence": 0.9, "date": "2025-07-01"}
    -- ]
    intermediate_goals JSONB DEFAULT '[]'::jsonb NOT NULL,

    -- Example terminal: [
    --   {"account": "Retirement", "threshold": 20000000, "confidence": 0.8}
    -- ]
    terminal_goals JSONB DEFAULT '[]'::jsonb NOT NULL,

    -- Withdrawals configuration (optional)
    -- Example: {
    --   "scheduled": [{"account": "Emergency", "amount": 1000000, "date": "2025-06-01"}],
    --   "stochastic": []
    -- }
    withdrawals JSONB,

    -- Simulation parameters
    start_date DATE NOT NULL,
    n_sims INTEGER DEFAULT 300 NOT NULL CHECK (n_sims >= 100 AND n_sims <= 10000),
    seed INTEGER,

    -- Optimization parameters
    t_max INTEGER DEFAULT 240 NOT NULL CHECK (t_max >= 12 AND t_max <= 600),
    t_min INTEGER DEFAULT 12 NOT NULL CHECK (t_min >= 1),
    solver VARCHAR(20) DEFAULT 'ECOS' NOT NULL CHECK (solver IN ('ECOS', 'SCS', 'CLARABEL', 'MOSEK')),
    objective VARCHAR(30) DEFAULT 'balanced' NOT NULL CHECK (objective IN ('risky', 'balanced', 'conservative', 'risky_turnover')),

    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,

    -- Constraints
    CONSTRAINT scenarios_name_profile_unique UNIQUE (profile_id, name),
    CONSTRAINT scenarios_t_min_le_t_max CHECK (t_min <= t_max)
);

-- Index for faster profile lookups
CREATE INDEX IF NOT EXISTS idx_scenarios_profile_id ON scenarios(profile_id);

-- ============================================================================
-- JOBS TABLE
-- Tracks async simulation/optimization jobs with progress
-- ============================================================================
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id UUID REFERENCES scenarios(id) ON DELETE CASCADE NOT NULL,

    -- Job type: 'simulation' or 'optimization'
    job_type VARCHAR(20) NOT NULL CHECK (job_type IN ('simulation', 'optimization')),

    -- Status tracking
    status VARCHAR(20) DEFAULT 'pending' NOT NULL
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),

    -- Progress tracking (0-100)
    progress INTEGER DEFAULT 0 NOT NULL CHECK (progress >= 0 AND progress <= 100),
    current_step VARCHAR(100),

    -- Error information
    error_message TEXT,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_jobs_scenario_id ON jobs(scenario_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);

-- ============================================================================
-- RESULTS TABLE
-- Stores simulation/optimization results
-- ============================================================================
CREATE TABLE IF NOT EXISTS results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE NOT NULL UNIQUE,

    -- Result type matches job_type
    result_type VARCHAR(20) NOT NULL CHECK (result_type IN ('simulation', 'optimization')),

    -- Optimization results (null for simulation)
    -- allocation_policy: [[0.6, 0.4], [0.55, 0.45], ...] shape (T, M)
    allocation_policy JSONB,
    optimal_horizon INTEGER,
    objective_value DOUBLE PRECISION,
    feasible BOOLEAN,
    solve_time DOUBLE PRECISION,

    -- Diagnostics from optimizer
    diagnostics JSONB,

    -- Simulation results (summary stats, not full trajectories)
    -- Example: {"mean": 15000000, "median": 14500000, "std": 2000000, "percentiles": {...}}
    summary_stats JSONB,

    -- Goal satisfaction status
    -- Example: [{"goal": "Emergency by 2025-07", "satisfied": true, "actual_prob": 0.92}]
    goal_status JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Index for job lookup
CREATE INDEX IF NOT EXISTS idx_results_job_id ON results(job_id);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- Ensures users can only access their own data
-- ============================================================================

-- Enable RLS on all tables (idempotent)
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE scenarios ENABLE ROW LEVEL SECURITY;
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE results ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- POLICIES — profiles
-- ============================================================================
DROP POLICY IF EXISTS "Users can view own profiles" ON profiles;
CREATE POLICY "Users can view own profiles" ON profiles
    FOR SELECT USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own profiles" ON profiles;
CREATE POLICY "Users can insert own profiles" ON profiles
    FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own profiles" ON profiles;
CREATE POLICY "Users can update own profiles" ON profiles
    FOR UPDATE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own profiles" ON profiles;
CREATE POLICY "Users can delete own profiles" ON profiles
    FOR DELETE USING (auth.uid() = user_id);

-- ============================================================================
-- POLICIES — scenarios
-- ============================================================================
DROP POLICY IF EXISTS "Users can view own scenarios" ON scenarios;
CREATE POLICY "Users can view own scenarios" ON scenarios
    FOR SELECT USING (
        profile_id IN (SELECT id FROM profiles WHERE user_id = auth.uid())
    );

DROP POLICY IF EXISTS "Users can insert own scenarios" ON scenarios;
CREATE POLICY "Users can insert own scenarios" ON scenarios
    FOR INSERT WITH CHECK (
        profile_id IN (SELECT id FROM profiles WHERE user_id = auth.uid())
    );

DROP POLICY IF EXISTS "Users can update own scenarios" ON scenarios;
CREATE POLICY "Users can update own scenarios" ON scenarios
    FOR UPDATE USING (
        profile_id IN (SELECT id FROM profiles WHERE user_id = auth.uid())
    );

DROP POLICY IF EXISTS "Users can delete own scenarios" ON scenarios;
CREATE POLICY "Users can delete own scenarios" ON scenarios
    FOR DELETE USING (
        profile_id IN (SELECT id FROM profiles WHERE user_id = auth.uid())
    );

-- ============================================================================
-- POLICIES — jobs
-- ============================================================================
DROP POLICY IF EXISTS "Users can view own jobs" ON jobs;
CREATE POLICY "Users can view own jobs" ON jobs
    FOR SELECT USING (
        scenario_id IN (
            SELECT s.id FROM scenarios s
            JOIN profiles p ON s.profile_id = p.id
            WHERE p.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users can insert own jobs" ON jobs;
CREATE POLICY "Users can insert own jobs" ON jobs
    FOR INSERT WITH CHECK (
        scenario_id IN (
            SELECT s.id FROM scenarios s
            JOIN profiles p ON s.profile_id = p.id
            WHERE p.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users can update own jobs" ON jobs;
CREATE POLICY "Users can update own jobs" ON jobs
    FOR UPDATE USING (
        scenario_id IN (
            SELECT s.id FROM scenarios s
            JOIN profiles p ON s.profile_id = p.id
            WHERE p.user_id = auth.uid()
        )
    );

-- ============================================================================
-- POLICIES — results
-- ============================================================================
DROP POLICY IF EXISTS "Users can view own results" ON results;
CREATE POLICY "Users can view own results" ON results
    FOR SELECT USING (
        job_id IN (
            SELECT j.id FROM jobs j
            JOIN scenarios s ON j.scenario_id = s.id
            JOIN profiles p ON s.profile_id = p.id
            WHERE p.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users can insert own results" ON results;
CREATE POLICY "Users can insert own results" ON results
    FOR INSERT WITH CHECK (
        job_id IN (
            SELECT j.id FROM jobs j
            JOIN scenarios s ON j.scenario_id = s.id
            JOIN profiles p ON s.profile_id = p.id
            WHERE p.user_id = auth.uid()
        )
    );

-- ============================================================================
-- SERVICE ROLE POLICIES
-- Allow the Python backend (using service_role key) to update jobs/results
-- ============================================================================
DROP POLICY IF EXISTS "Service role can update jobs" ON jobs;
CREATE POLICY "Service role can update jobs" ON jobs
    FOR UPDATE USING (auth.jwt() ->> 'role' = 'service_role');

DROP POLICY IF EXISTS "Service role can insert results" ON results;
CREATE POLICY "Service role can insert results" ON results
    FOR INSERT WITH CHECK (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- REALTIME
-- Enable realtime updates for jobs table (for progress tracking)
-- ============================================================================
DO $$
BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE jobs;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;

-- ============================================================================
-- FUNCTIONS
-- Helper functions for common operations
-- ============================================================================

-- Function to update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE OR REPLACE TRIGGER update_profiles_updated_at
    BEFORE UPDATE ON profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER update_scenarios_updated_at
    BEFORE UPDATE ON scenarios
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
