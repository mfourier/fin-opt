-- ============================================================================
-- FinOpt Migration 002: add 'proportional' optimization objective
-- ============================================================================
-- The CVaROptimizer gained a "proportional" objective which is now the default
-- everywhere (library, API, CLI, web wizard). The scenarios.objective CHECK
-- constraint from 001 predates it, so inserts with objective='proportional'
-- fail with: violates check constraint "scenarios_objective_check".
--
-- This migration widens the allowed set and aligns the column default.
-- Idempotent: safe to run multiple times.
-- ============================================================================

ALTER TABLE scenarios DROP CONSTRAINT IF EXISTS scenarios_objective_check;

ALTER TABLE scenarios ADD CONSTRAINT scenarios_objective_check
    CHECK (objective IN ('risky', 'balanced', 'conservative', 'risky_turnover', 'proportional'));

ALTER TABLE scenarios ALTER COLUMN objective SET DEFAULT 'proportional';
