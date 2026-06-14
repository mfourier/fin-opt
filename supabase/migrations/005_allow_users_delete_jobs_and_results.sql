-- ============================================================================
-- FinOpt schema patch
-- Allow users to delete their own jobs and saved results from the UI.
-- ============================================================================

-- ============================================================================
-- POLICIES — jobs
-- ============================================================================
DROP POLICY IF EXISTS "Users can delete own jobs" ON jobs;
CREATE POLICY "Users can delete own jobs" ON jobs
    FOR DELETE USING (
        scenario_id IN (
            SELECT s.id FROM scenarios s
            JOIN profiles p ON s.profile_id = p.id
            WHERE p.user_id = auth.uid()
        )
    );

-- ============================================================================
-- POLICIES — results
-- ============================================================================
DROP POLICY IF EXISTS "Users can delete own results" ON results;
CREATE POLICY "Users can delete own results" ON results
    FOR DELETE USING (
        job_id IN (
            SELECT j.id FROM jobs j
            JOIN scenarios s ON j.scenario_id = s.id
            JOIN profiles p ON s.profile_id = p.id
            WHERE p.user_id = auth.uid()
        )
    );
