import { PlanResults } from '@/components/finopt/PlanResults'
import { ThemeToggle } from '@/components/ThemeToggle'
import { mockProfile, mockScenario, mockResult } from '@/mocks/plan'

/**
 * Standalone preview of the redesigned "My plan" (results) screen, harvested
 * from the Lovable redesign and rendered with mock data. This route is for
 * visual review only — it is not wired to real jobs/Supabase yet.
 */
export default function PlanPreviewPage() {
  return (
    <div style={{ fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif' }}>
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-5xl px-4 py-10">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Preview · mock data</p>
              <h1 className="text-lg font-semibold text-foreground">My plan (redesign)</h1>
            </div>
            <ThemeToggle />
          </div>

          <PlanResults
            profile={mockProfile}
            scenario={mockScenario}
            result={mockResult}
            jobStatus="completed"
            onExportJSON={() => alert('Export JSON (mock)')}
            onExportCSV={() => alert('Export CSV (mock)')}
            onRecalculate={() => alert('Recalculate (mock)')}
            onAdjustGoals={() => alert('Adjust goals (mock)')}
          />
        </div>
      </div>
    </div>
  )
}
