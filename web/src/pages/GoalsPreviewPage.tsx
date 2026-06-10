import { useState } from 'react'
import { GoalsWizard } from '@/components/finopt/GoalsWizard'
import { Button } from '@/components/ui/button'
import { mockProfiles } from '@/mocks/plan'
import type { ScenarioDraft } from '@/mocks/types'

/**
 * Standalone preview of the redesigned "My goals" wizard, harvested from the
 * Lovable redesign and rendered with mock profiles. Visual review only — not
 * wired to Supabase / scenario creation yet (Phase B will do that).
 */
export default function GoalsPreviewPage() {
  const [dark, setDark] = useState(false)

  const handleCalculate = (draft: ScenarioDraft) => {
    console.log('[FinOpt] Calculate plan with draft:', draft)
    alert(`Calculate plan "${draft.name}" with ${draft.terminal_goals.length} goal(s).`)
  }

  return (
    <div
      className={dark ? 'dark' : ''}
      style={{ fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif' }}
    >
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-4xl px-4 py-8">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Preview · mock data</p>
              <h1 className="text-lg font-semibold text-foreground">My goals wizard (redesign)</h1>
            </div>
            <Button variant="outline" size="sm" onClick={() => setDark((d) => !d)}>
              {dark ? 'Light mode' : 'Dark mode'}
            </Button>
          </div>

          <GoalsWizard
            profiles={mockProfiles}
            onCalculate={handleCalculate}
            onCancel={() => alert('Cancel (mock)')}
          />
        </div>
      </div>
    </div>
  )
}
