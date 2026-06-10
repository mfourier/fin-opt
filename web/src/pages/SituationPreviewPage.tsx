import { useState } from 'react'
import { SituationForm } from '@/components/finopt/SituationForm'
import { Button } from '@/components/ui/button'
import { mockProfile } from '@/mocks/plan'
import type { ProfileDraft } from '@/mocks/types'

/**
 * Standalone preview of the redesigned "My situation" screen, harvested from the
 * Lovable redesign and rendered with a mock profile. Visual review only — not
 * wired to Supabase yet (Phase B will replace the ProfilesPage modal).
 */
export default function SituationPreviewPage() {
  const [dark, setDark] = useState(false)

  const handleSave = (draft: ProfileDraft) => {
    console.log('[FinOpt] Save situation:', draft)
    alert(`Save situation "${draft.name}" with ${draft.accounts_config.length} account(s).`)
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
              <h1 className="text-lg font-semibold text-foreground">My situation (redesign)</h1>
            </div>
            <Button variant="outline" size="sm" onClick={() => setDark((d) => !d)}>
              {dark ? 'Light mode' : 'Dark mode'}
            </Button>
          </div>

          <SituationForm
            initialProfile={mockProfile}
            onSave={handleSave}
            onCancel={() => alert('Cancel (mock)')}
          />
        </div>
      </div>
    </div>
  )
}
