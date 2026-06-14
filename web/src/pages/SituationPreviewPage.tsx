import { SituationForm } from '@/components/finopt/SituationForm'
import { ThemeToggle } from '@/components/ThemeToggle'
import { mockProfile } from '@/mocks/plan'
import type { ProfileDraft } from '@/mocks/types'

/**
 * Standalone preview of the redesigned "My situation" screen, harvested from the
 * Lovable redesign and rendered with a mock profile. Visual review only — not
 * wired to Supabase yet (Phase B will replace the ProfilesPage modal).
 */
export default function SituationPreviewPage() {
  const handleSave = (draft: ProfileDraft) => {
    console.log('[FinOpt] Save situation:', draft)
    alert(`Save situation "${draft.name}" with ${draft.accounts_config.length} account(s).`)
  }

  return (
    <div style={{ fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif' }}>
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-4xl px-4 py-8">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Preview · mock data</p>
              <h1 className="text-lg font-semibold text-foreground">My situation (redesign)</h1>
            </div>
            <ThemeToggle />
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
