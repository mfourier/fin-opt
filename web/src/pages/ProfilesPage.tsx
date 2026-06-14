import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { supabase } from '../lib/supabase'
import { useAuthStore } from '../lib/store'
import { useToast } from '../components/Toast'
import { SituationForm } from '@/components/finopt/SituationForm'
import { Button } from '@/components/ui/button'
import type { Profile, ProfileInsert } from '../types/database'
import type { ProfileDraft } from '@/mocks/types'

export default function ProfilesPage() {
  const queryClient = useQueryClient()
  const user = useAuthStore((state) => state.user)
  const toast = useToast()
  const [showForm, setShowForm] = useState(false)
  const [editingProfile, setEditingProfile] = useState<Profile | null>(null)

  const { data: profiles, isLoading } = useQuery({
    // Scoped by user so a cached list can never leak across accounts.
    queryKey: ['profiles', user?.id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .order('created_at', { ascending: false })
      if (error) throw error
      return data as Profile[]
    },
    enabled: !!user,
  })

  const createMutation = useMutation({
    mutationFn: async (profile: ProfileInsert) => {
      const { data, error } = await supabase.from('profiles').insert(profile).select().single()
      if (error) throw error
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] })
      resetForm()
      toast.success('Situation saved', 'Your situation has been created.')
    },
    onError: (error: unknown) => {
      toast.error('Failed to save situation', getProfileSaveErrorMessage(error))
    },
  })

  const updateMutation = useMutation({
    mutationFn: async ({ id, ...profile }: Partial<Profile> & { id: string }) => {
      const { data, error } = await supabase
        .from('profiles')
        .update(profile)
        .eq('id', id)
        .select()
        .single()
      if (error) throw error
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] })
      resetForm()
      toast.success('Situation updated', 'Your changes have been saved.')
    },
    onError: (error: unknown) => {
      toast.error('Failed to update situation', getProfileSaveErrorMessage(error))
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase.from('profiles').delete().eq('id', id)
      if (error) throw error
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] })
      toast.success('Situation deleted', 'The situation has been removed.')
    },
    onError: (error: Error) => {
      toast.error('Failed to delete situation', error.message)
    },
  })

  const resetForm = () => {
    setShowForm(false)
    setEditingProfile(null)
  }

  // Close the form overlay with the Escape key.
  useEffect(() => {
    if (!showForm) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') resetForm()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [showForm])

  const handleEdit = (profile: Profile) => {
    setEditingProfile(profile)
    setShowForm(true)
  }

  const handleSave = async (draft: ProfileDraft) => {
    const nextName = draft.name.trim()
    const duplicate = profiles?.find((profile) =>
      profile.name.trim() === nextName && profile.id !== editingProfile?.id,
    )

    if (duplicate) {
      toast.error(
        editingProfile ? 'Name already in use' : 'Situation already exists',
        'Choose a different situation name or edit the existing one.',
      )
      return
    }

    try {
      if (editingProfile) {
        await updateMutation.mutateAsync({
          id: editingProfile.id,
          name: nextName,
          description: draft.description,
          income_config: draft.income_config,
          accounts_config: draft.accounts_config,
          correlation_matrix: draft.correlation_matrix,
        })
      } else {
        await createMutation.mutateAsync({
          user_id: user!.id,
          name: nextName,
          description: draft.description,
          income_config: draft.income_config,
          accounts_config: draft.accounts_config,
          correlation_matrix: draft.correlation_matrix,
        })
      }
    } catch {
      // Mutations surface errors via toast.
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">My situation</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Your income and investment accounts — the basis your plans are built on.
          </p>
        </div>
        <Button
          type="button"
          onClick={() => {
            setEditingProfile(null)
            setShowForm(true)
          }}
          className="rounded-md"
        >
          New situation
        </Button>
      </div>

      {/* Form overlay (redesigned) */}
      {showForm && (
        <div
          className="fixed inset-0 z-50 overflow-y-auto bg-background"
          style={{ fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif' }}
        >
          <div className="mx-auto w-full max-w-4xl px-4 py-6 sm:px-6">
            <div className="mb-2 flex justify-end">
              <Button variant="ghost" size="sm" onClick={resetForm}>
                Close
              </Button>
            </div>
            <SituationForm
              initialProfile={editingProfile ?? undefined}
              onSave={handleSave}
              onCancel={resetForm}
            />
          </div>
        </div>
      )}

      {/* Situations list */}
      <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
        {isLoading ? (
          <div className="p-6 text-center text-muted-foreground">Loading...</div>
        ) : profiles?.length === 0 ? (
          <div className="p-6 text-center text-muted-foreground">
            No situations yet. Create one to get started.
          </div>
        ) : (
          <div className="divide-y divide-border">
            {profiles?.map((profile) => (
              <div
                key={profile.id}
                className="flex items-center justify-between gap-4 p-6 transition-colors hover:bg-muted/30"
              >
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-medium text-foreground">{profile.name}</h3>
                    {profile.is_demo && (
                      <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                        Demo
                      </span>
                    )}
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {profile.description || 'No description'}
                  </p>
                  <div className="mt-1 flex flex-wrap gap-2 text-xs text-muted-foreground">
                    <span>{profile.accounts_config.length} accounts</span>
                    <span>|</span>
                    <span>Salary: ${profile.income_config.fixed?.base?.toLocaleString() ?? 0}/mo</span>
                    {profile.income_config.variable && (
                      <>
                        <span>|</span>
                        <span>Variable: ${profile.income_config.variable.base?.toLocaleString() ?? 0}/mo</span>
                      </>
                    )}
                  </div>
                </div>
                {profile.is_demo ? (
                  <span className="text-xs text-muted-foreground">Read-only example</span>
                ) : (
                  <div className="flex gap-2">
                    <Button type="button" variant="outline" size="sm" onClick={() => handleEdit(profile)}>
                      Edit
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="border-danger/30 text-danger hover:bg-danger-soft hover:text-danger"
                      onClick={() => {
                        if (confirm('Delete this situation?')) {
                          deleteMutation.mutate(profile.id)
                        }
                      }}
                    >
                      Delete
                    </Button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function getProfileSaveErrorMessage(error: unknown): string {
  if (isUniqueConstraintError(error)) {
    return 'A situation with that name already exists in your account. Use a different name or edit the existing one.'
  }

  if (error instanceof Error) {
    return error.message
  }

  return 'An unexpected error occurred while saving your situation.'
}

function isUniqueConstraintError(error: unknown): error is { code?: string; message?: string } {
  if (!error || typeof error !== 'object') return false
  const maybeError = error as { code?: string; message?: string }
  return maybeError.code === '23505'
    || maybeError.message?.includes('profiles_name_user_unique') === true
}
