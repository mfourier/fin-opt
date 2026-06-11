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
          <h1 className="text-2xl font-bold text-gray-900">My situation</h1>
          <p className="mt-1 text-sm text-gray-500">
            Your income and investment accounts — the basis your plans are built on.
          </p>
        </div>
        <button
          onClick={() => {
            setEditingProfile(null)
            setShowForm(true)
          }}
          className="rounded-md bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700"
        >
          New situation
        </button>
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
      <div className="rounded-lg bg-white shadow">
        {isLoading ? (
          <div className="p-6 text-center text-gray-500">Loading...</div>
        ) : profiles?.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            No situations yet. Create one to get started.
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {profiles?.map((profile) => (
              <div key={profile.id} className="flex items-center justify-between p-6">
                <div>
                  <h3 className="font-medium text-gray-900">{profile.name}</h3>
                  <p className="mt-1 text-sm text-gray-500">{profile.description || 'No description'}</p>
                  <div className="mt-1 flex flex-wrap gap-2 text-xs text-gray-400">
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
                <div className="flex gap-2">
                  <button
                    onClick={() => handleEdit(profile)}
                    className="rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => {
                      if (confirm('Delete this situation?')) {
                        deleteMutation.mutate(profile.id)
                      }
                    }}
                    className="rounded-md border border-red-300 bg-white px-3 py-1.5 text-sm text-red-700 hover:bg-red-50"
                  >
                    Delete
                  </button>
                </div>
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
