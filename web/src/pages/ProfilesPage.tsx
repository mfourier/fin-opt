import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { supabase } from '../lib/supabase'
import { useAuthStore } from '../lib/store'
import type { Profile, ProfileInsert, AccountConfig, IncomeConfig } from '../types/database'

const defaultIncomeConfig: IncomeConfig = {
  fixed: {
    base: 1500000,
    annual_growth: 0.03,
  },
  contribution_rate_fixed: 0.3,
  contribution_rate_variable: 1.0,
}

const defaultAccounts: AccountConfig[] = [
  { name: 'Conservative', annual_return: 0.08, annual_volatility: 0.09, initial_wealth: 0 },
  { name: 'Aggressive', annual_return: 0.14, annual_volatility: 0.15, initial_wealth: 0 },
]

export default function ProfilesPage() {
  const queryClient = useQueryClient()
  const user = useAuthStore((state) => state.user)
  const [showForm, setShowForm] = useState(false)
  const [editingProfile, setEditingProfile] = useState<Profile | null>(null)

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    income_config: defaultIncomeConfig,
    accounts_config: defaultAccounts,
  })

  const { data: profiles, isLoading } = useQuery({
    queryKey: ['profiles'],
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
      const { data, error } = await supabase
        .from('profiles')
        .insert(profile)
        .select()
        .single()
      if (error) throw error
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] })
      resetForm()
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
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase.from('profiles').delete().eq('id', id)
      if (error) throw error
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] })
    },
  })

  const resetForm = () => {
    setShowForm(false)
    setEditingProfile(null)
    setFormData({
      name: '',
      description: '',
      income_config: defaultIncomeConfig,
      accounts_config: defaultAccounts,
    })
  }

  const handleEdit = (profile: Profile) => {
    setEditingProfile(profile)
    setFormData({
      name: profile.name,
      description: profile.description,
      income_config: profile.income_config,
      accounts_config: profile.accounts_config,
    })
    setShowForm(true)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (editingProfile) {
      updateMutation.mutate({ id: editingProfile.id, ...formData })
    } else {
      createMutation.mutate({
        user_id: user!.id,
        ...formData,
        correlation_matrix: null,
      })
    }
  }

  const updateAccount = (index: number, field: keyof AccountConfig, value: string | number) => {
    const newAccounts = [...formData.accounts_config]
    newAccounts[index] = { ...newAccounts[index], [field]: value }
    setFormData({ ...formData, accounts_config: newAccounts })
  }

  const addAccount = () => {
    setFormData({
      ...formData,
      accounts_config: [
        ...formData.accounts_config,
        { name: 'New Account', annual_return: 0.10, annual_volatility: 0.12, initial_wealth: 0 },
      ],
    })
  }

  const removeAccount = (index: number) => {
    setFormData({
      ...formData,
      accounts_config: formData.accounts_config.filter((_, i) => i !== index),
    })
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Profiles</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage your financial profiles (income sources and investment accounts).
          </p>
        </div>
        <button
          onClick={() => setShowForm(true)}
          className="rounded-md bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700"
        >
          New Profile
        </button>
      </div>

      {/* Form Modal */}
      {showForm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
          <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-lg bg-white p-6 shadow-xl">
            <h2 className="mb-4 text-lg font-medium text-gray-900">
              {editingProfile ? 'Edit Profile' : 'Create Profile'}
            </h2>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700">Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  required
                  className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Description</label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  rows={2}
                  className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                />
              </div>

              {/* Income Config */}
              <div>
                <h3 className="mb-2 font-medium text-gray-900">Fixed Income</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-gray-600">Base Monthly Income</label>
                    <input
                      type="number"
                      value={formData.income_config.fixed?.base ?? 0}
                      onChange={(e) => setFormData({
                        ...formData,
                        income_config: {
                          ...formData.income_config,
                          fixed: { ...formData.income_config.fixed!, base: Number(e.target.value) },
                        },
                      })}
                      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600">Annual Growth Rate</label>
                    <input
                      type="number"
                      step="0.01"
                      value={formData.income_config.fixed?.annual_growth ?? 0}
                      onChange={(e) => setFormData({
                        ...formData,
                        income_config: {
                          ...formData.income_config,
                          fixed: { ...formData.income_config.fixed!, annual_growth: Number(e.target.value) },
                        },
                      })}
                      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    />
                  </div>
                </div>
              </div>

              {/* Accounts */}
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="font-medium text-gray-900">Investment Accounts</h3>
                  <button
                    type="button"
                    onClick={addAccount}
                    className="text-sm text-primary-600 hover:text-primary-500"
                  >
                    + Add Account
                  </button>
                </div>
                <div className="space-y-4">
                  {formData.accounts_config.map((account, index) => (
                    <div key={index} className="rounded-md border border-gray-200 p-4">
                      <div className="mb-2 flex items-center justify-between">
                        <input
                          type="text"
                          value={account.name}
                          onChange={(e) => updateAccount(index, 'name', e.target.value)}
                          className="rounded-md border-gray-300 text-sm font-medium focus:border-primary-500 focus:ring-primary-500"
                        />
                        {formData.accounts_config.length > 1 && (
                          <button
                            type="button"
                            onClick={() => removeAccount(index)}
                            className="text-sm text-red-600 hover:text-red-500"
                          >
                            Remove
                          </button>
                        )}
                      </div>
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <label className="block text-xs text-gray-500">Annual Return</label>
                          <input
                            type="number"
                            step="0.01"
                            value={account.annual_return}
                            onChange={(e) => updateAccount(index, 'annual_return', Number(e.target.value))}
                            className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-500">Annual Volatility</label>
                          <input
                            type="number"
                            step="0.01"
                            value={account.annual_volatility}
                            onChange={(e) => updateAccount(index, 'annual_volatility', Number(e.target.value))}
                            className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-500">Initial Wealth</label>
                          <input
                            type="number"
                            value={account.initial_wealth}
                            onChange={(e) => updateAccount(index, 'initial_wealth', Number(e.target.value))}
                            className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex justify-end gap-3">
                <button
                  type="button"
                  onClick={resetForm}
                  className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={createMutation.isPending || updateMutation.isPending}
                  className="rounded-md bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50"
                >
                  {editingProfile ? 'Update' : 'Create'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Profiles List */}
      <div className="rounded-lg bg-white shadow">
        {isLoading ? (
          <div className="p-6 text-center text-gray-500">Loading...</div>
        ) : profiles?.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            No profiles yet. Create one to get started.
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {profiles?.map((profile) => (
              <div key={profile.id} className="flex items-center justify-between p-6">
                <div>
                  <h3 className="font-medium text-gray-900">{profile.name}</h3>
                  <p className="mt-1 text-sm text-gray-500">{profile.description || 'No description'}</p>
                  <p className="mt-1 text-xs text-gray-400">
                    {profile.accounts_config.length} accounts | Base income: ${profile.income_config.fixed?.base?.toLocaleString() ?? 0}
                  </p>
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
                      if (confirm('Delete this profile?')) {
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
