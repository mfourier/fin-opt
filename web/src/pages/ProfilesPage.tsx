import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { supabase } from '../lib/supabase'
import { useAuthStore } from '../lib/store'
import type {
  Profile,
  ProfileInsert,
  AccountConfig,
  IncomeConfig,
  FixedIncomeConfig,
  VariableIncomeConfig,
  SalaryRaise,
} from '../types/database'

// Default seasonality factors (12 months)
const defaultSeasonality = [0, 0, 0, 0.6, 1.0, 1.16, 1.0, 1.1, 0.5, 0.9, 0.85, 1.0]
const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

const defaultIncomeConfig: IncomeConfig = {
  fixed: {
    base: 1500000,
    annual_growth: 0.03,
  },
  contribution_rate_fixed: 0.3,
  contribution_rate_variable: 1.0,
}

const defaultAccounts: AccountConfig[] = [
  { name: 'Conservative', display_name: 'Conservative Fund', annual_return: 0.08, annual_volatility: 0.09, initial_wealth: 0 },
  { name: 'Aggressive', display_name: 'Aggressive Fund', annual_return: 0.14, annual_volatility: 0.15, initial_wealth: 0 },
]

export default function ProfilesPage() {
  const queryClient = useQueryClient()
  const user = useAuthStore((state) => state.user)
  const [showForm, setShowForm] = useState(false)
  const [editingProfile, setEditingProfile] = useState<Profile | null>(null)

  // Expandable sections
  const [showVariableIncome, setShowVariableIncome] = useState(false)
  const [showSeasonality, setShowSeasonality] = useState(false)
  const [showAdvancedContributions, setShowAdvancedContributions] = useState(false)

  // Salary raises management
  const [salaryRaises, setSalaryRaises] = useState<SalaryRaise[]>([])

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
    setShowVariableIncome(false)
    setShowSeasonality(false)
    setShowAdvancedContributions(false)
    setSalaryRaises([])
    setFormData({
      name: '',
      description: '',
      income_config: defaultIncomeConfig,
      accounts_config: defaultAccounts,
    })
  }

  const handleEdit = (profile: Profile) => {
    setEditingProfile(profile)

    // Extract salary raises from the profile
    const raises: SalaryRaise[] = []
    if (profile.income_config.fixed?.salary_raises) {
      Object.entries(profile.income_config.fixed.salary_raises).forEach(([date, amount]) => {
        raises.push({ date, amount })
      })
    }
    setSalaryRaises(raises)

    // Check if variable income exists
    setShowVariableIncome(!!profile.income_config.variable)
    setShowSeasonality(!!profile.income_config.variable?.seasonality)

    // Check if advanced contributions
    const hasArrayContributions =
      Array.isArray(profile.income_config.contribution_rate_fixed) ||
      Array.isArray(profile.income_config.contribution_rate_variable)
    setShowAdvancedContributions(hasArrayContributions)

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

    // Build income_config with salary raises
    const incomeConfig = { ...formData.income_config }

    // Convert salary raises array to Record
    if (salaryRaises.length > 0 && incomeConfig.fixed) {
      incomeConfig.fixed = {
        ...incomeConfig.fixed,
        salary_raises: salaryRaises.reduce((acc, raise) => {
          acc[raise.date] = raise.amount
          return acc
        }, {} as Record<string, number>)
      }
    }

    // Remove variable if not enabled
    if (!showVariableIncome) {
      delete incomeConfig.variable
    }

    // Remove seasonality if not enabled
    if (!showSeasonality && incomeConfig.variable) {
      delete incomeConfig.variable.seasonality
    }

    const finalFormData = {
      ...formData,
      income_config: incomeConfig,
    }

    if (editingProfile) {
      updateMutation.mutate({ id: editingProfile.id, ...finalFormData })
    } else {
      createMutation.mutate({
        user_id: user!.id,
        ...finalFormData,
        correlation_matrix: null,
      })
    }
  }

  // Fixed Income helpers
  const updateFixedIncome = (field: keyof FixedIncomeConfig, value: number) => {
    setFormData({
      ...formData,
      income_config: {
        ...formData.income_config,
        fixed: { ...formData.income_config.fixed!, [field]: value },
      },
    })
  }

  // Variable Income helpers
  const updateVariableIncome = (field: keyof VariableIncomeConfig, value: number | number[] | undefined) => {
    setFormData({
      ...formData,
      income_config: {
        ...formData.income_config,
        variable: {
          ...formData.income_config.variable!,
          [field]: value,
        },
      },
    })
  }

  const initializeVariableIncome = () => {
    setFormData({
      ...formData,
      income_config: {
        ...formData.income_config,
        variable: {
          base: 500000,
          sigma: 0.2,
        },
      },
    })
    setShowVariableIncome(true)
  }

  // Salary Raise helpers
  const addSalaryRaise = () => {
    const nextYear = new Date().getFullYear() + 1
    setSalaryRaises([...salaryRaises, { date: `${nextYear}-03-01`, amount: 100000 }])
  }

  const updateSalaryRaise = (index: number, field: keyof SalaryRaise, value: string | number) => {
    const newRaises = [...salaryRaises]
    newRaises[index] = { ...newRaises[index], [field]: value }
    setSalaryRaises(newRaises)
  }

  const removeSalaryRaise = (index: number) => {
    setSalaryRaises(salaryRaises.filter((_, i) => i !== index))
  }

  // Seasonality helpers
  const initializeSeasonality = () => {
    setFormData({
      ...formData,
      income_config: {
        ...formData.income_config,
        variable: {
          ...formData.income_config.variable!,
          seasonality: [...defaultSeasonality],
        },
      },
    })
    setShowSeasonality(true)
  }

  const updateSeasonalityMonth = (monthIndex: number, value: number) => {
    const currentSeasonality = formData.income_config.variable?.seasonality ?? [...defaultSeasonality]
    const newSeasonality = [...currentSeasonality]
    newSeasonality[monthIndex] = value
    updateVariableIncome('seasonality', newSeasonality)
  }

  // Contribution rate helpers
  const getContributionRateValue = (type: 'fixed' | 'variable'): number => {
    const rate = type === 'fixed'
      ? formData.income_config.contribution_rate_fixed
      : formData.income_config.contribution_rate_variable
    return Array.isArray(rate) ? rate[0] : rate
  }

  const getContributionRateArray = (type: 'fixed' | 'variable'): number[] => {
    const rate = type === 'fixed'
      ? formData.income_config.contribution_rate_fixed
      : formData.income_config.contribution_rate_variable
    return Array.isArray(rate) ? rate : Array(12).fill(rate)
  }

  const updateContributionRate = (type: 'fixed' | 'variable', value: number | number[]) => {
    setFormData({
      ...formData,
      income_config: {
        ...formData.income_config,
        [type === 'fixed' ? 'contribution_rate_fixed' : 'contribution_rate_variable']: value,
      },
    })
  }

  const toggleAdvancedContributions = () => {
    if (showAdvancedContributions) {
      // Convert back to scalar (use first value)
      const fixedScalar = getContributionRateValue('fixed')
      const varScalar = getContributionRateValue('variable')
      setFormData({
        ...formData,
        income_config: {
          ...formData.income_config,
          contribution_rate_fixed: fixedScalar,
          contribution_rate_variable: varScalar,
        },
      })
    } else {
      // Convert to array
      const fixedArray = getContributionRateArray('fixed')
      const varArray = getContributionRateArray('variable')
      setFormData({
        ...formData,
        income_config: {
          ...formData.income_config,
          contribution_rate_fixed: fixedArray,
          contribution_rate_variable: varArray,
        },
      })
    }
    setShowAdvancedContributions(!showAdvancedContributions)
  }

  // Account helpers
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
        { name: 'NewAccount', display_name: 'New Account', annual_return: 0.10, annual_volatility: 0.12, initial_wealth: 0 },
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
          <div className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-lg bg-white p-6 shadow-xl">
            <h2 className="mb-4 text-lg font-medium text-gray-900">
              {editingProfile ? 'Edit Profile' : 'Create Profile'}
            </h2>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Basic Info */}
              <div className="grid grid-cols-2 gap-4">
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
                  <input
                    type="text"
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                  />
                </div>
              </div>

              {/* Fixed Income Section */}
              <div className="rounded-lg border border-gray-200 p-4">
                <h3 className="mb-3 font-medium text-gray-900">Fixed Income (Salary)</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-gray-600">Base Monthly Income</label>
                    <input
                      type="number"
                      value={formData.income_config.fixed?.base ?? 0}
                      onChange={(e) => updateFixedIncome('base', Number(e.target.value))}
                      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600">Annual Growth Rate</label>
                    <input
                      type="number"
                      step="0.01"
                      value={formData.income_config.fixed?.annual_growth ?? 0}
                      onChange={(e) => updateFixedIncome('annual_growth', Number(e.target.value))}
                      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    />
                    <p className="mt-1 text-xs text-gray-500">e.g., 0.03 = 3% per year</p>
                  </div>
                </div>

                {/* Salary Raises */}
                <div className="mt-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-gray-700">Scheduled Salary Raises</label>
                    <button
                      type="button"
                      onClick={addSalaryRaise}
                      className="text-sm text-primary-600 hover:text-primary-500"
                    >
                      + Add Raise
                    </button>
                  </div>
                  {salaryRaises.length > 0 && (
                    <div className="mt-2 space-y-2">
                      {salaryRaises.map((raise, index) => (
                        <div key={index} className="flex items-center gap-2">
                          <input
                            type="date"
                            value={raise.date}
                            onChange={(e) => updateSalaryRaise(index, 'date', e.target.value)}
                            className="rounded-md border border-gray-300 px-2 py-1 text-sm"
                          />
                          <input
                            type="number"
                            value={raise.amount}
                            onChange={(e) => updateSalaryRaise(index, 'amount', Number(e.target.value))}
                            placeholder="Amount"
                            className="w-32 rounded-md border border-gray-300 px-2 py-1 text-sm"
                          />
                          <button
                            type="button"
                            onClick={() => removeSalaryRaise(index)}
                            className="text-red-500 hover:text-red-600"
                          >
                            &times;
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Variable Income Section */}
              <div className="rounded-lg border border-gray-200 p-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium text-gray-900">Variable Income (Bonuses/Commissions)</h3>
                  {!showVariableIncome ? (
                    <button
                      type="button"
                      onClick={initializeVariableIncome}
                      className="text-sm text-primary-600 hover:text-primary-500"
                    >
                      + Enable
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => setShowVariableIncome(false)}
                      className="text-sm text-red-600 hover:text-red-500"
                    >
                      Disable
                    </button>
                  )}
                </div>

                {showVariableIncome && (
                  <div className="mt-4 space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm text-gray-600">Base Amount (monthly)</label>
                        <input
                          type="number"
                          value={formData.income_config.variable?.base ?? 0}
                          onChange={(e) => updateVariableIncome('base', Number(e.target.value))}
                          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm text-gray-600">Volatility (sigma)</label>
                        <input
                          type="number"
                          step="0.01"
                          value={formData.income_config.variable?.sigma ?? 0.2}
                          onChange={(e) => updateVariableIncome('sigma', Number(e.target.value))}
                          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                        />
                        <p className="mt-1 text-xs text-gray-500">e.g., 0.2 = 20% monthly variation</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="block text-sm text-gray-600">Annual Growth (optional)</label>
                        <input
                          type="number"
                          step="0.01"
                          value={formData.income_config.variable?.annual_growth ?? ''}
                          onChange={(e) => updateVariableIncome('annual_growth', e.target.value ? Number(e.target.value) : undefined)}
                          placeholder="0.03"
                          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm text-gray-600">Floor (min)</label>
                        <input
                          type="number"
                          value={formData.income_config.variable?.floor ?? ''}
                          onChange={(e) => updateVariableIncome('floor', e.target.value ? Number(e.target.value) : undefined)}
                          placeholder="Optional"
                          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm text-gray-600">Cap (max)</label>
                        <input
                          type="number"
                          value={formData.income_config.variable?.cap ?? ''}
                          onChange={(e) => updateVariableIncome('cap', e.target.value ? Number(e.target.value) : undefined)}
                          placeholder="Optional"
                          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                        />
                      </div>
                    </div>

                    {/* Seasonality */}
                    <div>
                      <div className="flex items-center justify-between">
                        <label className="text-sm font-medium text-gray-700">Seasonality (12-month factors)</label>
                        {!showSeasonality ? (
                          <button
                            type="button"
                            onClick={initializeSeasonality}
                            className="text-sm text-primary-600 hover:text-primary-500"
                          >
                            + Enable
                          </button>
                        ) : (
                          <button
                            type="button"
                            onClick={() => setShowSeasonality(false)}
                            className="text-sm text-red-600 hover:text-red-500"
                          >
                            Disable
                          </button>
                        )}
                      </div>
                      {showSeasonality && (
                        <div className="mt-2 grid grid-cols-6 gap-2">
                          {monthNames.map((month, index) => (
                            <div key={month}>
                              <label className="block text-xs text-gray-500">{month}</label>
                              <input
                                type="number"
                                step="0.1"
                                value={formData.income_config.variable?.seasonality?.[index] ?? defaultSeasonality[index]}
                                onChange={(e) => updateSeasonalityMonth(index, Number(e.target.value))}
                                className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                              />
                            </div>
                          ))}
                        </div>
                      )}
                      <p className="mt-1 text-xs text-gray-500">Multiplier per month. e.g., 1.0 = normal, 0.5 = half, 2.0 = double</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Contribution Rates */}
              <div className="rounded-lg border border-gray-200 p-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium text-gray-900">Contribution Rates</h3>
                  <button
                    type="button"
                    onClick={toggleAdvancedContributions}
                    className="text-sm text-primary-600 hover:text-primary-500"
                  >
                    {showAdvancedContributions ? 'Use Simple (single value)' : 'Use Monthly (12 values)'}
                  </button>
                </div>
                <p className="mt-1 text-xs text-gray-500">Fraction of income contributed to investments each month</p>

                {!showAdvancedContributions ? (
                  <div className="mt-3 grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-gray-600">Fixed Income Rate</label>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        value={getContributionRateValue('fixed')}
                        onChange={(e) => updateContributionRate('fixed', Number(e.target.value))}
                        className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                      />
                      <p className="mt-1 text-xs text-gray-500">e.g., 0.3 = save 30% of salary</p>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-600">Variable Income Rate</label>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        value={getContributionRateValue('variable')}
                        onChange={(e) => updateContributionRate('variable', Number(e.target.value))}
                        className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                      />
                      <p className="mt-1 text-xs text-gray-500">e.g., 1.0 = save 100% of bonuses</p>
                    </div>
                  </div>
                ) : (
                  <div className="mt-3 space-y-4">
                    <div>
                      <label className="block text-sm text-gray-600">Fixed Income Rate (by month)</label>
                      <div className="mt-2 grid grid-cols-6 gap-2">
                        {monthNames.map((month, index) => (
                          <div key={`fixed-${month}`}>
                            <label className="block text-xs text-gray-500">{month}</label>
                            <input
                              type="number"
                              step="0.01"
                              min="0"
                              max="1"
                              value={getContributionRateArray('fixed')[index]}
                              onChange={(e) => {
                                const arr = [...getContributionRateArray('fixed')]
                                arr[index] = Number(e.target.value)
                                updateContributionRate('fixed', arr)
                              }}
                              className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-600">Variable Income Rate (by month)</label>
                      <div className="mt-2 grid grid-cols-6 gap-2">
                        {monthNames.map((month, index) => (
                          <div key={`var-${month}`}>
                            <label className="block text-xs text-gray-500">{month}</label>
                            <input
                              type="number"
                              step="0.01"
                              min="0"
                              max="1"
                              value={getContributionRateArray('variable')[index]}
                              onChange={(e) => {
                                const arr = [...getContributionRateArray('variable')]
                                arr[index] = Number(e.target.value)
                                updateContributionRate('variable', arr)
                              }}
                              className="mt-1 block w-full rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Investment Accounts */}
              <div className="rounded-lg border border-gray-200 p-4">
                <div className="mb-3 flex items-center justify-between">
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
                      <div className="mb-3 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={account.name}
                            onChange={(e) => updateAccount(index, 'name', e.target.value)}
                            placeholder="ID (no spaces)"
                            className="w-32 rounded-md border-gray-300 text-sm font-medium focus:border-primary-500 focus:ring-primary-500"
                          />
                          <input
                            type="text"
                            value={account.display_name ?? ''}
                            onChange={(e) => updateAccount(index, 'display_name', e.target.value)}
                            placeholder="Display Name"
                            className="w-48 rounded-md border-gray-300 text-sm focus:border-primary-500 focus:ring-primary-500"
                          />
                        </div>
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
                          <p className="mt-1 text-xs text-gray-400">e.g., 0.08 = 8%</p>
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
                          <p className="mt-1 text-xs text-gray-400">e.g., 0.12 = 12%</p>
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

              {/* Form Actions */}
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
                  <div className="mt-1 flex flex-wrap gap-2 text-xs text-gray-400">
                    <span>{profile.accounts_config.length} accounts</span>
                    <span>|</span>
                    <span>Fixed: ${profile.income_config.fixed?.base?.toLocaleString() ?? 0}/mo</span>
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
