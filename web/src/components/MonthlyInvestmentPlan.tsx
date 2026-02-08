import { useMemo, useState } from 'react'
import type { CashFlowStats } from '../types/database'

interface MonthlyInvestmentPlanProps {
  allocation: number[][]
  accountNames: string[]
  cashFlow: CashFlowStats
  startDate?: string
}

const ACCOUNT_COLORS = [
  '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899',
]

const formatCurrency = (value: number) => {
  if (Math.abs(value) >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(2)}M`
  }
  if (Math.abs(value) >= 1_000) {
    return `$${(value / 1_000).toFixed(0)}K`
  }
  return `$${value.toFixed(0)}`
}

type Grouping = 'monthly' | 'quarterly' | 'yearly'

export default function MonthlyInvestmentPlan({
  allocation,
  accountNames,
  cashFlow,
  startDate,
}: MonthlyInvestmentPlanProps) {
  const T = allocation.length

  // Auto-select grouping based on horizon
  const defaultGrouping: Grouping = T > 60 ? 'yearly' : T > 24 ? 'quarterly' : 'monthly'
  const [grouping, setGrouping] = useState<Grouping>(defaultGrouping)

  // Build contribution lookup by display name
  const contributionsByName = useMemo(() => {
    const map: Record<string, number[]> = {}
    for (const acc of cashFlow.contributions_by_account) {
      const name = acc.display_name || acc.account
      map[name] = acc.mean
    }
    return map
  }, [cashFlow])

  // Build row data
  const rows = useMemo(() => {
    type Row = {
      label: string
      monthRange: [number, number]
      total: number
      accounts: { name: string; amount: number; pct: number }[]
    }

    const result: Row[] = []
    const groupSize = grouping === 'yearly' ? 12 : grouping === 'quarterly' ? 3 : 1

    for (let start = 0; start < T; start += groupSize) {
      const end = Math.min(start + groupSize, T)
      const count = end - start

      // Label
      let label: string
      if (startDate) {
        const d1 = new Date(startDate)
        d1.setMonth(d1.getMonth() + start)
        if (grouping === 'monthly') {
          label = d1.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
        } else {
          const d2 = new Date(startDate)
          d2.setMonth(d2.getMonth() + end - 1)
          label = `${d1.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })} – ${d2.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })}`
        }
      } else {
        label = grouping === 'monthly' ? `Month ${start}` : `Months ${start}–${end - 1}`
      }

      // Aggregate contributions across the group
      let total = 0
      for (let t = start; t < end; t++) {
        total += cashFlow.contributions_mean[t] ?? 0
      }

      const accounts = accountNames.map((name, m) => {
        let amount = 0
        for (let t = start; t < end; t++) {
          amount += contributionsByName[name]?.[t] ?? 0
        }
        // Average allocation percentage for the period
        let pctSum = 0
        for (let t = start; t < end; t++) {
          pctSum += allocation[t]?.[m] ?? 0
        }
        const pct = pctSum / count
        return { name, amount, pct }
      })

      result.push({ label, monthRange: [start, end - 1], total, accounts })
    }

    return result
  }, [allocation, accountNames, cashFlow, contributionsByName, startDate, grouping, T])

  // Grand totals
  const grandTotal = useMemo(() => {
    const total = cashFlow.contributions_mean.reduce((s, v) => s + v, 0)
    const accounts = accountNames.map((name) => {
      const contribs = contributionsByName[name]
      const amount = contribs ? contribs.reduce((s, v) => s + v, 0) : 0
      return { name, amount }
    })
    return { total, accounts }
  }, [cashFlow, accountNames, contributionsByName])

  return (
    <div className="space-y-3">
      {/* Header with grouping toggle */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-gray-500">
          {grouping === 'monthly' ? 'Month-by-month' : grouping === 'quarterly' ? 'Quarterly' : 'Yearly'} breakdown of how much to invest in each account.
        </p>
        <div className="flex rounded-md border border-gray-200">
          {(['monthly', 'quarterly', 'yearly'] as Grouping[]).map((g) => (
            <button
              key={g}
              onClick={() => setGrouping(g)}
              className={`px-3 py-1 text-xs capitalize ${grouping === g ? 'bg-gray-100 font-medium' : ''}`}
            >
              {g}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="max-h-96 overflow-auto rounded-lg border border-gray-200">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="sticky top-0 bg-gray-50">
            <tr>
              <th className="px-4 py-2 text-left font-medium text-gray-500">Period</th>
              <th className="px-4 py-2 text-right font-medium text-gray-500">Total</th>
              {accountNames.map((name, m) => (
                <th key={m} className="px-4 py-2 text-right font-medium text-gray-500">
                  <span className="flex items-center justify-end gap-1.5">
                    <span
                      className="inline-block h-2.5 w-2.5 rounded"
                      style={{ backgroundColor: ACCOUNT_COLORS[m % ACCOUNT_COLORS.length] }}
                    />
                    {name}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 bg-white">
            {rows.map((row, i) => (
              <tr key={i} className="hover:bg-gray-50">
                <td className="whitespace-nowrap px-4 py-2 font-medium text-gray-700">{row.label}</td>
                <td className="whitespace-nowrap px-4 py-2 text-right text-gray-900">
                  {formatCurrency(row.total)}
                </td>
                {row.accounts.map((acc, m) => (
                  <td key={m} className="whitespace-nowrap px-4 py-2 text-right">
                    <span className="font-medium text-gray-900">{formatCurrency(acc.amount)}</span>
                    <span className="ml-1.5 text-gray-400">({(acc.pct * 100).toFixed(0)}%)</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
          {/* Totals row */}
          <tfoot className="sticky bottom-0 bg-gray-50">
            <tr className="font-semibold">
              <td className="px-4 py-2 text-gray-700">Total</td>
              <td className="px-4 py-2 text-right text-gray-900">{formatCurrency(grandTotal.total)}</td>
              {grandTotal.accounts.map((acc, m) => (
                <td key={m} className="px-4 py-2 text-right text-gray-900">
                  {formatCurrency(acc.amount)}
                </td>
              ))}
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  )
}
