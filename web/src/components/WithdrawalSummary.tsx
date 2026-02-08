import type { WithdrawalsConfig } from '../types/database'

interface WithdrawalSummaryProps {
  withdrawals: WithdrawalsConfig | null
  startDate?: string
}

const formatCurrency = (value: number) => {
  if (Math.abs(value) >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(1)}M`
  }
  if (Math.abs(value) >= 1_000) {
    return `$${(value / 1_000).toFixed(0)}K`
  }
  return `$${value.toFixed(0)}`
}

const formatDate = (dateStr: string) => {
  const d = new Date(dateStr + 'T00:00:00')
  return d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
}

export default function WithdrawalSummary({ withdrawals, startDate }: WithdrawalSummaryProps) {
  if (!withdrawals) return null

  const scheduled = withdrawals.scheduled ?? []
  const stochastic = withdrawals.stochastic ?? []

  if (scheduled.length === 0 && stochastic.length === 0) return null

  // Compute total scheduled withdrawals
  const totalScheduled = scheduled.reduce((sum, w) => sum + w.amount, 0)
  const totalStochasticExpected = stochastic.reduce((sum, w) => sum + w.base_amount, 0)

  // Resolve month offset for a date string
  const resolveMonth = (dateStr: string): string => {
    if (!startDate) return ''
    const start = new Date(startDate + 'T00:00:00')
    const target = new Date(dateStr + 'T00:00:00')
    const months = (target.getFullYear() - start.getFullYear()) * 12 + (target.getMonth() - start.getMonth())
    return `Month ${months}`
  }

  return (
    <div className="space-y-4">
      {/* Summary stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-lg bg-red-50 p-3 text-center">
          <p className="text-xs text-red-600">Total Events</p>
          <p className="text-lg font-semibold text-red-800">{scheduled.length + stochastic.length}</p>
        </div>
        <div className="rounded-lg bg-red-50 p-3 text-center">
          <p className="text-xs text-red-600">Scheduled Total</p>
          <p className="text-lg font-semibold text-red-800">{formatCurrency(totalScheduled)}</p>
        </div>
        <div className="rounded-lg bg-red-50 p-3 text-center">
          <p className="text-xs text-red-600">Expected Variable</p>
          <p className="text-lg font-semibold text-red-800">
            {stochastic.length > 0 ? `~${formatCurrency(totalStochasticExpected)}` : '-'}
          </p>
        </div>
      </div>

      {/* Scheduled withdrawals table */}
      {scheduled.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-medium text-gray-700">Scheduled Withdrawals</h4>
          <div className="overflow-hidden rounded-lg border border-gray-200">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium uppercase text-gray-500">Date</th>
                  <th className="px-4 py-2 text-left text-xs font-medium uppercase text-gray-500">Account</th>
                  <th className="px-4 py-2 text-right text-xs font-medium uppercase text-gray-500">Amount</th>
                  <th className="px-4 py-2 text-left text-xs font-medium uppercase text-gray-500">Description</th>
                  {startDate && (
                    <th className="px-4 py-2 text-right text-xs font-medium uppercase text-gray-500">Offset</th>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 bg-white">
                {scheduled
                  .sort((a, b) => a.date.localeCompare(b.date))
                  .map((w, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="whitespace-nowrap px-4 py-2 text-sm text-gray-900">{formatDate(w.date)}</td>
                      <td className="whitespace-nowrap px-4 py-2 text-sm text-gray-600">{w.account}</td>
                      <td className="whitespace-nowrap px-4 py-2 text-right text-sm font-medium text-red-600">
                        -{formatCurrency(w.amount)}
                      </td>
                      <td className="px-4 py-2 text-sm text-gray-500">{w.description || '-'}</td>
                      {startDate && (
                        <td className="whitespace-nowrap px-4 py-2 text-right text-xs text-gray-400">
                          {resolveMonth(w.date)}
                        </td>
                      )}
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Stochastic withdrawals table */}
      {stochastic.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-medium text-gray-700">Variable Withdrawals</h4>
          <div className="overflow-hidden rounded-lg border border-gray-200">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium uppercase text-gray-500">Date/Month</th>
                  <th className="px-4 py-2 text-left text-xs font-medium uppercase text-gray-500">Account</th>
                  <th className="px-4 py-2 text-right text-xs font-medium uppercase text-gray-500">Expected</th>
                  <th className="px-4 py-2 text-right text-xs font-medium uppercase text-gray-500">Range</th>
                  <th className="px-4 py-2 text-left text-xs font-medium uppercase text-gray-500">Description</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 bg-white">
                {stochastic.map((w, i) => {
                  const rangeMin = w.floor ?? Math.max(0, w.base_amount - 2 * w.sigma)
                  const rangeMax = w.cap ?? w.base_amount + 2 * w.sigma
                  return (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="whitespace-nowrap px-4 py-2 text-sm text-gray-900">
                        {w.date ? formatDate(w.date) : `Month ${w.month}`}
                      </td>
                      <td className="whitespace-nowrap px-4 py-2 text-sm text-gray-600">{w.account}</td>
                      <td className="whitespace-nowrap px-4 py-2 text-right text-sm font-medium text-orange-600">
                        ~{formatCurrency(w.base_amount)}
                      </td>
                      <td className="whitespace-nowrap px-4 py-2 text-right text-xs text-gray-500">
                        {formatCurrency(rangeMin)} - {formatCurrency(rangeMax)}
                      </td>
                      <td className="px-4 py-2 text-sm text-gray-500">{w.description || '-'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
