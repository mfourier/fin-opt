import { useMemo, useState } from 'react'
import type { CashFlowStats } from '../types/database'

interface AllocationHeatmapProps {
  allocation: number[][]
  accountNames: string[]
  startDate?: string
  cashFlow?: CashFlowStats
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

// Color palette for different accounts
const ACCOUNT_COLORS = [
  { main: '#3B82F6', light: '#DBEAFE', name: 'blue' },    // Blue
  { main: '#10B981', light: '#D1FAE5', name: 'green' },   // Green
  { main: '#F59E0B', light: '#FEF3C7', name: 'amber' },   // Amber
  { main: '#EF4444', light: '#FEE2E2', name: 'red' },     // Red
  { main: '#8B5CF6', light: '#EDE9FE', name: 'purple' },  // Purple
  { main: '#EC4899', light: '#FCE7F3', name: 'pink' },    // Pink
]

export default function AllocationHeatmap({ allocation, accountNames, startDate, cashFlow }: AllocationHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{ t: number; m: number } | null>(null)
  const [viewMode, setViewMode] = useState<'heatmap' | 'stacked'>('heatmap')

  // Build lookup: accountName -> mean contribution array for enriched tooltips
  const contributionsByName = useMemo(() => {
    if (!cashFlow) return null
    const map: Record<string, number[]> = {}
    for (const acc of cashFlow.contributions_by_account) {
      const displayName = acc.display_name || acc.account
      map[displayName] = acc.mean
    }
    return map
  }, [cashFlow])

  const stats = useMemo(() => {
    if (!allocation || allocation.length === 0) return null

    const T = allocation.length

    return accountNames.map((name, m) => {
      const values = allocation.map(row => row[m] ?? 0)
      const avg = values.reduce((a, b) => a + b, 0) / T
      const min = Math.min(...values)
      const max = Math.max(...values)
      const initial = values[0] ?? 0
      const final = values[T - 1] ?? 0
      const trend = final - initial

      // Average dollar amount for this account
      const contribs = cashFlow?.contributions_by_account.find(
        a => (a.display_name || a.account) === name
      )?.mean
      const avgAmount = contribs ? contribs.reduce((a, b) => a + b, 0) / contribs.length : null

      return { name, avg, min, max, initial, final, trend, avgAmount, color: ACCOUNT_COLORS[m % ACCOUNT_COLORS.length] }
    })
  }, [allocation, accountNames, cashFlow])

  if (!allocation || allocation.length === 0) {
    return <p className="text-gray-500">No allocation data available</p>
  }

  const T = allocation.length
  const M = allocation[0]?.length ?? 0

  // Generate color based on allocation percentage with account-specific color
  const getColor = (value: number, accountIndex: number) => {
    const color = ACCOUNT_COLORS[accountIndex % ACCOUNT_COLORS.length]
    const intensity = value
    // Interpolate between light and main color
    const r1 = parseInt(color.light.slice(1, 3), 16)
    const g1 = parseInt(color.light.slice(3, 5), 16)
    const b1 = parseInt(color.light.slice(5, 7), 16)
    const r2 = parseInt(color.main.slice(1, 3), 16)
    const g2 = parseInt(color.main.slice(3, 5), 16)
    const b2 = parseInt(color.main.slice(5, 7), 16)

    const r = Math.round(r1 + (r2 - r1) * intensity)
    const g = Math.round(g1 + (g2 - g1) * intensity)
    const b = Math.round(b1 + (b2 - b1) * intensity)

    return `rgb(${r}, ${g}, ${b})`
  }

  // Calculate display months based on horizon length
  const step = T > 60 ? Math.ceil(T / 30) : T > 24 ? 2 : 1
  const displayMonths = Array.from({ length: Math.ceil(T / step) }, (_, i) => i * step)

  // Format month label
  const formatMonth = (t: number) => {
    if (startDate) {
      const date = new Date(startDate)
      date.setMonth(date.getMonth() + t)
      return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
    }
    if (t % 12 === 0) return `Y${t / 12}`
    return ''
  }

  return (
    <div className="space-y-4">
      {/* View Mode Toggle */}
      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          {accountNames.map((name, m) => (
            <span
              key={m}
              className="flex items-center gap-1 text-xs"
            >
              <span
                className="h-3 w-3 rounded"
                style={{ backgroundColor: ACCOUNT_COLORS[m % ACCOUNT_COLORS.length].main }}
              />
              {name}
            </span>
          ))}
        </div>
        <div className="flex rounded-md border border-gray-200">
          <button
            onClick={() => setViewMode('heatmap')}
            className={`px-3 py-1 text-xs ${viewMode === 'heatmap' ? 'bg-gray-100 font-medium' : ''}`}
          >
            Heatmap
          </button>
          <button
            onClick={() => setViewMode('stacked')}
            className={`px-3 py-1 text-xs ${viewMode === 'stacked' ? 'bg-gray-100 font-medium' : ''}`}
          >
            Stacked
          </button>
        </div>
      </div>

      {viewMode === 'heatmap' ? (
        <div className="overflow-x-auto">
          <div className="min-w-fit">
            {/* Heatmap */}
            <div className="flex">
              {/* Y-axis labels (accounts) */}
              <div className="flex flex-col pr-2">
                <div className="h-6" />
                {accountNames.map((name, m) => (
                  <div key={m} className="flex h-10 items-center text-xs font-medium text-gray-700">
                    <span
                      className="mr-2 h-2 w-2 rounded-full"
                      style={{ backgroundColor: ACCOUNT_COLORS[m % ACCOUNT_COLORS.length].main }}
                    />
                    {name || `Account ${m}`}
                  </div>
                ))}
              </div>

              {/* Grid */}
              <div className="flex-1">
                {/* X-axis labels (months) */}
                <div className="flex">
                  {displayMonths.map((t) => (
                    <div key={t} className="h-6 w-5 text-center text-[10px] text-gray-500">
                      {formatMonth(t)}
                    </div>
                  ))}
                </div>

                {/* Cells */}
                {Array.from({ length: M }).map((_, m) => (
                  <div key={m} className="flex">
                    {displayMonths.map((t) => {
                      const value = allocation[t]?.[m] ?? 0
                      const isHovered = hoveredCell?.t === t && hoveredCell?.m === m
                      return (
                        <div
                          key={t}
                          className={`h-10 w-5 border border-white transition-all ${isHovered ? 'ring-2 ring-gray-400 ring-offset-1' : ''}`}
                          style={{ backgroundColor: getColor(value, m) }}
                          onMouseEnter={() => setHoveredCell({ t, m })}
                          onMouseLeave={() => setHoveredCell(null)}
                        />
                      )
                    })}
                  </div>
                ))}
              </div>
            </div>

            {/* Tooltip */}
            {hoveredCell && (() => {
              const pct = (allocation[hoveredCell.t]?.[hoveredCell.m] ?? 0) * 100
              const amount = contributionsByName?.[accountNames[hoveredCell.m]]?.[hoveredCell.t]
              const total = cashFlow?.contributions_mean[hoveredCell.t]
              return (
                <div className="mt-2 rounded-md bg-gray-800 px-3 py-2 text-xs text-white">
                  <span className="font-medium">Month {hoveredCell.t}</span>
                  {startDate && (
                    <span className="ml-2 text-gray-300">
                      ({formatMonth(hoveredCell.t)})
                    </span>
                  )}
                  {total != null && (
                    <span className="ml-2 text-gray-300">
                      • Total: {formatCurrency(total)}
                    </span>
                  )}
                  <span className="mx-2">|</span>
                  <span>{accountNames[hoveredCell.m]}</span>
                  <span className="mx-2">:</span>
                  <span className="font-semibold">{pct.toFixed(1)}%</span>
                  {amount != null && (
                    <span className="ml-2 text-gray-300">→ {formatCurrency(amount)}</span>
                  )}
                </div>
              )
            })()}
          </div>
        </div>
      ) : (
        /* Stacked Bar View */
        <div className="overflow-x-auto">
          <div className="flex h-48 items-end gap-px">
            {displayMonths.map((t) => (
              <div
                key={t}
                className="flex h-full w-5 flex-col justify-end"
                onMouseEnter={() => setHoveredCell({ t, m: 0 })}
                onMouseLeave={() => setHoveredCell(null)}
              >
                {accountNames.map((_, m) => {
                  const value = allocation[t]?.[m] ?? 0
                  return (
                    <div
                      key={m}
                      style={{
                        height: `${value * 100}%`,
                        backgroundColor: ACCOUNT_COLORS[m % ACCOUNT_COLORS.length].main,
                      }}
                    />
                  )
                })}
              </div>
            ))}
          </div>
          {/* X-axis labels */}
          <div className="flex gap-px">
            {displayMonths.map((t) => (
              <div key={t} className="w-5 text-center text-[10px] text-gray-500">
                {t % 12 === 0 ? `Y${t / 12}` : ''}
              </div>
            ))}
          </div>

          {/* Tooltip for stacked view */}
          {hoveredCell && (() => {
            const total = cashFlow?.contributions_mean[hoveredCell.t]
            return (
              <div className="mt-2 rounded-md bg-gray-800 px-3 py-2 text-xs text-white">
                <span className="font-medium">Month {hoveredCell.t}</span>
                {total != null && (
                  <span className="ml-2 text-gray-300">• Total: {formatCurrency(total)}</span>
                )}
                <span className="ml-4">
                  {accountNames.map((name, m) => {
                    const pct = ((allocation[hoveredCell.t]?.[m] ?? 0) * 100).toFixed(0)
                    const amount = contributionsByName?.[name]?.[hoveredCell.t]
                    return (
                      <span key={m} className="ml-2">
                        {name}: {pct}%
                        {amount != null && (
                          <span className="text-gray-300"> ({formatCurrency(amount)})</span>
                        )}
                      </span>
                    )
                  })}
                </span>
              </div>
            )
          })()}
        </div>
      )}

      {/* Summary Stats Cards */}
      {stats && (
        <div className="grid grid-cols-2 gap-3 pt-4 sm:grid-cols-3 lg:grid-cols-6">
          {stats.map((stat, m) => (
            <div
              key={m}
              className="rounded-lg border p-3"
              style={{ borderColor: stat.color.main, borderLeftWidth: '4px' }}
            >
              <p className="text-sm font-medium text-gray-900">{stat.name}</p>
              <div className="mt-1 space-y-1 text-xs text-gray-500">
                <div className="flex justify-between">
                  <span>Average:</span>
                  <span className="font-medium text-gray-700">{(stat.avg * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Range:</span>
                  <span className="text-gray-600">{(stat.min * 100).toFixed(0)}-{(stat.max * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Trend:</span>
                  <span className={stat.trend > 0.01 ? 'text-green-600' : stat.trend < -0.01 ? 'text-red-600' : 'text-gray-600'}>
                    {stat.trend > 0 ? '+' : ''}{(stat.trend * 100).toFixed(1)}%
                  </span>
                </div>
                {stat.avgAmount != null && (
                  <div className="flex justify-between border-t border-gray-100 pt-1 mt-1">
                    <span>Avg/mo:</span>
                    <span className="font-medium text-gray-700">{formatCurrency(stat.avgAmount)}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
