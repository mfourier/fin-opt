import { useMemo, useState } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts'
import type { SummaryStats, GoalStatus } from '../types/database'

interface PerAccountStats {
  account: string
  display_name: string
  mean: number[]
  p10: number[]
  p25: number[]
  p50: number[]
  p75: number[]
  p90: number[]
}

interface WealthTrajectoryChartProps {
  summaryStats: SummaryStats
  startDate?: string
  goalStatus?: GoalStatus[] | null
  optimalHorizon?: number | null
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

// Colors for bands
const BAND_COLOR = '#3B82F6'  // Blue
const ACCOUNT_COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']

export default function WealthTrajectoryChart({
  summaryStats,
  startDate,
  goalStatus,
  optimalHorizon: _optimalHorizon,
}: WealthTrajectoryChartProps) {
  void _optimalHorizon // Reserved for future timeline annotations
  const [viewMode, setViewMode] = useState<'total' | 'per_account'>('total')
  const [selectedAccount, setSelectedAccount] = useState<number>(0)

  const perAccount = (summaryStats as unknown as { per_account?: PerAccountStats[] })?.per_account
  const totalWealth = summaryStats as unknown as {
    total_wealth?: { mean: number[]; p10: number[]; p25: number[]; p50: number[]; p75: number[]; p90: number[] }
  }

  const totalData = totalWealth?.total_wealth
  const hasPerAccount = perAccount && perAccount.length > 0
  const hasTotalData = totalData && totalData.mean && totalData.mean.length > 0

  const chartData = useMemo(() => {
    if (viewMode === 'total' && hasTotalData) {
      const data = totalData!
      return data.mean.map((_, t) => {
        const point: Record<string, number | string> = { month: t }

        if (startDate) {
          const d = new Date(startDate)
          d.setMonth(d.getMonth() + t)
          point.date = d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
        }

        point.p10 = data.p10[t]
        point.p25 = data.p25[t]
        point.p50 = data.p50[t]
        point.p75 = data.p75[t]
        point.p90 = data.p90[t]
        point.mean = data.mean[t]

        // Bands for area chart: we need the "range" between levels
        point.band_10_25 = data.p25[t] - data.p10[t]
        point.band_25_50 = data.p50[t] - data.p25[t]
        point.band_50_75 = data.p75[t] - data.p50[t]
        point.band_75_90 = data.p90[t] - data.p75[t]
        point.base = data.p10[t]

        return point
      })
    }

    if (viewMode === 'per_account' && hasPerAccount) {
      const acc = perAccount![selectedAccount]
      if (!acc) return []

      return acc.mean.map((_, t) => {
        const point: Record<string, number | string> = { month: t }

        if (startDate) {
          const d = new Date(startDate)
          d.setMonth(d.getMonth() + t)
          point.date = d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
        }

        point.p10 = acc.p10[t]
        point.p25 = acc.p25[t]
        point.p50 = acc.p50[t]
        point.p75 = acc.p75[t]
        point.p90 = acc.p90[t]
        point.mean = acc.mean[t]

        point.band_10_25 = acc.p25[t] - acc.p10[t]
        point.band_25_50 = acc.p50[t] - acc.p25[t]
        point.band_50_75 = acc.p75[t] - acc.p50[t]
        point.band_75_90 = acc.p90[t] - acc.p75[t]
        point.base = acc.p10[t]

        return point
      })
    }

    return []
  }, [viewMode, totalData, hasTotalData, perAccount, hasPerAccount, selectedAccount, startDate])

  if (!hasTotalData && !hasPerAccount) {
    return <p className="text-gray-500">No wealth trajectory data available</p>
  }

  const color = viewMode === 'per_account'
    ? ACCOUNT_COLORS[selectedAccount % ACCOUNT_COLORS.length]
    : BAND_COLOR

  // Find goal thresholds for reference lines
  const goalThresholds = goalStatus?.filter(g =>
    viewMode === 'total' || (viewMode === 'per_account' && perAccount && g.account === perAccount[selectedAccount]?.account)
  ) ?? []

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex rounded-md border border-gray-200">
            <button
              onClick={() => setViewMode('total')}
              className={`px-3 py-1.5 text-xs ${viewMode === 'total' ? 'bg-gray-100 font-medium' : ''}`}
            >
              Total Wealth
            </button>
            {hasPerAccount && (
              <button
                onClick={() => setViewMode('per_account')}
                className={`px-3 py-1.5 text-xs ${viewMode === 'per_account' ? 'bg-gray-100 font-medium' : ''}`}
              >
                Per Account
              </button>
            )}
          </div>

          {viewMode === 'per_account' && hasPerAccount && (
            <select
              value={selectedAccount}
              onChange={(e) => setSelectedAccount(Number(e.target.value))}
              className="rounded-md border border-gray-300 px-2 py-1 text-sm"
            >
              {perAccount!.map((acc, i) => (
                <option key={i} value={i}>{acc.display_name || acc.account}</option>
              ))}
            </select>
          )}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-4 rounded" style={{ backgroundColor: color, opacity: 0.2 }}></span>
            P10-P90
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-4 rounded" style={{ backgroundColor: color, opacity: 0.4 }}></span>
            P25-P75
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-1 w-4 rounded" style={{ backgroundColor: color }}></span>
            Median
          </span>
        </div>
      </div>

      {/* Chart */}
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis
              dataKey={startDate ? 'date' : 'month'}
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: '#E5E7EB' }}
              interval="preserveStartEnd"
            />
            <YAxis
              tickFormatter={formatCurrency}
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: '#E5E7EB' }}
              width={70}
            />
            <Tooltip
              formatter={(value: number, name: string) => {
                const labels: Record<string, string> = {
                  mean: 'Mean',
                  p50: 'Median (P50)',
                  p10: 'P10',
                  p25: 'P25',
                  p75: 'P75',
                  p90: 'P90',
                }
                return [formatCurrency(value), labels[name] || name]
              }}
              labelFormatter={(label) => startDate ? String(label) : `Month ${label}`}
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: '1px solid #E5E7EB',
                borderRadius: '6px',
                fontSize: '12px',
              }}
            />

            {/* Fan chart: stacked bands from P10 to P90 */}
            {/* Base (invisible): P10 */}
            <Area
              type="monotone"
              dataKey="base"
              stackId="fan"
              stroke="none"
              fill="transparent"
            />
            {/* Band P10-P25 */}
            <Area
              type="monotone"
              dataKey="band_10_25"
              stackId="fan"
              stroke="none"
              fill={color}
              fillOpacity={0.12}
              name="p10"
            />
            {/* Band P25-P50 */}
            <Area
              type="monotone"
              dataKey="band_25_50"
              stackId="fan"
              stroke="none"
              fill={color}
              fillOpacity={0.25}
              name="p25"
            />
            {/* Band P50-P75 */}
            <Area
              type="monotone"
              dataKey="band_50_75"
              stackId="fan"
              stroke="none"
              fill={color}
              fillOpacity={0.25}
              name="p75"
            />
            {/* Band P75-P90 */}
            <Area
              type="monotone"
              dataKey="band_75_90"
              stackId="fan"
              stroke="none"
              fill={color}
              fillOpacity={0.12}
              name="p90"
            />

            {/* Median line */}
            <Area
              type="monotone"
              dataKey="p50"
              stroke={color}
              strokeWidth={2}
              fill="none"
              dot={false}
              name="p50"
            />

            {/* Mean line (dashed) */}
            <Area
              type="monotone"
              dataKey="mean"
              stroke={color}
              strokeWidth={1.5}
              strokeDasharray="5 5"
              fill="none"
              dot={false}
              name="mean"
            />

            {/* Goal threshold reference lines */}
            {viewMode === 'total' && goalThresholds.map((goal, i) => (
              <ReferenceLine
                key={i}
                y={goal.threshold}
                stroke="#EF4444"
                strokeDasharray="4 4"
                strokeWidth={1.5}
                label={{
                  value: `${goal.account}: ${formatCurrency(goal.threshold)}`,
                  position: 'right',
                  fontSize: 10,
                  fill: '#EF4444',
                }}
              />
            ))}

            <Legend content={() => null} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Final Wealth Stats */}
      {chartData.length > 0 && (
        <div className="grid grid-cols-5 gap-3 rounded-lg bg-gray-50 p-4 text-center">
          {[
            { label: 'P10 (Pessimistic)', key: 'p10' },
            { label: 'P25', key: 'p25' },
            { label: 'Median', key: 'p50' },
            { label: 'P75', key: 'p75' },
            { label: 'P90 (Optimistic)', key: 'p90' },
          ].map(({ label, key }) => {
            const lastPoint = chartData[chartData.length - 1]
            const value = lastPoint?.[key] as number | undefined
            return (
              <div key={key}>
                <p className="text-xs text-gray-500">{label}</p>
                <p className="text-sm font-semibold text-gray-900">
                  {value !== undefined ? formatCurrency(value) : '-'}
                </p>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
