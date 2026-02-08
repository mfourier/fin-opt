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
} from 'recharts'
import type { CashFlowStats } from '../types/database'

interface AllocationChartProps {
  allocation: number[][]
  accountNames: string[]
  startDate?: string
  cashFlow?: CashFlowStats
}

// Color palette matching the heatmap
const ACCOUNT_COLORS = [
  '#3B82F6', // Blue
  '#10B981', // Green
  '#F59E0B', // Amber
  '#EF4444', // Red
  '#8B5CF6', // Purple
  '#EC4899', // Pink
]

const formatCurrency = (value: number) => {
  if (Math.abs(value) >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(1)}M`
  }
  if (Math.abs(value) >= 1_000) {
    return `$${(value / 1_000).toFixed(0)}K`
  }
  return `$${value.toFixed(0)}`
}

export default function AllocationChart({ allocation, accountNames, startDate, cashFlow }: AllocationChartProps) {
  const [viewMode, setViewMode] = useState<'percentage' | 'amount'>('percentage')
  const showToggle = !!cashFlow

  // Build a lookup: accountName -> mean contribution array
  const contributionsByName = useMemo(() => {
    if (!cashFlow) return null
    const map: Record<string, number[]> = {}
    for (const acc of cashFlow.contributions_by_account) {
      const displayName = acc.display_name || acc.account
      map[displayName] = acc.mean
    }
    return map
  }, [cashFlow])

  const chartData = useMemo(() => {
    if (!allocation || allocation.length === 0) return []

    const T = allocation.length
    // Sample data for large horizons
    const step = T > 60 ? Math.ceil(T / 60) : 1

    const data = []
    for (let t = 0; t < T; t += step) {
      const point: Record<string, number | string> = { month: t }

      // Add date label if startDate provided
      if (startDate) {
        const date = new Date(startDate)
        date.setMonth(date.getMonth() + t)
        point.date = date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
      }

      if (viewMode === 'amount' && contributionsByName) {
        // Dollar amounts per account
        accountNames.forEach((name) => {
          point[name] = contributionsByName[name]?.[t] ?? 0
        })
        // Total contribution for tooltip
        point._total = cashFlow!.contributions_mean[t] ?? 0
      } else {
        // Allocation percentages for each account
        accountNames.forEach((name, m) => {
          point[name] = (allocation[t]?.[m] ?? 0) * 100
        })
      }

      data.push(point)
    }

    return data
  }, [allocation, accountNames, startDate, viewMode, contributionsByName, cashFlow])

  if (!allocation || allocation.length === 0) {
    return <p className="text-gray-500">No allocation data available</p>
  }

  const isAmount = viewMode === 'amount'

  return (
    <div className="space-y-2">
      {/* View Mode Toggle */}
      {showToggle && (
        <div className="flex justify-end">
          <div className="flex rounded-md border border-gray-200">
            <button
              onClick={() => setViewMode('percentage')}
              className={`px-3 py-1 text-xs ${viewMode === 'percentage' ? 'bg-gray-100 font-medium' : ''}`}
            >
              Percentage
            </button>
            <button
              onClick={() => setViewMode('amount')}
              className={`px-3 py-1 text-xs ${viewMode === 'amount' ? 'bg-gray-100 font-medium' : ''}`}
            >
              Amount ($)
            </button>
          </div>
        </div>
      )}

      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            stackOffset={isAmount ? 'none' : 'expand'}
            margin={{ top: 10, right: 30, left: isAmount ? 10 : 0, bottom: 0 }}
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
              tickFormatter={isAmount ? formatCurrency : (value) => `${Math.round(value * 100)}%`}
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: '#E5E7EB' }}
              domain={isAmount ? [0, 'auto'] : [0, 1]}
              width={isAmount ? 70 : undefined}
            />
            <Tooltip
              formatter={(value: number, name: string) => {
                if (name === '_total') return [null, null]
                if (isAmount) {
                  return [formatCurrency(value), name]
                }
                return [`${value.toFixed(1)}%`, name]
              }}
              labelFormatter={(label, payload) => {
                const dateLabel = startDate ? String(label) : `Month ${label}`
                if (isAmount && payload && payload.length > 0) {
                  const total = (payload[0]?.payload as Record<string, number>)?._total
                  if (total != null) {
                    return `${dateLabel}  •  Total: ${formatCurrency(total)}`
                  }
                }
                return dateLabel
              }}
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: '1px solid #E5E7EB',
                borderRadius: '6px',
                fontSize: '12px',
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: '12px' }}
              iconType="square"
              payload={accountNames.map((name, m) => ({
                value: name,
                type: 'square' as const,
                color: ACCOUNT_COLORS[m % ACCOUNT_COLORS.length],
              }))}
            />
            {accountNames.map((name, m) => (
              <Area
                key={name}
                type="monotone"
                dataKey={name}
                stackId="1"
                stroke={ACCOUNT_COLORS[m % ACCOUNT_COLORS.length]}
                fill={ACCOUNT_COLORS[m % ACCOUNT_COLORS.length]}
                fillOpacity={0.8}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
