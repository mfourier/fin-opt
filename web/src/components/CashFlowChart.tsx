import { useMemo, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts'
import type { CashFlowStats } from '../types/database'

interface CashFlowChartProps {
  cashFlow: CashFlowStats
  startDate?: string
}

const CONTRIBUTION_COLOR = '#10B981' // Green
const WITHDRAWAL_COLOR = '#EF4444'   // Red
const ACCOUNT_COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']

const formatCurrency = (value: number) => {
  if (Math.abs(value) >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(1)}M`
  }
  if (Math.abs(value) >= 1_000) {
    return `$${(value / 1_000).toFixed(0)}K`
  }
  return `$${value.toFixed(0)}`
}

export default function CashFlowChart({ cashFlow, startDate }: CashFlowChartProps) {
  const [viewMode, setViewMode] = useState<'net' | 'by_account'>('net')

  const hasWithdrawals = cashFlow.withdrawals_mean && cashFlow.withdrawals_mean.some(v => v > 0)

  const chartData = useMemo(() => {
    return cashFlow.contributions_mean.map((contrib, t) => {
      const point: Record<string, number | string> = { month: t }

      if (startDate) {
        const d = new Date(startDate + 'T00:00:00')
        d.setMonth(d.getMonth() + t)
        point.date = d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
      }

      if (viewMode === 'net') {
        point.contributions = contrib
        const withdrawal = cashFlow.withdrawals_mean?.[t] ?? 0
        point.withdrawals = -withdrawal  // Negative for downward bars
        point.net = contrib - withdrawal
      } else {
        // By account: show contribution per account as stacked bars
        for (const acc of cashFlow.contributions_by_account) {
          point[`contrib_${acc.account}`] = acc.mean[t] ?? 0
        }
        // Withdrawals per account (negative)
        if (cashFlow.withdrawals_by_account) {
          for (const acc of cashFlow.withdrawals_by_account) {
            const val = acc.mean[t] ?? 0
            if (val > 0) {
              point[`wd_${acc.account}`] = -val
            }
          }
        }
      }

      return point
    })
  }, [cashFlow, startDate, viewMode])

  // Sample data for large horizons to keep the chart readable
  const sampledData = useMemo(() => {
    if (chartData.length <= 48) return chartData
    const step = Math.ceil(chartData.length / 48)
    return chartData.filter((_, i) => i % step === 0 || i === chartData.length - 1)
  }, [chartData])

  // Compute totals for summary
  const totalContributions = cashFlow.contributions_mean.reduce((s, v) => s + v, 0)
  const totalWithdrawals = cashFlow.withdrawals_mean?.reduce((s, v) => s + v, 0) ?? 0

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex rounded-md border border-gray-200">
            <button
              onClick={() => setViewMode('net')}
              className={`px-3 py-1.5 text-xs ${viewMode === 'net' ? 'bg-gray-100 font-medium' : ''}`}
            >
              Net Cash Flow
            </button>
            <button
              onClick={() => setViewMode('by_account')}
              className={`px-3 py-1.5 text-xs ${viewMode === 'by_account' ? 'bg-gray-100 font-medium' : ''}`}
            >
              By Account
            </button>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <span className="inline-block h-3 w-3 rounded-sm" style={{ backgroundColor: CONTRIBUTION_COLOR }}></span>
            Contributions
          </span>
          {hasWithdrawals && (
            <span className="flex items-center gap-1">
              <span className="inline-block h-3 w-3 rounded-sm" style={{ backgroundColor: WITHDRAWAL_COLOR }}></span>
              Withdrawals
            </span>
          )}
        </div>
      </div>

      {/* Chart */}
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sampledData}
            margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis
              dataKey={startDate ? 'date' : 'month'}
              tick={{ fontSize: 10 }}
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
                const absValue = Math.abs(value)
                const prefix = value < 0 ? '-' : ''
                const label = name.startsWith('contrib_')
                  ? `${name.replace('contrib_', '')} (contrib)`
                  : name.startsWith('wd_')
                    ? `${name.replace('wd_', '')} (withdrawal)`
                    : name === 'contributions' ? 'Contributions'
                    : name === 'withdrawals' ? 'Withdrawals'
                    : name === 'net' ? 'Net Flow'
                    : name
                return [`${prefix}${formatCurrency(absValue)}`, label]
              }}
              labelFormatter={(label) => startDate ? String(label) : `Month ${label}`}
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: '1px solid #E5E7EB',
                borderRadius: '6px',
                fontSize: '12px',
              }}
            />
            <ReferenceLine y={0} stroke="#9CA3AF" strokeWidth={1} />

            {viewMode === 'net' ? (
              <>
                <Bar dataKey="contributions" fill={CONTRIBUTION_COLOR} radius={[2, 2, 0, 0]} />
                {hasWithdrawals && (
                  <Bar dataKey="withdrawals" fill={WITHDRAWAL_COLOR} radius={[0, 0, 2, 2]} />
                )}
              </>
            ) : (
              <>
                {/* Stacked contribution bars by account */}
                {cashFlow.contributions_by_account.map((acc, i) => (
                  <Bar
                    key={`contrib_${acc.account}`}
                    dataKey={`contrib_${acc.account}`}
                    stackId="contrib"
                    fill={ACCOUNT_COLORS[i % ACCOUNT_COLORS.length]}
                    radius={i === cashFlow.contributions_by_account.length - 1 ? [2, 2, 0, 0] : undefined}
                    name={`contrib_${acc.account}`}
                  />
                ))}
                {/* Stacked withdrawal bars by account */}
                {cashFlow.withdrawals_by_account?.map((acc, i) => (
                  <Bar
                    key={`wd_${acc.account}`}
                    dataKey={`wd_${acc.account}`}
                    stackId="wd"
                    fill={ACCOUNT_COLORS[i % ACCOUNT_COLORS.length]}
                    fillOpacity={0.5}
                    stroke={ACCOUNT_COLORS[i % ACCOUNT_COLORS.length]}
                    strokeDasharray="3 3"
                    name={`wd_${acc.account}`}
                  />
                ))}
              </>
            )}
            <Legend content={() => null} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-3 gap-3 rounded-lg bg-gray-50 p-4 text-center">
        <div>
          <p className="text-xs text-gray-500">Total Contributions</p>
          <p className="text-sm font-semibold text-green-700">{formatCurrency(totalContributions)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Total Withdrawals</p>
          <p className="text-sm font-semibold text-red-700">
            {totalWithdrawals > 0 ? `-${formatCurrency(totalWithdrawals)}` : '-'}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Net Cash Flow</p>
          <p className={`text-sm font-semibold ${totalContributions - totalWithdrawals >= 0 ? 'text-green-700' : 'text-red-700'}`}>
            {formatCurrency(totalContributions - totalWithdrawals)}
          </p>
        </div>
      </div>
    </div>
  )
}
