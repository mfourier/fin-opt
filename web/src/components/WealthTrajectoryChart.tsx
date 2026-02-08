import { useMemo, useState } from 'react'
import {
  AreaChart,
  LineChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts'
import type { SummaryStats, GoalStatus, WithdrawalsConfig, SampledTrajectories } from '../types/database'

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

interface WithdrawalMarker {
  month: number
  label: string
  amount: number
  account: string
  type: 'scheduled' | 'stochastic'
}

interface WealthTrajectoryChartProps {
  summaryStats: SummaryStats
  startDate?: string
  goalStatus?: GoalStatus[] | null
  optimalHorizon?: number | null
  withdrawals?: WithdrawalsConfig | null
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

type ChartStyle = 'percentiles' | 'montecarlo'

export default function WealthTrajectoryChart({
  summaryStats,
  startDate,
  goalStatus,
  optimalHorizon: _optimalHorizon,
  withdrawals,
}: WealthTrajectoryChartProps) {
  void _optimalHorizon // Reserved for future timeline annotations

  const perAccount = (summaryStats as unknown as { per_account?: PerAccountStats[] })?.per_account
  const totalWealth = summaryStats as unknown as {
    total_wealth?: { mean: number[]; p10: number[]; p25: number[]; p50: number[]; p75: number[]; p90: number[] }
  }
  const trajectories = (summaryStats as unknown as { trajectories?: SampledTrajectories })?.trajectories

  const totalData = totalWealth?.total_wealth
  const hasPerAccount = perAccount && perAccount.length > 0

  const [viewMode, setViewMode] = useState<'total' | 'per_account' | 'all_accounts'>(hasPerAccount ? 'all_accounts' : 'total')
  const [chartStyle, setChartStyle] = useState<ChartStyle>('percentiles')
  const [selectedAccount, setSelectedAccount] = useState<number>(0)
  const hasTotalData = totalData && totalData.mean && totalData.mean.length > 0
  const hasTrajectories = trajectories && trajectories.total && trajectories.total.length > 0

  // Helper to get date label for a month offset
  const getDateLabel = (t: number) => {
    if (!startDate) return undefined
    const d = new Date(startDate + 'T00:00:00')
    d.setMonth(d.getMonth() + t)
    return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
  }

  // Percentile fan chart data
  const percentileData = useMemo(() => {
    if (viewMode === 'total' && hasTotalData) {
      const data = totalData!
      return data.mean.map((_, t) => {
        const point: Record<string, number | string> = { month: t }
        const dateLabel = getDateLabel(t)
        if (dateLabel) point.date = dateLabel

        point.p10 = data.p10[t]
        point.p25 = data.p25[t]
        point.p50 = data.p50[t]
        point.p75 = data.p75[t]
        point.p90 = data.p90[t]
        point.mean = data.mean[t]

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
        const dateLabel = getDateLabel(t)
        if (dateLabel) point.date = dateLabel

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

  // Monte Carlo trajectory data
  const mcData = useMemo(() => {
    if (!hasTrajectories) return { data: [], nTrajectories: 0 }

    let rawTrajectories: number[][]
    if (viewMode === 'total') {
      rawTrajectories = trajectories!.total
    } else {
      const accTrajs = trajectories!.per_account?.[selectedAccount]
      if (!accTrajs) return { data: [], nTrajectories: 0 }
      rawTrajectories = accTrajs.trajectories
    }

    if (!rawTrajectories || rawTrajectories.length === 0) return { data: [], nTrajectories: 0 }

    const nTrajectories = rawTrajectories.length
    const T = rawTrajectories[0].length

    // Build data: each row is { month, date?, traj_0, traj_1, ..., percentile stats }
    const data = Array.from({ length: T }, (_, t) => {
      const point: Record<string, number | string> = { month: t }
      const dateLabel = getDateLabel(t)
      if (dateLabel) point.date = dateLabel

      for (let i = 0; i < nTrajectories; i++) {
        point[`t${i}`] = rawTrajectories[i][t]
      }

      // Reuse full-simulation percentile stats for tooltip consistency
      if (viewMode === 'total' && hasTotalData) {
        point.p10 = totalData!.p10[t]
        point.p25 = totalData!.p25[t]
        point.p50 = totalData!.p50[t]
        point.p75 = totalData!.p75[t]
        point.p90 = totalData!.p90[t]
        point.mean = totalData!.mean[t]
      } else if (viewMode === 'per_account' && hasPerAccount) {
        const acc = perAccount![selectedAccount]
        if (acc) {
          point.p10 = acc.p10[t]
          point.p25 = acc.p25[t]
          point.p50 = acc.p50[t]
          point.p75 = acc.p75[t]
          point.p90 = acc.p90[t]
          point.mean = acc.mean[t]
        }
      }

      return point
    })

    return { data, nTrajectories }
  }, [viewMode, trajectories, hasTrajectories, selectedAccount, startDate, totalData, hasTotalData, perAccount, hasPerAccount])

  // All-accounts percentile data: overlays fan charts for every account
  const allAccountsPercentileData = useMemo(() => {
    if (viewMode !== 'all_accounts' || !hasPerAccount) return []

    const T = perAccount![0].mean.length
    return Array.from({ length: T }, (_, t) => {
      const point: Record<string, number | string> = { month: t }
      const dateLabel = getDateLabel(t)
      if (dateLabel) point.date = dateLabel

      for (let m = 0; m < perAccount!.length; m++) {
        const acc = perAccount![m]
        point[`acc${m}_base`] = acc.p10[t]
        point[`acc${m}_band_10_25`] = acc.p25[t] - acc.p10[t]
        point[`acc${m}_band_25_50`] = acc.p50[t] - acc.p25[t]
        point[`acc${m}_band_50_75`] = acc.p75[t] - acc.p50[t]
        point[`acc${m}_band_75_90`] = acc.p90[t] - acc.p75[t]
        point[`acc${m}_p50`] = acc.p50[t]
        point[`acc${m}_mean`] = acc.mean[t]
        point[`acc${m}_p10`] = acc.p10[t]
        point[`acc${m}_p25`] = acc.p25[t]
        point[`acc${m}_p75`] = acc.p75[t]
        point[`acc${m}_p90`] = acc.p90[t]
      }
      return point
    })
  }, [viewMode, perAccount, hasPerAccount, startDate])

  // All-accounts Monte Carlo data: overlays trajectories for every account
  const allAccountsMCData = useMemo(() => {
    if (viewMode !== 'all_accounts' || !hasTrajectories || !trajectories?.per_account) {
      return { data: [] as Record<string, number | string>[], nPerAccount: [] as number[] }
    }

    const nAccounts = trajectories.per_account.length
    const nPerAccount = trajectories.per_account.map(a => a.trajectories.length)
    const T = trajectories.per_account[0]?.trajectories[0]?.length ?? 0
    if (T === 0) return { data: [], nPerAccount }

    const data = Array.from({ length: T }, (_, t) => {
      const point: Record<string, number | string> = { month: t }
      const dateLabel = getDateLabel(t)
      if (dateLabel) point.date = dateLabel

      for (let m = 0; m < nAccounts; m++) {
        const accTrajs = trajectories.per_account[m].trajectories
        for (let i = 0; i < accTrajs.length; i++) {
          point[`acc${m}_t${i}`] = accTrajs[i][t]
        }
        // Add percentile stats for tooltip
        if (hasPerAccount) {
          const acc = perAccount![m]
          point[`acc${m}_p50`] = acc.p50[t]
          point[`acc${m}_mean`] = acc.mean[t]
        }
      }
      return point
    })

    return { data, nPerAccount }
  }, [viewMode, trajectories, hasTrajectories, perAccount, hasPerAccount, startDate])

  // Compute withdrawal markers from scenario data
  const withdrawalMarkers = useMemo((): WithdrawalMarker[] => {
    if (!withdrawals || !startDate) return []

    const markers: WithdrawalMarker[] = []
    const start = new Date(startDate + 'T00:00:00')

    for (const w of withdrawals.scheduled ?? []) {
      const target = new Date(w.date + 'T00:00:00')
      const month = (target.getFullYear() - start.getFullYear()) * 12 + (target.getMonth() - start.getMonth())
      if (month >= 0) {
        markers.push({
          month,
          label: w.description || `${w.account} withdrawal`,
          amount: w.amount,
          account: w.account,
          type: 'scheduled',
        })
      }
    }

    for (const w of withdrawals.stochastic ?? []) {
      let month: number
      if (w.date) {
        const target = new Date(w.date + 'T00:00:00')
        month = (target.getFullYear() - start.getFullYear()) * 12 + (target.getMonth() - start.getMonth())
      } else if (w.month !== undefined) {
        month = w.month
      } else {
        continue
      }
      if (month >= 0) {
        markers.push({
          month,
          label: w.description || `${w.account} variable`,
          amount: w.base_amount,
          account: w.account,
          type: 'stochastic',
        })
      }
    }

    return markers.sort((a, b) => a.month - b.month)
  }, [withdrawals, startDate])

  // Filter markers for current view mode
  const visibleMarkers = useMemo(() => {
    if (viewMode === 'total' || viewMode === 'all_accounts') return withdrawalMarkers
    if (!perAccount) return []
    const selectedAcc = perAccount[selectedAccount]
    if (!selectedAcc) return []
    return withdrawalMarkers.filter(m => m.account === selectedAcc.account)
  }, [withdrawalMarkers, viewMode, perAccount, selectedAccount])

  if (!hasTotalData && !hasPerAccount) {
    return <p className="text-gray-500">No wealth trajectory data available</p>
  }

  const color = viewMode === 'per_account'
    ? ACCOUNT_COLORS[selectedAccount % ACCOUNT_COLORS.length]
    : BAND_COLOR

  // Find goal thresholds for reference lines
  const goalThresholds = goalStatus?.filter(g =>
    viewMode === 'total' || viewMode === 'all_accounts' || (viewMode === 'per_account' && perAccount && g.account === perAccount[selectedAccount]?.account)
  ) ?? []

  // Shared axis/grid/tooltip config for both chart types
  const xAxisProps = {
    dataKey: startDate ? 'date' : 'month',
    tick: { fontSize: 11 },
    tickLine: false,
    axisLine: { stroke: '#E5E7EB' },
    interval: 'preserveStartEnd' as const,
  }

  const yAxisProps = {
    tickFormatter: formatCurrency,
    tick: { fontSize: 11 },
    tickLine: false,
    axisLine: { stroke: '#E5E7EB' },
    width: 70,
  }

  // Withdrawal reference line helper
  const withdrawalRefLines = visibleMarkers.map((marker, i) => (
    <ReferenceLine
      key={`wd-${i}`}
      x={startDate ? (() => {
        const d = new Date(startDate + 'T00:00:00')
        d.setMonth(d.getMonth() + marker.month)
        return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
      })() : marker.month}
      stroke={marker.type === 'scheduled' ? '#DC2626' : '#EA580C'}
      strokeDasharray="3 3"
      strokeWidth={1}
      label={{
        value: `${marker.type === 'stochastic' ? '~' : ''}${formatCurrency(marker.amount)}`,
        position: 'top',
        fontSize: 9,
        fill: marker.type === 'scheduled' ? '#DC2626' : '#EA580C',
      }}
    />
  ))

  // Goal reference lines
  const goalRefLines = (viewMode === 'total' || viewMode === 'all_accounts') ? goalThresholds.map((goal, i) => {
    // In all_accounts mode, color-code goal lines by account
    const goalColor = viewMode === 'all_accounts' && hasPerAccount
      ? ACCOUNT_COLORS[(perAccount!.findIndex(a => a.account === goal.account || a.display_name === goal.account)) % ACCOUNT_COLORS.length] || '#EF4444'
      : '#EF4444'
    return (
      <ReferenceLine
        key={`goal-${i}`}
        y={goal.threshold}
        stroke={goalColor}
        strokeDasharray="4 4"
        strokeWidth={1.5}
        label={{
          value: `${goal.account}: ${formatCurrency(goal.threshold)}`,
          position: 'right',
          fontSize: 10,
          fill: goalColor,
        }}
      />
    )
  }) : []

  // Active chart data for final stats
  const activeData = chartStyle === 'percentiles' ? percentileData : mcData.data

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Account toggle */}
          <div className="flex rounded-md border border-gray-200">
            <button
              onClick={() => setViewMode('total')}
              className={`px-3 py-1.5 text-xs ${viewMode === 'total' ? 'bg-gray-100 font-medium' : ''}`}
            >
              Total Wealth
            </button>
            {hasPerAccount && (
              <>
                <button
                  onClick={() => setViewMode('per_account')}
                  className={`px-3 py-1.5 text-xs ${viewMode === 'per_account' ? 'bg-gray-100 font-medium' : ''}`}
                >
                  Per Account
                </button>
                <button
                  onClick={() => setViewMode('all_accounts')}
                  className={`px-3 py-1.5 text-xs ${viewMode === 'all_accounts' ? 'bg-gray-100 font-medium' : ''}`}
                >
                  All Accounts
                </button>
              </>
            )}
          </div>

          {/* Chart style toggle */}
          <div className="flex rounded-md border border-gray-200">
            <button
              onClick={() => setChartStyle('percentiles')}
              className={`px-3 py-1.5 text-xs ${chartStyle === 'percentiles' ? 'bg-gray-100 font-medium' : ''}`}
            >
              Percentiles
            </button>
            {hasTrajectories && (
              <button
                onClick={() => setChartStyle('montecarlo')}
                className={`px-3 py-1.5 text-xs ${chartStyle === 'montecarlo' ? 'bg-gray-100 font-medium' : ''}`}
              >
                Monte Carlo
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
        <div className="flex flex-wrap items-center gap-3 text-xs text-gray-500">
          {viewMode === 'all_accounts' && hasPerAccount ? (
            <>
              {perAccount!.map((acc, m) => (
                <span key={m} className="flex items-center gap-1">
                  <span
                    className="inline-block h-2 w-4 rounded"
                    style={{ backgroundColor: ACCOUNT_COLORS[m % ACCOUNT_COLORS.length] }}
                  ></span>
                  {acc.display_name || acc.account}
                </span>
              ))}
            </>
          ) : chartStyle === 'percentiles' ? (
            <>
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
            </>
          ) : (
            <>
              <span className="flex items-center gap-1">
                <span className="inline-block h-1 w-4 rounded" style={{ backgroundColor: color, opacity: 0.15 }}></span>
                Trajectories ({mcData.nTrajectories})
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-1 w-4 rounded" style={{ backgroundColor: color }}></span>
                Median
              </span>
            </>
          )}
          {visibleMarkers.length > 0 && (
            <span className="flex items-center gap-1">
              <span className="inline-block h-4 w-0 border-l-2 border-dashed border-red-600"></span>
              Withdrawals
            </span>
          )}
        </div>
      </div>

      {/* Chart */}
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          {viewMode === 'all_accounts' && hasPerAccount ? (
            chartStyle === 'montecarlo' && allAccountsMCData.data.length > 0 ? (
              <LineChart
                data={allAccountsMCData.data}
                margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis {...xAxisProps} />
                <YAxis {...yAxisProps} />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload || payload.length === 0) return null
                    const point = payload[0]?.payload
                    if (!point) return null
                    const dateLabel = startDate ? String(label) : `Month ${label}`
                    return (
                      <div style={{
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        border: '1px solid #E5E7EB',
                        borderRadius: '6px',
                        fontSize: '12px',
                        padding: '8px 12px',
                      }}>
                        <p style={{ fontWeight: 600, marginBottom: 4 }}>{dateLabel}</p>
                        {perAccount!.map((acc, m) => (
                          <p key={m} style={{ margin: '2px 0', display: 'flex', justifyContent: 'space-between', gap: 16 }}>
                            <span style={{ color: ACCOUNT_COLORS[m % ACCOUNT_COLORS.length] }}>
                              {acc.display_name || acc.account}
                            </span>
                            <span style={{ fontWeight: 500 }}>
                              {formatCurrency(point[`acc${m}_p50`] as number ?? 0)}
                            </span>
                          </p>
                        ))}
                      </div>
                    )
                  }}
                />
                {/* Trajectories for each account */}
                {perAccount!.map((_, m) => {
                  const accColor = ACCOUNT_COLORS[m % ACCOUNT_COLORS.length]
                  const nTrajs = allAccountsMCData.nPerAccount[m] ?? 0
                  return Array.from({ length: nTrajs }, (__, i) => (
                    <Line
                      key={`acc${m}_t${i}`}
                      type="monotone"
                      dataKey={`acc${m}_t${i}`}
                      stroke={accColor}
                      strokeWidth={0.8}
                      strokeOpacity={0.1}
                      dot={false}
                      activeDot={false}
                      isAnimationActive={false}
                      name={`acc${m}_t${i}`}
                    />
                  ))
                }).flat()}
                {/* Median lines on top */}
                {perAccount!.map((acc, m) => (
                  <Line
                    key={`acc${m}_median`}
                    type="monotone"
                    dataKey={`acc${m}_p50`}
                    stroke={ACCOUNT_COLORS[m % ACCOUNT_COLORS.length]}
                    strokeWidth={2.5}
                    dot={false}
                    name={acc.display_name || acc.account}
                    isAnimationActive={false}
                  />
                ))}
                {goalRefLines}
                {withdrawalRefLines}
                <Legend content={() => null} />
              </LineChart>
            ) : (
              <AreaChart
                data={allAccountsPercentileData}
                margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis {...xAxisProps} />
                <YAxis {...yAxisProps} />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload || payload.length === 0) return null
                    const point = payload[0]?.payload
                    if (!point) return null
                    const dateLabel = startDate ? String(label) : `Month ${label}`
                    return (
                      <div style={{
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        border: '1px solid #E5E7EB',
                        borderRadius: '6px',
                        fontSize: '12px',
                        padding: '8px 12px',
                      }}>
                        <p style={{ fontWeight: 600, marginBottom: 4 }}>{dateLabel}</p>
                        {perAccount!.map((acc, m) => {
                          const accColor = ACCOUNT_COLORS[m % ACCOUNT_COLORS.length]
                          return (
                            <div key={m} style={{ margin: '4px 0' }}>
                              <p style={{ color: accColor, fontWeight: 600, marginBottom: 2 }}>
                                {acc.display_name || acc.account}
                              </p>
                              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, marginLeft: 8 }}>
                                <span style={{ color: '#6B7280' }}>Median</span>
                                <span style={{ fontWeight: 500 }}>{formatCurrency(point[`acc${m}_p50`] as number)}</span>
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, marginLeft: 8 }}>
                                <span style={{ color: '#6B7280' }}>P25–P75</span>
                                <span>{formatCurrency(point[`acc${m}_p25`] as number)} – {formatCurrency(point[`acc${m}_p75`] as number)}</span>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    )
                  }}
                />
                {/* Fan chart per account */}
                {perAccount!.map((_, m) => {
                  const accColor = ACCOUNT_COLORS[m % ACCOUNT_COLORS.length]
                  const stackId = `fan_${m}`
                  return [
                    <Area key={`${m}_base`} type="monotone" dataKey={`acc${m}_base`} stackId={stackId} stroke="none" fill="transparent" />,
                    <Area key={`${m}_b1`} type="monotone" dataKey={`acc${m}_band_10_25`} stackId={stackId} stroke="none" fill={accColor} fillOpacity={0.08} name={`acc${m}_p10`} />,
                    <Area key={`${m}_b2`} type="monotone" dataKey={`acc${m}_band_25_50`} stackId={stackId} stroke="none" fill={accColor} fillOpacity={0.18} name={`acc${m}_p25`} />,
                    <Area key={`${m}_b3`} type="monotone" dataKey={`acc${m}_band_50_75`} stackId={stackId} stroke="none" fill={accColor} fillOpacity={0.18} name={`acc${m}_p75`} />,
                    <Area key={`${m}_b4`} type="monotone" dataKey={`acc${m}_band_75_90`} stackId={stackId} stroke="none" fill={accColor} fillOpacity={0.08} name={`acc${m}_p90`} />,
                  ]
                }).flat()}
                {/* Median lines on top */}
                {perAccount!.map((acc, m) => (
                  <Area
                    key={`acc${m}_median`}
                    type="monotone"
                    dataKey={`acc${m}_p50`}
                    stroke={ACCOUNT_COLORS[m % ACCOUNT_COLORS.length]}
                    strokeWidth={2}
                    fill="none"
                    dot={false}
                    name={acc.display_name || acc.account}
                  />
                ))}
                {goalRefLines}
                {withdrawalRefLines}
                <Legend content={() => null} />
              </AreaChart>
            )
          ) : chartStyle === 'percentiles' ? (
            <AreaChart
              data={percentileData}
              margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
              <XAxis {...xAxisProps} />
              <YAxis {...yAxisProps} />
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
              <Area type="monotone" dataKey="base" stackId="fan" stroke="none" fill="transparent" />
              <Area type="monotone" dataKey="band_10_25" stackId="fan" stroke="none" fill={color} fillOpacity={0.12} name="p10" />
              <Area type="monotone" dataKey="band_25_50" stackId="fan" stroke="none" fill={color} fillOpacity={0.25} name="p25" />
              <Area type="monotone" dataKey="band_50_75" stackId="fan" stroke="none" fill={color} fillOpacity={0.25} name="p75" />
              <Area type="monotone" dataKey="band_75_90" stackId="fan" stroke="none" fill={color} fillOpacity={0.12} name="p90" />

              {/* Median line */}
              <Area type="monotone" dataKey="p50" stroke={color} strokeWidth={2} fill="none" dot={false} name="p50" />
              {/* Mean line (dashed) */}
              <Area type="monotone" dataKey="mean" stroke={color} strokeWidth={1.5} strokeDasharray="5 5" fill="none" dot={false} name="mean" />

              {goalRefLines}
              {withdrawalRefLines}
              <Legend content={() => null} />
            </AreaChart>
          ) : (
            <LineChart
              data={mcData.data}
              margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
              <XAxis {...xAxisProps} />
              <YAxis {...yAxisProps} />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload || payload.length === 0) return null
                  const point = payload[0]?.payload
                  if (!point) return null
                  const dateLabel = startDate ? String(label) : `Month ${label}`
                  const stats = [
                    { label: 'P90', value: point.p90 },
                    { label: 'P75', value: point.p75 },
                    { label: 'Median (P50)', value: point.p50 },
                    { label: 'Mean', value: point.mean },
                    { label: 'P25', value: point.p25 },
                    { label: 'P10', value: point.p10 },
                  ]
                  return (
                    <div style={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #E5E7EB',
                      borderRadius: '6px',
                      fontSize: '12px',
                      padding: '8px 12px',
                    }}>
                      <p style={{ fontWeight: 600, marginBottom: 4 }}>{dateLabel}</p>
                      {stats.map(s => (
                        <p key={s.label} style={{ margin: '2px 0', display: 'flex', justifyContent: 'space-between', gap: 16 }}>
                          <span style={{ color: '#6B7280' }}>{s.label}</span>
                          <span style={{ fontWeight: 500 }}>{formatCurrency(s.value as number)}</span>
                        </p>
                      ))}
                    </div>
                  )
                }}
              />

              {/* Individual trajectories */}
              {Array.from({ length: mcData.nTrajectories }, (_, i) => (
                <Line
                  key={`t${i}`}
                  type="monotone"
                  dataKey={`t${i}`}
                  stroke={color}
                  strokeWidth={0.8}
                  strokeOpacity={0.12}
                  dot={false}
                  activeDot={false}
                  isAnimationActive={false}
                  name={`t${i}`}
                />
              ))}

              {/* Median line on top */}
              <Line
                type="monotone"
                dataKey="p50"
                stroke={color}
                strokeWidth={2.5}
                dot={false}
                name="p50"
                isAnimationActive={false}
              />

              {goalRefLines}
              {withdrawalRefLines}
              <Legend content={() => null} />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Monte Carlo info badge */}
      {chartStyle === 'montecarlo' && hasTrajectories && (
        <p className="text-center text-xs text-gray-400">
          Showing {trajectories!.n_sampled} of {trajectories!.n_total} simulated trajectories
        </p>
      )}

      {/* Final Wealth Stats — total / per_account percentile */}
      {viewMode !== 'all_accounts' && activeData.length > 0 && chartStyle === 'percentiles' && (
        <div className="grid grid-cols-5 gap-3 rounded-lg bg-gray-50 p-4 text-center">
          {[
            { label: 'P10 (Pessimistic)', key: 'p10' },
            { label: 'P25', key: 'p25' },
            { label: 'Median', key: 'p50' },
            { label: 'P75', key: 'p75' },
            { label: 'P90 (Optimistic)', key: 'p90' },
          ].map(({ label, key }) => {
            const lastPoint = percentileData[percentileData.length - 1]
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

      {/* Final Wealth Stats — total / per_account Monte Carlo */}
      {viewMode !== 'all_accounts' && activeData.length > 0 && chartStyle === 'montecarlo' && mcData.nTrajectories > 0 && (
        <div className="grid grid-cols-3 gap-3 rounded-lg bg-gray-50 p-4 text-center">
          {(() => {
            const lastValues = Array.from({ length: mcData.nTrajectories }, (_, i) => {
              const last = mcData.data[mcData.data.length - 1]
              return (last?.[`t${i}`] as number) ?? 0
            }).sort((a, b) => a - b)
            const min = lastValues[0]
            const max = lastValues[lastValues.length - 1]
            const median = lastValues[Math.floor(lastValues.length / 2)]
            return [
              { label: 'Min (sampled)', value: min },
              { label: 'Median', value: median },
              { label: 'Max (sampled)', value: max },
            ].map(({ label, value }) => (
              <div key={label}>
                <p className="text-xs text-gray-500">{label}</p>
                <p className="text-sm font-semibold text-gray-900">{formatCurrency(value)}</p>
              </div>
            ))
          })()}
        </div>
      )}

      {/* Final Wealth Stats — all_accounts */}
      {viewMode === 'all_accounts' && hasPerAccount && (
        <div className="grid gap-3 rounded-lg bg-gray-50 p-4" style={{ gridTemplateColumns: `repeat(${perAccount!.length}, 1fr)` }}>
          {perAccount!.map((acc, m) => {
            const accColor = ACCOUNT_COLORS[m % ACCOUNT_COLORS.length]
            const lastT = acc.mean.length - 1
            return (
              <div key={m} className="text-center">
                <p className="text-xs font-medium" style={{ color: accColor }}>
                  {acc.display_name || acc.account}
                </p>
                <p className="mt-1 text-sm font-semibold text-gray-900">
                  {formatCurrency(acc.p50[lastT])}
                </p>
                <p className="text-xs text-gray-400">
                  {formatCurrency(acc.p25[lastT])} – {formatCurrency(acc.p75[lastT])}
                </p>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
