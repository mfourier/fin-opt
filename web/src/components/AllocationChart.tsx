import { useMemo } from 'react'
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

interface AllocationChartProps {
  allocation: number[][]
  accountNames: string[]
  startDate?: string
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

export default function AllocationChart({ allocation, accountNames, startDate }: AllocationChartProps) {
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

      // Add allocation percentages for each account
      accountNames.forEach((name, m) => {
        point[name] = (allocation[t]?.[m] ?? 0) * 100
      })

      data.push(point)
    }

    return data
  }, [allocation, accountNames, startDate])

  if (!allocation || allocation.length === 0) {
    return <p className="text-gray-500">No allocation data available</p>
  }

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={chartData}
          stackOffset="expand"
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
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
            tickFormatter={(value) => `${Math.round(value * 100)}%`}
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: '#E5E7EB' }}
            domain={[0, 1]}
          />
          <Tooltip
            formatter={(value: number) => [`${value.toFixed(1)}%`, '']}
            labelFormatter={(label) => startDate ? label : `Month ${label}`}
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
  )
}
